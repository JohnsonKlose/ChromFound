import argparse
import json
import os
import pickle
import random

import scanpy as sc
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from src.data.dataset_ds import DatasetMultiPad
from src.models.chromfd_mixer import PretrainModelMambaLM
from src.utils.model_utils import ModelUtils
from src.utils.tb_utils import setup_logging


def warmup_lambda(current_step, warmup_steps=1000):
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    return 1.0


def load_data(file_path):
    if file_path.endswith('.h5ad'):
        print(f"Reading h5ad file from {file_path}")
        adata = sc.read_h5ad(file_path)
        return adata
    else:
        raise ValueError("Unsupported file format. Please provide a .h5ad file.")


def init_weight(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        torch.nn.init.zeros_(m.bias)


class FocalLoss(torch.nn.Module):
    """
    Focal Loss as described in https://arxiv.org/abs/1708.02002
    """
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha          # Balance factor
        self.gamma = gamma          # Modulating factor
        self.reduction = reduction  # Reduction method: 'mean', 'sum', 'none'

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # Probabilities of the predicted classes
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class FinetuneModelMambaCellType(PretrainModelMambaLM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        max_length = self.model_args["max_length"]
        self.post_backbone_dropout = torch.nn.Dropout(p=0.3)
        self.feature_projection = torch.nn.Sequential(
            torch.nn.Linear(self.model_args["embedding_dim"], 256),
            torch.nn.GELU(),
            torch.nn.Dropout(p=0.3),
            torch.nn.Linear(256, 1),
            torch.nn.GELU()
        )
        self.feature_projection.apply(init_weight)
        in_feature = max_length
        self.ft_cell_type_projection = torch.nn.Sequential(
            torch.nn.Linear(in_feature, 1024),
            torch.nn.GELU(),
            torch.nn.Dropout(p=0.3),
            torch.nn.Linear(1024, 512),
            torch.nn.GELU(),
            torch.nn.Dropout(p=0.3),
            torch.nn.Linear(512, 128),
            torch.nn.GELU(),
            torch.nn.Dropout(p=0.3),
            torch.nn.Linear(128, self.model_args["cell_type_num"])
        )
        self.ft_cell_type_projection.apply(init_weight)

        for name, param in self.mask_token_prediction.named_parameters():
            param.requires_grad = False

    def forward(self, value, chromosome, hg38_start, hg38_end, **kwargs):
        x = self.embedding(value, chromosome.long(), hg38_start.long(), hg38_end.long())
        x = self.backbone(x)
        x = self.feature_projection(x)
        x = torch.squeeze(x, dim=-1)
        x = self.post_backbone_dropout(x)
        x_cell_type_prediction = self.ft_cell_type_projection(x)
        return x_cell_type_prediction


def evaluate_finetune_model(model, val_dataloader, criterion, device):
    model.eval()
    eval_loss = 0
    accuracy = 0
    data_shape = 0
    eval_steps = 0
    eval_f1_score = 0
    cell_type_label_list = []
    cell_type_pred_list = []
    with torch.no_grad():
        for val_batch in val_dataloader:
            value, chromosome, pos_start, pos_end, cell_type = val_batch
            value = value.to(device)
            chromosome = chromosome.to(device)
            cell_type_gpu = cell_type.to(device)
            pos_start = pos_start.to(device)
            pos_end = pos_end.to(device)
            cell_type_output = model(value, chromosome, pos_start, pos_end)
            tmp_loss_cell_type_prediction = criterion(cell_type_output, cell_type_gpu)
            cell_type_pred = torch.argmax(cell_type_output, dim=-1)

            cell_type_label_list.extend(cell_type.detach().cpu().numpy().tolist())
            cell_type_pred_list.extend(cell_type_pred.detach().cpu().numpy().tolist())

            tmp_f1_score = f1_score(cell_type, cell_type_pred.cpu().numpy(), average='macro')
            eval_f1_score += tmp_f1_score
            accuracy += torch.sum(cell_type_pred == cell_type_gpu).item()
            data_shape += cell_type.size(0)
            eval_loss += tmp_loss_cell_type_prediction.item()
            eval_steps += 1
        eval_loss = eval_loss / eval_steps

    eval_loss_tensor = torch.tensor(eval_loss).to(device)
    eval_loss = eval_loss_tensor.item()

    eval_f1_score = f1_score(cell_type_label_list, cell_type_pred_list, average='macro')
    eval_f1_score_tensor = torch.tensor(eval_f1_score).to(device)
    eval_f1_score = eval_f1_score_tensor.item()

    accuracy = accuracy_score(cell_type_label_list, cell_type_pred_list)
    accuracy_tensor = torch.tensor(accuracy).to(device)
    accuracy = accuracy_tensor.item()

    return eval_loss, eval_f1_score, accuracy, cell_type_label_list, cell_type_pred_list


def cell_type_finetune(
        model,
        finetune_args,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        optimizer,
        lr_scheduler,
        device,
        logger
):
    model = model.to(device)
    cell_type_criterion = FocalLoss(alpha=1, gamma=2, reduction='mean')
    step = 0
    best_f1_score = 0.0
    for eph in range(finetune_args.get("epoch")):
        for batch in train_dataloader:
            model.train()
            value, chromosome, pos_start, pos_end, cell_type = batch
            value = value.to(device)
            chromosome = chromosome.to(device)
            pos_start = pos_start.to(device)
            pos_end = pos_end.to(device)
            cell_type = cell_type.to(device)
            cell_type_output = model(value, chromosome, pos_start, pos_end)
            # Compute Focal Loss
            loss_cell_type_prediction = cell_type_criterion(cell_type_output, cell_type)
            loss_cell_type_prediction.backward()
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()
            if step % finetune_args.get("loss_evaluate", 10) == 0:
                accuracy = torch.sum(torch.argmax(cell_type_output, dim=-1) == cell_type).item() / cell_type.size(0)
                logger.info(
                    f"[Train] loss at epoch {eph} step {step}: {loss_cell_type_prediction.item()}, "
                    f"accuracy: {accuracy:.4f}, lr: {optimizer.param_groups[0]['lr']}"
                )
            if step % finetune_args.get("val_evaluate", 10) == 0:
                eval_loss, eval_f1_score, eval_accuracy, eval_cell_type_label_list, eval_cell_type_pred_list = \
                    evaluate_finetune_model(model, val_dataloader, cell_type_criterion, device)
                test_loss, test_f1_score, test_accuracy, eval_cell_type_label_list, eval_cell_type_pred_list = \
                    evaluate_finetune_model(model, test_dataloader, cell_type_criterion, device)
                logger.info(
                    f"[Evaluate] loss at epoch {eph} step {step}: {eval_loss}, "
                    f"cell type accuracy: {eval_accuracy:.4f}, f1 score: {eval_f1_score:.4f}, "
                    f"lr: {optimizer.param_groups[0]['lr']:.6f}"
                )
                logger.info(
                    f"[Test] loss at epoch {eph} step {step}: {test_loss}, "
                    f"cell type accuracy: {test_accuracy:.4f}, f1 score: {test_f1_score:.4f}, "
                    f"lr: {optimizer.param_groups[0]['lr']:.6f}"
                )
                if eval_f1_score > best_f1_score:
                    best_f1_score = eval_f1_score
                    with open(os.path.join(
                            finetune_args["log_path"], f"cell_type_label_pred.pkl"), "wb") as f:
                        pickle.dump((eval_cell_type_label_list, eval_cell_type_pred_list), f)
                    logger.info(
                        f"[Evaluate] best validation f1_score: {best_f1_score:.4f} at epoch {eph} step {step}, "
                        f"test accuracy: {test_accuracy:.4f}, f1_score: {test_f1_score:.4f}"
                    )
                    torch.save(model.state_dict(), os.path.join(finetune_args["log_path"], "best_model.pt"))
            step += 1
        torch.save(model.state_dict(), os.path.join(finetune_args["log_path"], f"epoch_{eph}.pt"))


def main_finetune():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, help='local rank passed from distributed launcher', default=0)
    parser.add_argument("--batch_size", type=int, default=16, help="batch size for training")
    parser.add_argument("--learning_rate", type=float, required=True, help="learning rate for finetune")
    parser.add_argument("--pretrain_checkpoint_path", type=str, required=True, help="path to pretrain checkpoint")
    parser.add_argument("--pretrain_model_file", type=str, required=True, help="file name of pre-trained model")
    parser.add_argument('--pretrain_config_file', type=str, required=True, help='file name of pre-trained config')
    parser.add_argument("--cell_type_col", type=str, required=True, help="cell type column name")
    parser.add_argument("--epoch", type=int, required=True, help="epoch for training")
    parser.add_argument("--train_file_path", type=str, required=True, help="train file path")
    parser.add_argument("--test_file_path", type=str, required=True, help="validation file path")
    parser.add_argument("--log_path", type=str, required=True, help="log path")
    parser.add_argument("--load_pretrain_ckpt", action="store_true", default=True, help="load pre-trained model")
    args = parser.parse_args()

    with open(os.path.join(args.pretrain_checkpoint_path, args.pretrain_config_file), 'r') as file:
        pretrain_config = yaml.safe_load(file)

    device = torch.device(f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    pretrain_data_args = pretrain_config["data_args"]
    pretrain_model_args = pretrain_config["model_args"]
    log_path = args.log_path
    chromosome_vocab = ModelUtils.get_chromosome_vocab(
        os.path.join(args.pretrain_checkpoint_path, "chromosome_vocab.yaml")
    )
    pretrain_data_args["chromosome_vocab"] = chromosome_vocab
    adata_train_val = load_data(args.train_file_path)
    adata_test = load_data(args.test_file_path)

    adata_train_val.obs["tag"] = "train"
    adata_test.obs["tag"] = "test"
    adata_concat = sc.AnnData.concatenate(adata_train_val, adata_test)
    adata_train_val = adata_concat[adata_concat.obs["tag"] == "train"]
    adata_test = adata_concat[adata_concat.obs["tag"] == "test"]
    max_length = adata_concat.shape[1]

    cell_type = list(set(adata_train_val.obs[args.cell_type_col].unique().tolist() + adata_test.obs[
        args.cell_type_col].unique().tolist()))
    cell_type_map = {cell_type: idx for idx, cell_type in enumerate(sorted(cell_type))}

    if not os.path.exists(log_path):
        os.mkdir(log_path)
    os.system(f"cp {os.path.join(args.pretrain_checkpoint_path, args.pretrain_config_file)} {log_path}")
    os.system(f"cp {os.path.join(args.pretrain_checkpoint_path, 'chromosome_vocab.yaml')} {log_path}")

    log_file_path = os.path.join(log_path, "finetune.log")

    finetune_logger = setup_logging(log_file_path)

    finetune_logger.info('PretrainLogger is configured and ready.')
    finetune_logger.info(f"args from parser: {args}")
    finetune_logger.info(f"max length for cell type finetune: {max_length}")

    with open(os.path.join(log_path, "cell_type_map.json"), "w") as f:
        json.dump(cell_type_map, f)

    pretrain_data_args['cell_type_map'] = cell_type_map
    pretrain_model_args["cell_type_num"] = len(cell_type_map)
    pretrain_data_args['cell_type_col'] = args.cell_type_col
    pretrain_data_args["feature_num"] = adata_train_val.shape[1]
    pretrain_model_args["feature_num"] = adata_train_val.shape[1]
    pretrain_model_args["batch_size"] = args.batch_size
    pretrain_data_args["max_length"] = max_length
    pretrain_model_args["max_length"] = max_length
    pretrain_model_args["device"] = device
    pretrain_model_args["mask_ratio"] = 0.0
    pretrain_data_args["return_batch_label"] = False

    idx_list = [i for i in range(adata_train_val.X.shape[0])]
    random.shuffle(idx_list)
    split_idx = int(len(idx_list) * 0.9)
    train_idx = idx_list[:split_idx]
    val_idx = idx_list[split_idx:]
    adata_train = adata_train_val[train_idx]
    adata_val = adata_train_val[val_idx]

    train_dataset = DatasetMultiPad(*[adata_train], **pretrain_data_args)
    val_dataset = DatasetMultiPad(*[adata_val], **pretrain_data_args)
    test_dataset = DatasetMultiPad(*[adata_test], **pretrain_data_args)
    # Print dataset lengths
    print(f"Train Dataset Length: {len(train_dataset)}")
    print(f"Validation Dataset Length: {len(val_dataset)}")
    print(f"Test Dataset Length: {len(test_dataset)}")

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True
    )

    model = FinetuneModelMambaCellType(**pretrain_model_args)
    model = model.to(device)
    finetune_logger.info(f'Model parameters: {model}')
    optimizer_params = {
        "lr": args.learning_rate,
        "betas": (0.8, 0.999),
        "eps": 1e-8,
        "weight_decay": 1e-6
    }
    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_params)
    lr_scheduler = LambdaLR(optimizer, lr_lambda=lambda step: warmup_lambda(step, 200))
    if args.load_pretrain_ckpt:
        state_dict = torch.load(str(os.path.join(args.pretrain_checkpoint_path, args.pretrain_model_file)))
        missing_keys, unexpected_keys = model.load_state_dict(state_dict['module'], strict=False)
        if missing_keys:
            print("Missing keys (not found in checkpoint):")
            for key in missing_keys:
                print(f"  {key}")
        if unexpected_keys:
            print("Unexpected keys (found in checkpoint but not in model):")
            for key in unexpected_keys:
                print(f"  {key}")

    finetune_config = {
        "pretrain_checkpoint_path": args.pretrain_checkpoint_path,
        "pretrain_model_name": args.pretrain_model_file,
        "pretrain_config_file": args.pretrain_config_file,
        "loss_evaluate": 20,
        "val_evaluate": 20,
        "log_path": log_path,
        "epoch": args.epoch,
    }
    cell_type_finetune(
        model,
        finetune_config,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        optimizer,
        lr_scheduler,
        device,
        finetune_logger
    )


if __name__ == '__main__':
    main_finetune()
