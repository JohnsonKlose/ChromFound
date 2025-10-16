import argparse
import logging
import os

import numpy as np
import scanpy as sc
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset_ds import DatasetMultiPad
from src.models.chromfd_mixer import PretrainModelMambaLM
from src.utils.model_utils import ModelUtils

logging.basicConfig(level=logging.INFO)


def load_data(file_path):
    if file_path.endswith('.h5ad'):
        logging.info(f"Reading h5ad file from {file_path}")
        adata = sc.read_h5ad(file_path)
        return adata
    else:
        raise ValueError("Unsupported file format. Please provide a .h5ad file.")


class EmbeddingModel(PretrainModelMambaLM):
    def __init__(self, **kwargs):
        super(EmbeddingModel, self).__init__(**kwargs)

    def forward(self, value, chromosome, hg38_start, hg38_end, **kwargs):
        x = self.embedding(value, chromosome.long(), hg38_start.long(), hg38_end.long())
        x = self.backbone(x)
        return x


def generate_embeddings(model, adata_dataloader, device):
    model.eval()
    feature_embeddings = []

    with torch.no_grad():
        for batch_data in tqdm(adata_dataloader):
            value, chromosome, pos_start, pos_end, cell_type = batch_data
            value = value.to(device)
            chromosome = chromosome.to(device)
            pos_start = pos_start.to(device)
            pos_end = pos_end.to(device)

            embedding = model(value, chromosome, pos_start, pos_end)
            embedding = embedding.mean(axis=-1)
            embedding = embedding.detach().cpu().numpy()
            feature_embeddings.append(embedding)

            del value, chromosome, pos_start, pos_end, embedding
            torch.cuda.empty_cache()
    feature_embeddings = np.concatenate(feature_embeddings, axis=0)
    return feature_embeddings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, help='GPU rank', default=0)
    parser.add_argument('--data_path', type=str, required=True, help='path of h5ad file')
    parser.add_argument('--pretrain_checkpoint_path', type=str, required=True, help='path to pre-trained model')
    parser.add_argument('--pretrain_model_file', type=str, required=True, help='file name of pre-trained model')
    parser.add_argument('--pretrain_config_file', type=str, required=True, help='file name of pre-trained config')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--cell_type_col', required=True, help='column name of cell type')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the updated h5ad file')
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)

    pretrain_path = args.pretrain_checkpoint_path
    pretrain_config_file = args.pretrain_config_file
    with open(os.path.join(pretrain_path, pretrain_config_file), 'r') as file:
        pretrain_config = yaml.safe_load(file)
    pretrain_model_args = pretrain_config['model_args']
    pretrain_data_args = pretrain_config["data_args"]
    chromosome_vocab = ModelUtils.get_chromosome_vocab(os.path.join(pretrain_path, "chromosome_vocab.yaml"))
    pretrain_data_args["chromosome_vocab"] = chromosome_vocab
    adata = load_data(args.data_path)
    max_length = adata.shape[1]
    cell_type = list(set(adata.obs['celltype'].unique().tolist()))
    cell_type_map = {cell_type: idx for idx, cell_type in enumerate(sorted(cell_type))}

    pretrain_data_args['cell_type_map'] = cell_type_map
    pretrain_model_args["cell_type_num"] = len(cell_type_map)
    pretrain_data_args['cell_type_col'] = args.cell_type_col
    pretrain_data_args["feature_num"] = adata.shape[1]
    pretrain_model_args["feature_num"] = adata.shape[1]
    pretrain_model_args["batch_size"] = args.batch_size
    pretrain_data_args["max_length"] = max_length
    pretrain_model_args["max_length"] = max_length
    pretrain_model_args["device"] = device
    pretrain_model_args["add_cls"] = pretrain_data_args["add_cls"]
    pretrain_model_args["mask_ratio"] = 0.0
    pretrain_data_args["return_batch_label"] = False

    model = EmbeddingModel(**pretrain_model_args)
    state_dict = torch.load(str(os.path.join(pretrain_path, args.pretrain_model_file)))
    model.load_state_dict(state_dict['module'])
    model = model.to(device)
    adataset = DatasetMultiPad(*[adata], **pretrain_data_args)
    adata_dataloader = DataLoader(
        adataset, batch_size=args.batch_size, shuffle=False, pin_memory=True
    )

    embeddings = generate_embeddings(model, adata_dataloader, device)
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
        logging.info(f"make directory {args.output_path}")
    adata.obsm['X_embedding'] = embeddings
    adata.write_h5ad(os.path.join(args.output_path, "embeddings.h5ad"))
    logging.info(f"Embeddings shape: {adata.obsm['X_embedding'].shape} saved to {args.output_path}")


if __name__ == '__main__':
    main()
