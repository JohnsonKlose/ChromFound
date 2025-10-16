import math
from dataclasses import dataclass, field
from functools import partial

import torch
import torch.nn as nn
from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.modules.mha import MHA
from mamba_ssm.modules.mlp import GatedMLP

from chromfd_block import Block
from embedding_model import PretrainEmbeddingSimple

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


@dataclass
class MambaConfig:
    d_model: int = 2560
    d_intermediate: int = 0
    n_layer: int = 64
    attn_layer_idx: list = field(default_factory=list)
    attn_cfg: dict = field(default_factory=dict)
    rms_norm: bool = True
    residual_in_fp32: bool = True
    fused_add_norm: bool = True
    tie_embeddings: bool = True


def create_block(
    d_model,
    seq_length,
    wpsa_heads,
    wpsa_window_size,
    shift_size,
    d_intermediate,
    attn_layer_idx=None,
    attn_cfg=None,
    norm_epsilon=1e-5,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
):
    if attn_layer_idx is None:
        attn_layer_idx = []
    if attn_cfg is None:
        attn_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    if layer_idx not in attn_layer_idx:
        mixer_cls = partial(
            Mamba,
            layer_idx=layer_idx,
            **factory_kwargs
        )
    else:
        mixer_cls = partial(MHA, layer_idx=layer_idx, **attn_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    if d_intermediate == 0:
        mlp_cls = nn.Identity
    else:
        mlp_cls = partial(
            GatedMLP, hidden_features=d_intermediate, out_features=d_model, **factory_kwargs
        )

    block = Block(
        d_model,
        mixer_cls,
        mlp_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
        wpsa_window_size=wpsa_window_size,
        shift_size=shift_size,
        wpsa_heads=wpsa_heads,
        seq_length=seq_length,
    )
    block.layer_idx = layer_idx
    return block


def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


class MaskTokenPrediction(torch.nn.Module):
    def __init__(self, **kwargs):
        super(MaskTokenPrediction, self).__init__()
        self.model_args = kwargs
        if self.model_args.get("linear_embedding", False):
            self.projection = torch.nn.Linear(self.model_args.get("encoder_dim"), 1)
        else:
            self.projection = torch.nn.Linear(self.model_args.get("encoder_dim"), self.model_args.get("value_size"))
            self.softmax = torch.nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.projection(x)


class MambaMixer(nn.Module):
    def __init__(
        self,
        d_model: int,
        seq_length: int,
        wpsa_heads: int,
        wpsa_window_size: int,
        shift_size: int,
        n_layer: int,
        d_intermediate: int,
        attn_layer_idx=None,
        attn_cfg=None,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = False,
        initializer_cfg=None,
        fused_add_norm=False,
        residual_in_fp32=False,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32

        # We change the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    seq_length,
                    wpsa_heads=wpsa_heads,
                    wpsa_window_size=wpsa_window_size,
                    shift_size=shift_size,
                    d_intermediate=d_intermediate,
                    attn_layer_idx=attn_layer_idx,
                    attn_cfg=attn_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
                n_residuals_per_layer=1 if d_intermediate == 0 else 2,  # 2 if we have MLP
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward(self, inputs, inference_params=None, **mixer_kwargs):
        hidden_states = inputs
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params, **mixer_kwargs
            )
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            hidden_states = layer_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
                is_rms_norm=isinstance(self.norm_f, RMSNorm)
            )
        return hidden_states


class PretrainModelMambaLM(torch.nn.Module):
    def __init__(self, **kwargs):
        super(PretrainModelMambaLM, self).__init__()
        self.model_args = kwargs
        factory_kwargs = {"device": self.model_args["device"], "dtype": torch.float32}
        self.embedding = PretrainEmbeddingSimple(
            embedding_dim=self.model_args.get("embedding_dim"),
            chromosome_size=self.model_args.get("chromosome_size"),
            embedding_dropout=self.model_args.get("embedding_dropout", 0),
            positional_embedding_type=self.model_args.get("positional_embedding_type"),
            positional_temp=self.model_args['positional_temp'],
            batch_size=self.model_args.get("batch_size"),
            seq_length=self.model_args.get("max_length"),
            device=self.model_args["device"],
            chromatin_embedding=self.model_args.get("chromatin_embedding", True)
        )
        self.backbone = MambaMixer(
            d_model=self.model_args.get("embedding_dim"),
            seq_length=self.model_args.get("max_length"),
            wpsa_heads=self.model_args.get("wpsa_heads"),
            wpsa_window_size=self.model_args.get("wpsa_window_size"),
            shift_size=self.model_args.get("shift_size"),
            n_layer=self.model_args.get("encoder_layers"),
            d_intermediate=MambaConfig.d_intermediate,
            attn_layer_idx=None,
            attn_cfg=None,
            rms_norm=MambaConfig.rms_norm,
            initializer_cfg=None,
            fused_add_norm=MambaConfig.fused_add_norm,
            residual_in_fp32=MambaConfig.residual_in_fp32,
            **factory_kwargs,
        )
        self.mask_token_prediction = MaskTokenPrediction(**self.model_args)

    def forward(self, value, chromosome, hg38_start, hg38_end, key_padding_mask=None):
        x = self.embedding(value, chromosome.long(), hg38_start.long(), hg38_end.long())
        x = self.backbone(x)
        logits = self.mask_token_prediction(x)

        return logits
