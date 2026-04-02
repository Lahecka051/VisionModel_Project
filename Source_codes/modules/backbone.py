# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified for SOD-DETR: SCA (Selective Cross-Attention) integration
# ------------------------------------------------------------------------
# Copied and modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Conditional DETR (https://github.com/Atten4Vis/ConditionalDETR)
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

"""
Backbone modules.
"""

import torch
import torch.nn.functional as F
from peft import PeftModel

from rfdetr.models.backbone.base import BackboneBase
from rfdetr.models.backbone.dinov2 import DinoV2
from rfdetr.models.backbone.projector import MultiScaleProjector
from rfdetr.util.logger import get_logger
from rfdetr.util.misc import NestedTensor

from rfdetr.models.sca import SCA

logger = get_logger()

__all__ = ["Backbone"]


class Backbone(BackboneBase):
    """backbone."""

    def __init__(
        self,
        name: str,
        pretrained_encoder: str = None,
        window_block_indexes: list = None,
        drop_path=0.0,
        out_channels=256,
        out_feature_indexes: list = None,
        projector_scale: list = None,
        use_cls_token: bool = False,
        freeze_encoder: bool = False,
        layer_norm: bool = False,
        target_shape: tuple[int, int] = (640, 640),
        rms_norm: bool = False,
        backbone_lora: bool = False,
        gradient_checkpointing: bool = False,
        load_dinov2_weights: bool = True,
        patch_size: int = 14,
        num_windows: int = 4,
        positional_encoding_size: int = 0,
        # SCA parameters
        sca_enabled: bool = False,
        sca_vit_dim: int = 384,
        sca_cnn_channels: int = 256,
        sca_num_heads: int = 6,
        sca_topk_ratio: float = 0.25,
        sca_alpha_init: float = 0.01,
        sca_sopm_loss_weight: float = 0.5,
        sca_use_attn_prior: bool = True,
    ):
        super().__init__()
        name_parts = name.split("_")
        assert name_parts[0] == "dinov2"
        use_registers = False
        if "registers" in name_parts:
            use_registers = True
            name_parts.remove("registers")
        use_windowed_attn = False
        if "windowed" in name_parts:
            use_windowed_attn = True
            name_parts.remove("windowed")
        assert len(name_parts) == 2, (
            "name should be dinov2, then either registers, windowed, both, or none, then the size"
        )
        self.encoder = DinoV2(
            size=name_parts[-1],
            out_feature_indexes=out_feature_indexes,
            shape=target_shape,
            use_registers=use_registers,
            use_windowed_attn=use_windowed_attn,
            gradient_checkpointing=gradient_checkpointing,
            load_dinov2_weights=load_dinov2_weights,
            patch_size=patch_size,
            num_windows=num_windows,
            positional_encoding_size=positional_encoding_size,
            drop_path_rate=drop_path,
        )
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.projector_scale = projector_scale
        assert len(self.projector_scale) > 0
        assert sorted(self.projector_scale) == self.projector_scale, (
            "only support projector scale P3/P4/P5/P6 in ascending order."
        )
        level2scalefactor = dict(P3=2.0, P4=1.0, P5=0.5, P6=0.25)
        scale_factors = [level2scalefactor[lvl] for lvl in self.projector_scale]

        self.projector = MultiScaleProjector(
            in_channels=self.encoder._out_feature_channels,
            out_channels=out_channels,
            scale_factors=scale_factors,
            layer_norm=layer_norm,
            rms_norm=rms_norm,
        )

        self._export = False

        # SCA module
        self.sca_enabled = sca_enabled
        self._sca_loss_cache = {}

        if self.sca_enabled:
            self.sca = SCA(
                vit_dim=sca_vit_dim,
                cnn_channels=sca_cnn_channels,
                num_heads=sca_num_heads,
                topk_ratio=sca_topk_ratio,
                alpha_init=sca_alpha_init,
                sopm_loss_weight=sca_sopm_loss_weight,
                use_attn_prior=sca_use_attn_prior,
            )
            self.sca.register_attn_hook(self.encoder, num_heads=sca_num_heads)
        else:
            self.sca = None

    def export(self):
        self._export = True
        self._forward_origin = self.forward
        self.forward = self.forward_export

        if isinstance(self.encoder, PeftModel):
            logger.info("Merging and unloading LoRA weights")
            self.encoder.merge_and_unload()

    def forward(self, tensor_list: NestedTensor):
        """ """
        feats = self.encoder(tensor_list.tensors)

        if self.sca_enabled and self.sca is not None:
            feats, _, sca_losses = self.sca(
                feats=feats,
                images=tensor_list.tensors,
                targets=None,
            )
            self._sca_loss_cache = sca_losses
        else:
            self._sca_loss_cache = {}

        feats = self.projector(feats)
        out = []
        for feat in feats:
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=feat.shape[-2:]).to(torch.bool)[0]
            out.append(NestedTensor(feat, mask))
        return out

    def forward_export(self, tensors: torch.Tensor):
        feats = self.encoder(tensors)

        if self.sca_enabled and self.sca is not None:
            feats, _, _ = self.sca(
                feats=feats,
                images=tensors,
                targets=None,
            )

        feats = self.projector(feats)
        out_feats = []
        out_masks = []
        for feat in feats:
            b, _, h, w = feat.shape
            out_masks.append(torch.zeros((b, h, w), dtype=torch.bool, device=feat.device))
            out_feats.append(feat)
        return out_feats, out_masks

    @property
    def sca_losses(self):
        return self._sca_loss_cache

    def get_named_param_lr_pairs(self, args, prefix: str = "backbone.0"):
        num_layers = args.out_feature_indexes[-1] + 1
        backbone_key = "backbone.0.encoder"
        named_param_lr_pairs = {}
        for n, p in self.named_parameters():
            n = prefix + "." + n
            if backbone_key in n and p.requires_grad:
                lr = (
                    args.lr_encoder
                    * get_dinov2_lr_decay_rate(
                        n,
                        lr_decay_rate=args.lr_vit_layer_decay,
                        num_layers=num_layers,
                    )
                    * args.lr_component_decay**2
                )
                wd = args.weight_decay * get_dinov2_weight_decay_rate(n)
                named_param_lr_pairs[n] = {
                    "params": p,
                    "lr": lr,
                    "weight_decay": wd,
                }
            elif "sca" in n and p.requires_grad:
                if "cnn_branch" in n:
                    lr = getattr(args, 'lr_sca_cnn', 1e-3)
                else:
                    lr = getattr(args, 'lr_sca', 1e-4)
                wd = args.weight_decay
                if "bias" in n or "norm" in n or "alpha" in n:
                    wd = 0.0
                named_param_lr_pairs[n] = {
                    "params": p,
                    "lr": lr,
                    "weight_decay": wd,
                }

        return named_param_lr_pairs


def get_dinov2_lr_decay_rate(name, lr_decay_rate=1.0, num_layers=12):
    """
    Calculate lr decay rate for different ViT blocks.
    """
    layer_id = num_layers + 1
    if name.startswith("backbone"):
        if "embeddings" in name:
            layer_id = 0
        elif ".layer." in name and ".residual." not in name:
            layer_id = int(name[name.find(".layer.") :].split(".")[2]) + 1
    return lr_decay_rate ** (num_layers + 1 - layer_id)


def get_dinov2_weight_decay_rate(name, weight_decay_rate=1.0):
    if (
        ("gamma" in name)
        or ("pos_embed" in name)
        or ("rel_pos" in name)
        or ("bias" in name)
        or ("norm" in name)
        or ("embeddings" in name)
    ):
        weight_decay_rate = 0.0
    return weight_decay_rate
