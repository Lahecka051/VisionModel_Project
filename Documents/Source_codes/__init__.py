# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied and modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# ------------------------------------------------------------------------

# @changelog
# [v1.0.0] RF-DETR original
# [v2.0.0] 2026-03-10 - VIVID-Det: ssfa params in build_backbone()
# [v2.1.0] 2026-03-11 - VIVID-Det: tfcm params + Joiner temporal_mode

from typing import Callable, Dict, List

import torch
from torch import nn

from rfdetr.models.backbone.backbone import *
from rfdetr.models.position_encoding import build_position_encoding
from rfdetr.util.misc import NestedTensor


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self._export = False

    def forward(self, tensor_list: NestedTensor, temporal_mode: bool = False):
        """ """
        # [v2.1.0] temporal_mode passed to Backbone for TFCM
        x = self[0](tensor_list, temporal_mode=temporal_mode)
        pos = []
        for x_ in x:
            pos.append(self[1](x_, align_dim_orders=False).to(x_.tensors.dtype))
        return x, pos

    def export(self):
        self._export = True
        self._forward_origin = self.forward
        self.forward = self.forward_export
        for name, m in self.named_modules():
            if hasattr(m, "export") and isinstance(m.export, Callable) and hasattr(m, "_export") and not m._export:
                m.export()

    def forward_export(self, inputs: torch.Tensor):
        feats, masks = self[0](inputs)
        poss = []
        for feat, mask in zip(feats, masks):
            poss.append(self[1](mask, align_dim_orders=False).to(feat.dtype))
        return feats, None, poss


def build_backbone(
    encoder,
    vit_encoder_num_layers,
    pretrained_encoder,
    window_block_indexes,
    drop_path,
    out_channels,
    out_feature_indexes,
    projector_scale,
    use_cls_token,
    hidden_dim,
    position_embedding,
    freeze_encoder,
    layer_norm,
    target_shape,
    rms_norm,
    backbone_lora,
    force_no_pretrain,
    gradient_checkpointing,
    load_dinov2_weights,
    patch_size,
    num_windows,
    positional_encoding_size,
    # ================================================================
    # [v2.0.0] VIVID-Det SSFA parameters (all have defaults)
    ssfa_enabled=False,
    ssfa_vit_dim=384,
    ssfa_cnn_channels=256,
    ssfa_num_heads=6,
    ssfa_topk_ratio=0.25,
    ssfa_alpha_init=0.01,
    ssfa_sopm_loss_weight=0.5,
    ssfa_use_attn_prior=True,
    # ================================================================
    # [v2.1.0] VIVID-Det TFCM parameters (all have defaults)
    tfcm_enabled=False,
    tfcm_embed_dim=384,
    tfcm_n_ref=4,
    tfcm_tau_init=0.03,
    tfcm_alpha_init=-10.0,
    tfcm_tau_small=0.3,
    # ================================================================
):
    """
    Useful args:
        - encoder: encoder name
        - lr_encoder:
        - dilation
        - use_checkpoint: for swin only for now

    """
    position_embedding = build_position_encoding(hidden_dim, position_embedding)

    backbone = Backbone(
        encoder,
        pretrained_encoder,
        window_block_indexes=window_block_indexes,
        drop_path=drop_path,
        out_channels=out_channels,
        out_feature_indexes=out_feature_indexes,
        projector_scale=projector_scale,
        use_cls_token=use_cls_token,
        layer_norm=layer_norm,
        freeze_encoder=freeze_encoder,
        target_shape=target_shape,
        rms_norm=rms_norm,
        backbone_lora=backbone_lora,
        gradient_checkpointing=gradient_checkpointing,
        load_dinov2_weights=load_dinov2_weights,
        patch_size=patch_size,
        num_windows=num_windows,
        positional_encoding_size=positional_encoding_size,
        # ============================================================
        # [v2.0.0] VIVID-Det SSFA
        ssfa_enabled=ssfa_enabled,
        ssfa_vit_dim=ssfa_vit_dim,
        ssfa_cnn_channels=ssfa_cnn_channels,
        ssfa_num_heads=ssfa_num_heads,
        ssfa_topk_ratio=ssfa_topk_ratio,
        ssfa_alpha_init=ssfa_alpha_init,
        ssfa_sopm_loss_weight=ssfa_sopm_loss_weight,
        ssfa_use_attn_prior=ssfa_use_attn_prior,
        # ============================================================
        # [v2.1.0] VIVID-Det TFCM
        tfcm_enabled=tfcm_enabled,
        tfcm_embed_dim=tfcm_embed_dim,
        tfcm_n_ref=tfcm_n_ref,
        tfcm_tau_init=tfcm_tau_init,
        tfcm_alpha_init=tfcm_alpha_init,
        tfcm_tau_small=tfcm_tau_small,
        # ============================================================
    )

    model = Joiner(backbone, position_embedding)
    return model
