# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
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

@changelog
[v1.0.0] RF-DETR original
[v2.0.0] 2026-03-10 - SOD SCA integration
  - __init__: SCA module init + attention hook registration
  - forward: SCA inserted between encoder and projector
  - _heatmap_cache: cached small object heatmap for SAQG/TFCM access
  - get_named_param_lr_pairs: SCA parameter groups added
[v2.1.0] 2026-03-11 - TFCM integration
  - __init__: TFCM module init (embed_dim=384, n_ref=4, tau=0.03, alpha=-10.0)
  - forward: TFCM inserted between encoder and SCA + temporal_mode param
  - forward_export: TFCM with temporal_mode=True for video inference
  - Heatmap cache update: SCA → tfcm.update_sopm_cache(heatmap)
  - get_named_param_lr_pairs: TFCM parameter groups (lr=2e-4, wd=0)
"""

import torch
import torch.nn.functional as F
from peft import PeftModel

from rfdetr.models.backbone.base import BackboneBase
from rfdetr.models.backbone.dinov2 import DinoV2
from rfdetr.models.backbone.projector import MultiScaleProjector
from rfdetr.util.logger import get_logger
from rfdetr.util.misc import NestedTensor

# ====================================================================
# SOD: SCA (Selective Cross-Attention) import
# TFCM import
# ====================================================================
from rfdetr.models.sca import SCA
from rfdetr.models.tfcm import TFCM
# ====================================================================

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
        # ============================================================
        # SOD SCA (Selective Cross-Attention) parameters
        sca_enabled: bool = False,  # default False: opt-in
        sca_vit_dim: int = 384,
        sca_cnn_channels: int = 256,
        sca_num_heads: int = 6,
        sca_topk_ratio: float = 0.25,
        sca_alpha_init: float = 0.01,
        sca_heatmap_loss_weight: float = 0.5,
        sca_use_attn_prior: bool = True,
        # ============================================================
        # TFCM parameters
        tfcm_enabled: bool = False,  # default False: Stage 2 opt-in
        tfcm_embed_dim: int = 384,
        tfcm_n_ref: int = 4,
        tfcm_tau_init: float = 0.03,
        tfcm_alpha_init: float = -10.0,  # sigmoid(-10)≈0.0 identity
        tfcm_tau_small: float = 0.3,
        # ============================================================
    ):
        super().__init__()
        # an example name here would be "dinov2_base" or "dinov2_registers_windowed_base"
        # if "registers" is in the name, then use_registers is set to True, otherwise it is set to False
        # similarly, if "windowed" is in the name, then use_windowed_attn is set to True, otherwise it is set to False
        # the last part of the name should be the size
        # and the start should be dinov2
        name_parts = name.split("_")
        assert name_parts[0] == "dinov2"
        # name_parts[-1]
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
        # build encoder + projector as backbone module
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.projector_scale = projector_scale
        assert len(self.projector_scale) > 0
        # x[0]
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

        # ==============================================================
        # SOD: SCA (Selective Cross-Attention) module init
        #
        # SCA sits between encoder and projector (NAS-Safe gap).
        # Enriches feats[0] (Block4 output) with CNN high-res features.
        # feats[1:] (Block 6,9,12) are untouched -> P4/P5 preserved.
        #
        # Heatmap is cached in self._heatmap_cache for SAQG/TFCM access.
        # Heatmap FocalLoss is computed externally in SetCriterion
        # (targets not available in backbone forward).
        # ==============================================================
        self.sca_enabled = sca_enabled
        self._heatmap_cache = None
        self._sca_loss_cache = {}

        if self.sca_enabled:
            self.sca = SCA(
                vit_dim=sca_vit_dim,
                cnn_channels=sca_cnn_channels,
                num_heads=sca_num_heads,
                topk_ratio=sca_topk_ratio,
                alpha_init=sca_alpha_init,
                heatmap_loss_weight=sca_heatmap_loss_weight,
                use_attn_prior=sca_use_attn_prior,
            )
            # Register DINOv2 Block4 attention hook
            self.sca.register_attn_hook(self.encoder, num_heads=sca_num_heads)
        else:
            self.sca = None
        # ==============================================================

        # ==============================================================
        # TFCM module init
        #
        # TFCM sits between encoder and SCA (encoder → TFCM → SCA).
        # Video mode: cosine correspondence → Scale-Aware temporal fusion.
        # Image mode: complete bypass, cost 0%.
        # Stage 2 only: α=sigmoid(-10)≈0.0 → identity start.
        # ==============================================================
        self.tfcm_enabled = tfcm_enabled
        if self.tfcm_enabled:
            self.tfcm = TFCM(
                embed_dim=tfcm_embed_dim,
                n_ref=tfcm_n_ref,
                tau_init=tfcm_tau_init,
                alpha_init=tfcm_alpha_init,
                tau_small=tfcm_tau_small,
            )
        else:
            self.tfcm = None
        # ==============================================================

    def export(self):
        self._export = True
        self._forward_origin = self.forward
        self.forward = self.forward_export

        if isinstance(self.encoder, PeftModel):
            logger.info("Merging and unloading LoRA weights")
            self.encoder.merge_and_unload()

    def forward(self, tensor_list: NestedTensor, temporal_mode: bool = False):
        """ """
        # (H, W, B, C)
        feats = self.encoder(tensor_list.tensors)

        # ==============================================================
        # TFCM insertion (encoder → SCA gap)
        #
        # TFCM enriches feats[0] with temporal info from past frames.
        # Video mode: cosine correspondence → Scale-Aware fusion.
        # Image mode (temporal_mode=False): complete bypass.
        # feats[1:] pass through unchanged.
        # ==============================================================
        if self.tfcm_enabled and self.tfcm is not None:
            feats[0] = self.tfcm(feats[0], temporal_mode=temporal_mode)
        # ==============================================================

        # ==============================================================
        # SOD: SCA insertion (encoder -> projector gap)
        #
        # SCA enriches feats[0] with CNN high-res info via selective
        # cross-attention. feats[1:] pass through unchanged.
        #
        # targets=None here: Heatmap FocalLoss is computed in SetCriterion
        # using self._heatmap_cache. This avoids changing forward signature.
        #
        # Bypass: sca_enabled=False or alpha converges to 0
        #         -> mathematically equivalent to original
        # ==============================================================
        if self.sca_enabled and self.sca is not None:
            feats, heatmap, sca_losses = self.sca(
                feats=feats,
                images=tensor_list.tensors,
                targets=None,
            )
            self._heatmap_cache = heatmap
            self._sca_loss_cache = sca_losses
            # Heatmap 캐시 → 다음 프레임의 TFCM Scale-Aware 마스크용
            if self.tfcm_enabled and self.tfcm is not None:
                self.tfcm.update_sopm_cache(heatmap)
        else:
            self._heatmap_cache = None
            self._sca_loss_cache = {}
        # ==============================================================

        feats = self.projector(feats)
        # x: [(B, C, H, W)]
        out = []
        for feat in feats:
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=feat.shape[-2:]).to(torch.bool)[0]
            out.append(NestedTensor(feat, mask))
        return out

    def forward_export(self, tensors: torch.Tensor):
        feats = self.encoder(tensors)

        # TFCM in export mode (always temporal_mode=True for video)
        if self.tfcm_enabled and self.tfcm is not None:
            feats[0] = self.tfcm(feats[0], temporal_mode=True)

        # SOD: SCA in export mode (no targets, no loss)
        if self.sca_enabled and self.sca is not None:
            feats, heatmap, _ = self.sca(
                feats=feats,
                images=tensors,
                targets=None,
            )
            self._heatmap_cache = heatmap
            # Heatmap cache for next frame
            if self.tfcm_enabled and self.tfcm is not None:
                self.tfcm.update_sopm_cache(heatmap)

        feats = self.projector(feats)
        out_feats = []
        out_masks = []
        for feat in feats:
            # x: [(B, C, H, W)]
            b, _, h, w = feat.shape
            out_masks.append(torch.zeros((b, h, w), dtype=torch.bool, device=feat.device))
            out_feats.append(feat)
        return out_feats, out_masks

    # ------------------------------------------------------------------
    # Small object heatmap access for SAQG / TFCM
    # ------------------------------------------------------------------
    @property
    def sopm(self):
        """Return small object heatmap from last forward pass.

        Access from lwdetr.py:
            heatmap = self.backbone.sopm  (or self.backbone[0].sopm via Joiner)
        """
        return self._heatmap_cache

    @property
    def sca_losses(self):
        """Return SCA auxiliary losses (loss_heatmap).

        Access from training loop or SetCriterion:
            sca_loss = model.backbone.sca_losses
            losses.update(sca_loss)
        """
        return self._sca_loss_cache
    # ------------------------------------------------------------------

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

            # ============================================================
            # SOD: SCA (Selective Cross-Attention) parameter groups
            #
            # CNN Branch: lr=1e-3 (random init, fast convergence)
            # CrossAttn / Heatmap / alpha: lr=1e-4 (same as decoder)
            # ============================================================
            elif "sca" in n and p.requires_grad:
                if "cnn_branch" in n:
                    lr = getattr(args, 'lr_sca_cnn', 1e-3)
                else:
                    lr = getattr(args, 'lr_sca', 1e-4)

                wd = args.weight_decay
                # No weight decay for bias, norm, alpha
                if "bias" in n or "norm" in n or "alpha" in n:
                    wd = 0.0

                named_param_lr_pairs[n] = {
                    "params": p,
                    "lr": lr,
                    "weight_decay": wd,
                }
            # ============================================================

            # ============================================================
            # TFCM parameter groups
            #
            # TFCM has only 2 params (log_tau, alpha_raw).
            # lr=2e-4 (new module, faster learning in Stage 2).
            # No weight decay for scalar parameters.
            # ============================================================
            elif "tfcm" in n and p.requires_grad:
                lr = getattr(args, 'lr_tfcm', 2e-4)
                named_param_lr_pairs[n] = {
                    "params": p,
                    "lr": lr,
                    "weight_decay": 0.0,  # scalar params: no weight decay
                }
            # ============================================================

        return named_param_lr_pairs


def get_dinov2_lr_decay_rate(name, lr_decay_rate=1.0, num_layers=12):
    """
    Calculate lr decay rate for different ViT blocks.

    Args:
        name (string): parameter name.
        lr_decay_rate (float): base lr decay rate.
        num_layers (int): number of ViT blocks.
    Returns:
        lr decay rate for the given parameter.
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
