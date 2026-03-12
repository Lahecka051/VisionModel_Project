"""
SSFA: Small-object Selective Fusion Attention

VIVID-Det 모듈 2. Backbone→Projector 간극에 삽입.
CNN Branch 고해상도 특징 + ViT Attention Prior → SOPM → Top-K 선택적 Cross-Attention.

@changelog
[v1.0.0] 2026-03-10 - 초기 구현. VIVID-Det v2.2 기술검증 보고서 기반.
  - CNNBranch: stride-8, GroupNorm(32), 256ch
  - AttentionPriorExtractor: DINOv2 Block4 Self-Attn hook
  - SOPMHead: concat(D3, attn_prior) → Sigmoid
  - SelectiveCrossAttention: Top-K 25%, 2D sinusoidal PE
  - SOPMFocalLoss: GT small heatmap 직접 감독
  - SSFA: 통합 모듈, alpha=0.01 초기화
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# 2D Sinusoidal Positional Encoding
# ---------------------------------------------------------------------------

def build_2d_sincos_pe(h: int, w: int, d_model: int, temperature: float = 10000.0) -> torch.Tensor:
    """2D sinusoidal positional encoding 생성.

    DETR / ViTDet 방식. d_model의 절반을 y축, 나머지 절반을 x축에 할당.

    Args:
        h: 세로 그리드 크기.
        w: 가로 그리드 크기.
        d_model: 임베딩 차원 (4의 배수여야 함).
        temperature: sin/cos 주파수 스케일.

    Returns:
        [h*w, d_model] 텐서.
    """
    assert d_model % 4 == 0, f"d_model({d_model})은 4의 배수여야 합니다."

    half = d_model // 2
    quarter = d_model // 4

    pos_y = torch.arange(h, dtype=torch.float32).unsqueeze(1).expand(h, w)
    pos_x = torch.arange(w, dtype=torch.float32).unsqueeze(0).expand(h, w)

    dim_t = torch.arange(quarter, dtype=torch.float32)
    dim_t = temperature ** (2 * dim_t / half)  # [quarter]

    # y 축: sin/cos 교대 → [h, w, half]
    pe_y_sin = torch.sin(pos_y.unsqueeze(-1) / dim_t)  # [h, w, quarter]
    pe_y_cos = torch.cos(pos_y.unsqueeze(-1) / dim_t)
    pe_y = torch.stack([pe_y_sin, pe_y_cos], dim=-1).reshape(h, w, half)

    # x 축: sin/cos 교대 → [h, w, half]
    pe_x_sin = torch.sin(pos_x.unsqueeze(-1) / dim_t)
    pe_x_cos = torch.cos(pos_x.unsqueeze(-1) / dim_t)
    pe_x = torch.stack([pe_x_sin, pe_x_cos], dim=-1).reshape(h, w, half)

    pe = torch.cat([pe_y, pe_x], dim=-1)  # [h, w, d_model]
    return pe.reshape(h * w, d_model)


# ---------------------------------------------------------------------------
# Attention Prior Extractor (DINOv2 Block4 Self-Attention Hook)
# ---------------------------------------------------------------------------

class AttentionPriorExtractor:
    """DINOv2 Block4 Self-Attention에서 CLS->patch attention map을 추출.

    HuggingFace DINOv2는 별도 query/key/value Linear를 사용하며
    SDPA로 attention을 계산하므로 weight를 반환하지 않는다.
    query와 key Linear 출력을 각각 hook으로 캡처하여 수동 계산한다.

    실제 구조 (RF-DETR + HuggingFace DINOv2):
        encoder.encoder.layer.3.attention.attention
            .query  (Linear)
            .key    (Linear)
            .value  (Linear)
        타입: Dinov2WithRegistersSdpaSelfAttention
    """

    def __init__(self) -> None:
        self._q_output: Optional[torch.Tensor] = None
        self._k_output: Optional[torch.Tensor] = None
        self._num_heads: int = 6  # ViT-S/14
        self._handles: List = []

    def register(self, attn_module: nn.Module, num_heads: int = 6) -> None:
        """attention 모듈의 query/key Linear 출력을 캡처하는 hook 등록.

        Args:
            attn_module: Dinov2WithRegistersSdpaSelfAttention 모듈.
                         `.query` 와 `.key` (nn.Linear) 속성 필요.
            num_heads: attention head 수 (ViT-S=6).
        """
        self._num_heads = num_heads

        if hasattr(attn_module, 'query') and hasattr(attn_module, 'key'):
            self._handles.append(
                attn_module.query.register_forward_hook(self._hook_q)
            )
            self._handles.append(
                attn_module.key.register_forward_hook(self._hook_k)
            )
        else:
            raise AttributeError(
                f"attn_module({type(attn_module).__name__})에 "
                "'query'/'key' 속성이 없습니다."
            )

    def _hook_q(self, module: nn.Module, input: tuple, output: torch.Tensor) -> None:
        self._q_output = output

    def _hook_k(self, module: nn.Module, input: tuple, output: torch.Tensor) -> None:
        self._k_output = output

    def extract(self, num_patches: int) -> Optional[torch.Tensor]:
        """캡처된 Q/K에서 CLS->patch attention prior를 계산.

        Args:
            num_patches: 패치 토큰 수 (CLS/register 제외).

        Returns:
            [B, 1, num_patches] attention prior (softmax 적용됨).
            캡처 실패 시 None.
        """
        if self._q_output is None or self._k_output is None:
            return None

        q = self._q_output  # [B, N_total, dim]
        k = self._k_output  # [B, N_total, dim]
        B, N_total, dim = q.shape
        head_dim = dim // self._num_heads

        # Multi-head reshape
        q = q.reshape(B, N_total, self._num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(B, N_total, self._num_heads, head_dim).permute(0, 2, 1, 3)
        # [B, num_heads, N_total, head_dim]

        # CLS token (index 0) -> all patch tokens attention
        # N_total = 1 (CLS) + num_patches + num_registers
        # patch tokens start at index 1, registers follow after patches
        q_cls = q[:, :, 0:1, :]                   # [B, heads, 1, head_dim]
        k_patches = k[:, :, 1:1+num_patches, :]   # [B, heads, num_patches, head_dim]

        scale = head_dim ** -0.5
        attn = (q_cls @ k_patches.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)  # [B, heads, 1, num_patches]

        # head mean -> [B, 1, num_patches]
        attn_prior = attn.mean(dim=1)

        # clear cache (re-captured every forward)
        self._q_output = None
        self._k_output = None

        return attn_prior.detach()

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()


# ---------------------------------------------------------------------------
# CNN Branch (stride-8, GroupNorm)
# ---------------------------------------------------------------------------

class CNNBranch(nn.Module):
    """경량 CNN Branch: stride-8 고해상도 특징 추출.

    3단 Conv3x3(stride 2) + 1단 Conv3x3(stride 1) 정제.
    GroupNorm(32)으로 작은 batch에서도 안정적.

    M(576): [B,3,576,576] → [B,256,72,72]
    L(704): [B,3,704,704] → [B,256,88,88]

    파라미터: ~1.5M
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 256) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            # Stage 1: stride 2 → H/2
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(32, 64),
            nn.ReLU(inplace=True),

            # Stage 2: stride 2 → H/4
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(32, 128),
            nn.ReLU(inplace=True),

            # Stage 3: stride 2 → H/8
            nn.Conv2d(128, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),

            # 정제 레이어: stride 1 (해상도 유지, 수용 영역 확장)
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 원본 이미지 [B, 3, H, W].

        Returns:
            D3 특징맵 [B, 256, H/8, W/8].
        """
        return self.stem(x)


# ---------------------------------------------------------------------------
# SOPM Head (Small Object Possibility Map)
# ---------------------------------------------------------------------------

class SOPMHead(nn.Module):
    """SOPM 생성: concat(D3_reduced, Attn_prior) → Conv1x1 → Sigmoid.

    D3 (256ch) → 1x1 Conv → 1ch activation.
    Attn Prior (1ch, 선택적) 와 concat 후 최종 1ch Sigmoid.
    """

    def __init__(self, cnn_channels: int = 256) -> None:
        super().__init__()
        # D3 256ch → 1ch 차원 축소
        self.d3_reduce = nn.Conv2d(cnn_channels, 1, kernel_size=1, bias=True)
        # concat(d3_1ch, attn_prior_1ch) = 2ch → 1ch SOPM
        self.fuse = nn.Conv2d(2, 1, kernel_size=1, bias=True)

    def forward(
        self,
        d3: torch.Tensor,
        attn_prior: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            d3: CNN 특징맵 [B, 256, H_d3, W_d3].
            attn_prior: ViT attention prior [B, 1, H_d3, W_d3].
                        None이면 D3만으로 SOPM 생성 (ablation C1).

        Returns:
            SOPM [B, 1, H_d3, W_d3], 값 범위 (0, 1).
        """
        d3_act = self.d3_reduce(d3)  # [B, 1, H, W]

        if attn_prior is not None:
            fused = torch.cat([d3_act, attn_prior], dim=1)  # [B, 2, H, W]
            sopm = torch.sigmoid(self.fuse(fused))
        else:
            sopm = torch.sigmoid(d3_act)

        return sopm


# ---------------------------------------------------------------------------
# Selective Cross-Attention with 2D Sinusoidal PE
# ---------------------------------------------------------------------------

class SelectiveCrossAttention(nn.Module):
    """Top-K 선택적 Cross-Attention.

    Q = ViT 전체 토큰 (P3_raw), K/V = SOPM 상위 25% CNN 토큰.
    2D sinusoidal PE를 Q/K에 각각 추가 (피어리뷰 M1 반영).

    파라미터: ~0.5M (d_model=384 기준)
    """

    def __init__(
        self,
        vit_dim: int = 384,
        cnn_dim: int = 256,
        d_attn: int = 384,
        num_heads: int = 6,
        topk_ratio: float = 0.25,
    ) -> None:
        super().__init__()
        self.d_attn = d_attn
        self.num_heads = num_heads
        self.head_dim = d_attn // num_heads
        self.scale = self.head_dim ** -0.5
        self.topk_ratio = topk_ratio

        # Q/K/V/Out 투영
        self.q_proj = nn.Linear(vit_dim, d_attn, bias=True)
        self.k_proj = nn.Linear(cnn_dim, d_attn, bias=True)
        self.v_proj = nn.Linear(cnn_dim, d_attn, bias=True)
        self.out_proj = nn.Linear(d_attn, vit_dim, bias=True)

        # PE 캐시 (해상도별 lazy init)
        self._pe_cache: dict = {}

    def _get_pe(self, h: int, w: int, device: torch.device) -> torch.Tensor:
        """해상도별 2D sinusoidal PE를 캐시하여 반환.

        Returns:
            [h*w, d_attn] PE 텐서.
        """
        key = (h, w, device)
        if key not in self._pe_cache:
            pe = build_2d_sincos_pe(h, w, self.d_attn).to(device)
            self._pe_cache[key] = pe
        return self._pe_cache[key]

    def forward(
        self,
        vit_feats: torch.Tensor,
        cnn_feats_2d: torch.Tensor,
        sopm: torch.Tensor,
        h_vit: int,
        w_vit: int,
    ) -> torch.Tensor:
        """
        Args:
            vit_feats: ViT 패치 토큰 [B, N_vit, vit_dim]. N_vit = h_vit * w_vit.
            cnn_feats_2d: CNN D3 특징맵 [B, 256, H_d3, W_d3].
            sopm: SOPM [B, 1, H_d3, W_d3].
            h_vit: ViT 패치 그리드 세로 크기.
            w_vit: ViT 패치 그리드 가로 크기.

        Returns:
            Cross-Attention 출력 [B, N_vit, vit_dim].
        """
        B, _, H_d3, W_d3 = cnn_feats_2d.shape
        N_vit = vit_feats.shape[1]

        # --- Top-K 선택 ---
        sopm_flat = sopm.reshape(B, H_d3 * W_d3)        # [B, N_cnn]
        K = max(1, int(sopm_flat.shape[1] * self.topk_ratio))
        _, topk_indices = sopm_flat.topk(K, dim=1)       # [B, K]

        # CNN 특징 flatten → Top-K 인덱싱
        cnn_flat = cnn_feats_2d.flatten(2).permute(0, 2, 1)  # [B, N_cnn, 256]
        topk_idx_exp = topk_indices.unsqueeze(-1).expand(B, K, cnn_flat.shape[-1])
        cnn_selected = cnn_flat.gather(1, topk_idx_exp)  # [B, K, 256]

        # --- Q/K/V 투영 ---
        q = self.q_proj(vit_feats)    # [B, N_vit, d_attn]
        k = self.k_proj(cnn_selected) # [B, K, d_attn]
        v = self.v_proj(cnn_selected) # [B, K, d_attn]

        # --- 2D Sinusoidal PE 추가 ---
        pe_q = self._get_pe(h_vit, w_vit, q.device)[:N_vit]  # [N_vit, d_attn]
        q = q + pe_q.unsqueeze(0)

        # K의 PE: 선택된 CNN 토큰의 원래 위치 기반
        pe_k_full = self._get_pe(H_d3, W_d3, k.device)       # [N_cnn, d_attn]
        pe_k_selected = pe_k_full[topk_indices[0]]  # [K, d_attn] (배치 내 동일 인덱스 가정)
        # 배치별 인덱스가 다를 수 있으므로, 배치 루프 또는 gather 사용
        pe_k_full_exp = pe_k_full.unsqueeze(0).expand(B, -1, -1)  # [B, N_cnn, d_attn]
        topk_idx_pe = topk_indices.unsqueeze(-1).expand(B, K, self.d_attn)
        pe_k = pe_k_full_exp.gather(1, topk_idx_pe)  # [B, K, d_attn]
        k = k + pe_k

        # --- Multi-Head Attention ---
        q = q.reshape(B, N_vit, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(B, K, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B, K, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        # [B, num_heads, N, head_dim]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, N_vit, K]
        attn = attn.softmax(dim=-1)
        out = attn @ v  # [B, num_heads, N_vit, head_dim]

        out = out.permute(0, 2, 1, 3).reshape(B, N_vit, self.d_attn)
        out = self.out_proj(out)  # [B, N_vit, vit_dim]

        return out


# ---------------------------------------------------------------------------
# SOPM FocalLoss (학습 전용)
# ---------------------------------------------------------------------------

class SOPMFocalLoss(nn.Module):
    """SOPM 감독용 FocalLoss.

    GT 소형 bbox 중심에 가우시안 히트맵을 생성하여 SOPM을 직접 감독.
    추론 시 사용하지 않음 (비용 0%).

    L_sopm = FocalLoss(SOPM, GT_heatmap, alpha=0.25, gamma=2.0)
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        small_area_threshold: int = 1024,  # 32^2: COCO small 기준
        gaussian_sigma: float = 2.0,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.small_area_threshold = small_area_threshold
        self.gaussian_sigma = gaussian_sigma

    @torch.no_grad()
    def build_gt_heatmap(
        self,
        targets: List[dict],
        h: int,
        w: int,
        device: torch.device,
    ) -> torch.Tensor:
        """GT 소형 bbox 중심에 가우시안 히트맵 생성.

        Args:
            targets: 타겟 리스트. 각 dict에 'boxes' (cxcywh, normalized) 포함.
            h: 히트맵 세로 크기 (= SOPM 해상도).
            w: 히트맵 가로 크기.
            device: 텐서 디바이스.

        Returns:
            GT 히트맵 [B, 1, h, w], 값 범위 [0, 1].
        """
        B = len(targets)
        heatmap = torch.zeros(B, 1, h, w, device=device)

        # 가우시안 커널 좌표 (미리 생성)
        yy = torch.arange(h, dtype=torch.float32, device=device).unsqueeze(1)  # [h, 1]
        xx = torch.arange(w, dtype=torch.float32, device=device).unsqueeze(0)  # [1, w]

        for b in range(B):
            boxes = targets[b]['boxes']  # [N_obj, 4] cxcywh normalized
            if boxes.numel() == 0:
                continue

            # 면적 기반 소형 객체 필터링 (normalized → pixel 면적 추정)
            # boxes는 [0,1] 정규화. 면적 = w_box * h_box * (img_w * img_h)
            # 실제 이미지 크기 대신 SOPM 해상도 기준으로 필터링
            bw = boxes[:, 2] * w  # SOPM 해상도 기준 너비
            bh = boxes[:, 3] * h  # SOPM 해상도 기준 높이
            areas_sopm = bw * bh  # SOPM 공간에서의 면적

            # 원본 이미지 기준 area ≤ 1024에 대응하는 SOPM 면적 임계값
            # stride-8 기준: sopm_area = orig_area / 64
            sopm_threshold = self.small_area_threshold / 64.0
            small_mask = areas_sopm <= sopm_threshold

            if not small_mask.any():
                continue

            small_boxes = boxes[small_mask]
            cx = small_boxes[:, 0] * w  # SOPM x 좌표
            cy = small_boxes[:, 1] * h  # SOPM y 좌표

            for i in range(cx.shape[0]):
                gaussian = torch.exp(
                    -((xx - cx[i]) ** 2 + (yy - cy[i]) ** 2)
                    / (2.0 * self.gaussian_sigma ** 2)
                )
                heatmap[b, 0] = torch.max(heatmap[b, 0], gaussian)

        return heatmap.clamp(0.0, 1.0)

    def forward(
        self,
        sopm: torch.Tensor,
        targets: List[dict],
    ) -> torch.Tensor:
        """
        Args:
            sopm: SOPM 예측 [B, 1, H, W].
            targets: GT 타겟 리스트.

        Returns:
            FocalLoss 스칼라.
        """
        _, _, h, w = sopm.shape
        gt_heatmap = self.build_gt_heatmap(targets, h, w, sopm.device)

        # Binary Focal Loss
        pred = sopm.clamp(1e-6, 1.0 - 1e-6)
        bce = -(gt_heatmap * torch.log(pred) + (1 - gt_heatmap) * torch.log(1 - pred))

        # Focal 가중치
        p_t = gt_heatmap * pred + (1 - gt_heatmap) * (1 - pred)
        focal_weight = (1 - p_t) ** self.gamma

        # Alpha 가중치
        alpha_t = gt_heatmap * self.alpha + (1 - gt_heatmap) * (1 - self.alpha)

        loss = (alpha_t * focal_weight * bce).mean()
        return loss


# ---------------------------------------------------------------------------
# SSFA 통합 모듈
# ---------------------------------------------------------------------------

class SSFA(nn.Module):
    """Small-object Selective Fusion Attention.

    Backbone→Projector 간극에서 ViT 특징에 CNN 고해상도 정보를 선택적 융합.
    SOPM (Small Object Possibility Map) 생성하여 SAQG, TFCM에 전달.

    설계 근거:
    - alpha=0.01 초기화: 피어리뷰 C1 반영, gradient cold start 방지.
    - 2D sinusoidal PE: 피어리뷰 M1 반영, Q/K 해상도 차이 보정.
    - Top-K 25%: 배치 처리 효율 + 배경 노이즈 차단.
    - FocalLoss 직접 감독: SOPM 빠른 수렴 보장 (v5.0 이후).

    추론 비용: +1~3% (FocalLoss는 학습 전용).
    파라미터: ~3.1M (CNN ~1.5M + SOPM ~0.1M + CrossAttn ~0.5M + alpha 1개).
    """

    def __init__(
        self,
        vit_dim: int = 384,
        cnn_channels: int = 256,
        num_heads: int = 6,
        topk_ratio: float = 0.25,
        alpha_init: float = 0.01,
        sopm_loss_weight: float = 0.5,
        use_attn_prior: bool = True,
    ) -> None:
        """
        Args:
            vit_dim: ViT 출력 차원 (ViT-S=384).
            cnn_channels: CNN Branch 출력 채널 수.
            num_heads: Cross-Attention head 수.
            topk_ratio: Top-K 선택 비율 (0.25 = 상위 25%).
            alpha_init: 게이팅 파라미터 초기값 (피어리뷰: 0.01).
            sopm_loss_weight: L_sopm 가중치 (λ_sopm).
            use_attn_prior: ViT Attn Prior 사용 여부 (ablation C1 용).
        """
        super().__init__()

        self.use_attn_prior = use_attn_prior
        self.sopm_loss_weight = sopm_loss_weight

        # 게이팅 파라미터: clamp(0, 1) 적용
        # [v1.0.0] alpha=0.01 초기화 (피어리뷰 C1: cold start 방지)
        self.alpha = nn.Parameter(torch.tensor(alpha_init))

        # 서브 모듈
        self.cnn_branch = CNNBranch(in_channels=3, out_channels=cnn_channels)
        self.sopm_head = SOPMHead(cnn_channels=cnn_channels)
        self.cross_attn = SelectiveCrossAttention(
            vit_dim=vit_dim,
            cnn_dim=cnn_channels,
            d_attn=vit_dim,
            num_heads=num_heads,
            topk_ratio=topk_ratio,
        )
        self.sopm_focal_loss = SOPMFocalLoss()

        # Attention Prior 추출기 (hook 기반)
        self.attn_extractor = AttentionPriorExtractor()

    def register_attn_hook(self, encoder: nn.Module, num_heads: int = 6) -> None:
        """DINOv2 encoder에 attention hook을 등록.

        호출 시점: Backbone.__init__에서 encoder 초기화 직후.

        Args:
            encoder: DINOv2 encoder 모듈.
            num_heads: ViT attention head 수.

        실제 구조 (RF-DETR + HuggingFace DINOv2):
            encoder.encoder.layer.3.attention.attention
                -> Dinov2WithRegistersSdpaSelfAttention
                -> .query (Linear), .key (Linear), .value (Linear)
        """
        attn_module = None
        candidates = [
            # RF-DETR HuggingFace DINOv2 (confirmed)
            lambda: encoder.encoder.encoder.layer[3].attention.attention,
            # Windowed variant
            lambda: encoder.encoder.layer[3].attention.attention,
            # timm style
            lambda: encoder.blocks[3].attn,
        ]

        for candidate in candidates:
            try:
                attn_module = candidate()
                # Verify it has query/key attributes
                if hasattr(attn_module, 'query') and hasattr(attn_module, 'key'):
                    break
                attn_module = None
            except (AttributeError, IndexError):
                continue

        if attn_module is None:
            print(
                "[SSFA WARNING] DINOv2 Block4 attention module not found.\n"
                "  Falling back to D3-only SOPM (no Attn Prior).\n"
                "  Check encoder structure and update register_attn_hook()."
            )
            self.use_attn_prior = False
            return

        self.attn_extractor.register(attn_module, num_heads)

    def forward(
        self,
        feats: List[torch.Tensor],
        images: torch.Tensor,
        targets: Optional[List[dict]] = None,
    ) -> Tuple[List[torch.Tensor], torch.Tensor, dict]:
        """SSFA forward pass.

        Args:
            feats: DINOv2 encoder 출력. 4개 텐서 리스트 (out_feature_indexes=[3,6,9,12]).
                   각 텐서: [B, N_patches, vit_dim] 또는 [B, N_patches+1, vit_dim] (CLS 포함).
            images: 원본 이미지 [B, 3, H, W].
            targets: GT 타겟 (학습 시만 필요). None이면 SOPM loss 미계산.

        Returns:
            (enriched_feats, sopm, loss_dict)
            - enriched_feats: 수정된 feats 리스트 (feats[0]만 변경).
            - sopm: [B, 1, H_d3, W_d3] SOPM.
            - loss_dict: {"loss_sopm": tensor} (학습 시) 또는 {} (추론 시).
        """
        B = images.shape[0]
        loss_dict = {}

        # --- feats[0]: [B, C, H_vit, W_vit] 공간 형식 ---
        feat0 = feats[0]  # [B, 384, H_vit, W_vit]
        _, C_vit, h_vit, w_vit = feat0.shape

        # 토큰 형식으로 변환 (Cross-Attention용)
        # [B, C, H, W] -> [B, H*W, C]
        vit_patches = feat0.flatten(2).permute(0, 2, 1)  # [B, N_vit, C]

        # --- CNN Branch: 고해상도 특징 추출 ---
        d3 = self.cnn_branch(images)  # [B, 256, H/8, W/8]
        _, _, H_d3, W_d3 = d3.shape

        # --- Attention Prior 추출 ---
        attn_prior_2d = None
        if self.use_attn_prior:
            num_patches = h_vit * w_vit
            attn_prior = self.attn_extractor.extract(num_patches)  # [B, 1, N_patches]
            if attn_prior is not None:
                # ViT가 주목하지 않는 영역 = 소형 객체 후보 -> 역전
                attn_prior_inv = 1.0 - attn_prior  # [B, 1, N_patches]
                attn_prior_2d_vit = attn_prior_inv.reshape(B, 1, h_vit, w_vit)
                # D3 해상도로 리사이즈
                attn_prior_2d = F.interpolate(
                    attn_prior_2d_vit,
                    size=(H_d3, W_d3),
                    mode='bilinear',
                    align_corners=False,
                )

        # --- SOPM 생성 ---
        sopm = self.sopm_head(d3, attn_prior_2d)  # [B, 1, H_d3, W_d3]

        # --- SOPM FocalLoss (학습 시) ---
        if targets is not None and self.training:
            loss_sopm = self.sopm_focal_loss(sopm, targets)
            loss_dict["loss_sopm"] = loss_sopm * self.sopm_loss_weight

        # --- Selective Cross-Attention ---
        alpha_clamped = self.alpha.clamp(0.0, 1.0)
        cross_attn_out = self.cross_attn(
            vit_feats=vit_patches,
            cnn_feats_2d=d3,
            sopm=sopm,
            h_vit=h_vit,
            w_vit=w_vit,
        )  # [B, N_vit, C]

        # --- 게이팅 잔차 연결 ---
        enriched_patches = vit_patches + alpha_clamped * cross_attn_out

        # 공간 형식으로 복원: [B, N_vit, C] -> [B, C, H_vit, W_vit]
        enriched_feat0 = enriched_patches.permute(0, 2, 1).reshape(B, C_vit, h_vit, w_vit)

        # --- feats 리스트 재구성 (feats[0]만 교체) ---
        enriched_feats = list(feats)  # shallow copy
        enriched_feats[0] = enriched_feat0

        return enriched_feats, sopm, loss_dict
