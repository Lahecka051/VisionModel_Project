"""
TFCM: Temporal Feature Correspondence Module

VIVID-Det 모듈 1. Backbone encoder → SSFA 사이에 삽입.
DINOv2 자기지도 특징 공간에서 프레임 간 대응을 추출하여
소형 객체 영역에만 선택적으로 시간적 융합을 수행한다.

설계 근거:
  - DINO-Tracker: DINOv2 패치 토큰의 프레임 간 의미적 대응 능력
  - TransVOD: N_ref=4 참조 프레임 최적
  - Utrecht: 전체 시계열 융합 → 소형 객체 성능 저하. Scale-Aware 필수
  - WACV 2025: softmax τ=0.01~0.03 최적 (DINOv2 cosine sim 범위)
  - SimSC: learnable temperature > fixed temperature

핵심 동작:
  비디오 모드: Memory Buffer(FIFO N=4) → cosine similarity →
               softmax(τ=0.03 learnable) → SOPM(t-1) Scale-Aware Mask →
               α 융합 (α=0.0 초기화, 학습)
  이미지 모드: 완전 bypass, cost 0%
  Cold start:  t=1 bypass, t≥2 점진 활성 (α_eff = α × N_avail/N_max)

@changelog
[v1.0.0] 2026-03-11 - 초기 구현. VIVID-Det v2.2 기술검증 보고서 기반.
  - FeatureMemoryBuffer: FIFO N=4, detach 저장
  - TFCM: cosine sim → softmax(τ learnable) → α fusion
  - Scale-Aware mask via SOPM(t-1), τ_small=0.3
  - Cold start gradual activation
  - α=0.0 초기화 (Stage 2 identity start)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any


class FeatureMemoryBuffer:
    """FIFO Feature Memory Buffer.

    backbone 출력을 detach하여 최대 N 프레임 저장.
    학습 시 과거 특징에 대한 gradient 역전파를 차단한다.

    Attributes:
        max_size: 최대 저장 프레임 수 (기본 4, TransVOD ablation 기반).
        buffer: 저장된 특징 리스트 [B, C, H, W].
    """

    def __init__(self, max_size: int = 4):
        self.max_size = max_size
        self.buffer: list[torch.Tensor] = []

    def push(self, feat: torch.Tensor) -> None:
        """특징을 버퍼에 추가. detach하여 gradient 차단."""
        self.buffer.append(feat.detach())
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)

    def get_all(self) -> list[torch.Tensor]:
        """버퍼의 모든 특징 반환."""
        return self.buffer

    @property
    def size(self) -> int:
        return len(self.buffer)

    def clear(self) -> None:
        """버퍼 초기화. 비연속 시퀀스 전환 시 호출."""
        self.buffer.clear()


class TFCM(nn.Module):
    """Temporal Feature Correspondence Module.

    DINOv2 특징 공간에서 프레임 간 cosine similarity 기반 대응을 추출하고,
    SOPM(t-1) Scale-Aware 마스크를 적용하여 소형 객체 영역에만 시간적 융합.

    Args:
        embed_dim: backbone 출력 채널 수 (DINOv2 ViT-S = 384).
        n_ref: 참조 프레임 수 (기본 4).
        tau_init: softmax temperature 초기값 (기본 0.03).
        tau_min: temperature clamp 하한 (기본 0.005).
        tau_max: temperature clamp 상한 (기본 0.1).
        alpha_init: sigmoid 입력 초기값 (기본 -10.0, sigmoid(-10)≈0.0 → identity).
        tau_small: SOPM 임계값 (기본 0.3).
    """

    def __init__(
        self,
        embed_dim: int = 384,
        n_ref: int = 4,
        tau_init: float = 0.03,
        tau_min: float = 0.005,
        tau_max: float = 0.1,
        alpha_init: float = -10.0,
        tau_small: float = 0.3,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.n_ref = n_ref
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.tau_small = tau_small

        # learnable temperature (v2.2: WACV 2025 + SimSC 근거)
        # log 공간에서 학습하여 양수 보장
        self._log_tau = nn.Parameter(torch.tensor(tau_init).log())

        # learnable fusion ratio (sigmoid(-10.0) ≈ 0.0 → Stage 2 identity start)
        # 학습 진행에 따라 sigmoid 출력이 점진적으로 증가
        self._alpha_raw = nn.Parameter(torch.tensor(alpha_init))

        # Feature Memory Buffer (non-parameter, 모델 저장 대상 아님)
        self.memory_buffer = FeatureMemoryBuffer(max_size=n_ref)

        # SOPM(t-1) 캐시 (non-parameter)
        self._sopm_cache: Optional[torch.Tensor] = None

    @property
    def tau(self) -> torch.Tensor:
        """Clamped learnable temperature."""
        return self._log_tau.exp().clamp(self.tau_min, self.tau_max)

    @property
    def alpha(self) -> torch.Tensor:
        """Clamped fusion ratio. Sigmoid로 (0,1) 범위 보장."""
        return torch.sigmoid(self._alpha_raw)

    def update_sopm_cache(self, sopm: torch.Tensor) -> None:
        """SSFA에서 생성된 SOPM을 캐시에 저장.

        backbone forward에서 SSFA 호출 후 이 메서드를 호출하여
        다음 프레임의 TFCM이 사용할 SOPM(t-1)을 갱신한다.

        Args:
            sopm: [B, 1, H_sopm, W_sopm] SOPM 맵 (Sigmoid 출력).
        """
        self._sopm_cache = sopm.detach()

    def clear_state(self) -> None:
        """비연속 시퀀스 전환 시 전체 상태 초기화."""
        self.memory_buffer.clear()
        self._sopm_cache = None

    def _compute_correspondence(
        self,
        feat_cur: torch.Tensor,
        feat_ref: torch.Tensor,
    ) -> torch.Tensor:
        """프레임 간 cosine similarity 기반 대응 가중치 계산.

        Args:
            feat_cur: [B, N, C] 현재 프레임 토큰 (L2-normalized).
            feat_ref: [B, N, C] 참조 프레임 토큰 (L2-normalized).

        Returns:
            weights: [B, N, N] softmax(cosine_sim / τ) 가중치.
        """
        # cosine similarity: [B, N, N]
        sim = torch.bmm(feat_cur, feat_ref.transpose(1, 2))

        # softmax with learnable temperature
        tau = self.tau
        weights = F.softmax(sim / tau, dim=-1)

        return weights

    def _warp_features(
        self,
        feat_cur_norm: torch.Tensor,
        ref_features: list[torch.Tensor],
    ) -> torch.Tensor:
        """대응 기반 워프: 과거 프레임 특징을 현재 좌표계로 변환 후 평균.

        Args:
            feat_cur_norm: [B, N, C] L2-normalized 현재 토큰.
            ref_features: 참조 프레임 특징 리스트, 각 [B, C, H, W].

        Returns:
            warped: [B, N, C] 워프된 특징 평균.
        """
        B, N, C = feat_cur_norm.shape
        warped_sum = torch.zeros(B, N, C, device=feat_cur_norm.device,
                                 dtype=feat_cur_norm.dtype)

        for feat_ref_spatial in ref_features:
            # [B, C, H, W] → [B, N, C]
            feat_ref = feat_ref_spatial.flatten(2).transpose(1, 2)
            # L2 normalize for cosine similarity
            feat_ref_norm = F.normalize(feat_ref, dim=-1)

            # [B, N, N] correspondence weights
            weights = self._compute_correspondence(feat_cur_norm, feat_ref_norm)

            # [B, N, C] = [B, N, N] @ [B, N, C]
            warped = torch.bmm(weights, feat_ref)
            warped_sum = warped_sum + warped

        # 참조 프레임 수로 평균
        warped_avg = warped_sum / len(ref_features)
        return warped_avg

    def _build_scale_mask(
        self,
        H: int,
        W: int,
        device: torch.device,
        dtype: torch.dtype,
        B: int,
    ) -> torch.Tensor:
        """SOPM(t-1) 기반 Scale-Aware 마스크 생성.

        SOPM > τ_small 영역만 1, 나머지 0.
        SOPM 캐시가 없으면 (cold start) 전체 1 (모든 토큰 융합 허용).

        Args:
            H, W: 현재 특징 맵 해상도.
            device, dtype: 텐서 속성.
            B: 배치 크기.

        Returns:
            mask: [B, H*W, 1] 마스크.
        """
        if self._sopm_cache is None:
            # cold start: SOPM 없음 → 전체 허용
            return torch.ones(B, H * W, 1, device=device, dtype=dtype)

        # SOPM을 현재 특징 해상도로 보간
        # _sopm_cache: [B, 1, H_sopm, W_sopm]
        sopm_resized = F.interpolate(
            self._sopm_cache, size=(H, W), mode="bilinear", align_corners=False
        )
        # [B, 1, H, W] → [B, H*W, 1]
        sopm_flat = sopm_resized.flatten(2).transpose(1, 2)

        mask = (sopm_flat > self.tau_small).to(dtype)
        return mask

    def forward(
        self,
        feats: torch.Tensor,
        temporal_mode: bool = False,
    ) -> torch.Tensor:
        """TFCM forward.

        Args:
            feats: [B, C, H, W] backbone encoder 출력.
            temporal_mode: True이면 비디오 모드 (시퀀스 메타데이터 기반 결정).
                          False이면 이미지 모드 → bypass.

        Returns:
            enhanced: [B, C, H, W] 시간적 융합된 (또는 bypass된) 특징.
        """
        # ── 이미지 모드: 완전 bypass ──
        if not temporal_mode:
            return feats

        B, C, H, W = feats.shape

        # ── 비디오 모드 ──
        ref_features = self.memory_buffer.get_all()

        # 현재 프레임을 버퍼에 추가 (detach)
        self.memory_buffer.push(feats)

        # cold start: 참조 프레임 없으면 bypass
        n_avail = len(ref_features)
        if n_avail == 0:
            return feats

        # ── 대응 기반 워프 ──
        # [B, C, H, W] → [B, N, C] (N = H*W)
        feat_tokens = feats.flatten(2).transpose(1, 2)
        feat_tokens_norm = F.normalize(feat_tokens, dim=-1)

        # 워프된 특징: [B, N, C]
        warped = self._warp_features(feat_tokens_norm, ref_features)

        # ── Scale-Aware 마스크 ──
        # [B, N, 1]
        scale_mask = self._build_scale_mask(H, W, feats.device, feats.dtype, B)

        # ── Cold start gradual activation ──
        # α_eff = α × (N_available / N_max)
        alpha = self.alpha
        alpha_eff = alpha * (n_avail / self.n_ref)

        # ── 융합 ──
        # F_enhanced(i) = M(i) * [α_eff * warped(i) + (1-α_eff) * F_t(i)]
        #               + (1-M(i)) * F_t(i)
        fused = alpha_eff * warped + (1.0 - alpha_eff) * feat_tokens

        enhanced_tokens = scale_mask * fused + (1.0 - scale_mask) * feat_tokens

        # [B, N, C] → [B, C, H, W]
        enhanced = enhanced_tokens.transpose(1, 2).reshape(B, C, H, W)

        return enhanced
