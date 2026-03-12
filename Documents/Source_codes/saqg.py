"""
SAQG: Small-object Aware Query Generator

VIVID-Det 모듈 3. Decoder 진입 단계에서 SOPM 기반 쿼리 재배치.
총 쿼리 수 N_total 불변 (NAS-Safe). 배치 비율만 적응적 조정.

동작 흐름:
  1. SOPM 통계 -> MLP -> 적응적 비율 (0.5~0.9)
  2. N_small 쿼리: SOPM 고밀도 영역에서 좌표 샘플링
  3. N_general 쿼리: 원본 refpoint_embed 유지
  4. 반환: [B, N_total, 4] per-image 쿼리 위치

NAS-Safe: 쿼리 수 불변, 배치만 조정. 추론 비용 +5~10%.
파라미터: ~0.03M

@changelog
[v1.0.0] 2026-03-10 - Initial implementation. VIVID-Det v2.2 spec.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class SAQG(nn.Module):
    """Small-object Aware Query Generator.

    SOPM 밀도 기반으로 reference point 좌표를 재배치한다.
    N_total 쿼리 중 N_small개를 SOPM 고밀도 영역에 배치하고,
    나머지 N_general개는 원본 학습 가능 refpoint를 유지한다.

    삽입 위치: lwdetr.py LWDETR.forward, transformer 호출 직전.
    SOPM 의존: backbone._sopm_cache (SSFA 출력).
    """

    def __init__(
        self,
        scale_factor: float = 4.0,
        min_ratio: float = 0.5,
        max_ratio: float = 0.9,
        sopm_threshold: float = 0.3,
    ) -> None:
        """
        Args:
            scale_factor: density_ratio에 곱하는 스케일 팩터.
                          COCO 소형 객체 비율(~41%) 고려 시 3~5 범위.
            min_ratio: 소형 쿼리 최소 비율 (0.5 = 50%).
            max_ratio: 소형 쿼리 최대 비율 (0.9 = 90%).
            sopm_threshold: SOPM 값이 이 이상인 영역만 밀도 계산에 사용.
        """
        super().__init__()
        self.scale_factor = scale_factor
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.sopm_threshold = sopm_threshold

        # MLP Ratio Predictor: SOPM 통계 (3D) -> 적응적 비율
        # 입력: [mean, max, density_ratio] of SOPM
        self.ratio_mlp = nn.Sequential(
            nn.Linear(3, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

        # Small-object w/h prior: 소형 객체의 초기 크기 추정
        # refpoint의 w,h 좌표 (normalized). 학습 가능.
        # 초기값: 소형 객체 기준 ~20px / 576px ≈ 0.035
        self.small_wh_prior = nn.Parameter(torch.tensor([0.035, 0.035]))

    def _compute_ratio(self, sopm: torch.Tensor) -> torch.Tensor:
        """SOPM 통계에서 이미지별 적응적 비율 계산.

        Args:
            sopm: [B, 1, H, W] SOPM.

        Returns:
            [B] 텐서, 각 이미지의 small_query_ratio (0.5~0.9).
        """
        B = sopm.shape[0]
        sopm_flat = sopm.reshape(B, -1)  # [B, H*W]

        # 통계량 3D: mean, max, density_ratio
        s_mean = sopm_flat.mean(dim=1, keepdim=True)         # [B, 1]
        s_max = sopm_flat.max(dim=1, keepdim=True).values    # [B, 1]
        density = (sopm_flat > self.sopm_threshold).float().mean(dim=1, keepdim=True)  # [B, 1]

        stats = torch.cat([s_mean, s_max, density], dim=1)   # [B, 3]
        ratio_raw = self.ratio_mlp(stats).squeeze(-1)        # [B]
        ratio = torch.sigmoid(ratio_raw)  # (0, 1)

        # clamp to [min_ratio, max_ratio]
        ratio = ratio * (self.max_ratio - self.min_ratio) + self.min_ratio

        return ratio

    def _sample_positions(
        self,
        sopm: torch.Tensor,
        n_small: int,
    ) -> torch.Tensor:
        """SOPM 밀도 기반으로 소형 쿼리 좌표를 샘플링.

        SOPM 값을 확률로 사용하여 multinomial 샘플링.
        학습 시: multinomial (확률적, gradient 흐름 불가하지만 SOPM은 detach).
        추론 시: top-k (결정적).

        Args:
            sopm: [B, 1, H, W] SOPM.
            n_small: 샘플링할 좌표 수.

        Returns:
            [B, n_small, 4] reference points (cx, cy, w, h) normalized.
        """
        B, _, H, W = sopm.shape
        sopm_flat = sopm.reshape(B, H * W)  # [B, H*W]

        # 확률 분포 생성 (min clamp으로 zero division 방지)
        probs = sopm_flat.clamp(min=1e-6)
        probs = probs / probs.sum(dim=1, keepdim=True)  # [B, H*W]

        if self.training:
            # 학습: multinomial 샘플링 (다양성 확보)
            indices = torch.multinomial(probs, n_small, replacement=False)  # [B, n_small]
        else:
            # 추론: top-k (결정적)
            _, indices = probs.topk(n_small, dim=1)  # [B, n_small]

        # 인덱스 -> (y, x) 좌표 -> normalized (cx, cy)
        y_idx = indices // W  # [B, n_small]
        x_idx = indices % W   # [B, n_small]

        # grid 중심 좌표로 정규화 [0, 1]
        cx = (x_idx.float() + 0.5) / W  # [B, n_small]
        cy = (y_idx.float() + 0.5) / H  # [B, n_small]

        # w, h: 학습 가능 prior
        wh = self.small_wh_prior.abs().unsqueeze(0).unsqueeze(0)  # [1, 1, 2]
        wh = wh.expand(B, n_small, 2)  # [B, n_small, 2]

        # [B, n_small, 4] (cx, cy, w, h)
        ref_small = torch.stack([cx, cy], dim=-1)  # [B, n_small, 2]
        ref_small = torch.cat([ref_small, wh], dim=-1)  # [B, n_small, 4]

        return ref_small

    def forward(
        self,
        refpoint_embed_weight: torch.Tensor,
        sopm: Optional[torch.Tensor],
        batch_size: int,
    ) -> torch.Tensor:
        """SOPM 기반 쿼리 재배치.

        Args:
            refpoint_embed_weight: 원본 reference point 임베딩.
                학습 시: [N_total * group_detr, 4]
                추론 시: [N_total, 4]
            sopm: [B, 1, H, W] SOPM from SSFA. None이면 bypass.
            batch_size: 현재 배치 크기.

        Returns:
            [B, N, 4] per-image reference points.
            sopm이 None이면 원본을 batch-expand하여 반환.
        """
        N = refpoint_embed_weight.shape[0]

        # --- SOPM이 없으면 bypass (원본 batch-expand) ---
        if sopm is None:
            return refpoint_embed_weight.unsqueeze(0).expand(batch_size, -1, -1)

        B = batch_size

        # --- 적응적 비율 계산 ---
        ratio = self._compute_ratio(sopm.detach())  # [B], detach: SOPM gradient 차단

        # 배치 내 평균 비율 사용 (쿼리 수는 배치 내 동일해야 텐서 정합)
        avg_ratio = ratio.mean().item()
        n_small = max(1, min(int(round(avg_ratio * N)), N - 1))
        n_general = N - n_small

        # --- N_small: SOPM 밀도 기반 좌표 샘플링 ---
        ref_small = self._sample_positions(sopm.detach(), n_small)  # [B, n_small, 4]

        # --- N_general: 원본 refpoint의 뒷부분 유지 ---
        # 원본 refpoint를 batch-expand
        ref_general = refpoint_embed_weight[n_small:N, :].unsqueeze(0).expand(B, -1, -1)
        # [B, n_general, 4]

        # --- 병합 ---
        ref_combined = torch.cat([ref_small, ref_general], dim=1)  # [B, N, 4]

        return ref_combined
