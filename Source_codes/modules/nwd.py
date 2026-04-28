# ------------------------------------------------------------------------
# SOD: Small Object Detection
# ------------------------------------------------------------------------
# NWD: Normalized Wasserstein Distance
# bbox를 2D Gaussian으로 모델링하여 Hungarian matching cost에 적용.
# Reference: J. Wang et al., "A Normalized Gaussian Wasserstein Distance
#            for Tiny Object Detection," CVPR, 2022.
# ------------------------------------------------------------------------

"""
NWD: bbox를 2D Gaussian N(cx, cy, w/2, h/2)으로 모델링하여
Wasserstein-2 거리를 계산하고, exp(-W2/C)로 정규화한다.

논문 식 (9), (10):
소형 객체(16x16)에서 1px 오차: IoU ~0.68, NWD ~0.95
→ matching 단계에서 소형 객체에 안정적인 cost를 제공한다.
"""

import torch


def box_to_gaussian(boxes):
    """cxcywh 박스를 2D Gaussian 파라미터로 변환. 논문 식 (7), (8).

    Args:
        boxes: (N, 4) tensor [cx, cy, w, h], 정규화 좌표 [0, 1]

    Returns:
        mu: (N, 2) 중심 [cx, cy]
        sigma: (N, 2) 표준편차 [w/2, h/2]
    """
    mu = boxes[:, :2]
    sigma = boxes[:, 2:] / 2
    return mu, sigma


def wasserstein2_pairwise(mu1, sigma1, mu2, sigma2):
    """두 Gaussian 집합 간 pairwise Wasserstein-2 거리의 제곱. 논문 식 (9).

    W2^2 = ||mu1 - mu2||^2 + ||sigma1 - sigma2||^2

    대각 공분산 가정 (Σ = diag(σ_w², σ_h²))에서 closed-form.

    Args:
        mu1: (N, 2), sigma1: (N, 2)  — 예측 박스
        mu2: (M, 2), sigma2: (M, 2)  — GT 박스

    Returns:
        w2_sq: (N, M) pairwise W2 거리의 제곱
    """
    # ||mu1_i - mu2_j||^2
    mu_diff_sq = torch.cdist(mu1, mu2, p=2).pow(2)

    # ||sigma1_i - sigma2_j||^2
    sigma_diff_sq = torch.cdist(sigma1, sigma2, p=2).pow(2)

    return mu_diff_sq + sigma_diff_sq


def nwd_pairwise(pred_boxes, tgt_boxes, C=0.5):
    """Pairwise NWD (Normalized Wasserstein Distance). 논문 식 (10).

    NWD = exp(-sqrt(W2^2) / C)

    값 범위: [0, 1]. 1이면 완전 일치, 0이면 완전 불일치.

    Args:
        pred_boxes: (N, 4) [cx, cy, w, h] 정규화 좌표
        tgt_boxes: (M, 4) [cx, cy, w, h] 정규화 좌표
        C: 정규화 상수. 0.5 (논문 기준).

    Returns:
        nwd: (N, M) pairwise NWD 값
    """
    mu1, sigma1 = box_to_gaussian(pred_boxes)
    mu2, sigma2 = box_to_gaussian(tgt_boxes)

    w2_sq = wasserstein2_pairwise(mu1, sigma1, mu2, sigma2)
    w2 = w2_sq.clamp(min=1e-8).sqrt()

    nwd = torch.exp(-w2 / C)
    return nwd
