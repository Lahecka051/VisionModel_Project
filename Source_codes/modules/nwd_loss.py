# ------------------------------------------------------------------------
# SOD: Small Object Detection
# ------------------------------------------------------------------------
# NWD Loss: Scale-Adaptive NWD/CIoU Loss
# Patch-size-based NWD/CIoU branching for bbox regression loss.
# Replaces GIoU loss in SetCriterion.loss_boxes.
# ------------------------------------------------------------------------

"""
NWD Loss: loss_giou를 패치 수 기반 NWD/CIoU 분기로 교체.

소형 객체 (normalized area <= threshold):
  - NWD loss 사용. 1-2px bbox 오차에도 안정적인 gradient 제공.
  - GIoU/CIoU는 극소형 bbox에서 IoU 급락으로 gradient 불안정.

대형 객체 (normalized area > threshold):
  - CIoU loss 사용. bbox 정밀 회귀에 최적화된 표준 loss.

threshold = 0.0031 ≈ (4*16/576)^2
  576x576 입력에서 패치 4개 이하 면적에 해당.

학습 전용 모듈, 추론 비용 0%, 추가 파라미터 0.
"""

import math
import torch
from rfdetr.util import box_ops


def nwd_loss_elementwise(pred_boxes, target_boxes, C=0.5):
    """Element-wise NWD loss (1 - NWD) for matched pairs.

    Args:
        pred_boxes: (N, 4) [cx, cy, w, h] normalized [0,1]
        target_boxes: (N, 4) [cx, cy, w, h] normalized [0,1]
        C: normalization constant

    Returns:
        loss: (N,) element-wise NWD loss in [0, 1]
    """
    mu1, mu2 = pred_boxes[:, :2], target_boxes[:, :2]
    sigma1, sigma2 = pred_boxes[:, 2:] / 2, target_boxes[:, 2:] / 2

    center_dist_sq = (mu1 - mu2).pow(2).sum(dim=-1)
    sigma_dist_sq = (sigma1 - sigma2).pow(2).sum(dim=-1)
    w2 = (center_dist_sq + sigma_dist_sq).clamp(min=1e-8).sqrt()

    nwd = torch.exp(-w2 / C)
    return 1.0 - nwd


def ciou_loss_elementwise(pred_boxes, target_boxes):
    """Element-wise CIoU loss for matched pairs.

    Args:
        pred_boxes: (N, 4) [cx, cy, w, h] normalized [0,1]
        target_boxes: (N, 4) [cx, cy, w, h] normalized [0,1]

    Returns:
        loss: (N,) element-wise CIoU loss
    """
    pred_xyxy = box_ops.box_cxcywh_to_xyxy(pred_boxes)
    tgt_xyxy = box_ops.box_cxcywh_to_xyxy(target_boxes)

    inter_x1 = torch.max(pred_xyxy[:, 0], tgt_xyxy[:, 0])
    inter_y1 = torch.max(pred_xyxy[:, 1], tgt_xyxy[:, 1])
    inter_x2 = torch.min(pred_xyxy[:, 2], tgt_xyxy[:, 2])
    inter_y2 = torch.min(pred_xyxy[:, 3], tgt_xyxy[:, 3])
    inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

    pred_area = (pred_xyxy[:, 2] - pred_xyxy[:, 0]) * (pred_xyxy[:, 3] - pred_xyxy[:, 1])
    tgt_area = (tgt_xyxy[:, 2] - tgt_xyxy[:, 0]) * (tgt_xyxy[:, 3] - tgt_xyxy[:, 1])
    union_area = pred_area + tgt_area - inter_area

    iou = inter_area / union_area.clamp(min=1e-8)

    enclose_x1 = torch.min(pred_xyxy[:, 0], tgt_xyxy[:, 0])
    enclose_y1 = torch.min(pred_xyxy[:, 1], tgt_xyxy[:, 1])
    enclose_x2 = torch.max(pred_xyxy[:, 2], tgt_xyxy[:, 2])
    enclose_y2 = torch.max(pred_xyxy[:, 3], tgt_xyxy[:, 3])

    enclose_diag_sq = (enclose_x2 - enclose_x1).pow(2) + (enclose_y2 - enclose_y1).pow(2)
    center_dist_sq = (pred_boxes[:, 0] - target_boxes[:, 0]).pow(2) + \
                     (pred_boxes[:, 1] - target_boxes[:, 1]).pow(2)
    rho = center_dist_sq / enclose_diag_sq.clamp(min=1e-8)

    w_pred, h_pred = pred_boxes[:, 2], pred_boxes[:, 3]
    w_tgt, h_tgt = target_boxes[:, 2], target_boxes[:, 3]
    v = (4.0 / (math.pi ** 2)) * (
        torch.atan(w_tgt / h_tgt.clamp(min=1e-8)) -
        torch.atan(w_pred / h_pred.clamp(min=1e-8))
    ).pow(2)
    with torch.no_grad():
        alpha = v / (1.0 - iou + v).clamp(min=1e-8)

    ciou = iou - rho - alpha * v
    return 1.0 - ciou


def nwd_adaptive_loss(pred_boxes, target_boxes, threshold=0.0031, nwd_C=0.5):
    """Scale-Adaptive NWD/CIoU Loss.

    패치 수 기반 분기:
      area <= threshold -> NWD loss (소형 객체)
      area > threshold  -> CIoU loss (대형 객체)

    Args:
        pred_boxes: (N, 4) [cx, cy, w, h] normalized [0,1]
        target_boxes: (N, 4) [cx, cy, w, h] normalized [0,1]
        threshold: normalized area threshold (default 0.0031 ~ 4 patches at 16/576)
        nwd_C: NWD normalization constant

    Returns:
        loss: (N,) element-wise loss
    """
    N = pred_boxes.shape[0]
    if N == 0:
        return pred_boxes.new_zeros(0)

    with torch.no_grad():
        gt_area = target_boxes[:, 2] * target_boxes[:, 3]
        small_mask = gt_area <= threshold

    loss = pred_boxes.new_zeros(N)

    if small_mask.any():
        loss[small_mask] = nwd_loss_elementwise(
            pred_boxes[small_mask], target_boxes[small_mask], C=nwd_C
        )

    large_mask = ~small_mask
    if large_mask.any():
        loss[large_mask] = ciou_loss_elementwise(
            pred_boxes[large_mask], target_boxes[large_mask]
        )

    return loss
