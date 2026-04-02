# SOD-DETR 소스파일 배치 가이드

SOD-DETR은 RF-DETR(rfdetr 1.5.2) 기반의 군 경계 감시 소형 객체 탐지 프레임워크이다.
SCA(Selective Cross-Attention)와 NWD(Normalized Wasserstein Distance) 두 개 모듈을 적용하여 소형 객체의 bbox 정밀도를 개선한다.

## 사전 요구사항

```bash
pip install rfdetr==1.5.2
```

## 배치 경로

모든 파일은 rfdetr 패키지 내 `rfdetr/models/` 디렉토리에 배치한다.

```bash
# 경로 확인
python -c "import rfdetr; import os; print(os.path.join(os.path.dirname(rfdetr.__file__), 'models'))"
```

## 파일 구성

### 원본 교체 (3개)

기존 RF-DETR 파일을 덮어쓴다.

| 파일 | 설명 |
|------|------|
| `backbone.py` | SCA 모듈 통합 (encoder→projector 간극) |
| `lwdetr.py` | SCA 파라미터를 backbone에 전달 |
| `matcher.py` | 헝가리안 정합 비용에 NWD 항 추가 |

### 신규 추가 (2개)

RF-DETR 원본에 존재하지 않는 파일이다.

| 파일 | 설명 |
|------|------|
| `sca.py` | SCA 전체 (CNN Branch + 히트맵 + Top-K + Cross-Attention) |
| `nwd_matching.py` | NWD pairwise 거리 계산 |

### 수정 없음

`__init__.py`, `transformer.py`, `position_encoding.py`, `segmentation_head.py`, `backbone/`, `ops/`는 원본 그대로 사용한다.

## 모듈 설명

### SCA (Selective Cross-Attention)

DINOv2 블록 4의 self-attention map을 사전 지식으로 활용하여, 경량 CNN 분기(stride-8)의 상위 25% 토큰만 선택적으로 ViT 특징에 교차 어텐션으로 융합한다.

- CNN Branch: Conv3x3(stride 2) x3 + Conv3x3(stride 1) x1, GroupNorm, 출력 [B, 256, 72, 72]
- 소형 객체 히트맵: CNN 특징과 블록 4 어텐션 맵을 결합(Conv1x1 + Sigmoid)
- Top-K 선택: 히트맵 상위 25% 위치의 CNN 토큰만 선택
- 교차 어텐션: 선택된 CNN 토큰을 K/V, ViT 전체 토큰을 Q로 2D sinusoidal PE 적용
- 게이팅 잔차 연결: P3' = P3 + α_gate · CA(Q, K, V), α = 0.01 초기화

관련 파일: `sca.py`, `backbone.py`

### NWD (Normalized Wasserstein Distance)

헝가리안 정합의 비용 함수에서 GIoU를 NWD로 대체한다. bbox를 2D 가우시안 분포 N(cx, cy, w/2, h/2)로 모델링하여 Wasserstein-2 거리를 계산하고, 지수 함수로 0~1 범위의 유사도로 정규화한다. 소형 객체에서 1~2px 오차에도 IoU가 급변하는 문제를 완화하여 안정적인 GT 할당을 제공한다. 학습 시에만 동작하므로 추론 비용 0%, 추가 파라미터 0개이다.

관련 파일: `nwd_matching.py`, `matcher.py`

## 원본 복원

```bash
pip install --force-reinstall rfdetr==1.5.2
```
