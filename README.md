# SOD-DETR

## 선택적 특징 융합을 적용한 트랜스포머 기반 소형 객체 탐지

**Transformer Based Small Object Detection with Selective Feature Fusion**

| 항목 | 내용 |
|------|------|
| 게재지 | 국방품질연구논집 |
| 논문번호 | J1_202600061 |
| 국문 제목 | 선택적 특징 융합을 적용한 트랜스포머 기반 소형 객체 탐지 |
| 영문 제목 | Transformer Based Small Object Detection with Selective Feature Fusion |

군 경계 작전 환경에서의 소형 객체 탐지를 위한 RF-DETR 기반 프레임워크이다. **SCA**(Selective Cross-Attention) 모듈과 **NWD**(Normalized Wasserstein Distance) matching cost를 적용하여 Person, Bird와 같은 소형 객체의 탐지 정밀도를 개선한다.

---

## Highlights

- **SCA (Selective Cross-Attention)**: DINOv2 Block 4의 self-attention map을 사전 지식으로 활용하고, 경량 CNN 분기(stride-8)의 상위 25% 토큰만 ViT 특징에 교차 어텐션으로 융합한다.
- **NWD (Normalized Wasserstein Distance)**: 헝가리안 정합 비용에서 GIoU를 NWD로 대체하여 소형 객체의 1~2px 위치 오차로 IoU가 급변하는 문제를 완화한다. 학습 시에만 동작하므로 추론 비용과 추가 파라미터는 증가하지 않는다.
- **AGX Orin TensorRT FP16 배포**: Baseline 대비 latency overhead +7.0%(6.97 ms -> 7.46 ms) 수준으로 온디바이스 적용 가능성을 확인했다.

| Model | mAP@50:95 | Person AP | Total FP | FP 감소율 |
|-------|-----------|-----------|----------|-----------|
| RF-DETR-M (Baseline) | 0.902 | 0.673 | 121 | - |
| SOD-SCA | 0.904 | 0.704 | 83 | -31.4% |
| **SOD-SCA+NWD** | **0.906** | **0.713** | **74** | **-38.8%** |

전체 벤치마크 결과는 [benchmark/Result.md](benchmark/Result.md)에서 확인할 수 있다.

---

## Repository Structure

```text
Defense_Quality_Research_Council/
|-- Source_codes/
|   |-- modules/                  # SOD-DETR 핵심 모듈 및 RF-DETR 패치 파일
|   |   |-- sca.py                # SCA: Selective Cross-Attention
|   |   |-- nwd.py                # NWD: pairwise Wasserstein distance
|   |   |-- backbone.py           # SCA 통합 버전
|   |   |-- lwdetr.py             # SCA/NWD 파라미터 전달
|   |   |-- matcher.py            # NWD matching cost
|   |   `-- SOD-DETR_소스파일_배치_가이드.md
|   `-- train/                    # RF-DETR, SOD-DETR, YOLO 학습 스크립트
|-- RF-DETR/
|   |-- SOD/                      # SOD-DETR 학습 로그 및 best checkpoint
|   |-- rfdetr_m/                 # RF-DETR-M 학습 로그 및 best checkpoint
|   `-- rfdetr_l/                 # RF-DETR-L 학습 로그 및 best checkpoint
|-- YOLO/
|   |-- V8/
|   |-- V11/
|   |-- V12/                      # YOLO 비교군 학습 결과
|   `-- data.yaml
|-- benchmark/
|   `-- Result.md                 # RTX 5090 / AGX Orin 벤치마크 결과
|-- Documents/
|   |-- Figure.pptx
|   |-- 논문유사도검사결과_확인서.pdf
|   `-- 선택적 특징 융합을 적용한 트랜스포머 기반 소형 객체 탐지.pdf
|-- requirements.txt
`-- README.md
```

대용량 체크포인트와 실험 산출물은 Git LFS 포인터로 관리한다. 중간 체크포인트와 벤치마크 이미지 등 반복 생성 가능한 파일은 저장소 크기 최적화를 위해 정리했다.

---

## Datasets

AI-HUB의 군 경계 작전 환경 데이터셋 2종을 사용한다.

| 구분 | 내용 |
|------|------|
| 클래스 | Fishing_Boat, Merchant_Ship, Warship, Person, Bird, Fixed_Wing, Rotary_Wing, UAV, Leaflet, Trash_Bomb |
| 테스트 영상 | 20,172건 |
| GT 인스턴스 | 41,472건 |
| 실데이터 | [AI-HUB ID 71858](https://aihub.or.kr/aihubdata/data/view.do?srchOptnCnd=OPTNCND001&currMenu=115&topMenu=100&searchKeyword=%EA%B5%B0%EA%B2%BD%EA%B3%84&aihubDataSe=data&dataSetSn=71858) |
| 합성 데이터 | [AI-HUB ID 71856](https://aihub.or.kr/aihubdata/data/view.do?srchOptnCnd=OPTNCND001&currMenu=115&topMenu=100&searchKeyword=%EA%B5%B0%EA%B2%BD%EA%B3%84&aihubDataSe=data&dataSetSn=71856) |

---

## Installation

```bash
pip install -r requirements.txt
pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128
```

PyTorch와 CUDA 버전은 실행 환경에 맞게 조정한다.

---

## SOD-DETR Module Placement

`Source_codes/modules/`의 파일을 설치된 `rfdetr` 패키지의 `rfdetr/models/` 경로에 배치한다.

```bash
python -c "import rfdetr, os; print(os.path.join(os.path.dirname(rfdetr.__file__), 'models'))"
```

| 파일 | 동작 |
|------|------|
| `sca.py`, `nwd.py` | 신규 추가 |
| `backbone.py`, `lwdetr.py`, `matcher.py` | RF-DETR 원본 파일 교체 |

상세한 배치 절차는 [Source_codes/modules/SOD-DETR_소스파일_배치_가이드.md](Source_codes/modules/SOD-DETR_%EC%86%8C%EC%8A%A4%ED%8C%8C%EC%9D%BC_%EB%B0%B0%EC%B9%98_%EA%B0%80%EC%9D%B4%EB%93%9C.md)를 참고한다.

---

## Training

학습 스크립트의 `dataset_dir`, `output_dir`, `pretrain_weights` 등 경로를 실행 환경에 맞게 수정한 후 실행한다.

```bash
python Source_codes/train/train_rfdetr_baseline.py --size medium
python Source_codes/train/train_sod_sca_only.py
python Source_codes/train/train_sod_sca_nwd.py
python Source_codes/train/train_yolov8.py
python Source_codes/train/train_yolov11.py
python Source_codes/train/train_yolov12.py
```

---

## Benchmark Environment

| 항목 | 사양 |
|------|------|
| GPU | NVIDIA GeForce RTX 5090 32GB |
| 온디바이스 | NVIDIA Jetson AGX Orin 64GB, TensorRT FP16 |
| PyTorch | 2.10.0+cu128 |
| CUDA | 12.8 |
| 속도 측정 | warmup 50, 200 runs x 10 rounds, IQR 1.5x trimming, median of means |

---

## Documents

- 벤치마크 상세: [benchmark/Result.md](benchmark/Result.md)
- 발표/그림 자료: [Documents/Figure.pptx](Documents/Figure.pptx)
- 논문 유사도 확인서: [Documents/논문유사도검사결과_확인서.pdf](Documents/%EB%85%BC%EB%AC%B8%EC%9C%A0%EC%82%AC%EB%8F%84%EA%B2%80%EC%82%AC%EA%B2%B0%EA%B3%BC_%ED%99%95%EC%9D%B8%EC%84%9C.pdf)
- 논문 PDF: [Documents/선택적 특징 융합을 적용한 트랜스포머 기반 소형 객체 탐지.pdf](Documents/%EC%84%A0%ED%83%9D%EC%A0%81%20%ED%8A%B9%EC%A7%95%20%EC%9C%B5%ED%95%A9%EC%9D%84%20%EC%A0%81%EC%9A%A9%ED%95%9C%20%ED%8A%B8%EB%9E%9C%EC%8A%A4%ED%8F%AC%EB%A8%B8%20%EA%B8%B0%EB%B0%98%20%EC%86%8C%ED%98%95%20%EA%B0%9D%EC%B2%B4%20%ED%83%90%EC%A7%80.pdf)

---

## Acknowledgements

본 코드는 다음 오픈소스 프로젝트를 기반으로 한다.

- [RF-DETR](https://github.com/roboflow/rf-detr) (Roboflow, Apache 2.0)
- [LW-DETR](https://github.com/Atten4Vis/LW-DETR) (Baidu)
- [DETR](https://github.com/facebookresearch/detr) (Facebook AI Research)
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- NWD: J. Wang et al., "A Normalized Gaussian Wasserstein Distance for Tiny Object Detection," CVPR 2022.
