"""
RF-DETR Medium / Large 학습

사전 준비:
  pip install rfdetr
  python yolo2coco.py  (COCO 포맷 변환)

실행:
  python train_rfdetr.py --size medium
  python train_rfdetr.py --size large

RF-DETR COCO 디렉토리 구조 (valid 필수):
  rfdetr_data/
    train/
      _annotations.coco.json
      image1.jpg ...
    valid/
      _annotations.coco.json
      image1.jpg ...
"""

import argparse
from rfdetr import RFDETRMedium, RFDETRLarge

# ── 설정 ──
DATASET_DIR = r"C:\gitnconda\Swin-Transformer\rfdetr_data"
EPOCHS = 100
RESUME_PATH = r"./runs_compare/rfdetr_l/checkpoint.pth"  # 체크포인트 경로

def train(size):
    if size == "medium":
        model = RFDETRMedium()
        batch = 16
        grad_accum = 1
        output = "./runs_compare/rfdetr_m"
    elif size == "large":
        model = RFDETRLarge()
        batch = 4
        grad_accum = 4   # effective batch = 4*4 = 16
        output = "./runs_compare/rfdetr_l"
    else:
        raise ValueError(f"Unknown size: {size}")

    print(f"\n=== RF-DETR {size.upper()} 학습 시작 ===")
    print(f"  Batch: {batch}, Grad Accum: {grad_accum}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Output: {output}")

    model.train(
        dataset_dir=DATASET_DIR,
        epochs=EPOCHS,
        batch_size=batch,
        grad_accum_steps=grad_accum,
        lr=1e-4,
        resume=RESUME_PATH, # 체크포인트 학습 재개 (OPTIONAL)
        output_dir=output,
        device="cuda",
        early_stopping=True,
        early_stopping_patience=15,
        num_workers=8,
    )

    print(f"\n=== RF-DETR {size.upper()} 학습 완료 ===")
    print(f"  결과: {output}")
    print(f"  Best weights: {output}/checkpoint_best_total.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=str, choices=["medium", "large"],
                        required=True)
    args = parser.parse_args()
    train(args.size)