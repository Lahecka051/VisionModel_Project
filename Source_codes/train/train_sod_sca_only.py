"""
SOD-DETR: SCA only 학습

SCA (Selective Cross-Attention)만 적용, NWD 비활성 (GIoU 유지)
RF-DETR-M baseline 가중치 위에 파인튜닝

실행:
  python train_sod_sca_only.py
"""

from rfdetr import RFDETRMedium

def main():
    model = RFDETRMedium(
        pretrain_weights=r"C:\gitnconda\Swin-Transformer\runs_compare\rfdetr_m\checkpoint_best_total.pth",
        # SCA
        sca_enabled=True,
        # NWD 비활성 (기본값: cost_nwd=0, GIoU 유지)
    )

    model.train(
        dataset_dir=r"C:\gitnconda\Swin-Transformer\rfdetr_data",
        epochs=100,
        batch_size=8,
        grad_accum_steps=2,
        lr=1e-4,
        output_dir=r"C:\gitnconda\Swin-Transformer\runs_compare\sod_sca_only",
        device="cuda",
        early_stopping=True,
        early_stopping_patience=10,
        num_workers=8,
        tensorboard=True,
    )


if __name__ == "__main__":
    main()
