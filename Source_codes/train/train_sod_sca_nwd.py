"""
SOD-DETR: SCA + NWD 학습

SCA (Selective Cross-Attention) + NWD (Normalized Wasserstein Distance) matching cost
RF-DETR-M baseline 가중치 위에 파인튜닝

실행:
  python train_sod_sca_nwd.py
"""

from rfdetr import RFDETRMedium

def main():
    model = RFDETRMedium(
        pretrain_weights=r"C:\gitnconda\Swin-Transformer\runs_compare\rfdetr_m\checkpoint_best_total.pth",
        # SCA
        sca_enabled=True,
        # NWD matching cost (GIoU 대체)
        set_cost_nwd=2,
        set_cost_giou=0,
    )

    model.train(
        dataset_dir=r"C:\gitnconda\Swin-Transformer\rfdetr_data",
        epochs=100,
        batch_size=8,
        grad_accum_steps=2,
        lr=1e-4,
        output_dir=r"C:\gitnconda\Swin-Transformer\runs_compare\sod_sca_nwd",
        device="cuda",
        early_stopping=True,
        early_stopping_patience=10,
        num_workers=8,
        tensorboard=True,
    )


if __name__ == "__main__":
    main()
