from ultralytics import YOLO
import pandas as pd # 결과 정리를 위해 추가

def main():
    # 1. 설정
    YOLO_DIR = r"C:\env\yolo_format\Merged_Dataset"
    DATA_YAML = f"{YOLO_DIR}/data.yaml"
    
    # 학습할 모델 스케일 리스트
    model_scales = ['n', 's', 'm', 'l']
    results_summary = []

    for scale in model_scales:
        model_name = f"yolo11{scale}"
        print(f"\n {model_name} 학습 시작...")

        # 모델 로드 (yolo11n.pt, yolo11s.pt ...)
        model = YOLO(f"{model_name}.pt")

        # 2. 학습 (Train)
        model.train(
            data=DATA_YAML,
            epochs=100,
            batch=16, # 메모리 부족 시 l 모델에서는 줄여야 할 수도 있습니다.
            imgsz=640,
            device=0,
            project="./runs_compare",
            name=model_name,
            exist_ok=True,
            # 소형 객체 최적화 및 Augmentation
            mosaic=1.0,
            mixup=0.1,
            copy_paste=0.1,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            flipud=0.5,
            fliplr=0.5,
            # 학습 안정화
            warmup_epochs=3,
            patience=20,
            save_period=10,
        )

        # 3. 검증 (Validation)
        print(f"📊 {model_name} 검증 중...")
        metrics = model.val(data=DATA_YAML, imgsz=640, device=0)
        
        # 결과 저장
        results_summary.append({
            "Model": model_name,
            "mAP50": round(metrics.box.map50, 4),
            "mAP50-95": round(metrics.box.map, 4)
        })

    # 4. 최종 결과 출력
    print("\n" + "="*30)
    print("      YOLOv11 비교 결과")
    print("="*30)
    df = pd.DataFrame(results_summary)
    print(df.to_string(index=False))
    print("="*30)

if __name__ == '__main__':
    main()