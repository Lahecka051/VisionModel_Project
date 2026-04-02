import torch
import gc
from ultralytics import YOLO

# =============================================
# 군 경계 작전 환경 내 인식 데이터만으로 학습되는 코드
# =============================================
# 데이터 설정 파일 경로
DATA_YAML = r"C:\github\VLA\yolo_format\data.yaml"

# 결과 저장 경로 (프로젝트 루트)
PROJECT_DIR = r"C:\github\VLA\yolo_format\runs"

# 비교할 모델 리스트
model_list = ['yolov8n.pt','yolov8s.pt'] # 'yolov8m.pt', 'yolov8l.pt'

# 공통 하이퍼파라미터 (사용자 지정 값)
HYPERPARAMS = {
    'epochs': 100,         # 총 에폭 수
    'imgsz': 640,         # 이미지 크기
    'batch': 32,          # 배치 크기
    'device': 0,          # GPU 장치 번호
    'workers': 16,        # 데이터 로딩 워커 수

    'optimizer': 'AdamW', # 옵티마이저
    'lr0': 0.0004,        # 학습률 지정
    'weight_decay': 0.0001, # 가중치 감쇠 지정
    'exist_ok': True,     # 덮어쓰기 허용 여부
    'patience': 25,       # 25 epoch 동안 성능 향상 없으면 조기 종료 (시간 절약)
    'verbose': True      # 학습 로그 자세히 출력
}

def train_sequence():
    print(f"학습 시퀸스 시작: {model_list}")
    print(f"결과 저장 위치: {PROJECT_DIR}")
    print("="*50)

    for model_name in model_list:
        run_name = f"{model_name.split('.')[0]}_custom" # 예: yolov8n_custom
        
        print(f"\n [START] {run_name} 모델 학습 시작...")
        
        try:
            # 1. 모델 로드
            model = YOLO(model_name)
            
            # 2. 학습 시작
            # data.yaml 경로가 정확한지 꼭 확인하세요!
            results = model.train(
                data=DATA_YAML,
                project=PROJECT_DIR,
                name=run_name,
                **HYPERPARAMS
            )
            
            print(f"[DONE] {run_name} 학습 완료! (Best mAP50-95: {results.box.map:.4f})")

        except Exception as e:
            print(f"[ERROR] {run_name} 학습 중 오류 발생: {e}")
        
        finally:
            # 3. 메모리 정리 (매우 중요!)
            # 모델 변수를 삭제하고 GPU 캐시를 비워야 다음 큰 모델 학습 시 VRAM 부족이 안 뜹니다.
            del model
            gc.collect()
            torch.cuda.empty_cache()
            print("GPU 메모리 정리 완료.\n")

    print("="*50)
    print("모든 모델의 학습이 종료되었습니다.")

if __name__ == '__main__':
    train_sequence()