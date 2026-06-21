import os
import mlflow
from ultralytics import YOLO,settings

# 1. model config
MODEL_NAME = "yolo26m"
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. paths config
PRETRAINED_MODEL = os.path.join(CURRENT_DIR, "pretrained_models", f"{MODEL_NAME}.pt")
DATA_CONFIG_PATH = os.path.join(CURRENT_DIR, "../../../../../datasets/ingeniia_services_smart_checkout_img_v1.0.0_training_20260120/split_data/data_augmentation/data.yaml")
LOCAL_RUNS_DIR = os.path.join(CURRENT_DIR, "src/training/train/runs/YOLO_Detection")

# 3. mlflow config
MLFLOW_DB_PATH = os.path.join(CURRENT_DIR, "mlflow.db")
mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH}")

# 4. ultralytics config
settings.update({"mlflow": True, "runs_dir": LOCAL_RUNS_DIR})

# 5. train
def train_detection():
    print(f"🚀 Iniciando entrenamiento de Detección con {MODEL_NAME}...")
    
    # load model (GPU)
    model = YOLO(PRETRAINED_MODEL).to('cuda')
    
    # mlflow config
    os.environ["MLFLOW_EXPERIMENT_NAME"] = "Xrays_Detection_Experiments"
    os.environ["MLFLOW_RUN_NAME"] = "02_Detection_Baseline_YOLO26"

    # train
    model.train(
        data=DATA_CONFIG_PATH,
        epochs=100,
        batch=32,
        imgsz=640,
        patience=10,
        task='detect',
        project=LOCAL_RUNS_DIR,
        name=f"{MODEL_NAME}_detect",
        exist_ok=True,
        save=True
    )

    print("✅ Entrenamiento de Detección finalizado y registrado en MLflow.")

if __name__ == '__main__':
    train_detection()