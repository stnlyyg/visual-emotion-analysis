from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent

# Checkpoints
DATASET_CHECKPOINT = "FastJobs/Visual_Emotional_Analysis"
MODEL_CHECKPOINT = "google/vit-base-patch16-224-in21k"

# Data Processing
DATA_DIR = ROOT_DIR / "data"

PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_TRAIN_DATA = PROCESSED_DIR / "train"
PROCESSED_TEST_DATA = PROCESSED_DIR / "test"

DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

BRIGHTNESS = 0.2
CONTRAST = 0.2
SATURATION = 0.2
HUE = 0.2

# Log
LOG_DIR = ROOT_DIR / "log"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Training
TRAINED_MODEL_PATH = ROOT_DIR / "model"
TRAINED_MODEL_PATH.mkdir(parents=True, exist_ok=True)

BEST_MODEL_CHECKPOINT = TRAINED_MODEL_PATH / "checkpoint-160"

TrainingArgs = {
    "output_dir": str(TRAINED_MODEL_PATH),
    "remove_unused_columns": False,
    "eval_strategy": "epoch",
    "save_strategy": "epoch",
    "learning_rate": 1e-4,
    "per_device_train_batch_size": 8,
    "gradient_accumulation_steps": 4,
    "per_device_eval_batch_size": 8,
    "num_train_epochs": 10,
    "warmup_ratio": 0.5,
    "logging_steps": 1000,
    "load_best_model_at_end": True,
    "metric_for_best_model": "accuracy",
    "save_total_limit": 1,
    "weight_decay": 0.1,
    "logging_dir": LOG_DIR,
    "logging_strategy": "steps",
    "logging_steps": 1000,
    "report_to": "tensorboard"
}

#Evaluation
EVAL_RESULT_DIR = ROOT_DIR / "eval_results"
EVAL_RESULT_DIR.mkdir(parents=True, exist_ok=True)

EVAL_ARGS = {
    "output_dir": EVAL_RESULT_DIR,
    "per_device_eval_batch_size": 8,
    "do_train": False,
    "do_eval": True
}