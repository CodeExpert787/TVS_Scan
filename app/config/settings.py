import os

# Base directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

# Create necessary directories
for directory in [DATA_DIR, MODELS_DIR, LOGS_DIR, STATIC_DIR, TEMPLATES_DIR]:
    os.makedirs(directory, exist_ok=True)

# Model settings
MODEL_PARAMS = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5]
}

# File paths
MODEL_PATH = os.path.join(MODELS_DIR, "pcos_model.joblib")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.joblib")
FIELD_NAMES_PATH = os.path.join(DATA_DIR, "field_names.json")

# Target column name
TARGET_COLUMN = "Jika anda sudah membuat Quiz Jenis PCOS, sila nyatakan Jenis PCOS anda._ If you had done the PCOS Type Quiz, please state your PCOS Type" 