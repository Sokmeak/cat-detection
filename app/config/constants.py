"""
Application Constants
====================
Configuration values and constants
"""

# Page configuration
PAGE_TITLE = "Cat Detector"
PAGE_ICON = "🔍"
LAYOUT = "wide"

# Model configuration
SUPPORTED_IMAGE_TYPES = ["jpg", "jpeg", "png", "bmp"]
DEFAULT_CONFIDENCE = 0.25
MIN_CONFIDENCE = 0.0
MAX_CONFIDENCE = 1.0
CONFIDENCE_STEP = 0.05

# Default model files
DEFAULT_MODELS = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"]

# Paths
RUNS_TRAIN_PATH = "runs/train"
WEIGHTS_SUBPATH = "weights/best.pt"

# Messages
NO_MODELS_ERROR = "⚠️ No trained models found"
MODEL_LOAD_ERROR = "Failed to load model"
MODEL_READY_SUCCESS = "✓ Model Ready"

# Tab labels
TAB_UPLOAD = "📤 Upload Image"
TAB_CAMERA = "📷 Camera"
