"""Shared paths and constants for the FitML Flask backend."""
import os

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)

UPLOADS_DIR = os.path.join(BACKEND_DIR, "uploads")
DB_PATH = os.path.join(BACKEND_DIR, "fitml.db")

CATALOG_DIR = os.path.join(PROJECT_ROOT, "data", "catalog")
METADATA_CSV = os.path.join(CATALOG_DIR, "metadata.csv")
MENS_CHART_CSV = os.path.join(PROJECT_ROOT, "data", "raw", "mens_size_charts.csv")

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "xgboost_weighted.joblib")
LABEL_ENCODER_PATH = os.path.join(PROJECT_ROOT, "models", "label_encoder.joblib")

# Privacy: uploaded photos live for 24h max (docs/privacy.md, RA 10173 framing)
UPLOAD_TTL_HOURS = 24

# Categories that need a full-body photo to composite
BOTTOM_CATEGORIES = {"jeans", "skirt", "shorts", "slacks", "trousers"}

MAX_UPLOAD_MB = 10
ALLOWED_PHOTO_EXT = {".jpg", ".jpeg", ".png", ".webp"}
