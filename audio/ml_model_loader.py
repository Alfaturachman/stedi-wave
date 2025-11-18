import os
import joblib
from django.conf import settings

GB_MODEL_PATH = os.path.join(settings.BASE_DIR, "ml_model", "trainedGBModel_new.pkl")

try:
    gbModel, classFolders = joblib.load(GB_MODEL_PATH)
    print("✓ Gradient Boosting model loaded")
except Exception as e:
    gbModel = None
    classFolders = None
    print(f"✗ Error loading GB model: {e}")
