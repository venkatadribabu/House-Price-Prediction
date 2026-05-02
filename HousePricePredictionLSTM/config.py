import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# Default Zillow city CSV; override with env ZILLOW_CSV or train.py --data
DEFAULT_DATA_PATH = Path(
    r"c:\Users\harsh\Downloads\City_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv"
)


def zillow_csv_path() -> Path:
    override = os.environ.get("ZILLOW_CSV", "").strip()
    return Path(override) if override else DEFAULT_DATA_PATH


ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

DEFAULT_LOOKBACK = 12
DEFAULT_HIDDEN = 128
DEFAULT_NUM_LAYERS = 1
DEFAULT_DROPOUT = 0.0
DEFAULT_EPOCHS = 100
DEFAULT_BATCH_SIZE = 64
DEFAULT_LR = 1e-3
DEFAULT_TRAIN_RATIO = 0.8
