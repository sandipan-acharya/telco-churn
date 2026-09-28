import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "prefect" / "flows"))

from telco_pipeline import telco_pipeline

if __name__ == "__main__":
    telco_pipeline.serve(
        name="telco-every-2-min",
        interval=120,
        tags=["assignment", "telco-churn-group-40"],
    )