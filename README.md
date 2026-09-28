# Telco Churn — minimum files

## Run 1.3 and 1.4 on your laptop

```bash
cd telco-min
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements-prefect.txt

python src/preprocess.py
python src/eda.py
```

Outputs:
- `data/processed/telco_preprocessed.csv`
- `reports/preprocess_report.json`
- `reports/eda_report.json`
- `reports/plots/*.png`

## Run 1.5 (Prefect Cloud + local worker)

1. Create a workspace at https://app.prefect.cloud and an API key.

```bash
prefect cloud login --key YOUR_API_KEY
prefect work-pool create local-process --type process
python prefect/flows/telco_pipeline.py
prefect deploy --all
```

2. In a second terminal (same folder, same venv):

```bash
prefect worker start --pool local-process
```

3. After screenshots, pause the 2-minute schedule:

```bash
prefect deployment pause "telco-churn-dataops/telco-every-2-min"
```

Then stop the worker with Ctrl+C.
