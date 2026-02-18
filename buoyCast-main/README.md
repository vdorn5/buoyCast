# buoyCast (AAI-540 MLOps)

This repository is set up as a **full-cycle SageMaker MLOps project** for wind-driven wave forecasting from **NOAA NDBC buoy data**.

**Goal (ML problem):** predict a physics-safe wave-energy proxy:

- **E\_star = Hs²** (non-negative)
- and report wave-height proxy **Hs = sqrt(E\_star)** for stakeholder-friendly metrics

## What “full cycle” means here

From scratch in SageMaker you can run:

1. **Live data gathering** (NOAA NDBC) → curated data in **S3** (no backup data required)
2. **Data engineering** → optional Athena external table over curated Parquet
3. **Feature engineering** → supervised lag features
4. **Feature Store** → Feature Group creation + ingestion (offline + optional online)
5. **SageMaker Pipeline (DAG)** → preprocess → train → evaluate → conditional register
6. **Model Registry** → Model Package Group + model versioning
7. **Deployment** → real-time endpoint + invocation output
8. **Monitoring** → CloudWatch dashboard + Model Monitor baseline/schedule
9. **CI** → GitHub Actions (lint + tests) for a “CI/CD DAG” demo

## Notebook run order (SageMaker Studio)

Run these notebooks in order:

- `notebooks/01_data_gathering_etl.ipynb`
- `notebooks/02_athena_curated_table.ipynb` (optional but good for the Design Doc)
- `notebooks/03_feature_store_ingest.ipynb`
- `notebooks/04_sagemaker_pipeline_train_register.ipynb`
- `notebooks/05_deploy_endpoint_monitor.ipynb`
- `notebooks/99_cleanup_resources.ipynb` (recommended to avoid endpoint costs)

## Local / dev install

```bash
pip install -r requirements.txt
pytest -q
ruff check src tests
```

## Repo layout (what matters for MLOps)

- `src/` — reusable library code (cleaning, feature engineering, physics guardrails)
- `notebooks/` — the “operator runbook” for SageMaker Studio
- `sagemaker_scripts/` — scripts used by **Processing**, **Training**, and **Inference**
- `pipelines/` — SageMaker Pipeline (DAG) definition
- `.github/workflows/` — GitHub Actions CI (lint + tests, docker build)

---

## Original project background

The original buoyCast concept is “Wind-Driven Wave Forecasting with Physics-Informed ML”.
This AAI-540 version focuses on the **MLOps system** and operability in SageMaker.
