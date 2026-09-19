# Engineering AI Assistant

This repository contains the assistant built for W15/W16 and the Week 17 MLOps extension. It is organized around a reproducible Python environment, MLflow experiment tracking, and Evidently-based monitoring for both the assistant and a churn-prediction pipeline.

## Features

- FastAPI backend with validation and rate limiting
- Streamlit UI and retrieval-grounded assistant
- Local TF-IDF RAG over project notes and documents
- Bounded agentic loop with tool use and explicit degraded-failure handling
- `uv`-managed reproducible environment and `pyproject.toml`
- MLflow experiment tracking for model and prompt experiments
- Evidently drift and regression monitoring for both Track A and Track B
- Docker support and optional Airflow-style orchestration examples

## Environment & Reproducibility

The project now uses `uv` to make setup deterministic from a clean clone. `uv` resolves transitive dependencies, locks them in a shared lockfile, and prevents the classic Python environment drift that happens when packages are installed ad hoc with `pip` or when different system interpreters are used across machines.

```bash
uv sync
source .venv/bin/activate
uvicorn app.main:app --reload
```

The repo also includes a fallback `requirements.txt` for environments that are not yet migrated, but the canonical path is:

```bash
uv sync
```

Then open the API at `http://localhost:8000/docs` and the UI at `http://localhost:8501`.

## Track A: Data Science MLOps

The churn pipeline is implemented in the training and monitoring scripts under the `scripts/` directory. The workflow is:

1. load or synthesize a churn dataset,
2. train multiple models with different hyperparameters,
3. log experiment metrics and artifacts to MLflow,
4. register the winning model,
5. run an Evidently drift check,
6. trigger retraining when drift crosses a configured threshold.

### MLflow strategy

The model comparison varies the model family and hyperparameters so they are meaningfully different runs rather than only random seeds. The training script logs accuracy, precision, recall, F1, ROC-AUC, confusion matrices, and ROC artifacts per run.

The best run is chosen from the MLflow comparison by balancing the imbalance-sensitive metrics. On a churn dataset, accuracy alone is misleading because false negatives matter and the target can be skewed. The winning model is therefore the one with the best F1 and ROC-AUC while still maintaining strong recall.

### Example tracked workflow

```bash
uv run python scripts/train_churn_model.py
uv run python scripts/evaluate_drift.py
```

The script creates the MLflow experiment and can register a model in the model registry when a run is selected as the champion.

### Verified model evidence

The latest successful pipeline run produced the following MLflow comparison:

| Model | F1 | ROC-AUC | Recall |
| --- | ---: | ---: | ---: |
| logistic_regression | 0.6115 | 0.8451 | 0.5597 |
| random_forest (150/8) | 0.5746 | 0.8412 | 0.5045 |
| random_forest (250/12) | 0.5745 | 0.8297 | 0.5187 |

The winning model was the logistic-regression variant, and it was registered as `telco-churn-model` in the MLflow model registry. The workspace artifact is stored at `artifacts/churn_drift_report.html` and the summary JSON is at `artifacts/churn_drift_summary.json`.

## Track B: Agentic AI MLOps

The assistant continues to use its prompt-driven agent loop, but now the prompt versions are explicitly tracked as first-class configuration artifacts. The repository includes a versioned prompt catalog in `app/providers.py` and under the `prompts/` directory.

### Prompt versioning

- `prompt_v1`: conservative baseline
- `prompt_v2`: bounded-agent policy
- `prompt_v3`: production-grade verification prompt

Each version is evaluated against the same regression set and the traces are retained as structured session metadata alongside the MLflow run metrics. The point is that each new prompt is a response to a concrete failure observed in the previous trace, not a speculative tweak.

### MLflow strategy

The comparison tracks:

- prompt version and configuration values,
- benchmark or task-completion metrics,
- tool-call correctness,
- token usage,
- pass/fail regression results from the Evidently test suite,
- representative traces and prompt artifacts.

A useful experiment comparison thus balances task completion and cost, not just answer quality.

```bash
uv run python scripts/track_agent_runs.py
```

The experiment was logged for all three prompt versions: `prompt_v1`, `prompt_v2`, and `prompt_v3`. Each run is recorded with its prompt version, average iteration count, token usage, and degraded-run counters inside MLflow.

## Monitoring & Drift Strategy

Evidently monitors the difference between a reference distribution and a current production-like distribution. For the churn pipeline, the reference set is treated as the training-time baseline and the current set is a later batch with injected drift such as MonthlyCharges shifts or contract skew.

The drift report checks:

- feature drift for the engineered columns,
- target drift for the churn rate,
- custom metrics such as mean MonthlyCharges shift or churn-rate difference inside a contract segment.

The latest report showed drift in the `MonthlyCharges` feature (0.4898 normalized Wasserstein distance) and in the `Contract` distribution (0.4319 Jensen-Shannon distance), with a drifted-columns share of 0.1 across the monitored profile. If drift exceeds a threshold, the system logs the event, marks the run as unstable, and can trigger a retraining workflow. The same idea applies to the agent track: if regression metrics degrade on the fixed benchmark set, the prompt version should not be promoted.

## Orchestration

A simple Airflow-style DAG is included at `dags/mlops_daily_check.py` to represent the optional scheduled workflow. It can:

- run the drift evaluation or agent regression check,
- compare the latest metrics to a threshold,
- raise an alert or log a retraining recommendation when the signal crosses the threshold.

This is intentionally small and easy to extend into a real DAG scheduler.

## Testing

```bash
pytest -q
```

This project keeps a small but meaningful regression suite for the offline assistant and for the versioned prompt catalog.

## Project layout

- `app/main.py`: API entry point
- `app/service.py`: orchestration, cache, fallback, and citations
- `app/rag.py`: chunking and retrieval
- `app/providers.py`: model/provider logic and prompt versions
- `app/tools.py`: safe allow-listed tools
- `scripts/`: MLflow, drift, and evaluation runners
- `dags/`: optional orchestration DAGs
- `evaluation/`: benchmark notes and summary artifacts