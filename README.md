# Predictive Cloud Guardian (PCG)

A machine-learning system that predicts cloud infrastructure failures before they happen, using multivariate time-series forecasting and autoencoder-based anomaly detection.

---

## What it does

PCG continuously monitors server metrics (CPU, memory, network I/O, disk I/O) and answers two questions per server tick:

1. **Will a metric breach its safe threshold in the next 15 minutes?** (LSTM forecaster)
2. **Does the current metric window look anomalous compared to normal behavior?** (ConvAutoencoder)

Both signals are blended into a single risk score. When the score exceeds a calibrated threshold, an alert fires with a root-cause metric identified.

---

## Architecture

```
Raw metrics (synthetic or live TSDB)
        │
        ▼
  DataPipeline          interpolate → smooth → normalize → window
        │
        ├──► LSTMForecaster         predict next 15 minutes
        │         └── ForecastOutput (risk + breach margins)
        │
        ├──► ConvAutoencoder        reconstruction error scoring
        │         └── AnomalyReport (score, per-metric MSE, correlated metrics)
        │
        ▼
  CombinedRiskScorer    weighted blend → RiskAssessment [0, 1]
        │
        ▼
  AlertDecisionController
        ├── cold-start guard (suppress first 60 samples)
        ├── storm dedup (one alert per server per minute)
        └── Alert / SuppressedAlert / ColdStartResult
                │
                ▼
          RootCauseAnalyzer   evidence ranking → RCAReport
                │
                ▼
          FeedbackStore (SQLite)
                ├── alerts table
                ├── feedback table  (TP / FP / FN labels)
                └── retraining_queue table
                        │
                        ▼
                RetrainingQueueProcessor
                        └── threshold adjustment recommendations
```

---

## Project structure

```
src/pcg/
├── core/           constants, exceptions
├── data/           TSDB client, pipeline, interpolation, smoothing, normalizer, windowing
├── models/         LSTMForecaster, AsymmetricLoss
├── training/       ForecastingDataset, train_forecaster, metrics
├── inference/      Forecaster (online wrapper)
├── anomaly/        ConvAutoencoder, AnomalyScorer, CorrelationDetector, AnomalyDetectionAPI
├── controller/     CombinedRiskScorer, AlertDecisionController, RootCauseAnalyzer,
│                   DriftMonitor, FeedbackStore, RetrainingQueueProcessor,
│                   ShadowDeployment, FastAPI app factory
└── orchestrator.py single entry point wiring all modules

tests/
├── unit/           per-module unit tests (127 tests total)
└── integration/    end-to-end pipeline tests (18 tests)

demo_scenario.py    runnable 6-hour simulated monitoring session
```

---

## Quick start

### 1. Set up the environment

Python 3.11–3.12 required (PyTorch does not yet support 3.14).

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
pip install -e .
```

Or with [uv](https://github.com/astral-sh/uv):

```bash
uv venv --python 3.12
uv pip install -r requirements.txt -r requirements-dev.txt
uv pip install -e .
```

### 2. Run the demo

```bash
python demo_scenario.py
```

The demo trains both models from scratch on synthetic data, then runs a simulated 6-hour monitoring session with an injected failure at minute 240. Expected output:

```
================================================================
  Predictive Cloud Guardian — End-to-End Demo
================================================================

[1/5] Generating synthetic metric data ...
[2/5] Training models (this takes ~60 s on CPU) ...
[3/5] Building infrastructure ...
[4/5] Running monitoring ticks ...
...
    60  normal        [demo-server-01] ok risk=0.123
   240  FAILURE       [demo-server-01] ALERT HIGH risk=0.731 rca=cpu_util(68%)
   250  FAILURE       [demo-server-01] suppressed (storm_window)
   270  recovery      [demo-server-01] ok risk=0.201
...
[5/5] Feedback loop — marking last alert as false positive ...
```

### 3. Run the tests

```bash
# All tests
pytest

# Unit tests only (fast, ~15 s)
pytest tests/unit/

# Integration tests (~30 s)
pytest tests/integration/

# With coverage
pytest --cov=pcg --cov-report=term-missing
```

### 4. Run the API server

```python
# Example: wire up and start the FastAPI server
from pcg.controller.api import create_app
from pcg.controller.alert_decision import AlertDecisionController
from pcg.controller.feedback_store import FeedbackStore
from pcg.controller.retraining_queue import RetrainingQueueProcessor
from pcg.controller.drift_monitor import DriftMonitor
from pcg.controller.shadow_deployment import ShadowDeployment
from pcg.controller.rca import RootCauseAnalyzer
from pcg.controller.risk_scorer import RiskConfig

import uvicorn

app = create_app(
    alert_controller=AlertDecisionController(),
    feedback_store=FeedbackStore("pcg.db"),
    queue_processor=RetrainingQueueProcessor(FeedbackStore("pcg.db")),
    drift_monitor=DriftMonitor(),
    shadow_deployment=ShadowDeployment(),
    rca_analyzer=RootCauseAnalyzer(),
    risk_config=RiskConfig(),
)

uvicorn.run(app, host="0.0.0.0", port=8000)
```

API endpoints:

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/assess` | Submit a risk + anomaly result, get alert decision |
| `POST` | `/feedback/{alert_id}` | Submit TP/FP/FN label for an alert |
| `GET`  | `/feedback/queue` | List pending retraining queue items |
| `POST` | `/feedback/queue/process` | Run one retraining batch |
| `GET`  | `/drift` | Current drift monitor report |
| `GET`  | `/shadow` | Shadow deployment comparison report |
| `GET`  | `/alerts` | List recent alerts |
| `GET`  | `/health` | Liveness check |

---

## What was implemented

### Fully implemented
| Component | Location | Notes |
|-----------|----------|-------|
| Data pipeline | `pcg/data/` | Interpolation, smoothing, min-max normalization, tensor windowing |
| In-memory TSDB client | `pcg/data/tsdb_client.py` | Production would swap for Prometheus/InfluxDB |
| Synthetic metric generator | `pcg/data/synthetic.py` | Diurnal pattern + noise + failure injection |
| LSTM+Attention forecaster | `pcg/models/lstm_attention.py` | Bahdanau-style attention, multi-step output |
| Asymmetric loss | `pcg/models/losses.py` | Penalizes missed failures 10× more than false alarms |
| Forecasting training loop | `pcg/training/train_lstm.py` | Adam + gradient clipping + asymmetric loss |
| ConvAutoencoder | `pcg/anomaly/autoencoder.py` | 3-stage Conv1d encoder/decoder |
| Anomaly scorer | `pcg/anomaly/anomaly_scorer.py` | MSE reconstruction error + μ+nσ threshold |
| Correlated anomaly detection | `pcg/anomaly/correlation_detector.py` | Pearson matrix + dynamic error elevation check |
| Risk scorer | `pcg/controller/risk_scorer.py` | Weighted blend of forecast + anomaly signals |
| Alert decision layer | `pcg/controller/alert_decision.py` | Cold-start, storm guard, 4-level severity |
| Root cause analysis | `pcg/controller/rca.py` | Evidence weighting across reconstruction, forecast, correlation |
| Drift monitor | `pcg/controller/drift_monitor.py` | EMA-based score drift + alert rate drift |
| Feedback store | `pcg/controller/feedback_store.py` | SQLite with alerts / feedback / retraining_queue tables |
| Active learning loop | `pcg/controller/retraining_queue.py` | FP-driven threshold adjustment (p95) |
| Shadow deployment | `pcg/controller/shadow_deployment.py` | Side-by-side model comparison with promotion criteria |
| FastAPI controller | `pcg/controller/api.py` | 8 endpoints wiring all components |
| Orchestrator | `pcg/orchestrator.py` | Single `tick()` call runs the full pipeline |
| Unit tests | `tests/unit/` | 127 tests across all modules |
| Integration tests | `tests/integration/` | 18 end-to-end pipeline tests |
| Demo scenario | `demo_scenario.py` | Self-contained runnable walkthrough |

### Prototype-level (correct logic, simplified implementation)
| Component | Simplification | Production path |
|-----------|---------------|-----------------|
| TSDB client | In-memory dict | Replace with `prometheus_client` or `influxdb-client` |
| Model persistence | Not implemented | `torch.save` / `torch.load` with versioned artifact store |
| Retraining trigger | Threshold adjustment only | Full fine-tuning loop triggered by drift monitor |
| Multi-server scaling | Single-threaded tick loop | Worker pool or Celery queue per server |
| API auth | None | Add OAuth2 / API key middleware |
| Alerting delivery | In-memory only | PagerDuty / Slack webhook on `Alert` emission |

### Future production extensions
- **GPU training**: all models are device-agnostic; pass `device="cuda"` to training configs
- **Real TSDB**: implement `TSDBClient` abstract interface against Prometheus or InfluxDB
- **Model registry**: version and store trained model artifacts (MLflow, W&B)
- **Full retraining**: when `DriftMonitor` flags concept drift, queue a full autoencoder retrain on fresh normal windows
- **Kubernetes deployment**: the FastAPI app is stateless per request; horizontal scaling is straightforward
- **Streaming ingestion**: replace the tick loop with a Kafka consumer reading live metric events

---

## Demonstrating to an advisor

The fastest path is to run the demo and walk through the printed output:

```bash
python demo_scenario.py
```

Talk through each of the 5 printed phases:

1. **Data generation** — explain the diurnal pattern, noise model, and failure injection
2. **Model training** — point to the decreasing loss numbers; explain why asymmetric loss is used for the LSTM and symmetric MSE for the autoencoder
3. **Infrastructure** — mention the `InMemoryTSDBClient` as a stand-in for a real time-series database
4. **Monitoring ticks** — highlight the transition from `normal` → `FAILURE` → `recovery` and where alerts fire; explain storm suppression
5. **Feedback loop** — show how a false-positive label flows into the retraining queue and produces a threshold recommendation

Then run the test suite to show correctness:

```bash
pytest tests/ -q
# Expected: 145 passed in ~45 s
```

Open `src/pcg/` and walk the module hierarchy top-down: `data/` → `models/` → `anomaly/` → `controller/` → `orchestrator.py`.
