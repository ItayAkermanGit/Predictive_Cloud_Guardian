#!/usr/bin/env python3
# End-to-end demo of the Predictive Cloud Guardian system.
#
# Simulates a 6-hour server monitoring session:
#   - Minutes 0-239   : normal operation (trains and calibrates all models)
#   - Minutes 240-269 : injected cpu_util + mem_util spike (anomaly window)
#   - Minutes 270+    : recovery
#
# Walkthrough printed to stdout, one line per tick.
# All models train on the fly from synthetic data — no pre-trained weights needed.
# Runtime: ~60 seconds on CPU.

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

# Allow running from project root without installing the package.
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch

from pcg.anomaly.anomaly_scorer import AnomalyScorer
from pcg.anomaly.correlation_detector import CorrelationDetector
from pcg.anomaly.synthetic_normal import NormalWindowDataset, build_normal_dataframe
from pcg.anomaly.train_autoencoder import AutoencoderTrainConfig, train_autoencoder
from pcg.anomaly.autoencoder import ConvAutoencoder
from pcg.controller.feedback_store import FeedbackStore
from pcg.controller.risk_scorer import RiskConfig
from pcg.core.constants import (
    METRIC_ORDER,
    N_METRICS,
    SAMPLE_INTERVAL_SECONDS,
    WINDOW_SIZE,
)
from pcg.data.normalizer import MinMaxNormalizer
from pcg.data.synthetic import FailureInjection, SyntheticConfig, SyntheticMetricGenerator
from pcg.data.tsdb_client import InMemoryTSDBClient
from pcg.data.pipeline import DataPipeline
from pcg.models.lstm_attention import LSTMForecaster
from pcg.orchestrator import PCGOrchestrator, build_orchestrator
from pcg.training.dataset import ForecastingDataset
from pcg.training.train_lstm import TrainConfig, train_forecaster


# ─── Configuration ────────────────────────────────────────────────────────────

SERVER_ID       = "demo-server-01"
TOTAL_MINUTES   = 360          # 6-hour simulated window
FAILURE_START   = 240          # minute at which the spike begins
FAILURE_MINUTES = 30           # duration of the spike

START_TIME = datetime(2024, 3, 1, 8, 0, 0)


# ─── Step 1: generate synthetic time-series ──────────────────────────────────

def _build_series(with_failure: bool, seed: int, minutes: int):
    failures = []
    if with_failure:
        failures = [
            FailureInjection("cpu_util", FAILURE_START,     FAILURE_MINUTES, magnitude=0.45),
            FailureInjection("mem_util", FAILURE_START + 5, FAILURE_MINUTES, magnitude=0.40),
        ]
    cfg = SyntheticConfig(seed=seed, noise_std=0.03, failures=failures)
    gen = SyntheticMetricGenerator(cfg)
    return gen.generate(start=START_TIME, minutes=minutes)


# ─── Step 2: train LSTM forecaster ───────────────────────────────────────────

def _train_forecaster(normal_df, normalizer: MinMaxNormalizer) -> LSTMForecaster:
    print("  [train] LSTM forecaster ...")
    dataset = ForecastingDataset(normal_df, normalizer, stride=1)
    model = LSTMForecaster(n_metrics=N_METRICS, hidden_size=64, num_layers=1)
    cfg = TrainConfig(epochs=15, batch_size=64, seed=42)
    history = train_forecaster(model, dataset, config=cfg)
    print(f"         final train_loss={history.train_loss[-1]:.5f}")
    return model


# ─── Step 3: train ConvAutoencoder ───────────────────────────────────────────

def _train_autoencoder(
    normal_df, normalizer: MinMaxNormalizer
) -> tuple[ConvAutoencoder, AnomalyScorer]:
    print("  [train] ConvAutoencoder ...")
    ds = NormalWindowDataset(normal_df, normalizer, stride=2)
    model = ConvAutoencoder(n_metrics=N_METRICS, latent_dim=16)
    cfg = AutoencoderTrainConfig(epochs=20, batch_size=64, seed=42)
    history = train_autoencoder(model, ds, config=cfg)
    print(f"         final train_loss={history.train_loss[-1]:.5f}")

    scorer = AnomalyScorer(model, n_sigma=3.0)
    scorer.fit_threshold(ds.all_windows)
    print(
        f"         threshold={scorer.threshold:.5f} "
        f"(μ={scorer.mu_normal:.5f} σ={scorer.sigma_normal:.6f})"
    )
    return model, scorer


# ─── Step 4: fit correlation detector ────────────────────────────────────────

def _fit_correlation(normal_df, normalizer: MinMaxNormalizer) -> CorrelationDetector:
    ds = NormalWindowDataset(normal_df, normalizer, stride=5)
    det = CorrelationDetector(corr_threshold=0.6)
    det.fit(ds.all_windows)
    return det


# ─── Step 5: build synthetic TSDB from the full series ───────────────────────

def _make_tsdb(full_df) -> InMemoryTSDBClient:
    client = InMemoryTSDBClient()
    client.upsert(SERVER_ID, full_df)
    return client


# ─── Main demo ────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 64)
    print("  Predictive Cloud Guardian — End-to-End Demo")
    print("=" * 64)

    # --- data ---
    print("\n[1/5] Generating synthetic metric data ...")
    normal_df = _build_series(with_failure=False, seed=42, minutes=TOTAL_MINUTES)
    full_df   = _build_series(with_failure=True,  seed=42, minutes=TOTAL_MINUTES)
    print(f"      {len(full_df)} rows generated ({TOTAL_MINUTES} minutes)")
    print(f"      Failure injected at minutes {FAILURE_START}–{FAILURE_START + FAILURE_MINUTES}")

    # --- normalizer ---
    normalizer = MinMaxNormalizer()
    normalizer.fit(normal_df)
    print(
        f"      Normalizer fitted. "
        f"cpu_util range [{normalizer.mins['cpu_util']:.3f}, {normalizer.maxs['cpu_util']:.3f}]"
    )

    # --- models ---
    print("\n[2/5] Training models (this takes ~60 s on CPU) ...")
    forecaster_model = _train_forecaster(normal_df, normalizer)
    ae_model, scorer = _train_autoencoder(normal_df, normalizer)
    corr_det         = _fit_correlation(normal_df, normalizer)
    print("      All models trained and calibrated.")

    # --- infrastructure ---
    print("\n[3/5] Building infrastructure ...")
    tsdb   = _make_tsdb(full_df)
    store  = FeedbackStore(db_path=":memory:")
    pipeline = DataPipeline(tsdb, normalizer)

    risk_cfg = RiskConfig(
        w_forecast=0.5,
        w_anomaly=0.5,
        risk_threshold=0.45,  # slightly lower for demo visibility
    )

    orchestrator = build_orchestrator(
        pipeline=pipeline,
        forecaster_model=forecaster_model,
        normalizer=normalizer,
        anomaly_scorer=scorer,
        correlation_detector=corr_det,
        risk_config=risk_cfg,
        db_path=":memory:",
    )
    # Replace the store so feedback ops use the same in-memory DB.
    orchestrator._store = store
    print("      Orchestrator ready.")

    # --- tick loop ---
    print("\n[4/5] Running monitoring ticks ...")
    print("-" * 64)
    print(f"{'Minute':>6}  {'Phase':<12}  {'Result'}")
    print("-" * 64)

    n_alerts = 0
    last_alert_id: int | None = None

    # Only tick from minute WINDOW_SIZE onward (need a full window to start).
    tick_minutes = range(WINDOW_SIZE, TOTAL_MINUTES)

    for minute in tick_minutes:
        now = START_TIME + timedelta(minutes=minute)

        try:
            result = orchestrator.tick(SERVER_ID, now=now)
        except Exception as exc:
            # Skip if the window isn't populated yet.
            print(f"{minute:>6}  {'skip':<12}  {exc}")
            continue

        phase = "normal"
        if FAILURE_START <= minute < FAILURE_START + FAILURE_MINUTES:
            phase = "FAILURE"
        elif minute >= FAILURE_START + FAILURE_MINUTES:
            phase = "recovery"

        # Print every 10th normal tick; print every failure/alert tick.
        if result.alerted or phase == "FAILURE" or minute % 20 == 0:
            line = f"{minute:>6}  {phase:<12}  {result.summary()}"
            print(line)

        if result.alerted and isinstance(result.decision, __import__('pcg.controller.alert_decision', fromlist=['Alert']).Alert):
            n_alerts += 1
            last_alert_id = result.alert_db_id

    print("-" * 64)
    print(f"      Total alerts fired: {n_alerts}")

    # --- feedback loop ---
    print("\n[5/5] Feedback loop — marking last alert as false positive ...")
    if last_alert_id is not None:
        fid = store.submit_feedback(
            alert_id=last_alert_id,
            label="false_positive",
            notes="demo: simulated operator feedback",
            submitted_by="demo_operator",
        )
        print(f"      Feedback #{fid} submitted → retraining queue length: {store.queue_length()}")

        from pcg.controller.retraining_queue import RetrainingQueueProcessor
        proc = RetrainingQueueProcessor(store)
        adjustments = proc.process_batch(current_threshold=risk_cfg.risk_threshold)
        if adjustments:
            for adj in adjustments:
                print(
                    f"      Threshold recommendation: "
                    f"{adj.current_threshold:.3f} → {adj.suggested_threshold:.3f} "
                    f"(based on {adj.based_on_n_false_positives} FP reports)"
                )
        else:
            print("      Not enough FP reports yet for a threshold recommendation.")
    else:
        print("      No alerts were persisted — skipping feedback step.")

    # --- drift report ---
    print("\n      Drift monitor report:")
    try:
        drift_rpt = orchestrator._drift.report()
        print(f"      {drift_rpt.summary()}")
    except RuntimeError as exc:
        print(f"      {exc}")

    print("\n" + "=" * 64)
    print("  Demo complete.")
    print("=" * 64)


if __name__ == "__main__":
    main()
