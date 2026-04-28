"""Project-wide constants traceable directly to the Phase 1 proposal.

These values are the AUTHORITATIVE numeric specification:
    * 60-minute sliding window (Problem 1)
    * 15-minute forecast horizon (Problem 1)
    * 5-sample moving-average smoothing (Problem 5)
    * Asymmetric loss alpha = 10 (Problem 3)
    * 3-sigma anomaly threshold (Problem 2)
    * 60-sample cold-start switch (Problem 8)
    * 60-second alert grouping window (Problem 9)
    * canonical multivariate metric ordering (Problem 7)

Defense note: every tensor in PCG has a column ordering identical to
METRIC_ORDER below. Treat any change here as a breaking change to all
serialized model artifacts and to the normalizer JSON.
"""

WINDOW_SIZE: int = 60                # length of input window in minutes
HORIZON_MINUTES: int = 15            # length of forecast horizon in minutes
SMOOTHING_WINDOW: int = 5            # trailing moving-average size in samples
ASYMMETRIC_LOSS_ALPHA: float = 10.0  # weight applied to false-negative residuals
ANOMALY_SIGMA: float = 3.0           # k in (mean + k*std) AE-error threshold
COLD_START_MIN_SAMPLES: int = 60     # contiguous samples needed to leave warmup
GROUPING_WINDOW_SECONDS: int = 60    # alert-storm grouping window length

# Canonical multivariate vector definition. The position of a metric here
# IS its column index in every (n_samples, N_METRICS) array and in every
# (B, WINDOW_SIZE, N_METRICS) tensor. Read once, treat as immutable.
METRIC_ORDER: tuple[str, ...] = (
    "cpu_util",
    "mem_util",
    "net_io",
    "disk_io",
)
N_METRICS: int = len(METRIC_ORDER)

# Sampling cadence: one row per minute, matching the proposal's per-minute
# controller execution. Any TSDB client must align to this grid.
SAMPLE_INTERVAL_SECONDS: int = 60
