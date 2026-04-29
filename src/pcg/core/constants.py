# Numeric constants used across the project.

WINDOW_SIZE: int = 60                # input window length in minutes
HORIZON_MINUTES: int = 15            # forecast horizon in minutes
SMOOTHING_WINDOW: int = 5            # moving average window size
ASYMMETRIC_LOSS_ALPHA: float = 10.0  # weight for false negatives
ANOMALY_SIGMA: float = 3.0           # 3-sigma threshold for anomaly score
COLD_START_MIN_SAMPLES: int = 60     # minimum samples to leave warmup
GROUPING_WINDOW_SECONDS: int = 60    # alert grouping window length

# Order of metrics in every tensor we build.
METRIC_ORDER: tuple[str, ...] = (
    "cpu_util",
    "mem_util",
    "net_io",
    "disk_io",
)
N_METRICS: int = len(METRIC_ORDER)

SAMPLE_INTERVAL_SECONDS: int = 60    # one sample per minute
