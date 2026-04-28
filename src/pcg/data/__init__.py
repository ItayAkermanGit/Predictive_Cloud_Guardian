"""Data pipeline (Phase 2).

Solves proposal Problems 1, 5, 7 end-to-end:
    * Problem 1 — 60-minute sliding window over multivariate metrics.
    * Problem 5 — linear interpolation for missing samples + moving-average
                  smoothing for transient noise.
    * Problem 7 — multivariate feature vector construction in a fixed,
                  documented column ordering (see core.constants.METRIC_ORDER).

Data flows from a TSDB client through interpolate → smooth → normalize →
window → tensor. The shape produced for every model head is exactly:

    torch.Tensor of shape (1, WINDOW_SIZE=60, N_METRICS=4), dtype float32

For training we stack many such windows along the batch axis.
"""
