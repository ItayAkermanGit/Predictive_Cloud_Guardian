# Active learning feedback loop — retraining queue processor.
#
# The feedback store accumulates false-positive records submitted by
# operators. This module processes those records and decides what to do:
#
# Strategy (lightweight, no full model retraining required):
#   1. Read pending false-positive items from the queue.
#   2. For each item, record the (server_id, risk_score) pair.
#   3. If enough false-positive risk scores accumulate for a server,
#      compute a suggested threshold adjustment:
#        suggested_threshold = percentile(fp_scores, 95)
#      This lifts the threshold so that 95% of past false positives
#      would have been suppressed, without requiring a GPU retraining run.
#   4. Emit ThresholdAdjustment recommendations that the API layer can
#      apply to the running RiskConfig.
#   5. Mark processed items as "done" in the queue.
#
# Full retraining (of the LSTM or autoencoder) is a heavier operation
# that happens separately when drift is detected. The threshold adjustment
# here is the fast, lightweight feedback path that operators can trigger.

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from .feedback_store import FeedbackStore, QueueItem


@dataclass(frozen=True)
class ThresholdAdjustment:
    # Recommended risk threshold change derived from false-positive analysis.

    server_id: str                  # "global" means applies to all servers
    current_threshold: float
    suggested_threshold: float
    based_on_n_false_positives: int
    computed_at: datetime
    rationale: str

    @property
    def delta(self) -> float:
        return self.suggested_threshold - self.current_threshold


@dataclass
class RetrainingQueueConfig:
    # Tuning knobs for the processor.

    # Process this many queue items per run.
    batch_size: int = 20

    # Minimum false positives needed before recommending a threshold change.
    min_fp_for_adjustment: int = 5

    # Percentile of false-positive risk scores used to compute new threshold.
    # p95 means: lift the threshold so 95% of past FPs would have been suppressed.
    fp_percentile: float = 95.0

    # Maximum upward adjustment allowed per run (prevents runaway threshold drift).
    max_adjustment: float = 0.20


class RetrainingQueueProcessor:
    # Reads pending false-positive records and produces threshold recommendations.

    def __init__(
        self,
        store: FeedbackStore,
        config: Optional[RetrainingQueueConfig] = None,
    ) -> None:
        self._store = store
        self._cfg = config or RetrainingQueueConfig()

        # Accumulates false-positive risk scores per server.
        # Keyed by server_id; "global" holds cross-server scores.
        self._fp_scores: dict[str, list[float]] = {}

    def process_batch(
        self, current_threshold: float
    ) -> list[ThresholdAdjustment]:
        # Drain up to batch_size pending items from the queue.
        # Returns a (possibly empty) list of recommended threshold adjustments.
        items = self._store.pending_queue_items(limit=self._cfg.batch_size)
        if not items:
            return []

        for item in items:
            self._store.mark_queue_item(item.id, "processing")
            self._ingest(item)
            self._store.mark_queue_item(item.id, "done")

        return self._compute_adjustments(current_threshold)

    def _ingest(self, item: QueueItem) -> None:
        # Record the false-positive risk score for this server.
        for key in (item.server_id, "global"):
            self._fp_scores.setdefault(key, []).append(item.combined_risk)

    def _compute_adjustments(
        self, current_threshold: float
    ) -> list[ThresholdAdjustment]:
        import statistics

        adjustments: list[ThresholdAdjustment] = []
        now = datetime.utcnow()

        for server_id, scores in self._fp_scores.items():
            n = len(scores)
            if n < self._cfg.min_fp_for_adjustment:
                continue

            scores_sorted = sorted(scores)
            # Percentile index — linear interpolation.
            idx = (self._cfg.fp_percentile / 100.0) * (n - 1)
            lo = int(idx)
            hi = min(lo + 1, n - 1)
            suggested = scores_sorted[lo] + (idx - lo) * (scores_sorted[hi] - scores_sorted[lo])

            # Cap how much the threshold can move upward in one batch.
            suggested = min(
                float(suggested),
                current_threshold + self._cfg.max_adjustment,
            )
            # Never suggest a threshold below the current one.
            if suggested <= current_threshold:
                continue

            adjustments.append(
                ThresholdAdjustment(
                    server_id=server_id,
                    current_threshold=current_threshold,
                    suggested_threshold=round(suggested, 4),
                    based_on_n_false_positives=n,
                    computed_at=now,
                    rationale=(
                        f"p{self._cfg.fp_percentile:.0f} of {n} false-positive "
                        f"risk scores = {suggested:.4f}; lifting threshold "
                        f"from {current_threshold:.4f} to {suggested:.4f} "
                        f"would suppress ~{self._cfg.fp_percentile:.0f}% of past FPs."
                    ),
                )
            )

        return adjustments

    def fp_score_count(self, server_id: str = "global") -> int:
        return len(self._fp_scores.get(server_id, []))
