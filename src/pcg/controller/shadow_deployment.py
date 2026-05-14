# Shadow deployment simulation.
#
# In shadow mode a "candidate" model runs alongside the "production" model.
# Both process the same input but only the production model's decisions are
# acted on. The candidate's decisions are logged for comparison.
#
# This lets us answer before promoting a new model:
#   - Does the candidate agree with production on normal windows?
#   - Does it catch anomalies that production misses?
#   - Does it produce fewer false positives on known-normal windows?
#
# Comparison metrics tracked per evaluation window:
#   agreement_rate   — fraction of windows where both models agree on alert/no-alert.
#   candidate_only   — windows where candidate alerts but production doesn't (potential FN coverage).
#   production_only  — windows where production alerts but candidate doesn't (potential regression).
#   risk_mae         — mean absolute difference in combined_risk between models.
#
# Promotion recommendation:
#   The candidate is recommended for promotion when, over a sufficient number
#   of shadow evaluations:
#     - agreement_rate >= min_agreement_rate
#     - production_only (regressions) == 0  OR production_only rate < max_regression_rate
#     - risk_mae <= max_risk_mae

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from .risk_scorer import RiskAssessment


@dataclass
class ShadowConfig:
    # Thresholds used to decide whether the candidate is ready to promote.

    # Minimum evaluations before a promotion recommendation is made.
    min_evaluations: int = 200

    # Fraction of windows where both models must agree.
    min_agreement_rate: float = 0.90

    # Max fraction of windows where production fires but candidate doesn't.
    max_regression_rate: float = 0.02

    # Max mean absolute difference in combined_risk between models.
    max_risk_mae: float = 0.05

    def __post_init__(self) -> None:
        if not 0 < self.min_agreement_rate <= 1:
            raise ValueError("min_agreement_rate must be in (0, 1]")
        if not 0 <= self.max_regression_rate < 1:
            raise ValueError("max_regression_rate must be in [0, 1)")


@dataclass
class ShadowComparison:
    # Result of comparing one production and candidate assessment.

    server_id: str
    evaluated_at: datetime
    production_risk: float
    candidate_risk: float
    production_alert: bool
    candidate_alert: bool

    @property
    def agree(self) -> bool:
        return self.production_alert == self.candidate_alert

    @property
    def candidate_only(self) -> bool:
        return self.candidate_alert and not self.production_alert

    @property
    def production_only(self) -> bool:
        return self.production_alert and not self.candidate_alert

    @property
    def risk_delta(self) -> float:
        return abs(self.production_risk - self.candidate_risk)


@dataclass(frozen=True)
class ShadowReport:
    # Aggregate statistics over all shadow evaluations.

    n_evaluations: int
    agreement_rate: float
    candidate_only_rate: float    # candidate fires, production doesn't
    production_only_rate: float   # production fires, candidate doesn't
    risk_mae: float
    ready_to_promote: bool
    promotion_blockers: list[str]  # human-readable reasons blocking promotion
    computed_at: datetime

    def summary(self) -> str:
        status = "PROMOTE" if self.ready_to_promote else "NOT READY"
        blockers = "; ".join(self.promotion_blockers) if self.promotion_blockers else "none"
        return (
            f"[shadow={status}] n={self.n_evaluations} "
            f"agree={self.agreement_rate:.1%} "
            f"cand_only={self.candidate_only_rate:.1%} "
            f"prod_only={self.production_only_rate:.1%} "
            f"risk_mae={self.risk_mae:.4f} "
            f"blockers={blockers}"
        )


class ShadowDeployment:
    # Manages shadow evaluation between a production and candidate scorer.

    def __init__(self, config: Optional[ShadowConfig] = None) -> None:
        self.cfg = config or ShadowConfig()
        self._comparisons: list[ShadowComparison] = []

    def evaluate(
        self,
        production: RiskAssessment,
        candidate: RiskAssessment,
        now: Optional[datetime] = None,
    ) -> ShadowComparison:
        # Record one paired evaluation. Both assessments must be for the same server.
        if production.server_id != candidate.server_id:
            raise ValueError(
                f"server_id mismatch: production={production.server_id!r} "
                f"candidate={candidate.server_id!r}"
            )
        if now is None:
            now = datetime.utcnow()

        comparison = ShadowComparison(
            server_id=production.server_id,
            evaluated_at=now,
            production_risk=production.combined_risk,
            candidate_risk=candidate.combined_risk,
            production_alert=production.should_alert,
            candidate_alert=candidate.should_alert,
        )
        self._comparisons.append(comparison)
        return comparison

    def report(self, now: Optional[datetime] = None) -> ShadowReport:
        if now is None:
            now = datetime.utcnow()

        n = len(self._comparisons)
        if n == 0:
            return ShadowReport(
                n_evaluations=0,
                agreement_rate=0.0,
                candidate_only_rate=0.0,
                production_only_rate=0.0,
                risk_mae=0.0,
                ready_to_promote=False,
                promotion_blockers=["no evaluations yet"],
                computed_at=now,
            )

        agree_count        = sum(1 for c in self._comparisons if c.agree)
        candidate_only     = sum(1 for c in self._comparisons if c.candidate_only)
        production_only    = sum(1 for c in self._comparisons if c.production_only)
        total_risk_delta   = sum(c.risk_delta for c in self._comparisons)

        agreement_rate       = agree_count / n
        candidate_only_rate  = candidate_only / n
        production_only_rate = production_only / n
        risk_mae             = total_risk_delta / n

        blockers: list[str] = []
        if n < self.cfg.min_evaluations:
            blockers.append(
                f"only {n}/{self.cfg.min_evaluations} evaluations completed"
            )
        if agreement_rate < self.cfg.min_agreement_rate:
            blockers.append(
                f"agreement {agreement_rate:.1%} < {self.cfg.min_agreement_rate:.1%}"
            )
        if production_only_rate > self.cfg.max_regression_rate:
            blockers.append(
                f"regression rate {production_only_rate:.1%} > {self.cfg.max_regression_rate:.1%}"
            )
        if risk_mae > self.cfg.max_risk_mae:
            blockers.append(
                f"risk_mae {risk_mae:.4f} > {self.cfg.max_risk_mae:.4f}"
            )

        return ShadowReport(
            n_evaluations=n,
            agreement_rate=agreement_rate,
            candidate_only_rate=candidate_only_rate,
            production_only_rate=production_only_rate,
            risk_mae=risk_mae,
            ready_to_promote=len(blockers) == 0,
            promotion_blockers=blockers,
            computed_at=now,
        )

    def reset(self) -> None:
        self._comparisons.clear()

    @property
    def n_evaluations(self) -> int:
        return len(self._comparisons)
