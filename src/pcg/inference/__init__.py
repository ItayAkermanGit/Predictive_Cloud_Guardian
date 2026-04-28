"""Online inference wrappers for trained PCG models.

Phase 3 ships the forecaster wrapper. The autoencoder anomaly detector
and the per-metric root-cause decomposer arrive in Phase 4 alongside
the controller that fuses both signals into a Decision JSON.
"""

from .forecaster import Forecaster, ForecastOutput

__all__ = ["Forecaster", "ForecastOutput"]
