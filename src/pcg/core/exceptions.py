"""Domain-specific exceptions for the data pipeline.

Typed exceptions let the controller distinguish *recoverable* states
(e.g. InsufficientHistoryError → fall back to the cold-start generic model
described in proposal Problem 8) from real bugs that should propagate.
"""


class PCGError(Exception):
    """Base class for all PCG-raised errors. Tests can catch this to assert
    that an error came from our code rather than from a dependency."""


class InsufficientHistoryError(PCGError):
    """Raised when fewer than WINDOW_SIZE valid samples exist for a server.

    The controller (Phase 4) catches this to route the request to the
    generic cold-start model (Problem 8) instead of the per-server model.
    """


class MetricSchemaError(PCGError):
    """Raised when an incoming DataFrame is missing a required metric column.

    A schema error indicates the upstream TSDB query was misconfigured —
    we deliberately fail loudly rather than silently drop columns.
    """


class NormalizerNotFittedError(PCGError):
    """Raised when ``transform()`` is called before ``fit()``.

    This guards against a common bug: forgetting to load saved min/max
    parameters before serving a model in production.
    """
