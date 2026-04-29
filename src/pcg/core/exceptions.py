# Custom exceptions used by the data pipeline.


class PCGError(Exception):
    # Base class for project errors.
    pass


class InsufficientHistoryError(PCGError):
    # Not enough samples to build a 60 minute window.
    pass


class MetricSchemaError(PCGError):
    # A required metric column is missing.
    pass


class NormalizerNotFittedError(PCGError):
    # Normalizer was used before calling fit().
    pass
