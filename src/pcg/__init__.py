"""Predictive Cloud Guardian (PCG) — proactive AIOps monitoring.

Top-level package. Submodules:
    core/   — cross-cutting constants, schemas, exceptions.
    data/   — Phase 2 data pipeline (this phase).
    models/, training/, inference/, controller/, alerting/, mlops/, api/
            — implemented in later phases.

The `__version__` string is intentionally kept here (and not pulled from
package metadata) so the runtime works even when PCG is run as a plain
sys.path entry rather than as an installed distribution.
"""

__version__ = "0.1.0"
