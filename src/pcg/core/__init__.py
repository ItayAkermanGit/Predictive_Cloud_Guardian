"""Cross-cutting primitives shared by every other PCG module.

Anything in here must be import-cheap and dependency-light: no torch, no
pandas. That keeps `from pcg.core import ...` fast and lets us reuse these
constants from configuration scripts and tests without paying ML import cost.
"""
