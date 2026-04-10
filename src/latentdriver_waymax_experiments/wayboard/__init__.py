from __future__ import annotations

from .data import RunRecord, SuiteRecord, discover_runs, discover_suites

__all__ = [
    "RunRecord",
    "SuiteRecord",
    "WaymaxBoard",
    "discover_runs",
    "discover_suites",
]


def __getattr__(name: str):
    if name == "WaymaxBoard":
        from .app import WaymaxBoard

        return WaymaxBoard
    raise AttributeError(name)
