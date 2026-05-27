"""Python package entry point for the EIDL Lumerical optimization helpers."""

from importlib import import_module

__all__ = [
    "Filters",
    "Lumerical_module",
    "Lumerical_utill",
    "Opt_MS2",
    "Sub_Mapping",
    "monitoring",
]


def __getattr__(name):
    if name in __all__:
        return import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _activate_monitoring_from_env():
    import os

    enabled = os.environ.get("MSOPT_PROFILE", "").lower() in {"1", "true", "yes", "on"}
    enabled = enabled or os.environ.get("EIDL_PROFILE", "").lower() in {"1", "true", "yes", "on"}
    if not enabled:
        return
    try:
        import_module(f"{__name__}.monitoring").activate_from_env()
    except Exception as exc:
        print(f"[Profiler] failed to activate monitoring: {exc}")


_activate_monitoring_from_env()
