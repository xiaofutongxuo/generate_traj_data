"""Enhanced trajectory GUI package."""

from .environment import setup_environment

setup_environment()


def __getattr__(name):
    if name == "TrajectoryViewerEnhanced":
        from .viewer import TrajectoryViewerEnhanced

        return TrajectoryViewerEnhanced
    if name in {"parse_args", "main"}:
        from .cli import main, parse_args

        return {"parse_args": parse_args, "main": main}[name]
    raise AttributeError(name)


__all__ = ["TrajectoryViewerEnhanced", "parse_args", "main", "setup_environment"]
