#!/usr/bin/env python3
"""Trajectory diversity annotation GUI entrypoint."""

from traj_annotation.cli import main, parse_args
from traj_annotation.viewer import TrajectoryViewerEnhanced

__all__ = ["TrajectoryViewerEnhanced", "parse_args", "main"]


if __name__ == "__main__":
    main()
