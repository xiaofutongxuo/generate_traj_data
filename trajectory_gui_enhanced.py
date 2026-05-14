#!/usr/bin/env python3
"""Compatibility entrypoint for the enhanced trajectory GUI.

The implementation lives in :mod:`traj_gui_enhanced`; this file preserves the
historical ``python trajectory_gui_enhanced.py ...`` command.
"""

from traj_gui_enhanced.cli import main, parse_args
from traj_gui_enhanced.viewer import TrajectoryViewerEnhanced

__all__ = ["TrajectoryViewerEnhanced", "parse_args", "main"]


if __name__ == "__main__":
    main()
