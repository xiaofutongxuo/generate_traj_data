#!/usr/bin/env python3
"""Compatibility entrypoint for Alpamayo trajectory inference."""

from traj_inference.runner import main, parse_args

__all__ = ["parse_args", "main"]


if __name__ == "__main__":
    main()
