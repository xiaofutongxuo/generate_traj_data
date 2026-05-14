"""Lazy Tk/PIL ImageTk imports for CLI help and non-GUI imports."""

from __future__ import annotations

import importlib

from PIL import Image


class _LazyModule:
    def __init__(self, module_name: str):
        self._module_name = module_name
        self._module = None

    def _load(self):
        if self._module is None:
            self._module = importlib.import_module(self._module_name)
        return self._module

    def __getattr__(self, name: str):
        return getattr(self._load(), name)


tk = _LazyModule("tkinter")
ttk = _LazyModule("tkinter.ttk")
messagebox = _LazyModule("tkinter.messagebox")
ImageTk = _LazyModule("PIL.ImageTk")

__all__ = ["Image", "ImageTk", "messagebox", "tk", "ttk"]
