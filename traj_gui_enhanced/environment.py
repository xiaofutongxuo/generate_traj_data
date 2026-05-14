"""Runtime environment setup for the enhanced trajectory GUI."""

import ctypes
from pathlib import Path
import os
import sys


def _prepend_env_path(name: str, value: Path) -> None:
    current = os.environ.get(name, "")
    value_str = str(value)
    parts = [part for part in current.split(":") if part]
    if value_str not in parts:
        os.environ[name] = ":".join([value_str] + parts)


def _configure_bundled_tk(repo_root: Path) -> None:
    """Expose the optional project-local Tk runtime when system Tk is absent."""
    runtime_root = repo_root / ".runtime" / "tk" / "usr"
    python_tag = f"python{sys.version_info.major}.{sys.version_info.minor}"
    py_lib = runtime_root / "lib" / python_tag
    dynload = py_lib / "lib-dynload"
    if not (py_lib.exists() and dynload.exists()):
        return

    for path in (dynload, py_lib):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

    lib_dirs = (runtime_root / "lib" / "x86_64-linux-gnu", runtime_root / "lib")
    for lib_dir in lib_dirs:
        if lib_dir.exists():
            _prepend_env_path("LD_LIBRARY_PATH", lib_dir)

    os.environ.setdefault("TCL_LIBRARY", str(runtime_root / "share" / "tcltk" / "tcl8.6"))
    os.environ.setdefault("TK_LIBRARY", str(runtime_root / "share" / "tcltk" / "tk8.6"))

    # LD_LIBRARY_PATH changes after process start do not reliably affect dlopen,
    # so preload the bundled libraries by full path before importing _tkinter.
    for lib_path in (
        runtime_root / "lib" / "x86_64-linux-gnu" / "libtk8.6.so",
        runtime_root / "lib" / "libBLT.2.5.so.8.6",
    ):
        if lib_path.exists():
            ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)


def setup_environment() -> None:
    """Apply the environment defaults used by the historical GUI script."""
    os.environ.setdefault("GENERATE_TRAJ_USE_TORCH", "0")
    if not os.environ.get("DISPLAY") and Path("/tmp/.X11-unix/X1").exists():
        os.environ["DISPLAY"] = ":1"
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        import tkinter  # noqa: F401
    except ModuleNotFoundError:
        sys.modules.pop("tkinter", None)
        sys.modules.pop("_tkinter", None)
        _configure_bundled_tk(repo_root)
