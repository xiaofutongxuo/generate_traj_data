import importlib
from pathlib import Path
import re
import unittest


class ProjectStructureTests(unittest.TestCase):
    def test_new_annotation_entrypoint_and_packages_import(self):
        repo_root = Path(__file__).resolve().parents[1]

        self.assertTrue((repo_root / "trajectory_annotator.py").exists())
        self.assertFalse((repo_root / "trajectory_gui_enhanced.py").exists())

        annotator = importlib.import_module("trajectory_annotator")
        annotation_cli = importlib.import_module("traj_annotation.cli")
        core_data = importlib.import_module("traj_core.data_loader")
        core_dynamics = importlib.import_module("traj_core.dynamics")
        inference_runner = importlib.import_module("traj_inference.runner")

        self.assertIs(annotator.main, annotation_cli.main)
        self.assertTrue(hasattr(core_data, "load_data"))
        self.assertTrue(hasattr(core_dynamics, "optimize_pseudo_gt_trajectory"))
        self.assertTrue(hasattr(inference_runner, "main"))

    def test_gui_requirements_exclude_vla_inference_dependencies(self):
        repo_root = Path(__file__).resolve().parents[1]
        requirements_path = repo_root / "requirements.txt"

        self.assertTrue(requirements_path.exists())
        package_names = set()
        for raw_line in requirements_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue
            package_names.add(re.split(r"[<>=!~;\\[]", line, maxsplit=1)[0].strip().lower())

        required_gui_packages = {
            "numpy",
            "pandas",
            "pyarrow",
            "scipy",
            "pillow",
            "opencv-python",
        }
        forbidden_inference_packages = {
            "torch",
            "torchvision",
            "transformers",
            "peft",
            "accelerate",
            "bitsandbytes",
        }

        self.assertTrue(required_gui_packages.issubset(package_names))
        self.assertTrue(forbidden_inference_packages.isdisjoint(package_names))


if __name__ == "__main__":
    unittest.main()
