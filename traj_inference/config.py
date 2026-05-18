# SPDX-License-Identifier: Apache-2.0
"""Configuration for VLM trajectory generation using Alpamayo 1.5."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


_REPO_ROOT = Path(__file__).resolve().parents[1]


def _default_model_path() -> str:
    local_model = Path("/home/ubuntu/Public/lxh/models/Alpamayo-1.5-10B")
    repo_model = _REPO_ROOT / "Alpamayo-1.5-10B"
    return os.environ.get(
        "ALPAMAYO_MODEL_PATH",
        str(local_model if local_model.exists() else repo_model),
    )


@dataclass
class ModelConfig:
    """Model configuration for Alpamayo 1.5."""
    model_path: str = field(default_factory=_default_model_path)
    traj_checkpoint: Optional[str] = None
    lora_adapter: Optional[str] = None
    dtype: str = "bfloat16"


@dataclass
class DataConfig:
    """Data configuration."""
    train_data_root: str = os.environ.get(
        "TRAIN_DATA_ROOT",
        str(_REPO_ROOT / "triplane_tokenization" / "data_cache" / "alpamayo_extracted"),
    )
    calibration_dir: str = os.environ.get(
        "CALIBRATION_DIR",
        str(_REPO_ROOT / "triplane_tokenization" / "cailibration"),
    )
    num_history_steps: int = 16
    num_future_steps: int = 64
    num_frames: int = 4
    target_image_hw: tuple[int, int] = (1280, 1920)
    heading_num_steps: int = 5
    min_heading_displacement_m: float = 0.2
    time_step: float = 0.1


@dataclass
class InferenceConfig:
    """Inference configuration."""
    num_traj_samples: int = 6
    num_traj_sets: int = 1
    top_p: float = 0.98
    top_k: Optional[int] = None
    temperature: float = 0.6
    max_vlm_generation_tokens: int = 1
    use_cot: bool = True
    use_expert_only: bool = False  # Use only expert branch without VLM CoT
    batch_size: int = 2
    candidate_stride: int = 3


@dataclass
class OutputConfig:
    """Output configuration."""
    output_dir: str = "./output"
    vis_output_dir: str = "./visualizations"
    num_vis_samples: int = 10
    vis_camera: str = "FC"
    vis_save_images: bool = True


@dataclass
class Config:
    """Main configuration class."""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "Config":
        """Create Config from dictionary."""
        def _dict_to_obj(d, cls):
            if isinstance(d, dict):
                field_names = {f.name for f in cls.__dataclass_fields__.values()}
                filtered = {k: v for k, v in d.items() if k in field_names}
                for key, value in filtered.items():
                    field_type = cls.__dataclass_fields__[key].type
                    if hasattr(field_type, "__args__"):
                        # Handle Union types
                        filtered[key] = value
                    elif "tuple" in str(field_type) and isinstance(value, list):
                        filtered[key] = tuple(value)
                obj = cls(**filtered)
                return obj
            return d

        cfg = cls()
        for section_name in ["model", "data", "inference", "output"]:
            if section_name in config_dict:
                section_cls = type(f"{section_name.title()}Config", (), {
                    "__annotations__": {
                        k: type(v) for k, v in config_dict[section_name].items()
                    }
                })
                setattr(cfg, section_name, _dict_to_obj(config_dict[section_name], section_cls))
        return cfg

    def to_dict(self) -> dict:
        """Convert Config to dictionary."""
        return {
            "model": {k: getattr(self.model, k) for k in dir(self.model) if not k.startswith("_")},
            "data": {k: getattr(self.data, k) for k in dir(self.data) if not k.startswith("_")},
            "inference": {k: getattr(self.inference, k) for k in dir(self.inference) if not k.startswith("_")},
            "output": {k: getattr(self.output, k) for k in dir(self.output) if not k.startswith("_")},
        }
