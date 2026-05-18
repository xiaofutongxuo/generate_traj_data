# SPDX-License-Identifier: Apache-2.0
"""Load Alpamayo 1.5 model and processor."""

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

_repo_root = Path(__file__).resolve().parents[1]
for alpamayo_src in (
    os.environ.get("ALPAMAYO_SRC"),
    "/home/ubuntu/Public/lxh/alpamayo_1.5/src",
    str(_repo_root / "alpamayo_1.5" / "src"),
):
    if alpamayo_src and Path(alpamayo_src).exists():
        sys.path.insert(0, alpamayo_src)
        break

import alpamayo1_5.helper as helper
from alpamayo1_5.checkpoint_utils import maybe_load_lora_adapter, maybe_load_traj_finetune_weights
from alpamayo1_5.models.token_utils import to_special_token


def load_alpamayo_model(model_path: str, traj_checkpoint: str = None, lora_adapter: str = None, device: str = "cuda") -> "Alpamayo1_5":
    """Load Alpamayo 1.5 model with optional checkpoints.

    Args:
        model_path: Path to the Alpamayo 1.5 model
        traj_checkpoint: Optional path to trajectory finetune weights
        lora_adapter: Optional path to LoRA adapter
        device: Device to load the model on

    Returns:
        Loaded Alpamayo 1.5 model
    """
    from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5

    print(f"Loading Alpamayo 1.5 model from {model_path}...")

    model = Alpamayo1_5.from_pretrained(model_path, dtype=torch.bfloat16)

    if lora_adapter:
        maybe_load_lora_adapter(
            model, lora_adapter, reporter=lambda msg: print(f"  LoRA: {msg}")
        )

    if traj_checkpoint:
        maybe_load_traj_finetune_weights(
            model, traj_checkpoint, reporter=lambda msg: print(f"  Traj: {msg}")
        )

    model = model.to(device)
    print(f"Model loaded successfully on {device}")

    return model


def get_processor(tokenizer):
    """Get the processor for the model.

    Args:
        tokenizer: The tokenizer from the loaded model

    Returns:
        Processor for the model
    """
    return helper.get_processor(tokenizer)


def build_inference_inputs(conv_data: dict, processor, use_cot: bool = False):
    """Build inference inputs from converted data.

    Args:
        conv_data: The converted data dictionary from data loader
        processor: The model processor
        use_cot: Whether to use Chain-of-Thought generation

    Returns:
        Processed inputs for the model
    """
    frames = conv_data["image_frames"].flatten(0, 1)
    msgs = helper.create_message(
        frames,
        camera_indices=conv_data["camera_indices"],
    )

    if use_cot:
        return processor.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=False,
            continue_final_message=True, return_dict=True, return_tensors="pt",
        )

    msgs[-1]["content"] = [{
        "type": "text",
        "text": (
            f"{to_special_token('cot_start')}"
            f"{to_special_token('cot_end')}"
            f"{to_special_token('traj_future_start')}"
        ),
    }]
    text = processor.apply_chat_template(
        msgs,
        tokenize=False,
        add_generation_prompt=False,
        continue_final_message=False,
    )
    return processor(
        text=[text],
        images=list(frames),
        return_tensors="pt",
    )
