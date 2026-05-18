# SPDX-License-Identifier: Apache-2.0
"""Main inference script for generating trajectories with Alpamayo 1.5 VLM.

This script generates multiple trajectory samples for each input frame using
the Alpamayo 1.5 Vision Language Model, projects them onto camera images
for visualization, and saves the results to parquet format.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import torch
except ModuleNotFoundError:
    torch = None

from traj_inference.config import Config, ModelConfig, DataConfig, InferenceConfig, OutputConfig
from traj_core.data_loader import (
    filter_t0s_with_full_future,
    get_dataset_names,
    get_clip_stems_from_dataset,
    load_data,
    to_device,
    get_t0_candidates,
)
from traj_core.frame_index import build_video_frame_t0_candidates
from traj_core.calibration_loader import (
    load_calibration_for_segment,
    CameraCalibration,
)
from traj_core.visualization import (
    load_image_from_frame,
    visualize_sample,
    create_trajectory_grid_visualization,
    draw_trajectory_on_image,
)
from traj_core.dynamics import optimize_pseudo_gt_trajectory, trajectory_components_from_xyz


MODEL_CAMERA_ORDER = ["FL", "FC", "FR", "RL", "RC", "RR", "FC_FAR"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate trajectories using Alpamayo 1.5 VLM"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to config JSON file"
    )
    parser.add_argument(
        "--model_path", type=str, default=None,
        help="Path to Alpamayo 1.5 model"
    )
    parser.add_argument(
        "--traj_checkpoint", type=str, default=None,
        help="Path to trajectory finetune checkpoint"
    )
    parser.add_argument(
        "--lora_adapter", type=str, default=None,
        help="Path to LoRA adapter"
    )
    parser.add_argument(
        "--train_data_root", type=str, default=None,
        help="Root directory for training data"
    )
    parser.add_argument(
        "--calibration_dir", type=str, default=None,
        help="Directory containing calibration files"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory for generated trajectories"
    )
    parser.add_argument(
        "--vis_output_dir", type=str, default=None,
        help="Output directory for visualizations"
    )
    parser.add_argument(
        "--datasets", type=str, default="",
        help="Comma-separated dataset names to process"
    )
    parser.add_argument(
        "--num_traj_samples", type=int, default=6,
        help="Number of trajectory samples per input"
    )
    parser.add_argument(
        "--num_vis_samples", type=int, default=10,
        help="Number of samples to visualize"
    )
    parser.add_argument(
        "--vis_camera", type=str, default="FC",
        choices=MODEL_CAMERA_ORDER,
        help="Camera to use for visualization"
    )
    parser.add_argument(
        "--min_speed_mps", type=float, default=2.0,
        help="Minimum speed for t0 candidates"
    )
    parser.add_argument(
        "--max_samples", type=int, default=0,
        help="Maximum number of samples to process (0 = all)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=2,
        help="Number of valid t0 frames to process per model batch"
    )
    parser.add_argument(
        "--candidate_stride", type=int, default=3,
        help="Process every Nth valid t0 candidate"
    )
    parser.add_argument(
        "--t0_source", type=str, default="video_frames",
        choices=["speed_candidates", "video_frames"],
        help=(
            "Source of t0 frames. speed_candidates keeps the legacy sparse "
            "speed-filtered candidates; video_frames uses master video timestamps."
        ),
    )
    parser.add_argument(
        "--frame_stride", type=int, default=1,
        help="Video-frame stride when --t0_source video_frames is used",
    )
    parser.add_argument(
        "--no_speed_filter", action="store_true",
        help="Disable min-speed filtering for legacy speed_candidates mode",
    )
    parser.add_argument(
        "--max_generation_tokens", type=int, default=None,
        help="Maximum VLM generation tokens (defaults to 256 with CoT, otherwise config value)"
    )
    parser.add_argument(
        "--use_cot", dest="use_cot", action="store_true", default=True,
        help="Use Chain-of-Thought generation"
    )
    parser.add_argument(
        "--no_cot", dest="use_cot", action="store_false",
        help="Disable Chain-of-Thought generation"
    )
    parser.add_argument(
        "-expert", "--expert", dest="use_expert_only", action="store_true", default=False,
        help="Use only expert branch without VLM CoT (faster, may reduce accuracy)"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.6,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top_p", type=float, default=0.98,
        help="Top-p sampling parameter"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device to use"
    )
    return parser.parse_args()


def load_config_from_args(args) -> Config:
    """Load configuration from command line arguments."""
    config = Config()

    if args.config:
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        config = Config.from_dict(config_dict)

    if args.model_path:
        config.model.model_path = args.model_path
    if args.traj_checkpoint:
        config.model.traj_checkpoint = args.traj_checkpoint
    if args.lora_adapter:
        config.model.lora_adapter = args.lora_adapter

    if args.train_data_root:
        config.data.train_data_root = args.train_data_root
    if args.calibration_dir:
        config.data.calibration_dir = args.calibration_dir

    if args.output_dir:
        config.output.output_dir = args.output_dir
    if args.vis_output_dir:
        config.output.vis_output_dir = args.vis_output_dir

    config.inference.num_traj_samples = args.num_traj_samples
    config.inference.temperature = args.temperature
    config.inference.top_p = args.top_p
    config.inference.use_cot = args.use_cot
    config.inference.use_expert_only = args.use_expert_only
    config.inference.batch_size = max(1, args.batch_size)
    config.inference.candidate_stride = max(1, args.candidate_stride)
    if args.max_generation_tokens is not None:
        config.inference.max_vlm_generation_tokens = args.max_generation_tokens
    elif config.inference.use_cot and config.inference.max_vlm_generation_tokens <= 1:
        config.inference.max_vlm_generation_tokens = 256

    config.output.num_vis_samples = args.num_vis_samples
    config.output.vis_camera = args.vis_camera

    return config


def extract_dataset_segment(clip_id: str) -> tuple[str, str]:
    """Extract dataset name and segment name from clip_id.

    Args:
        clip_id: Clip identifier (e.g., '2026-03-24-12-06-59')

    Returns:
        Tuple of (dataset_name, segment_name)
    """
    return clip_id, clip_id


def save_trajectory_results(
    results: dict,
    output_dir: Path,
) -> None:
    """Save trajectory generation results to disk in parquet format.

    Output is organized by dataset: output_dir/{dataset_name}/{clip_stem}.egomotion.parquet
    Each row is one trajectory (64 timesteps stored as lists).

    Args:
        results: Results dictionary containing trajectories and metadata
        output_dir: Base output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Group results by dataset and clip. Keep every t0 for the clip; each row
    # below is one sampled trajectory for one t0.
    by_clip = {}
    for result in results["results"]:
        key = (result["dataset_name"], result["clip_id"])
        by_clip.setdefault(key, []).append(result)

    total_files = 0
    total_rows = 0
    for (dataset_name, clip_id), clip_results in by_clip.items():
        dataset_dir = output_dir / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)

        traj_file = dataset_dir / f"{clip_id}.egomotion.parquet"

        rows = []
        for result in sorted(clip_results, key=lambda r: int(r["t0_us"])):
            t0_us = int(result["t0_us"])
            pred_trajs = result["predicted_trajectories"]  # [1, 1, num_samples, 64, 3]

            num_samples = pred_trajs.shape[2]
            num_steps = pred_trajs.shape[3]
            for sample_idx in range(num_samples):
                traj_xyz = np.asarray(pred_trajs[0, 0, sample_idx], dtype=np.float64)  # [num_steps, 3]
                optimization = optimize_pseudo_gt_trajectory(traj_xyz)
                components = trajectory_components_from_xyz(optimization.xyz)

                rows.append({
                    "t0_us": t0_us,
                    "sample_idx": sample_idx,
                    "source": "vla",
                    "timestamp": [t0_us + int((i + 1) * 100000) for i in range(num_steps)],
                    "qx": components["qx"].tolist(),
                    "qy": components["qy"].tolist(),
                    "qz": components["qz"].tolist(),
                    "qw": components["qw"].tolist(),
                    "x": components["x"].tolist(),
                    "y": components["y"].tolist(),
                    "z": components["z"].tolist(),
                    "vx": components["vx"].tolist(),
                    "vy": components["vy"].tolist(),
                    "vz": components["vz"].tolist(),
                    "curvature": components["curvature"].tolist(),
                })

        df = pd.DataFrame(rows)
        df.to_parquet(traj_file, index=False)
        total_files += 1
        total_rows += len(rows)
        print(
            f"  Saved {len(clip_results)} t0 frames / {len(rows)} trajectories "
            f"to {traj_file.relative_to(output_dir)}"
        )

    print(f"Saved {total_files} parquet files ({total_rows} trajectories) in {output_dir}")


def save_cot_results(results: list[dict], output_dir: Path) -> None:
    """Save generated CoT strings to a JSONL sidecar file."""
    cot_rows = []
    for result in results:
        cot_samples = result.get("cot_samples") or []
        if not cot_samples:
            continue

        for sample_idx, cot_text in enumerate(cot_samples):
            cot_rows.append({
                "dataset_name": result["dataset_name"],
                "clip_id": result["clip_id"],
                "t0_us": int(result["t0_us"]),
                "sample_idx": sample_idx,
                "cot": cot_text,
            })

    cot_file = output_dir / "cot.jsonl"
    if not cot_rows:
        if cot_file.exists():
            cot_file.unlink()
        print("No CoT outputs to save")
        return

    with open(cot_file, "w", encoding="utf-8") as f:
        for row in cot_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Saved {len(cot_rows)} CoT rows to {cot_file}")


def _move_to_device(data, device):
    """Recursively move tensors to device."""
    if isinstance(data, dict):
        return {k: _move_to_device(v, device) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return type(data)(_move_to_device(x, device) for x in data)
    elif hasattr(data, 'to'):
        return data.to(device, non_blocking=True)
    else:
        return data


def _batch_tokenized_inputs(tokenized_inputs: list[dict], pad_token_id: int) -> dict:
    """Merge per-sample processor outputs into one batched model input."""
    if torch is None:
        raise RuntimeError("PyTorch is required for Alpamayo inference.")
    if len(tokenized_inputs) == 1:
        return tokenized_inputs[0]

    batched = {}
    keys = tokenized_inputs[0].keys()
    for key in keys:
        values = [item[key] for item in tokenized_inputs]
        if not torch.is_tensor(values[0]):
            batched[key] = values
            continue

        if values[0].dim() == 2 and key in {"input_ids", "attention_mask", "labels"}:
            pad_value = 0
            if key == "input_ids":
                pad_value = pad_token_id
            elif key == "labels":
                pad_value = -100

            squeezed = [value.squeeze(0) for value in values]
            padded = torch.nn.utils.rnn.pad_sequence(
                squeezed, batch_first=True, padding_value=pad_value
            )
            batched[key] = padded
        else:
            batched[key] = torch.cat(values, dim=0)

    return batched


def sample_with_expert_only(
    model,
    conv_data: dict,
    processor,
    num_traj_samples: int = 6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate trajectories using only Expert branch + Diffusion, no VLM CoT.

    This method runs VLM forward pass once to get KV cache, then directly uses
    Expert branch with Diffusion sampling. Faster than full VLM CoT inference
    but may have slightly lower accuracy.

    Args:
        model: Alpamayo model
        conv_data: Converted data dictionary
        processor: Model processor
        num_traj_samples: Number of trajectory samples

    Returns:
        pred_xyz: (1, 1, num_samples, T, 3)
        pred_rot: (1, 1, num_samples, T, 3, 3)
    """
    from traj_inference.model_loader import build_inference_inputs

    import einops

    device = next(model.parameters()).device
    ego_history_xyz = conv_data["ego_history_xyz"].to(device)  # (B, 1, T, 3)
    ego_history_rot = conv_data["ego_history_rot"].to(device)  # (B, 1, T, 3, 3)
    B, n_traj_group, _, _ = ego_history_xyz.shape
    assert n_traj_group == 1, "Only one trajectory group is supported for inference."

    # Reuse the official no-CoT prompt format:
    #   history tokens + images + <|cot_start|><|cot_end|><|traj_future_start|>
    tokenized_data = build_inference_inputs(conv_data, processor, use_cot=False)
    tokenized_data = _move_to_device(dict(tokenized_data), device)
    input_ids = tokenized_data.pop("input_ids")

    # Fuse history trajectory info into token IDs
    traj_data = {
        "ego_history_xyz": ego_history_xyz,
        "ego_history_rot": ego_history_rot,
    }
    input_ids = model.fuse_traj_tokens(input_ids, traj_data)

    # Step 1: VLM forward pass only (no autoregressive generation), get KV cache
    vlm_kwargs = {
        "input_ids": input_ids,
        "use_cache": True,
        "return_dict": True,
        **tokenized_data,
    }

    with torch.no_grad():
        vlm_out = model.vlm(**vlm_kwargs)

    prompt_cache = vlm_out.past_key_values
    if prompt_cache is None:
        raise RuntimeError("VLM forward pass returned no KV cache")
    prefill_seq_len = prompt_cache.get_seq_length()

    # Get rope_deltas for position encoding
    if hasattr(model.vlm, "peft_config"):
        rope_src = model.vlm.base_model.model.model
    else:
        rope_src = model.vlm.model
    rope_deltas = getattr(rope_src, "rope_deltas", None)
    if rope_deltas is None:
        rope_deltas = torch.zeros((B, 1), dtype=torch.long, device=device)
    else:
        rope_deltas = rope_deltas.to(device)

    # Find <traj_future_start> position
    eos_id = model.tokenizer.convert_tokens_to_ids("<|traj_future_start|>")
    n_tokens = model.action_space.get_action_space_dims()[0]
    offset = model._find_eos_offset(
        sequences=input_ids,
        eos_token_id=eos_id,
        device=device,
    )

    n_samples_total = num_traj_samples
    total_batch = B * n_samples_total
    prompt_cache.batch_repeat_interleave(n_samples_total)
    offset = torch.repeat_interleave(offset, n_samples_total, dim=0)
    rope_deltas = torch.repeat_interleave(rope_deltas, n_samples_total, dim=0)
    prefix_mask = tokenized_data.get("attention_mask")
    if prefix_mask is not None:
        prefix_mask = torch.repeat_interleave(prefix_mask, n_samples_total, dim=0)

    position_ids, attention_mask = model._build_expert_pos_ids_and_attn_mask(
        offset=offset,
        rope_deltas=rope_deltas,
        kv_cache_seq_len=prefill_seq_len,
        n_diffusion_tokens=n_tokens,
        b_star=total_batch,
        device=device,
        prefix_mask=prefix_mask,
    )

    forward_kwargs = {}
    if getattr(model.config, "expert_non_causal_attention", False):
        forward_kwargs["is_causal"] = False

    # Define diffusion step function
    def step_fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        future_token_embeds = model.action_in_proj(x, t)
        if future_token_embeds.dim() == 2:
            future_token_embeds = future_token_embeds.view(b, n_tokens, -1)

        expert_out = model.expert(
            inputs_embeds=future_token_embeds,
            position_ids=position_ids,
            past_key_values=prompt_cache,
            attention_mask=attention_mask,
            use_cache=True,
            **forward_kwargs,
        )
        prompt_cache.crop(prefill_seq_len)
        last_hidden = expert_out.last_hidden_state[:, -n_tokens:]
        pred = model.action_out_proj(last_hidden).view(-1, *model.action_space.get_action_space_dims())
        return pred

    # Diffusion sampling
    sampled_action = model.diffusion.sample(
        batch_size=total_batch,
        step_fn=step_fn,
        device=device,
        return_all_steps=False,
    )

    # Convert action to trajectory
    hist_xyz_rep = einops.repeat(
        ego_history_xyz[:, -1], "b ... -> (b n) ...", n=n_samples_total
    )
    hist_rot_rep = einops.repeat(
        ego_history_rot[:, -1], "b ... -> (b n) ...", n=n_samples_total
    )
    pred_xyz, pred_rot = model.action_space.action_to_traj(sampled_action, hist_xyz_rep, hist_rot_rep)

    # Reshape to (B, 1, num_samples, T, 3), matching the standard rollout API.
    pred_xyz = einops.rearrange(pred_xyz, "(b n) ... -> b 1 n ...", n=n_samples_total)
    pred_rot = einops.rearrange(pred_rot, "(b n) ... -> b 1 n ...", n=n_samples_total)

    return pred_xyz, pred_rot


def process_sample_batch(
    model,
    processor,
    conv_batch: list[dict],
    config: Config,
    camera_calibs: dict[str, CameraCalibration],
) -> list[dict[str, Any]]:
    """Process a batch of samples and generate trajectories.

    Args:
        model: Alpamayo 1.5 model
        processor: Model processor
        conv_batch: Converted data dictionaries for the samples
        config: Configuration
        camera_calibs: Camera calibrations

    Returns:
        One result dictionary per sample
    """
    from traj_inference.model_loader import build_inference_inputs

    if torch is None:
        raise RuntimeError("PyTorch is required for Alpamayo inference.")

    results = []

    if config.inference.use_expert_only:
        # Expert-only mode: use Expert branch + Diffusion without VLM CoT
        expert_sample_chunk_size = max(1, int(os.environ.get("EXPERT_SAMPLE_CHUNK_SIZE", "1")))
        for conv_data in conv_batch:
            torch.cuda.manual_seed_all(42)
            xyz_chunks = []
            rot_chunks = []
            remaining_samples = config.inference.num_traj_samples
            while remaining_samples > 0:
                chunk_samples = min(expert_sample_chunk_size, remaining_samples)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pred_xyz_chunk, pred_rot_chunk = sample_with_expert_only(
                        model, conv_data, processor,
                        num_traj_samples=chunk_samples,
                    )
                xyz_chunks.append(pred_xyz_chunk.cpu())
                rot_chunks.append(pred_rot_chunk.cpu())
                del pred_xyz_chunk, pred_rot_chunk
                torch.cuda.empty_cache()
                remaining_samples -= chunk_samples

            pred_xyz = torch.cat(xyz_chunks, dim=2)
            pred_rot = torch.cat(rot_chunks, dim=2)

            pred_xyz_np = pred_xyz.float().numpy()
            pred_rot_np = pred_rot.float().numpy()

            gt_xyz_np = conv_data.get("ego_future_xyz", None)
            if gt_xyz_np is not None:
                gt_xyz_np = gt_xyz_np.cpu().float().numpy()

            results.append({
                "clip_id": conv_data["clip_id"],
                "dataset_name": conv_data.get("dataset_name", ""),
                "t0_us": conv_data["t0_us"],
                "num_samples": pred_xyz_np.shape[2],
                "trajectory_length": pred_xyz_np.shape[3],
                "predicted_trajectories": pred_xyz_np,
                "predicted_rotations": pred_rot_np,
                "gt_trajectory": gt_xyz_np,
                "cot_samples": [],
                "method": "expert_only",
                "t0_heading": {
                    "source": conv_data.get("t0_heading_source", "unknown"),
                    "disp_m": float(conv_data.get("t0_heading_disp_m", 0)),
                    "yaw_rad": float(conv_data.get("t0_heading_yaw_rad", 0)),
                },
            })
    else:
        # Standard VLM CoT mode
        tokenized_inputs = [
            build_inference_inputs(conv_data, processor, use_cot=config.inference.use_cot)
            for conv_data in conv_batch
        ]
        pad_token_id = getattr(model.tokenizer, "pad_token_id", 0)
        inp = _batch_tokenized_inputs(tokenized_inputs, pad_token_id=pad_token_id)

        mi = {
            "tokenized_data": _move_to_device(inp, "cuda"),
            "ego_history_xyz": torch.cat(
                [conv_data["ego_history_xyz"] for conv_data in conv_batch], dim=0
            ).to("cuda"),
            "ego_history_rot": torch.cat(
                [conv_data["ego_history_rot"] for conv_data in conv_batch], dim=0
            ).to("cuda"),
        }

        with torch.autocast("cuda", dtype=torch.bfloat16):
            model_outputs = model.sample_trajectories_from_data_with_vlm_rollout(
                data=copy.deepcopy(mi),
                top_p=config.inference.top_p,
                temperature=config.inference.temperature,
                num_traj_samples=config.inference.num_traj_samples,
                max_generation_length=config.inference.max_vlm_generation_tokens,
                return_extra=config.inference.use_cot,
            )

        if config.inference.use_cot:
            pred_xyz, pred_rot, extra = model_outputs
        else:
            pred_xyz, pred_rot = model_outputs
            extra = {}

        pred_xyz_np = pred_xyz.cpu().float().numpy()
        pred_rot_np = pred_rot.cpu().float().numpy()

        for batch_idx, conv_data in enumerate(conv_batch):
            gt_xyz_np = conv_data.get("ego_future_xyz", None)
            if gt_xyz_np is not None:
                gt_xyz_np = gt_xyz_np.cpu().float().numpy()

            cot_samples = []
            if "cot" in extra:
                cot_arr = extra["cot"][batch_idx]
                cot_samples = np.asarray(cot_arr).reshape(-1).tolist()

            results.append({
                "clip_id": conv_data["clip_id"],
                "dataset_name": conv_data.get("dataset_name", ""),
                "t0_us": conv_data["t0_us"],
                "num_samples": pred_xyz_np.shape[2],
                "trajectory_length": pred_xyz_np.shape[3],
                "predicted_trajectories": pred_xyz_np[batch_idx:batch_idx + 1],
                "predicted_rotations": pred_rot_np[batch_idx:batch_idx + 1],
                "gt_trajectory": gt_xyz_np,
                "cot_samples": cot_samples,
                "method": "vlm_cot" if config.inference.use_cot else "vlm_no_cot",
                "t0_heading": {
                    "source": conv_data.get("t0_heading_source", "unknown"),
                    "disp_m": float(conv_data.get("t0_heading_disp_m", 0)),
                    "yaw_rad": float(conv_data.get("t0_heading_yaw_rad", 0)),
                },
            })

    return results


def visualize_sample_wrapper(
    sample_result: dict,
    conv_data: dict,
    camera_calibs: dict[str, CameraCalibration],
    vis_camera: str,
    vis_output_dir: Path,
    sample_idx: int,
) -> None:
    """Create and save visualization for a sample.

    Args:
        sample_result: One result from process_sample_batch
        conv_data: Original converted data
        camera_calibs: Camera calibrations
        vis_camera: Camera name for visualization
        vis_output_dir: Output directory for visualizations
        sample_idx: Sample index for naming
    """
    if vis_camera not in camera_calibs:
        print(f"Warning: Camera {vis_camera} not in calibrations")
        return

    cam_idx = conv_data["camera_indices"]
    if len(cam_idx) == 0:
        return

    frame_idx = len(cam_idx) // 2
    frame = conv_data["image_frames"][frame_idx, 0]

    calib = camera_calibs[vis_camera]
    image = load_image_from_frame(frame)
    if image.shape[1] != calib.image_width or image.shape[0] != calib.image_height:
        image = cv2.resize(image, (calib.image_width, calib.image_height))

    pred_trajectories = [
        np.array(sample_result["predicted_trajectories"][0, 0, i])
        for i in range(sample_result["predicted_trajectories"].shape[2])
    ]

    gt_traj = None
    if sample_result["gt_trajectory"] is not None:
        gt_traj = np.array(sample_result["gt_trajectory"][0, 0])

    vis_img = visualize_sample(
        image,
        pred_trajectories,
        gt_traj,
        calib,
        save_path=vis_output_dir / f"sample_{sample_idx:04d}.png",
        show_timestamps=True,
    )


def main():
    args = parse_args()
    if torch is None:
        raise RuntimeError(
            "PyTorch is required for Alpamayo inference. "
            "Use trajectory_annotator.py for GUI-only annotation."
        )
    from traj_inference.model_loader import load_alpamayo_model, get_processor

    config = load_config_from_args(args)

    output_dir = Path(config.output.output_dir)
    vis_output_dir = Path(config.output.vis_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 60)
    print("Alpamayo 1.5 Trajectory Generation")
    print("=" * 60)
    print(f"Model path: {config.model.model_path}")
    print(f"Data root: {config.data.train_data_root}")
    print(f"Calibration dir: {config.data.calibration_dir}")
    print(f"Output dir: {config.output.output_dir}")
    print(f"Num trajectory samples: {config.inference.num_traj_samples}")
    print(f"CoT enabled: {config.inference.use_cot}")
    print(f"Batch size: {config.inference.batch_size}")
    if args.t0_source == "video_frames":
        print(f"t0 source: every video frame (frame_stride={max(1, int(args.frame_stride))})")
    else:
        speed_filter = "off" if args.no_speed_filter else f">= {args.min_speed_mps:.2f} m/s"
        print(f"t0 source: speed candidates (speed filter {speed_filter})")
        print(f"Candidate stride: every {config.inference.candidate_stride} valid t0")
    print(f"Max VLM generation tokens: {config.inference.max_vlm_generation_tokens}")
    print()

    print("[1/5] Loading Alpamayo 1.5 model...")
    model = load_alpamayo_model(
        model_path=config.model.model_path,
        traj_checkpoint=config.model.traj_checkpoint,
        lora_adapter=config.model.lora_adapter,
        device=args.device,
    )
    processor = get_processor(model.tokenizer)

    print("\n[2/5] Discovering datasets and clips...")
    datasets = get_dataset_names(config.data.train_data_root)
    print(f"Found {len(datasets)} datasets: {datasets}")

    if args.datasets:
        selected_datasets = [d.strip() for d in args.datasets.split(",")]
        datasets = [d for d in datasets if d in selected_datasets]
        print(f"Filtered to {len(datasets)} datasets: {datasets}")

    all_clips = []
    for dataset_name in datasets:
        dataset_path = Path(config.data.train_data_root) / dataset_name
        clip_stems = get_clip_stems_from_dataset(dataset_path)
        for clip_stem in clip_stems:
            all_clips.append((dataset_name, clip_stem))

    print(f"Found {len(all_clips)} clips across all datasets")

    if len(all_clips) == 0:
        print("No clips found. Exiting.")
        return

    print(f"\n[3/5] Loading calibrations...")
    all_calibrations = {}
    local_calibration_roots = [Path(config.data.train_data_root)]
    raw_calibration_root = Path(os.environ.get("RAW_TRAIN_DATA_ROOT", "/home/ubuntu/Public/train_data"))
    if raw_calibration_root.exists() and raw_calibration_root not in local_calibration_roots:
        local_calibration_roots.append(raw_calibration_root)
    for dataset_name in datasets:
        calib_dataset = dataset_name.replace('_converted', '')
        dataset_loaded = 0
        for dataset_name_inner, clip_stem in all_clips:
            if dataset_name_inner != dataset_name:
                continue
            last_error = None
            for local_calibration_root in local_calibration_roots:
                try:
                    calibs = load_calibration_for_segment(
                        config.data.calibration_dir,
                        calib_dataset,
                        clip_stem,
                        data_root=str(local_calibration_root),
                        target_image_hw=config.data.target_image_hw,
                    )
                    all_calibrations[(dataset_name, clip_stem)] = calibs
                    dataset_loaded += 1
                    break
                except Exception as e:
                    last_error = e
            else:
                print(f"Warning: Could not load calibration for {dataset_name}/{clip_stem}: {last_error}")
        if dataset_loaded:
            print(f"  Loaded calibration for {dataset_name}: {dataset_loaded} clips")
        else:
            print(f"Warning: No calibration loaded for {dataset_name}")

    print(f"Loaded calibrations for {len(all_calibrations)} clips")

    print("\n[4/5] Generating trajectories...")
    results = []
    vis_count = 0
    processed = 0
    pending_items = []
    start_time = time.time()

    pbar = tqdm(all_clips, desc="Processing clips")

    def flush_pending() -> None:
        nonlocal vis_count, processed, pending_items
        if not pending_items:
            return

        conv_batch = [item["conv_data"] for item in pending_items]
        batch_results = process_sample_batch(
            model,
            processor,
            conv_batch,
            config,
            pending_items[0]["calibs"],
        )

        for item, result in zip(pending_items, batch_results):
            results.append(result)
            processed += 1

            if vis_count < config.output.num_vis_samples:
                try:
                    visualize_sample_wrapper(
                        result,
                        item["conv_data"],
                        item["calibs"],
                        config.output.vis_camera,
                        vis_output_dir,
                        vis_count,
                    )
                    vis_count += 1
                except Exception as e:
                    print(f"Warning: Visualization failed: {e}")

        pending_items = []
        pbar.set_postfix({
            "processed": processed,
            "results": len(results),
        })

    for dataset_name, clip_stem in pbar:
        if args.max_samples > 0 and processed + len(pending_items) >= args.max_samples:
            break

        if (dataset_name, clip_stem) not in all_calibrations:
            continue

        try:
            calibs = all_calibrations[(dataset_name, clip_stem)]

            if args.t0_source == "video_frames":
                video_candidates = build_video_frame_t0_candidates(
                    config.data.train_data_root,
                    dataset_name,
                    clip_stem,
                    frame_stride=args.frame_stride,
                    num_history_steps=config.data.num_history_steps,
                    num_future_steps=config.data.num_future_steps,
                    require_full_history=False,
                    require_full_future=False,
                )
                t0_values = [int(t0) for t0 in video_candidates.t0_values]
                t0_values = filter_t0s_with_full_future(
                    config.data.train_data_root,
                    dataset_name,
                    t0_values,
                    num_future_steps=config.data.num_future_steps,
                    time_step=config.data.time_step,
                    clip_stem=clip_stem,
                )
                if not t0_values:
                    print(f"Warning: no video-frame t0 timestamps for {dataset_name}/{clip_stem}")
                    continue
            else:
                candidates = get_t0_candidates(
                    config.data.train_data_root,
                    dataset_name,
                    clip_stem,
                    min_speed_mps=-float("inf") if args.no_speed_filter else args.min_speed_mps,
                )
                t0_values = [int(t0) for t0, _speed in candidates] if candidates else [None]
                t0_values = t0_values[::config.inference.candidate_stride]

            for t0_us in t0_values:
                if args.max_samples > 0 and processed + len(pending_items) >= args.max_samples:
                    break

                conv_data = load_data(
                    data_root=config.data.train_data_root,
                    clip_stem=clip_stem,
                    dataset_name=dataset_name,
                    t0_us=t0_us,
                    num_history_steps=config.data.num_history_steps,
                    num_future_steps=config.data.num_future_steps,
                    time_step=config.data.time_step,
                    num_frames=config.data.num_frames,
                    target_image_hw=config.data.target_image_hw,
                    heading_num_steps=config.data.heading_num_steps,
                    min_heading_displacement_m=config.data.min_heading_displacement_m,
                )

                pending_items.append({
                    "conv_data": conv_data,
                    "calibs": calibs,
                })

                if len(pending_items) >= config.inference.batch_size:
                    flush_pending()

        except Exception as e:
            print(f"\nError processing {clip_stem}: {e}")
            import traceback
            traceback.print_exc()

    flush_pending()

    elapsed = time.time() - start_time
    if processed > 0:
        print(f"\nProcessed {processed} samples in {elapsed:.1f}s ({elapsed/processed:.2f}s per sample)")
    else:
        print(f"\nNo samples processed in {elapsed:.1f}s")

    print("\n[5/5] Saving results...")
    save_trajectory_results(
        {
            "config": {
                "model_path": config.model.model_path,
                "num_traj_samples": config.inference.num_traj_samples,
                "temperature": config.inference.temperature,
                "top_p": config.inference.top_p,
                "use_cot": config.inference.use_cot,
                "batch_size": config.inference.batch_size,
                "candidate_stride": config.inference.candidate_stride,
                "t0_source": args.t0_source,
                "frame_stride": max(1, int(args.frame_stride)),
                "no_speed_filter": bool(args.no_speed_filter),
            },
            "results": results,
        },
        output_dir,
    )
    save_cot_results(results, output_dir)

    print(f"\nDone! Generated {len(results)} samples")
    print(f"Trajectories saved to: {output_dir}")
    print(f"Visualizations saved to: {vis_output_dir}")


if __name__ == "__main__":
    main()
