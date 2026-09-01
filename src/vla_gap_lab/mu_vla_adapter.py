"""Inference adapter for the released mu-VLA MIKASA checkpoint."""

from __future__ import annotations

import json
import sys
import types
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn


def install_prismatic_remote_stubs() -> None:
    """Provide the two tiny Prismatic modules required by checkpoint remote code.

    Importing upstream ``prismatic`` eagerly imports its RLDS/TensorFlow training
    stack. Inference only needs action masks and constants, so isolate those
    definitions instead of pulling a training data stack into the simulator.
    """
    for name in list(sys.modules):
        if name == "prismatic" or name.startswith("prismatic."):
            del sys.modules[name]
    root = types.ModuleType("prismatic")
    root.__path__ = []
    training = types.ModuleType("prismatic.training")
    training.__path__ = []
    train_utils = types.ModuleType("prismatic.training.train_utils")
    vla = types.ModuleType("prismatic.vla")
    vla.__path__ = []
    constants = types.ModuleType("prismatic.vla.constants")

    class NormalizationType(str, Enum):
        NORMAL = "normal"
        BOUNDS = "bounds"
        BOUNDS_Q99 = "bounds_q99"

    def current_action_mask(token_ids: torch.Tensor) -> torch.Tensor:
        cumulative = torch.cumsum(token_ids != -100, dim=1)
        return (1 <= cumulative) & (cumulative <= 7) & (token_ids > 31743)

    def next_actions_mask(token_ids: torch.Tensor) -> torch.Tensor:
        cumulative = torch.cumsum(token_ids != -100, dim=1)
        return (cumulative > 7) & (token_ids > 31743)

    train_utils.get_current_action_mask = current_action_mask
    train_utils.get_next_actions_mask = next_actions_mask
    for name, value in {
        "ACTION_DIM": 7,
        "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
        "ACTION_TOKEN_BEGIN_IDX": 31743,
        "IGNORE_INDEX": -100,
        "NUM_ACTIONS_CHUNK": 8,
        "STOP_INDEX": 2,
        "NormalizationType": NormalizationType,
    }.items():
        setattr(constants, name, value)
    sys.modules.update(
        {
            "prismatic": root,
            "prismatic.training": training,
            "prismatic.training.train_utils": train_utils,
            "prismatic.vla": vla,
            "prismatic.vla.constants": constants,
        }
    )


def strip_ddp_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key.removeprefix("module."): value for key, value in state_dict.items()}


def normalize_bounds_q99(values: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
    low = np.asarray(stats["q01"])
    high = np.asarray(stats["q99"])
    mask = np.asarray(stats.get("mask", np.ones_like(low, dtype=bool)))
    normalized = np.where(mask, 2 * (values - low) / (high - low + 1e-8) - 1, values)
    return np.clip(normalized, -1, 1)


class _MLPResNetBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.ffn = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.ReLU())

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.ffn(value)


class _MLPResNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.mlp_resnet_blocks = nn.ModuleList([_MLPResNetBlock(hidden_dim) for _ in range(2)])
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.relu(self.fc1(self.layer_norm1(value)))
        for block in self.mlp_resnet_blocks:
            value = block(value)
        return self.fc2(self.layer_norm2(value))


class _L1RegressionActionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, action_dim: int = 7) -> None:
        super().__init__()
        self.model = _MLPResNet(input_dim * action_dim, hidden_dim, action_dim)

    def predict_action(self, states: torch.Tensor) -> torch.Tensor:
        return self.model(states.reshape(states.shape[0], 8, -1))


class _ProprioProjector(nn.Module):
    def __init__(self, llm_dim: int, proprio_dim: int = 7) -> None:
        super().__init__()
        self.fc1 = nn.Linear(proprio_dim, llm_dim)
        self.fc2 = nn.Linear(llm_dim, llm_dim)
        self.act_fn1 = nn.GELU()

    def forward(self, proprio: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act_fn1(self.fc1(proprio)))


class MuVLAPolicy:
    """Stateful, receding-horizon mu-VLA policy with diagnostic interventions."""

    chunk_size = 1

    def __init__(
        self,
        checkpoint: str | Path,
        instruction: str,
        *,
        mode: str = "normal",
        revision_step: int | None = None,
        load_in_4bit: bool = True,
    ) -> None:
        if mode not in {"normal", "freeze", "oracle_refresh"}:
            raise ValueError("mode must be normal, freeze, or oracle_refresh")
        install_prismatic_remote_stubs()
        from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig

        checkpoint = Path(checkpoint)
        quantization = None
        if load_in_4bit:
            quantization = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        self.processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)
        self.model = AutoModelForVision2Seq.from_pretrained(
            checkpoint,
            trust_remote_code=True,
            quantization_config=quantization,
            device_map={"": 0},
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        ).eval()
        statistics = json.loads((checkpoint / "dataset_statistics.json").read_text())
        self.model.norm_stats = statistics
        self.stats = statistics["mikasa_combined"]
        self.model.vision_backbone.set_num_images_in_input(2)

        self.action_head = _L1RegressionActionHead(
            input_dim=self.model.llm_dim,
            hidden_dim=self.model.llm_dim,
            action_dim=7,
        ).to(device="cuda", dtype=torch.bfloat16)
        self.action_head.load_state_dict(
            strip_ddp_prefix(
                torch.load(checkpoint / "action_head--150000_checkpoint.pt", weights_only=True)
            )
        )
        self.action_head.eval()
        self.proprio_projector = _ProprioProjector(self.model.llm_dim, proprio_dim=7).to(
            device="cuda", dtype=torch.bfloat16
        )
        self.proprio_projector.load_state_dict(
            strip_ddp_prefix(
                torch.load(
                    checkpoint / "proprio_projector--150000_checkpoint.pt", weights_only=True
                )
            )
        )
        self.proprio_projector.eval()
        initial = torch.load(checkpoint / "memory_module--150000_checkpoint.pt", weights_only=True)[
            "initial_memory"
        ]
        self.initial_memory = initial[None].to(device="cuda", dtype=torch.bfloat16)
        self.instruction = instruction
        self.mode = mode
        self.revision_step = revision_step
        self.inertia: list[float] = []
        self.candidate_inertia: list[float] = []
        self.update_norm: list[float] = []
        self.reset()

    def reset(self) -> None:
        self.memory = self.initial_memory.clone()
        self.step = 0
        self.inertia.clear()
        self.candidate_inertia.clear()
        self.update_norm.clear()

    def _inputs(self, obs: dict[str, Any]) -> tuple[Any, np.ndarray]:
        rgb = obs["rgb"].detach().cpu().numpy()[0]
        prompt = f"In: What action should the robot take to {self.instruction.lower()}?\nOut:"
        inputs = [
            self.processor(prompt, Image.fromarray(rgb[..., start : start + 3])).to(
                "cuda", dtype=torch.bfloat16
            )
            for start in (0, 3)
        ]
        inputs[0]["pixel_values"] = torch.cat(
            [inputs[0]["pixel_values"], inputs[1]["pixel_values"]], dim=1
        )
        proprio = obs["proprio"].detach().cpu().numpy()[0]
        return inputs[0], normalize_bounds_q99(proprio, self.stats["proprio"])

    @torch.inference_mode()
    def forward(self, obs: dict[str, Any]) -> torch.Tensor:
        inputs, proprio = self._inputs(obs)
        previous = self.memory
        input_memory = previous
        if self.mode == "oracle_refresh" and self.step == self.revision_step:
            input_memory = self.initial_memory
        actions, _, candidate = self.model.predict_action(
            **inputs,
            unnorm_key="mikasa_combined",
            proprio=proprio,
            proprio_projector=self.proprio_projector,
            action_head=self.action_head,
            memory_state=input_memory,
            use_memory_mask=True,
            attention_mask_mode="custom",
        )
        freeze = (
            self.mode == "freeze"
            and self.revision_step is not None
            and self.step >= self.revision_step
        )
        self.memory = previous if freeze else candidate
        self.inertia.append(
            float(
                F.cosine_similarity(
                    previous.float().flatten(1), self.memory.float().flatten(1)
                ).item()
            )
        )
        self.candidate_inertia.append(
            float(
                F.cosine_similarity(
                    previous.float().flatten(1), candidate.float().flatten(1)
                ).item()
            )
        )
        self.update_norm.append(
            float(torch.linalg.vector_norm(candidate.float() - previous.float()))
        )
        self.step += 1
        return torch.from_numpy(np.asarray(actions[:1], dtype=np.float32))
