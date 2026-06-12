from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn


@dataclass(frozen=True)
class LoraConfig:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "v_proj")

    @classmethod
    def from_dict(cls, raw_config: dict) -> "LoraConfig":
        target_modules = raw_config.get("target_modules", cls.target_modules)
        return cls(
            r=int(raw_config.get("r", cls.r)),
            alpha=int(raw_config.get("alpha", cls.alpha)),
            dropout=float(raw_config.get("dropout", cls.dropout)),
            target_modules=tuple(target_modules),
        )

    def to_dict(self) -> dict:
        return {
            "r": self.r,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "target_modules": list(self.target_modules),
        }


class LoRALinear(nn.Module):
    """Linear layer with a frozen base path and a trainable low-rank adapter."""

    def __init__(self, base_layer: nn.Linear, r: int, alpha: int, dropout: float) -> None:
        super().__init__()
        if r <= 0:
            raise ValueError("LoRA rank r must be positive")

        self.base = base_layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_A = nn.Linear(base_layer.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, base_layer.out_features, bias=False)

        for parameter in self.base.parameters():
            parameter.requires_grad = False

        self.reset_parameters()
        self.lora_A.to(device=base_layer.weight.device, dtype=base_layer.weight.dtype)
        self.lora_B.to(device=base_layer.weight.device, dtype=base_layer.weight.dtype)

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_B(self.lora_A(self.dropout(x))) * self.scaling


def _matches_target(module_name: str, target_modules: Iterable[str]) -> bool:
    leaf_name = module_name.rsplit(".", maxsplit=1)[-1]
    return any(leaf_name == target or module_name.endswith(target) for target in target_modules)


def apply_lora_to_model(model: nn.Module, config: LoraConfig) -> list[str]:
    """Replace target Linear layers in-place and return the replaced module names."""

    replaced: list[str] = []
    for parent_name, parent_module in list(model.named_modules()):
        for child_name, child_module in list(parent_module.named_children()):
            full_name = f"{parent_name}.{child_name}" if parent_name else child_name
            if isinstance(child_module, LoRALinear):
                continue
            if isinstance(child_module, nn.Linear) and _matches_target(full_name, config.target_modules):
                setattr(
                    parent_module,
                    child_name,
                    LoRALinear(
                        child_module,
                        r=config.r,
                        alpha=config.alpha,
                        dropout=config.dropout,
                    ),
                )
                replaced.append(full_name)

    if not replaced:
        raise ValueError(f"No Linear modules matched LoRA targets: {config.target_modules}")

    return replaced


def mark_only_lora_trainable(model: nn.Module) -> None:
    for name, parameter in model.named_parameters():
        parameter.requires_grad = ".lora_" in name


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if ".lora_" in name
    }


def load_lora_state_dict(model: nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    result = model.load_state_dict(state_dict, strict=False)
    expected = set(lora_state_dict(model))
    provided = set(state_dict)
    missing = sorted(expected - provided)
    unexpected = sorted(provided - expected)
    if missing or unexpected:
        raise RuntimeError(
            "LoRA adapter state mismatch: "
            f"missing={missing[:5]} unexpected={unexpected[:5]}"
        )
    non_lora_unexpected = [key for key in result.unexpected_keys if ".lora_" in key]
    if non_lora_unexpected:
        raise RuntimeError(f"Unexpected LoRA keys: {non_lora_unexpected[:5]}")

