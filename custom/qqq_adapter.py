from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from model import Kronos

from custom.lora import (
    LoraConfig,
    apply_lora_to_model,
    load_lora_state_dict,
    lora_state_dict,
    mark_only_lora_trainable,
)


class KronosLoraForecaster(nn.Module):
    """Kronos predictor plus a small next-return regression head."""

    def __init__(self, kronos: Kronos) -> None:
        super().__init__()
        self.kronos = kronos
        self.return_head = nn.Linear(kronos.d_model, 1)

    def forward(
        self,
        s1_ids: torch.Tensor,
        s2_ids: torch.Tensor,
        stamp: torch.Tensor,
        s1_targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        s1_logits, context = self.kronos.decode_s1(s1_ids, s2_ids, stamp)
        s1_condition = s1_targets if s1_targets is not None else s1_ids
        s2_logits = self.kronos.decode_s2(context, s1_condition)
        return_pred = self.return_head(context).squeeze(-1)
        return s1_logits, s2_logits, return_pred


def build_lora_forecaster(kronos: Kronos, lora_config: LoraConfig) -> tuple[KronosLoraForecaster, list[str]]:
    replaced = apply_lora_to_model(kronos, lora_config)
    mark_only_lora_trainable(kronos)
    forecaster = KronosLoraForecaster(kronos)
    return forecaster, replaced


def save_adapter(
    output_path: str | Path,
    forecaster: KronosLoraForecaster,
    lora_config: LoraConfig,
    metadata: dict[str, Any],
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "kronos_qqq_lora_adapter_v1",
        "lora_config": lora_config.to_dict(),
        "metadata": metadata,
        "adapter_state": lora_state_dict(forecaster.kronos),
        "return_head_state": {
            name: value.detach().cpu()
            for name, value in forecaster.return_head.state_dict().items()
        },
    }
    torch.save(payload, output_path)


def load_adapter(
    base_model_path: str | Path,
    adapter_path: str | Path,
    map_location: str | torch.device = "cpu",
) -> tuple[KronosLoraForecaster, dict[str, Any]]:
    adapter_path = Path(adapter_path)
    payload = torch.load(adapter_path, map_location=map_location)
    lora_config = LoraConfig.from_dict(payload["lora_config"])

    kronos = Kronos.from_pretrained(str(base_model_path))
    forecaster, _ = build_lora_forecaster(kronos, lora_config)
    load_lora_state_dict(forecaster.kronos, payload["adapter_state"])
    forecaster.return_head.load_state_dict(payload["return_head_state"])
    return forecaster, payload

