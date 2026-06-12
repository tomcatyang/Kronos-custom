from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model import Kronos, KronosTokenizer

from custom.lora import LoraConfig
from custom.qqq_adapter import KronosLoraForecaster, build_lora_forecaster, save_adapter
from custom.qqq_data import build_datasets


def resolve_path(path_value: str | Path, project_root: Path = PROJECT_ROOT) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else project_root / path


def load_config(config_path: str | Path) -> dict[str, Any]:
    config_path = resolve_path(config_path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid config file: {config_path}")
    return config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(device_config: str) -> torch.device:
    if device_config == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_config)


def trainable_parameter_summary(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total, trainable


def compute_losses(
    forecaster: KronosLoraForecaster,
    batch: dict[str, torch.Tensor],
    tokenizer: KronosTokenizer,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    x = batch["x"].to(device, non_blocking=True)
    stamp = batch["stamp"].to(device, non_blocking=True)
    return_targets = batch["return_targets"].to(device, non_blocking=True)

    with torch.no_grad():
        token_s1, token_s2 = tokenizer.encode(x, half=True)

    input_s1 = token_s1[:, :-1]
    input_s2 = token_s2[:, :-1]
    target_s1 = token_s1[:, 1:]
    target_s2 = token_s2[:, 1:]
    stamp_in = stamp[:, :-1, :]

    s1_logits, s2_logits, return_pred = forecaster(input_s1, input_s2, stamp_in, s1_targets=target_s1)

    loss_config = config.get("loss", {})
    lookback_window = int(config["data"].get("lookback_window", 512))
    predict_window = int(config["data"].get("predict_window", 1))
    token_loss_weight = float(loss_config.get("token_loss_weight", 1.0))
    return_mse_weight = float(loss_config.get("return_mse_weight", 1.0))
    direction_loss_weight = float(loss_config.get("direction_loss_weight", 0.0))
    return_scale = float(loss_config.get("return_scale", 100.0))

    if loss_config.get("loss_on_prediction_window_only", True):
        token_positions = torch.arange(input_s1.shape[1], device=device)
        start_position = max(0, lookback_window - 1)
        end_position = min(input_s1.shape[1], start_position + predict_window)
        mask = (token_positions >= start_position) & (token_positions < end_position)
        mask = mask.unsqueeze(0).expand_as(input_s1)
    else:
        mask = torch.ones_like(input_s1, dtype=torch.bool)

    token_loss_s1 = F.cross_entropy(s1_logits[mask], target_s1[mask])
    token_loss_s2 = F.cross_entropy(s2_logits[mask], target_s2[mask])
    token_loss = (token_loss_s1 + token_loss_s2) / 2.0

    scaled_returns = return_targets * return_scale
    return_mse = F.mse_loss(return_pred[mask], scaled_returns[mask])
    loss = token_loss_weight * token_loss + return_mse_weight * return_mse

    direction_loss = torch.zeros((), device=device)
    if direction_loss_weight > 0:
        direction_targets = (return_targets[mask] > 0).float()
        direction_loss = F.binary_cross_entropy_with_logits(return_pred[mask], direction_targets)
        loss = loss + direction_loss_weight * direction_loss

    return loss, {
        "loss": float(loss.detach().cpu()),
        "token_loss": float(token_loss.detach().cpu()),
        "token_loss_s1": float(token_loss_s1.detach().cpu()),
        "token_loss_s2": float(token_loss_s2.detach().cpu()),
        "return_mse": float(return_mse.detach().cpu()),
        "direction_loss": float(direction_loss.detach().cpu()),
    }


def run_epoch(
    forecaster: KronosLoraForecaster,
    tokenizer: KronosTokenizer,
    loader: DataLoader,
    config: dict[str, Any],
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    phase: str,
    max_steps: int | None = None,
) -> dict[str, float]:
    is_train = optimizer is not None
    forecaster.train(is_train)
    tokenizer.eval()

    totals: dict[str, float] = {}
    batch_count = 0
    training_config = config["training"]
    grad_clip = float(training_config.get("grad_clip", 1.0))
    log_interval = int(training_config.get("log_interval", 0) or 0)
    total_steps = len(loader) if max_steps is None else min(len(loader), max_steps)

    for batch_index, batch in enumerate(loader):
        if max_steps is not None and batch_index >= max_steps:
            break

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = compute_losses(forecaster, batch, tokenizer, config, device)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in forecaster.parameters() if parameter.requires_grad],
                    max_norm=grad_clip,
                )
            optimizer.step()
        else:
            with torch.no_grad():
                _, metrics = compute_losses(forecaster, batch, tokenizer, config, device)

        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + value
        batch_count += 1

        should_log = (
            log_interval > 0
            and (batch_count % log_interval == 0 or batch_count == total_steps)
        )
        if should_log:
            running = {key: value / batch_count for key, value in totals.items()}
            print(
                f"[{phase}] epoch={epoch} step={batch_count}/{total_steps} "
                f"loss={running['loss']:.6f} "
                f"token_loss={running['token_loss']:.6f} "
                f"return_mse={running['return_mse']:.6f}",
                flush=True,
            )

    if batch_count == 0:
        raise ValueError("DataLoader produced no batches")

    return {key: value / batch_count for key, value in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a QQQ LoRA adapter for Kronos.")
    parser.add_argument("--config", default="custom/configs/qqq_lora.yaml", help="YAML config path")
    parser.add_argument("--dry-run", action="store_true", help="Run one train and one validation step without saving")
    parser.add_argument("--max-train-steps", type=int, default=None, help="Limit train steps per epoch")
    args = parser.parse_args()

    config = load_config(args.config)
    training_config = config.get("training", {})
    set_seed(int(training_config.get("seed", 42)))
    device = select_device(str(training_config.get("device", "auto")))

    output_dir = resolve_path(config["model"]["output_dir"])
    adapter_path = output_dir / str(config["model"].get("adapter_name", "adapter.pt"))

    train_dataset, val_dataset = build_datasets(config["data"], PROJECT_ROOT)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(training_config.get("batch_size", 2)),
        shuffle=True,
        num_workers=int(training_config.get("num_workers", 0)),
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(training_config.get("batch_size", 2)),
        shuffle=False,
        num_workers=int(training_config.get("num_workers", 0)),
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    tokenizer_path = resolve_path(config["model"]["tokenizer_path"])
    base_model_path = resolve_path(config["model"]["base_model_path"])
    print(f"Device: {device}")
    print(f"Loading tokenizer: {tokenizer_path}")
    tokenizer = KronosTokenizer.from_pretrained(str(tokenizer_path)).to(device)
    tokenizer.eval()
    for parameter in tokenizer.parameters():
        parameter.requires_grad = False

    print(f"Loading base model: {base_model_path}")
    kronos = Kronos.from_pretrained(str(base_model_path)).to(device)
    lora_config = LoraConfig.from_dict(config.get("lora", {}))
    forecaster, replaced_modules = build_lora_forecaster(kronos, lora_config)
    forecaster.to(device)

    total_parameters, trainable_parameters = trainable_parameter_summary(forecaster)
    print(f"LoRA target modules: {len(replaced_modules)}")
    print(f"Trainable parameters: {trainable_parameters:,} / {total_parameters:,}")
    print(f"Train samples: {len(train_dataset)}, validation samples: {len(val_dataset)}")

    optimizer = torch.optim.AdamW(
        [parameter for parameter in forecaster.parameters() if parameter.requires_grad],
        lr=float(training_config.get("learning_rate", 1e-4)),
        weight_decay=float(training_config.get("weight_decay", 0.01)),
    )

    max_train_steps = args.max_train_steps
    if max_train_steps is None:
        raw_max_steps = training_config.get("max_train_steps")
        max_train_steps = None if raw_max_steps in (None, "") else int(raw_max_steps)
    if args.dry_run:
        max_train_steps = 1

    best_val_loss = float("inf")
    best_metadata: dict[str, Any] = {}
    epochs = 1 if args.dry_run else int(training_config.get("epochs", 3))
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(
            forecaster,
            tokenizer,
            train_loader,
            config,
            device,
            optimizer=optimizer,
            epoch=epoch,
            phase="train",
            max_steps=max_train_steps,
        )
        val_metrics = run_epoch(
            forecaster,
            tokenizer,
            val_loader,
            config,
            device,
            optimizer=None,
            epoch=epoch,
            phase="val",
            max_steps=1 if args.dry_run else None,
        )
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "train": train_metrics,
                    "val": val_metrics,
                },
                ensure_ascii=False,
                indent=2,
            )
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_metadata = {
                "base_model_path": str(base_model_path),
                "tokenizer_path": str(tokenizer_path),
                "data_path": str(resolve_path(config["data"]["data_path"])),
                "train_samples": len(train_dataset),
                "val_samples": len(val_dataset),
                "replaced_modules": replaced_modules,
                "epoch": epoch,
                "best_val_loss": best_val_loss,
            }
            if not args.dry_run:
                save_adapter(adapter_path, forecaster, lora_config, best_metadata)
                output_dir.mkdir(parents=True, exist_ok=True)
                resolved_config = copy.deepcopy(config)
                resolved_config["model"]["tokenizer_path"] = str(tokenizer_path)
                resolved_config["model"]["base_model_path"] = str(base_model_path)
                resolved_config["model"]["adapter_path"] = str(adapter_path)
                with (output_dir / "training_config.yaml").open("w", encoding="utf-8") as handle:
                    yaml.safe_dump(resolved_config, handle, allow_unicode=True, sort_keys=False)

    elapsed_seconds = time.time() - start_time
    if args.dry_run:
        print("Dry run completed; no adapter was saved.")
    else:
        print(f"Best validation loss: {best_val_loss:.6f}")
        print(f"Adapter saved to: {adapter_path}")
        print(f"Metadata: {json.dumps(best_metadata, ensure_ascii=False, indent=2)}")
    print(f"Elapsed seconds: {elapsed_seconds:.2f}")


if __name__ == "__main__":
    main()
