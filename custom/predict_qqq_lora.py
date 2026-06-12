from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model import KronosTokenizer
from model.kronos import calc_time_stamps, sample_from_logits

from custom.qqq_adapter import load_adapter
from custom.qqq_data import FEATURE_COLUMNS, load_ohlcva_csv


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


def next_timestamp(index: pd.Series, fallback_minutes: int = 5) -> pd.Timestamp:
    if len(index) >= 2:
        delta = index.iloc[-1] - index.iloc[-2]
    else:
        delta = pd.Timedelta(minutes=fallback_minutes)
    return index.iloc[-1] + delta


def prepare_context(df: pd.DataFrame, lookback: int, clip: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
    context = df.tail(lookback).copy()
    if len(context) < lookback:
        raise ValueError(f"Need at least {lookback} rows, got {len(context)}")

    feature_values = context[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float32)
    mean = feature_values.mean(axis=0)
    std = feature_values.std(axis=0)
    normalized = np.clip((feature_values - mean) / (std + 1e-5), -clip, clip).astype(np.float32)

    future_timestamp = next_timestamp(context["timestamps"])
    x_stamp = calc_time_stamps(context["timestamps"]).to_numpy(dtype=np.float32)
    y_stamp = calc_time_stamps(pd.Series([future_timestamp])).to_numpy(dtype=np.float32)
    return (
        torch.from_numpy(normalized).unsqueeze(0),
        torch.from_numpy(x_stamp).unsqueeze(0),
        torch.from_numpy(y_stamp).unsqueeze(0),
        mean,
        std,
    )


def predict_next(
    tokenizer: KronosTokenizer,
    forecaster,
    df: pd.DataFrame,
    lookback: int,
    clip: float,
    temperature: float,
    top_p: float,
    device: torch.device,
) -> dict[str, Any]:
    x, x_stamp, y_stamp, mean, std = prepare_context(df, lookback, clip)
    x = x.to(device)
    x_stamp = x_stamp.to(device)
    y_stamp = y_stamp.to(device)
    full_stamp = torch.cat([x_stamp, y_stamp], dim=1)

    tokenizer.eval()
    forecaster.eval()
    with torch.no_grad():
        token_s1, token_s2 = tokenizer.encode(x, half=True)
        s1_logits, context = forecaster.kronos.decode_s1(token_s1, token_s2, x_stamp)
        next_s1 = sample_from_logits(
            s1_logits[:, -1, :],
            temperature=temperature,
            top_k=0,
            top_p=top_p,
            sample_logits=True,
        )
        s2_logits = forecaster.kronos.decode_s2(context, next_s1)
        next_s2 = sample_from_logits(
            s2_logits[:, -1, :],
            temperature=temperature,
            top_k=0,
            top_p=top_p,
            sample_logits=True,
        )
        output_s1 = torch.cat([token_s1, next_s1], dim=1)
        output_s2 = torch.cat([token_s2, next_s2], dim=1)
        decoded = tokenizer.decode([output_s1, output_s2], half=True)
        next_bar = decoded[:, -1, :].squeeze(0).detach().cpu().numpy()
        next_bar = next_bar * (std + 1e-5) + mean

        _, _, return_pred = forecaster(output_s1[:, :-1], output_s2[:, :-1], full_stamp[:, :-1, :])
        scaled_return_pred = return_pred[:, -1].item()

    return {
        "temperature": temperature,
        "top_p": top_p,
        "predicted_bar": dict(zip(FEATURE_COLUMNS, [float(value) for value in next_bar])),
        "predicted_log_return": scaled_return_pred / 100.0,
    }


def calibrate_grid(
    tokenizer: KronosTokenizer,
    forecaster,
    df: pd.DataFrame,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, float]:
    """Pick temperature/top_p by recent directional hit rate."""

    inference_config = config.get("inference", {})
    temperature_grid = inference_config.get("temperature_grid", [0.1, 0.3, 0.5, 0.7, 0.9])
    top_p_grid = inference_config.get("top_p_grid", [0.8, 0.9, 0.95])
    calibration_lookback = int(inference_config.get("calibration_lookback", 10))
    data_config = config["data"]
    lookback = int(data_config.get("lookback_window", 512))
    clip = float(data_config.get("clip", 5.0))

    if len(df) < lookback + calibration_lookback + 1:
        return {"temperature": float(temperature_grid[0]), "top_p": float(top_p_grid[0]), "hit_rate": float("nan")}

    best = {"temperature": float(temperature_grid[0]), "top_p": float(top_p_grid[0]), "hit_rate": -1.0}
    for temperature, top_p in itertools.product(temperature_grid, top_p_grid):
        hits = 0
        trials = 0
        for end_index in range(len(df) - calibration_lookback, len(df)):
            historical = df.iloc[:end_index].copy()
            actual_return = np.log(df["close"].iloc[end_index] / df["close"].iloc[end_index - 1])
            prediction = predict_next(
                tokenizer,
                forecaster,
                historical,
                lookback=lookback,
                clip=clip,
                temperature=float(temperature),
                top_p=float(top_p),
                device=device,
            )
            hits += int(np.sign(prediction["predicted_log_return"]) == np.sign(actual_return))
            trials += 1
        hit_rate = hits / trials if trials else 0.0
        if hit_rate > best["hit_rate"]:
            best = {"temperature": float(temperature), "top_p": float(top_p), "hit_rate": hit_rate}

    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="Run QQQ next-bar prediction with a Kronos LoRA adapter.")
    parser.add_argument("--config", default="custom/configs/qqq_lora.yaml", help="YAML config path")
    parser.add_argument("--adapter", default=None, help="Adapter checkpoint path")
    parser.add_argument("--data", default=None, help="QQQ OHLCVA CSV path")
    parser.add_argument("--calibrate", action="store_true", help="Run rolling grid calibration before prediction")
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    tokenizer_path = resolve_path(config["model"]["tokenizer_path"])
    base_model_path = resolve_path(config["model"]["base_model_path"])
    adapter_path = resolve_path(args.adapter or (resolve_path(config["model"]["output_dir"]) / config["model"].get("adapter_name", "adapter.pt")))
    data_path = resolve_path(args.data or config["data"]["data_path"])

    tokenizer = KronosTokenizer.from_pretrained(str(tokenizer_path)).to(device)
    forecaster, adapter_payload = load_adapter(base_model_path, adapter_path, map_location=device)
    forecaster.to(device)

    df = load_ohlcva_csv(data_path, timestamp_col=str(config["data"].get("timestamp_col", "timestamps")))
    temperature = args.temperature
    top_p = args.top_p
    calibration = None
    if args.calibrate:
        calibration = calibrate_grid(tokenizer, forecaster, df, config, device)
        temperature = calibration["temperature"]
        top_p = calibration["top_p"]

    prediction = predict_next(
        tokenizer,
        forecaster,
        df,
        lookback=int(config["data"].get("lookback_window", 512)),
        clip=float(config["data"].get("clip", 5.0)),
        temperature=temperature,
        top_p=top_p,
        device=device,
    )
    result = {
        "adapter": str(adapter_path),
        "adapter_metadata": adapter_payload.get("metadata", {}),
        "data": str(data_path),
        "last_timestamp": str(df["timestamps"].iloc[-1]),
        "calibration": calibration,
        "prediction": prediction,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

