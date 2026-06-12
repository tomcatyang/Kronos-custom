from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
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
from model.kronos import calc_time_stamps

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


def select_device(device_config: str) -> torch.device:
    if device_config == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_config)


def split_boundaries(df: pd.DataFrame, data_config: dict[str, Any]) -> dict[str, int]:
    total_length = len(df)
    train_end = int(total_length * float(data_config.get("train_ratio", 0.8)))
    val_end = int(total_length * (float(data_config.get("train_ratio", 0.8)) + float(data_config.get("val_ratio", 0.1))))
    return {"train_end": train_end, "val_end": val_end, "total_length": total_length}


def parse_timestamp(value: str | None) -> pd.Timestamp | None:
    return None if value is None else pd.Timestamp(value)


def target_indices_for_backtest(
    df: pd.DataFrame,
    data_config: dict[str, Any],
    split: str,
    start: str | None,
    end: str | None,
) -> tuple[list[int], dict[str, Any]]:
    lookback = int(data_config.get("lookback_window", 512))
    boundaries = split_boundaries(df, data_config)
    train_end = boundaries["train_end"]
    val_end = boundaries["val_end"]
    total_length = boundaries["total_length"]

    if split == "val":
        raw_start, raw_end = train_end, val_end
    elif split == "test":
        raw_start, raw_end = val_end, total_length
    elif split == "custom":
        raw_start, raw_end = lookback, total_length
    else:
        raise ValueError("split must be one of: val, test, custom")

    start_ts = parse_timestamp(start)
    end_ts = parse_timestamp(end)
    target_indices = list(range(max(raw_start, lookback), raw_end))
    if start_ts is not None:
        target_indices = [idx for idx in target_indices if df["timestamps"].iloc[idx] >= start_ts]
    if end_ts is not None:
        target_indices = [idx for idx in target_indices if df["timestamps"].iloc[idx] <= end_ts]

    if target_indices and min(target_indices) < train_end:
        train_start_ts = df["timestamps"].iloc[0]
        train_end_ts = df["timestamps"].iloc[train_end - 1]
        raise ValueError(
            "Backtest target period overlaps the training period: "
            f"{train_start_ts} to {train_end_ts}"
        )

    if not target_indices:
        raise ValueError("No target bars selected for backtest")

    metadata = {
        **boundaries,
        "lookback_window": lookback,
        "train_start": str(df["timestamps"].iloc[0]),
        "train_end": str(df["timestamps"].iloc[train_end - 1]),
        "val_start": str(df["timestamps"].iloc[train_end]),
        "val_end": str(df["timestamps"].iloc[val_end - 1]),
        "test_start": str(df["timestamps"].iloc[val_end]),
        "test_end": str(df["timestamps"].iloc[-1]),
        "selected_start": str(df["timestamps"].iloc[target_indices[0]]),
        "selected_end": str(df["timestamps"].iloc[target_indices[-1]]),
        "selected_count": len(target_indices),
        "split": split,
    }
    return target_indices, metadata


def build_context_batch(
    df: pd.DataFrame,
    target_indices: list[int],
    lookback: int,
    clip: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    x_batch = []
    stamp_batch = []
    for target_idx in target_indices:
        context = df.iloc[target_idx - lookback : target_idx]
        features = context[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float32)
        mean = features.mean(axis=0)
        std = features.std(axis=0)
        normalized = np.clip((features - mean) / (std + 1e-5), -clip, clip).astype(np.float32)
        stamps = calc_time_stamps(context["timestamps"]).to_numpy(dtype=np.float32)
        x_batch.append(normalized)
        stamp_batch.append(stamps)

    return (
        torch.from_numpy(np.stack(x_batch, axis=0)),
        torch.from_numpy(np.stack(stamp_batch, axis=0)),
    )


def predict_returns(
    tokenizer: KronosTokenizer,
    forecaster,
    df: pd.DataFrame,
    target_indices: list[int],
    lookback: int,
    clip: float,
    batch_size: int,
    return_scale: float,
    device: torch.device,
    log_interval: int,
) -> np.ndarray:
    predictions: list[np.ndarray] = []
    tokenizer.eval()
    forecaster.eval()

    total_batches = math.ceil(len(target_indices) / batch_size)
    with torch.no_grad():
        for batch_number, start in enumerate(range(0, len(target_indices), batch_size), start=1):
            batch_indices = target_indices[start : start + batch_size]
            x, stamp = build_context_batch(df, batch_indices, lookback, clip)
            x = x.to(device)
            stamp = stamp.to(device)
            token_s1, token_s2 = tokenizer.encode(x, half=True)
            _, context = forecaster.kronos.decode_s1(token_s1, token_s2, stamp)
            scaled_pred = forecaster.return_head(context).squeeze(-1)[:, -1]
            predictions.append((scaled_pred / return_scale).detach().cpu().numpy())

            if log_interval > 0 and (batch_number % log_interval == 0 or batch_number == total_batches):
                completed = min(start + len(batch_indices), len(target_indices))
                print(
                    f"[backtest] batch={batch_number}/{total_batches} "
                    f"bars={completed}/{len(target_indices)}",
                    flush=True,
                )

    return np.concatenate(predictions, axis=0)


def max_drawdown(log_returns: np.ndarray) -> float:
    equity = np.exp(np.cumsum(log_returns))
    running_max = np.maximum.accumulate(equity)
    drawdowns = equity / running_max - 1.0
    return float(drawdowns.min()) if len(drawdowns) else 0.0


def summarize_results(
    results: pd.DataFrame,
    threshold: float,
    fee_bps: float,
    mode: str,
) -> dict[str, Any]:
    pred = results["predicted_log_return"].to_numpy(dtype=np.float64)
    actual = results["actual_log_return"].to_numpy(dtype=np.float64)
    signal = results["signal"].to_numpy(dtype=np.float64)
    strategy = results["strategy_log_return"].to_numpy(dtype=np.float64)
    nonzero_mask = signal != 0

    if len(pred) > 1 and np.std(pred) > 0 and np.std(actual) > 0:
        corr = float(np.corrcoef(pred, actual)[0, 1])
    else:
        corr = float("nan")

    if len(strategy) > 1 and np.std(strategy) > 0:
        bars_per_year = 252 * 78
        annualized_sharpe = float(np.mean(strategy) / np.std(strategy) * np.sqrt(bars_per_year))
    else:
        annualized_sharpe = float("nan")

    return {
        "bars": int(len(results)),
        "start": str(results["timestamp"].iloc[0]),
        "end": str(results["timestamp"].iloc[-1]),
        "threshold": threshold,
        "fee_bps": fee_bps,
        "mode": mode,
        "direction_hit_rate": float((np.sign(pred) == np.sign(actual)).mean()),
        "active_signal_rate": float(nonzero_mask.mean()),
        "active_hit_rate": float((np.sign(pred[nonzero_mask]) == np.sign(actual[nonzero_mask])).mean()) if nonzero_mask.any() else float("nan"),
        "mse": float(np.mean((pred - actual) ** 2)),
        "mae": float(np.mean(np.abs(pred - actual))),
        "correlation": corr,
        "avg_predicted_log_return": float(np.mean(pred)),
        "avg_actual_log_return": float(np.mean(actual)),
        "buy_hold_return": float(np.exp(np.sum(actual)) - 1.0),
        "strategy_return": float(np.exp(np.sum(strategy)) - 1.0),
        "strategy_avg_log_return": float(np.mean(strategy)),
        "strategy_win_rate": float((strategy > 0).mean()),
        "strategy_max_drawdown": max_drawdown(strategy),
        "strategy_annualized_sharpe_estimate": annualized_sharpe,
        "long_signals": int((signal > 0).sum()),
        "short_signals": int((signal < 0).sum()),
        "flat_signals": int((signal == 0).sum()),
    }


def save_outputs(results: pd.DataFrame, summary: dict[str, Any], output_dir: Path, split: str) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"qqq_lora_backtest_{split}_{timestamp}.csv"
    json_path = output_dir / f"qqq_lora_backtest_{split}_{timestamp}.json"
    results.to_csv(csv_path, index=False)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    return csv_path, json_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Continuous QQQ backtest for a Kronos LoRA adapter.")
    parser.add_argument("--config", default="custom/configs/qqq_lora.yaml", help="YAML config path")
    parser.add_argument("--adapter", default=None, help="Adapter checkpoint path")
    parser.add_argument("--data", default=None, help="QQQ OHLCVA CSV path")
    parser.add_argument("--split", choices=["val", "test", "custom"], default="test", help="Evaluation split")
    parser.add_argument("--start", default=None, help="Target start timestamp for custom filtering")
    parser.add_argument("--end", default=None, help="Target end timestamp for custom filtering")
    parser.add_argument("--max-steps", type=int, default=None, help="Limit selected target bars")
    parser.add_argument("--batch-size", type=int, default=32, help="Inference batch size")
    parser.add_argument("--threshold", type=float, default=0.0, help="Absolute log-return threshold for non-flat signals")
    parser.add_argument("--fee-bps", type=float, default=0.0, help="Per-unit position change transaction cost in bps")
    parser.add_argument("--mode", choices=["long-short", "long-only"], default="long-short", help="Signal mode")
    parser.add_argument("--device", default="auto", help="Device: auto, cpu, cuda, cuda:0, mps")
    parser.add_argument("--log-interval", type=int, default=10, help="Progress log interval in batches")
    parser.add_argument("--output-dir", default=None, help="Directory for CSV/JSON backtest outputs")
    args = parser.parse_args()

    config = load_config(args.config)
    data_config = config["data"]
    loss_config = config.get("loss", {})
    device = select_device(args.device)

    tokenizer_path = resolve_path(config["model"]["tokenizer_path"])
    base_model_path = resolve_path(config["model"]["base_model_path"])
    adapter_path = resolve_path(args.adapter or (resolve_path(config["model"]["output_dir"]) / config["model"].get("adapter_name", "adapter.pt")))
    data_path = resolve_path(args.data or data_config["data_path"])
    output_dir = resolve_path(args.output_dir or (resolve_path(config["model"]["output_dir"]) / "backtests"))

    df = load_ohlcva_csv(data_path, timestamp_col=str(data_config.get("timestamp_col", "timestamps")))
    target_indices, period_metadata = target_indices_for_backtest(df, data_config, args.split, args.start, args.end)
    if args.max_steps is not None:
        target_indices = target_indices[: args.max_steps]
        period_metadata["selected_count"] = len(target_indices)
        period_metadata["selected_end"] = str(df["timestamps"].iloc[target_indices[-1]])

    print(json.dumps({"period": period_metadata}, ensure_ascii=False, indent=2), flush=True)
    print(f"Device: {device}", flush=True)
    print(f"Loading tokenizer: {tokenizer_path}", flush=True)
    tokenizer = KronosTokenizer.from_pretrained(str(tokenizer_path)).to(device)
    print(f"Loading adapter: {adapter_path}", flush=True)
    forecaster, adapter_payload = load_adapter(base_model_path, adapter_path, map_location="cpu")
    forecaster.to(device)

    lookback = int(data_config.get("lookback_window", 512))
    clip = float(data_config.get("clip", 5.0))
    return_scale = float(loss_config.get("return_scale", 100.0))
    predicted_returns = predict_returns(
        tokenizer=tokenizer,
        forecaster=forecaster,
        df=df,
        target_indices=target_indices,
        lookback=lookback,
        clip=clip,
        batch_size=args.batch_size,
        return_scale=return_scale,
        device=device,
        log_interval=args.log_interval,
    )

    closes = df["close"].to_numpy(dtype=np.float64)
    actual_returns = np.log(closes[target_indices] / closes[np.array(target_indices) - 1])
    signal = np.where(predicted_returns > args.threshold, 1.0, np.where(predicted_returns < -args.threshold, -1.0, 0.0))
    if args.mode == "long-only":
        signal = np.where(signal > 0, 1.0, 0.0)
    position_change = np.abs(signal - np.concatenate([[0.0], signal[:-1]]))
    fee = position_change * (args.fee_bps / 10000.0)
    strategy_returns = signal * actual_returns - fee

    results = pd.DataFrame(
        {
            "timestamp": df["timestamps"].iloc[target_indices].astype(str).to_numpy(),
            "actual_close": closes[target_indices],
            "previous_close": closes[np.array(target_indices) - 1],
            "predicted_log_return": predicted_returns,
            "actual_log_return": actual_returns,
            "signal": signal.astype(int),
            "position_change": position_change,
            "fee_log_return": fee,
            "strategy_log_return": strategy_returns,
        }
    )

    summary = {
        "adapter": str(adapter_path),
        "adapter_metadata": adapter_payload.get("metadata", {}),
        "data": str(data_path),
        "period": period_metadata,
        "metrics": summarize_results(results, args.threshold, args.fee_bps, args.mode),
    }
    csv_path, json_path = save_outputs(results, summary, output_dir, args.split)
    summary["outputs"] = {"csv": str(csv_path), "json": str(json_path)}
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

