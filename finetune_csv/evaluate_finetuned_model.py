import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from config_loader import CustomFinetuneConfig
from model import Kronos, KronosPredictor, KronosTokenizer


FEATURE_COLS = ["open", "high", "low", "close", "volume", "amount"]
PRICE_COLS = ["open", "high", "low", "close"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_test_split(config: CustomFinetuneConfig) -> pd.DataFrame:
    df = pd.read_csv(config.data_path, parse_dates=["timestamps"])
    df = df.sort_values("timestamps").reset_index(drop=True)
    val_end = int(len(df) * (config.train_ratio + config.val_ratio))
    return df.iloc[val_end:].reset_index(drop=True)


def build_windows(test_df: pd.DataFrame, config: CustomFinetuneConfig, window_count: int):
    context_len = config.lookback_window
    pred_len = config.predict_window
    max_start = len(test_df) - context_len - pred_len
    if max_start < 0:
        raise ValueError(
            f"Test split is too short: rows={len(test_df)}, "
            f"lookback_window={context_len}, predict_window={pred_len}"
        )

    actual_window_count = min(window_count, max_start + 1)
    starts = np.linspace(0, max_start, actual_window_count, dtype=int)

    contexts = []
    futures = []
    x_timestamps = []
    y_timestamps = []
    for start in starts:
        context = test_df.iloc[start : start + context_len].copy()
        future = test_df.iloc[start + context_len : start + context_len + pred_len].copy()
        contexts.append(context[FEATURE_COLS].reset_index(drop=True))
        futures.append(future[FEATURE_COLS].reset_index(drop=True))
        x_timestamps.append(context["timestamps"].reset_index(drop=True))
        y_timestamps.append(future["timestamps"].reset_index(drop=True))

    return starts, contexts, futures, x_timestamps, y_timestamps


def summarize_metrics(name: str, pred: np.ndarray, actual: np.ndarray, contexts) -> dict:
    metrics = {}
    diff = pred - actual

    print(f"\n{name}")
    for idx, col in enumerate(FEATURE_COLS):
        col_diff = diff[:, :, idx]
        col_actual = actual[:, :, idx]
        mae = float(np.mean(np.abs(col_diff)))
        rmse = float(np.sqrt(np.mean(col_diff ** 2)))
        mape = float(np.mean(np.abs(col_diff) / (np.abs(col_actual) + 1e-9)) * 100)
        metrics[col] = {"mae": mae, "rmse": rmse, "mape_pct": mape}
        if col in PRICE_COLS:
            print(f"  {col}: MAE={mae:.6f}, RMSE={rmse:.6f}, MAPE={mape:.6f}%")

    volume_metrics = metrics["volume"]
    print(
        "  volume: "
        f"MAE={volume_metrics['mae']:.2f}, "
        f"RMSE={volume_metrics['rmse']:.2f}, "
        f"MAPE={volume_metrics['mape_pct']:.2f}%"
    )

    close_idx = FEATURE_COLS.index("close")
    last_close = np.array([float(context["close"].iloc[-1]) for context in contexts])
    actual_delta = actual[:, :, close_idx] - last_close[:, None]
    pred_delta = pred[:, :, close_idx] - last_close[:, None]
    final_actual_delta = actual[:, -1, close_idx] - last_close
    final_pred_delta = pred[:, -1, close_idx] - last_close

    close_direction_acc = float(np.mean(np.sign(pred_delta) == np.sign(actual_delta)) * 100)
    final_close_direction_acc = float(
        np.mean(np.sign(final_pred_delta) == np.sign(final_actual_delta)) * 100
    )
    metrics["close_direction_acc"] = close_direction_acc
    metrics["final_close_direction_acc"] = final_close_direction_acc

    print(f"  close_direction_acc={close_direction_acc:.2f}%")
    print(f"  final_close_direction_acc={final_close_direction_acc:.2f}%")
    return metrics


def build_naive_predictions(contexts, pred_len: int) -> np.ndarray:
    return np.stack(
        [
            np.repeat(
                context[FEATURE_COLS].iloc[[-1]].to_numpy(dtype=np.float64),
                pred_len,
                axis=0,
            )
            for context in contexts
        ]
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned Kronos CSV model.")
    parser.add_argument(
        "--config",
        default="finetune_csv/configs/config_qqq_1m_cpu.yaml",
        help="Path to finetune_csv YAML config.",
    )
    parser.add_argument(
        "--windows",
        type=int,
        default=100,
        help="Number of evenly spaced test windows to evaluate.",
    )
    parser.add_argument("--device", default="cpu", help="Torch device, e.g. cpu or cuda:0.")
    parser.add_argument("--top-k", type=int, default=1, help="Top-k sampling value.")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p sampling value.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature.")
    parser.add_argument("--sample-count", type=int, default=1, help="Samples per prediction.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = CustomFinetuneConfig(args.config)
    set_seed(config.seed)

    test_df = load_test_split(config)
    starts, contexts, futures, x_timestamps, y_timestamps = build_windows(
        test_df, config, args.windows
    )

    print(f"config={args.config}")
    print(f"tokenizer_path={config.tokenizer_best_model_path}")
    print(f"basemodel_path={config.basemodel_best_model_path}")
    print(f"test_rows={len(test_df)}, windows={len(starts)}")
    print(f"test_range={test_df['timestamps'].iloc[0]} -> {test_df['timestamps'].iloc[-1]}")
    print(f"lookback_window={config.lookback_window}, predict_window={config.predict_window}")

    tokenizer = KronosTokenizer.from_pretrained(config.tokenizer_best_model_path)
    model = Kronos.from_pretrained(config.basemodel_best_model_path)
    tokenizer.eval()
    model.eval()

    predictor = KronosPredictor(
        model,
        tokenizer,
        device=args.device,
        max_context=config.max_context,
        clip=config.clip,
    )

    with torch.no_grad():
        pred_dfs = predictor.predict_batch(
            df_list=contexts,
            x_timestamp_list=x_timestamps,
            y_timestamp_list=y_timestamps,
            pred_len=config.predict_window,
            T=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            sample_count=args.sample_count,
            verbose=False,
        )

    pred = np.stack([pred_df[FEATURE_COLS].to_numpy(dtype=np.float64) for pred_df in pred_dfs])
    actual = np.stack([future[FEATURE_COLS].to_numpy(dtype=np.float64) for future in futures])
    naive = build_naive_predictions(contexts, config.predict_window)

    model_metrics = summarize_metrics("model", pred, actual, contexts)
    naive_metrics = summarize_metrics("naive_last_value", naive, actual, contexts)

    print("\nimprovement_vs_naive")
    for col in PRICE_COLS:
        model_mae = model_metrics[col]["mae"]
        naive_mae = naive_metrics[col]["mae"]
        improvement = (naive_mae - model_mae) / naive_mae * 100 if naive_mae else float("nan")
        print(f"  {col}_mae_improvement={improvement:.2f}%")

    print("\nsample_windows")
    for i in range(min(5, len(starts))):
        print(
            "  "
            f"start={int(starts[i])}, "
            f"x_end={x_timestamps[i].iloc[-1]}, "
            f"y_end={y_timestamps[i].iloc[-1]}, "
            f"last_close={contexts[i]['close'].iloc[-1]:.4f}, "
            f"actual_final={actual[i, -1, FEATURE_COLS.index('close')]:.4f}, "
            f"pred_final={pred[i, -1, FEATURE_COLS.index('close')]:.4f}"
        )


if __name__ == "__main__":
    main()
