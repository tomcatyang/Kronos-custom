import argparse
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from config_loader import CustomFinetuneConfig
from model import Kronos, KronosPredictor, KronosTokenizer


FEATURE_COLS = ["open", "high", "low", "close", "volume", "amount"]
PRICE_COLS = ["open", "high", "low", "close"]
CLOSE_IDX = FEATURE_COLS.index("close")
DEFAULT_CANDIDATES = [
    {
        "name": "greedy_top_k_1",
        "T": 1.0,
        "top_k": 1,
        "top_p": 1.0,
        "description": "确定性贪心解码。每次只选择概率最高的 token，结果最稳定。",
    },
    {
        "name": "sample_top_k_3_T07",
        "T": 0.7,
        "top_k": 3,
        "top_p": 1.0,
        "description": "保守采样。只在概率最高的 3 个 token 中采样，并用较低温度降低随机性。",
    },
    {
        "name": "sample_top_k_5_T10",
        "T": 1.0,
        "top_k": 5,
        "top_p": 1.0,
        "description": "较宽松的 top-k 采样。在波动较大的窗口中允许更多候选 token。",
    },
    {
        "name": "sample_top_p_090_T07",
        "T": 0.7,
        "top_k": 0,
        "top_p": 0.90,
        "description": "核采样。候选集合按累计概率动态变化，模型越不确定候选越多。",
    },
]


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


def build_one_step_window(test_df: pd.DataFrame, start: int, context_len: int):
    context = test_df.iloc[start : start + context_len].copy()
    future = test_df.iloc[start + context_len : start + context_len + 1].copy()
    return (
        context[FEATURE_COLS].reset_index(drop=True),
        context["timestamps"].reset_index(drop=True),
        future[FEATURE_COLS].reset_index(drop=True),
        future["timestamps"].reset_index(drop=True),
    )


def predict_batch(predictor, windows, params: dict, seed: int) -> np.ndarray:
    set_seed(seed)
    contexts = [window[0] for window in windows]
    x_timestamps = [window[1] for window in windows]
    y_timestamps = [window[3] for window in windows]
    with torch.no_grad():
        pred_dfs = predictor.predict_batch(
            df_list=contexts,
            x_timestamp_list=x_timestamps,
            y_timestamp_list=y_timestamps,
            pred_len=1,
            T=params["T"],
            top_k=params["top_k"],
            top_p=params["top_p"],
            sample_count=1,
            verbose=False,
        )
    return np.stack([pred_df[FEATURE_COLS].to_numpy(dtype=np.float64) for pred_df in pred_dfs])


def close_mae(pred: np.ndarray, windows) -> float:
    actual = np.stack([window[2][FEATURE_COLS].to_numpy(dtype=np.float64) for window in windows])
    return float(np.mean(np.abs(pred[:, :, CLOSE_IDX] - actual[:, :, CLOSE_IDX])))


def calibration_score(pred: np.ndarray, windows, metric: str) -> float:
    actual = np.stack([window[2][FEATURE_COLS].to_numpy(dtype=np.float64) for window in windows])
    contexts = [window[0] for window in windows]
    last_close = np.array([float(context["close"].iloc[-1]) for context in contexts])
    pred_close = pred[:, 0, CLOSE_IDX]
    actual_close = actual[:, 0, CLOSE_IDX]
    pred_delta = pred_close - last_close
    actual_delta = actual_close - last_close

    if metric == "close_mae":
        return -float(np.mean(np.abs(pred_delta - actual_delta)))

    if metric == "direction_acc":
        return float(np.mean(np.sign(pred_delta) == np.sign(actual_delta)))

    if metric == "signed_return":
        actual_return = actual_delta / last_close
        return float(np.mean(np.sign(pred_delta) * actual_return))

    raise ValueError(f"Unsupported calibration metric: {metric}")


def print_candidates() -> None:
    print("candidate_params:")
    for candidate in DEFAULT_CANDIDATES:
        print(
            "  "
            f"{candidate['name']}: "
            f"T={candidate['T']}, top_k={candidate['top_k']}, top_p={candidate['top_p']} - "
            f"{candidate['description']}"
        )


def build_naive_predictions(contexts) -> np.ndarray:
    return np.stack(
        [
            np.repeat(
                context[FEATURE_COLS].iloc[[-1]].to_numpy(dtype=np.float64),
                1,
                axis=0,
            )
            for context in contexts
        ]
    )


def summarize_metrics(name: str, pred: np.ndarray, actual: np.ndarray, contexts) -> dict:
    diff = pred - actual
    metrics = {}
    print(f"\n{name}")

    for idx, col in enumerate(PRICE_COLS):
        col_diff = diff[:, :, idx]
        col_actual = actual[:, :, idx]
        mae = float(np.mean(np.abs(col_diff)))
        rmse = float(np.sqrt(np.mean(col_diff ** 2)))
        mape = float(np.mean(np.abs(col_diff) / (np.abs(col_actual) + 1e-9)) * 100)
        metrics[col] = {"mae": mae, "rmse": rmse, "mape_pct": mape}
        print(f"  {col}: MAE={mae:.6f}, RMSE={rmse:.6f}, MAPE={mape:.6f}%")

    last_close = np.array([float(context["close"].iloc[-1]) for context in contexts])
    actual_close = actual[:, 0, CLOSE_IDX]
    pred_close = pred[:, 0, CLOSE_IDX]
    actual_delta = actual_close - last_close
    pred_delta = pred_close - last_close
    signal = np.sign(pred_delta)
    nonzero_mask = signal != 0
    actual_return = actual_delta / last_close

    close_direction_acc = float(np.mean(signal == np.sign(actual_delta)) * 100)
    trade_direction_acc = (
        float(np.mean(signal[nonzero_mask] == np.sign(actual_delta[nonzero_mask])) * 100)
        if np.any(nonzero_mask)
        else float("nan")
    )
    avg_signed_return_bps = float(np.mean(signal * actual_return) * 10000)
    avg_long_only_return_bps = float(np.mean((pred_delta > 0).astype(float) * actual_return) * 10000)

    metrics["close_direction_acc"] = close_direction_acc
    metrics["trade_direction_acc"] = trade_direction_acc
    metrics["avg_signed_return_bps"] = avg_signed_return_bps
    metrics["avg_long_only_return_bps"] = avg_long_only_return_bps

    print(f"  close_direction_acc={close_direction_acc:.2f}%")
    print(f"  trade_direction_acc={trade_direction_acc:.2f}%")
    print(f"  avg_signed_return={avg_signed_return_bps:.4f} bps/step")
    print(f"  avg_long_only_return={avg_long_only_return_bps:.4f} bps/step")
    return metrics


def evaluate(
    config_path: str,
    windows: int,
    calibration_windows: int,
    calibration_metric: str,
    device: str,
) -> None:
    config = CustomFinetuneConfig(config_path)
    test_df = load_test_split(config)
    context_len = config.lookback_window
    min_start = calibration_windows
    max_start = len(test_df) - context_len - 1
    if max_start <= min_start:
        raise ValueError(
            f"Test split is too short: rows={len(test_df)}, "
            f"lookback_window={context_len}, calibration_windows={calibration_windows}"
        )

    actual_windows = min(windows, max_start - min_start + 1)
    starts = np.linspace(min_start, max_start, actual_windows, dtype=int)

    print(f"config={config_path}")
    print(f"tokenizer_path={config.tokenizer_best_model_path}")
    print(f"basemodel_path={config.basemodel_best_model_path}")
    print(f"test_rows={len(test_df)}, windows={len(starts)}, calibration_windows={calibration_windows}")
    print(f"calibration_metric={calibration_metric}")
    print(f"test_range={test_df['timestamps'].iloc[0]} -> {test_df['timestamps'].iloc[-1]}")
    print(f"lookback_window={context_len}, predict_window=1")
    print_candidates()

    tokenizer = KronosTokenizer.from_pretrained(config.tokenizer_best_model_path)
    model = Kronos.from_pretrained(config.basemodel_best_model_path)
    tokenizer.eval()
    model.eval()
    predictor = KronosPredictor(
        model,
        tokenizer,
        device=device,
        max_context=config.max_context,
        clip=config.clip,
    )

    selected_params = []
    rolling_preds = []
    fixed_preds = []
    actuals = []
    contexts = []
    fixed_params = DEFAULT_CANDIDATES[0]

    for index, start in enumerate(starts, 1):
        calibration_starts = [int(start - offset) for offset in range(calibration_windows, 0, -1)]
        calibration_set = [
            build_one_step_window(test_df, calibration_start, context_len)
            for calibration_start in calibration_starts
        ]

        scored_params = []
        for candidate_index, params in enumerate(DEFAULT_CANDIDATES):
            pred = predict_batch(
                predictor,
                calibration_set,
                params,
                seed=config.seed + index * 100 + candidate_index,
            )
            scored_params.append((calibration_score(pred, calibration_set, calibration_metric), params))

        best_score, best_params = max(scored_params, key=lambda item: item[0])
        selected_params.append(best_params["name"])

        current_window = build_one_step_window(test_df, int(start), context_len)
        rolling_pred = predict_batch(
            predictor,
            [current_window],
            best_params,
            seed=config.seed + index * 1000,
        )[0]
        fixed_pred = predict_batch(
            predictor,
            [current_window],
            fixed_params,
            seed=config.seed + index * 1000,
        )[0]

        rolling_preds.append(rolling_pred)
        fixed_preds.append(fixed_pred)
        actuals.append(current_window[2][FEATURE_COLS].to_numpy(dtype=np.float64))
        contexts.append(current_window[0])

        if index == 1 or index == len(starts) or index % 10 == 0:
            print(
                f"progress {index}/{len(starts)} "
                f"best={best_params['name']} cal_{calibration_metric}_score={best_score:.6f}"
            )

    rolling_pred = np.stack(rolling_preds)
    fixed_pred = np.stack(fixed_preds)
    actual = np.stack(actuals)
    naive = build_naive_predictions(contexts)

    rolling_metrics = summarize_metrics("rolling_one_step_calibrated_model", rolling_pred, actual, contexts)
    fixed_metrics = summarize_metrics("fixed_one_step_greedy_top_k_1_model", fixed_pred, actual, contexts)
    naive_metrics = summarize_metrics("naive_last_value", naive, actual, contexts)

    print("\nselected_params_count")
    for name, count in Counter(selected_params).most_common():
        print(f"  {name}: {count}")

    print("\nclose_mae_improvement_vs_naive")
    for name, metrics in [("rolling", rolling_metrics), ("fixed", fixed_metrics)]:
        model_mae = metrics["close"]["mae"]
        naive_mae = naive_metrics["close"]["mae"]
        improvement = (naive_mae - model_mae) / naive_mae * 100 if naive_mae else float("nan")
        print(f"  {name}: {improvement:.2f}%")

    print("\nsample_windows")
    for idx in range(min(5, len(starts))):
        y_index = int(starts[idx]) + context_len
        print(
            "  "
            f"start={int(starts[idx])}, "
            f"y={test_df['timestamps'].iloc[y_index]}, "
            f"last_close={contexts[idx]['close'].iloc[-1]:.4f}, "
            f"actual={actual[idx, 0, CLOSE_IDX]:.4f}, "
            f"rolling={rolling_pred[idx, 0, CLOSE_IDX]:.4f}, "
            f"fixed={fixed_pred[idx, 0, CLOSE_IDX]:.4f}"
        )


def parse_args():
    parser = argparse.ArgumentParser(description="对微调后的 Kronos CSV 模型做滚动 one-step 评估。")
    parser.add_argument(
        "--config",
        default="finetune_csv/configs/config_qqq_1m_cpu.yaml",
        help="finetune_csv YAML 配置文件路径。",
    )
    parser.add_argument("--windows", type=int, default=50, help="滚动测试点数量。")
    parser.add_argument(
        "--calibration-windows",
        type=int,
        default=10,
        help="用于选择推理参数的历史 one-step 校准窗口数量。",
    )
    parser.add_argument(
        "--calibration-metric",
        choices=["close_mae", "direction_acc", "signed_return"],
        default="signed_return",
        help=(
            "用于选择推理参数的校准指标。close_mae 最小化下一根 close 误差；"
            "direction_acc 最大化下一根方向准确率；signed_return 最大化 "
            "mean(sign(pred_close-last_close) * actual_return)。"
        ),
    )
    parser.add_argument("--device", default="cpu", help="Torch 设备，例如 cpu 或 cuda:0。")
    return parser.parse_args()


def main():
    args = parse_args()
    evaluate(
        args.config,
        args.windows,
        args.calibration_windows,
        args.calibration_metric,
        args.device,
    )


if __name__ == "__main__":
    main()
