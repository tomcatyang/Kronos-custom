from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


FEATURE_COLUMNS = ("open", "high", "low", "close", "volume", "amount")
TIME_COLUMNS = ("minute", "hour", "weekday", "day", "month")


def load_ohlcva_csv(data_path: str | Path, timestamp_col: str = "timestamps") -> pd.DataFrame:
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"QQQ data file not found: {data_path}")

    df = pd.read_csv(data_path)
    if timestamp_col not in df.columns:
        raise ValueError(f"CSV must contain timestamp column: {timestamp_col}")

    missing_columns = [column for column in FEATURE_COLUMNS if column not in df.columns]
    if missing_columns == ["amount"]:
        df["amount"] = df["close"] * df["volume"]
    elif missing_columns:
        raise ValueError(f"CSV is missing OHLCVA columns: {missing_columns}")

    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df = df.sort_values(timestamp_col).reset_index(drop=True)
    df = df.rename(columns={timestamp_col: "timestamps"})
    df = df[["timestamps", *FEATURE_COLUMNS]].copy()
    df[list(FEATURE_COLUMNS)] = df[list(FEATURE_COLUMNS)].astype("float32")
    df[list(FEATURE_COLUMNS)] = df[list(FEATURE_COLUMNS)].replace([np.inf, -np.inf], np.nan)
    df[list(FEATURE_COLUMNS)] = df[list(FEATURE_COLUMNS)].ffill().bfill()

    if df[list(FEATURE_COLUMNS)].isnull().any().any():
        raise ValueError("CSV still contains NaN values after forward/backward fill")

    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["minute"] = result["timestamps"].dt.minute
    result["hour"] = result["timestamps"].dt.hour
    result["weekday"] = result["timestamps"].dt.weekday
    result["day"] = result["timestamps"].dt.day
    result["month"] = result["timestamps"].dt.month
    return result


class QQQKlineDataset(Dataset):
    """Time-ordered QQQ OHLCVA windows for Kronos token prediction."""

    def __init__(
        self,
        data_path: str | Path,
        split: str,
        lookback_window: int,
        predict_window: int,
        train_ratio: float,
        val_ratio: float,
        clip: float,
        stride: int = 1,
        timestamp_col: str = "timestamps",
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be one of: train, val, test")
        if lookback_window < 2:
            raise ValueError("lookback_window must be at least 2")
        if predict_window < 1:
            raise ValueError("predict_window must be at least 1")
        if stride < 1:
            raise ValueError("stride must be at least 1")

        full_df = add_time_features(load_ohlcva_csv(data_path, timestamp_col=timestamp_col))
        total_length = len(full_df)
        train_end = int(total_length * train_ratio)
        val_end = int(total_length * (train_ratio + val_ratio))

        if split == "train":
            df = full_df.iloc[:train_end].copy()
        elif split == "val":
            df = full_df.iloc[train_end:val_end].copy()
        else:
            df = full_df.iloc[val_end:].copy()

        self.data_path = Path(data_path)
        self.split = split
        self.lookback_window = lookback_window
        self.predict_window = predict_window
        self.window = lookback_window + predict_window
        self.clip = clip
        self.stride = stride
        self.df = df.reset_index(drop=True)
        self.n_samples = (len(self.df) - self.window) // self.stride + 1

        if self.n_samples <= 0:
            raise ValueError(
                f"{split} split is too short for window={self.window}: "
                f"split_length={len(self.df)}"
            )

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        start = index * self.stride
        end = start + self.window
        window = self.df.iloc[start:end]

        features = window[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float32)
        past_features = features[: self.lookback_window]
        mean = past_features.mean(axis=0)
        std = past_features.std(axis=0)
        x = (features - mean) / (std + 1e-5)
        x = np.clip(x, -self.clip, self.clip).astype(np.float32)

        close = np.maximum(window["close"].to_numpy(dtype=np.float32), 1e-8)
        return_targets = np.diff(np.log(close)).astype(np.float32)
        stamps = window[list(TIME_COLUMNS)].to_numpy(dtype=np.float32)

        return {
            "x": torch.from_numpy(x),
            "stamp": torch.from_numpy(stamps),
            "return_targets": torch.from_numpy(return_targets),
        }


def build_datasets(config: dict, project_root: Path) -> tuple[QQQKlineDataset, QQQKlineDataset]:
    data_path = Path(config["data_path"])
    if not data_path.is_absolute():
        data_path = project_root / data_path

    common_kwargs = {
        "data_path": data_path,
        "lookback_window": int(config.get("lookback_window", 512)),
        "predict_window": int(config.get("predict_window", 1)),
        "train_ratio": float(config.get("train_ratio", 0.8)),
        "val_ratio": float(config.get("val_ratio", 0.1)),
        "clip": float(config.get("clip", 5.0)),
        "stride": int(config.get("stride", 1)),
        "timestamp_col": str(config.get("timestamp_col", "timestamps")),
    }
    return (
        QQQKlineDataset(split="train", **common_kwargs),
        QQQKlineDataset(split="val", **common_kwargs),
    )

