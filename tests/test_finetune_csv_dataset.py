import sys
from pathlib import Path

import numpy as np
import pandas as pd

KRONOS_ROOT = Path(__file__).resolve().parents[1]
FINETUNE_CSV_ROOT = KRONOS_ROOT / "finetune_csv"
if str(FINETUNE_CSV_ROOT) not in sys.path:
    sys.path.insert(0, str(FINETUNE_CSV_ROOT))

from finetune_base_model import CustomKlineDataset


def test_custom_kline_dataset_normalizes_with_lookback_window_only(tmp_path):
    data_path = tmp_path / "kline.csv"
    lookback_window = 3
    predict_window = 1

    rows = []
    values = [1.0, 2.0, 3.0, 1000.0, 2000.0, 3000.0]
    for i, value in enumerate(values):
        rows.append(
            {
                "timestamps": f"2024-01-01 09:{30 + i:02d}:00",
                "open": value,
                "high": value,
                "low": value,
                "close": value,
                "volume": value,
                "amount": value,
            }
        )
    pd.DataFrame(rows).to_csv(data_path, index=False)

    dataset = CustomKlineDataset(
        data_path=str(data_path),
        data_type="train",
        lookback_window=lookback_window,
        predict_window=predict_window,
        clip=10000.0,
        seed=0,
        train_ratio=1.0,
        val_ratio=0.0,
        test_ratio=0.0,
    )
    dataset.set_epoch_seed(-1)

    x_tensor, _ = dataset[0]
    obtained = x_tensor[:, 0].numpy()

    past_values = np.array(values[:lookback_window], dtype=np.float32)
    expected_mean = np.mean(past_values)
    expected_std = np.std(past_values)
    expected_values = np.array(
        values[: lookback_window + predict_window + 1],
        dtype=np.float32,
    )
    expected = (expected_values - expected_mean) / (expected_std + 1e-5)

    np.testing.assert_allclose(obtained, expected, rtol=1e-6)
