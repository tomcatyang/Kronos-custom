# Kronos 自定义 CSV 微调 QA

本文整理本项目在无 GPU 环境下运行 `finetune_csv` 微调流程时的常见问题。

## Q1: Kronos-Tokenizer-base 和 Kronos-base 分别是什么？

`Kronos-Tokenizer-base` 是 K 线 tokenizer / 量化器，负责把连续 K 线特征编码成离散 token，也负责把 token 解码回连续 K 线。

输入特征通常是：

```text
open, high, low, close, volume, amount
```

它的作用是：

```text
连续 K 线数据 -> 离散 token
离散 token -> 重建连续 K 线数据
```

`Kronos-base` 是预测模型 / predictor / basemodel，负责基于历史 token 做自回归预测：

```text
历史 token -> 未来 token
```

完整预测流程是：

```text
历史 OHLCV 数据
  -> Kronos-Tokenizer-base 编码成 token
  -> Kronos-base 预测未来 token
  -> Kronos-Tokenizer-base 解码成未来 OHLCV
```

## Q2: 后训练和微调的基本顺序是什么？

推荐顺序是：

```text
1. 微调 Kronos-Tokenizer-base
2. 微调 Kronos-base
```

如果数据量较小，或者只是先验证流程，可以先只微调 `Kronos-base`。如果市场分布和预训练数据差异明显，例如不同市场、不同周期、不同资产类别，再考虑微调 tokenizer。

## Q3: 没有 GPU 能不能训练？

可以启动训练，但速度会慢，尤其是 `Kronos-base` 有约 1 亿参数，CPU 微调耗时会比较长。

检查 CUDA 状态：

```bash
cd /home/dev/high-freq-trade-sys/Kronos-custom
source .venv/bin/activate

python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
PY
```

如果输出类似：

```text
cuda_available False
cuda_count 0
```

说明当前环境没有可用 GPU。CPU 配置里应设置：

```yaml
device:
  use_cuda: false
```

## Q4: 如何创建并使用独立 Python 环境？

使用项目自带脚本：

```bash
cd /home/dev/high-freq-trade-sys/Kronos-custom
bash init.sh
source .venv/bin/activate
```

后续 Kronos-custom 相关命令都建议基于这个环境运行。

## Q5: 已创建的 CPU smoke 配置在哪里？

配置文件：

```text
finetune_csv/configs/config_cpu_smoke.yaml
```

它是无 GPU 环境下的轻量训练配置，主要用于验证流程可运行。

关键配置：

```yaml
data:
  lookback_window: 128
  predict_window: 16
  max_context: 128

training:
  tokenizer_epochs: 1
  basemodel_epochs: 1
  batch_size: 1
  num_workers: 0

device:
  use_cuda: false
```

## Q6: 窗口参数由哪些字段控制？

窗口参数在配置文件的 `data` 段：

```yaml
data:
  lookback_window: 128
  predict_window: 16
  max_context: 128
```

含义：

```text
lookback_window
  使用多少根历史 K 线作为上下文。

predict_window
  训练样本里包含多少根未来 K 线。

max_context
  模型最大上下文长度。Kronos-base 通常建议不超过 512。
```

训练样本总长度是：

```text
lookback_window + predict_window + 1
```

例如：

```yaml
lookback_window: 128
predict_window: 16
```

对应每个训练样本：

```text
128 + 16 + 1 = 145 根 K 线
```

## Q7: epoch 参数由哪些字段控制？

epoch 参数在配置文件的 `training` 段：

```yaml
training:
  tokenizer_epochs: 1
  basemodel_epochs: 1
```

含义：

```text
tokenizer_epochs
  tokenizer 微调轮数。

basemodel_epochs
  Kronos-base / predictor 微调轮数。
```

无 GPU 环境建议先从 1 开始，确认流程稳定后再增加。

## Q8: predict_window 是否和实际预测长度有关？

有关，但不要求完全相等。

`predict_window` 是训练时每个样本包含的未来长度。实际推理时预测多少根 K 线，由 `predictor.predict(..., pred_len=xxx)` 控制。

建议：

```text
predict_window 约等于常用实际预测长度
```

例如：

```text
主要预测未来 8 根 5min K 线   -> predict_window: 8 或 16
主要预测未来 24 根 5min K 线  -> predict_window: 24 或 32
主要预测未来 48 根 5min K 线  -> predict_window: 48
```

不建议训练时 `predict_window` 很短，但推理时长期生成很长序列，否则误差更容易累积。

## Q9: clip、train_ratio、val_ratio、test_ratio 是什么？

`clip` 控制归一化后的极端值截断：

```yaml
clip: 5.0
```

数据会先归一化：

```text
x = (x - mean) / std
```

再截断到：

```text
[-5.0, 5.0]
```

作用是降低异常波动、极端成交量或脏数据对训练的影响。

切分参数：

```yaml
train_ratio: 0.85
val_ratio: 0.15
test_ratio: 0.0
```

表示按时间顺序切分：

```text
前 85% -> 训练集
后 15% -> 验证集
最后 0% -> 测试集
```

时间序列应按时间切分，不建议随机切分，否则可能产生未来数据泄漏。

## Q10: 如何启动 CPU smoke 训练？

已提供脚本：

```text
scripts/train_cpu_smoke.sh
```

进入项目根目录运行：

```bash
cd /home/dev/high-freq-trade-sys/Kronos-custom
```

完整 smoke 训练，包含 tokenizer 和 basemodel：

```bash
bash scripts/train_cpu_smoke.sh
```

只训练 tokenizer：

```bash
bash scripts/train_cpu_smoke.sh --skip-basemodel
```

只训练 basemodel：

```bash
bash scripts/train_cpu_smoke.sh --skip-tokenizer
```

跳过已有阶段：

```bash
bash scripts/train_cpu_smoke.sh --skip-existing
```

## Q11: --skip-existing 是什么意思？

`--skip-existing` 表示如果某个阶段的 `best_model` 已经存在，就跳过该阶段。

它会检查：

```text
finetune_csv/finetuned/HK_ali_09988_cpu_smoke/tokenizer/best_model
finetune_csv/finetuned/HK_ali_09988_cpu_smoke/basemodel/best_model
```

如果 tokenizer 已训练过，运行：

```bash
bash scripts/train_cpu_smoke.sh --skip-existing
```

会跳过 tokenizer，继续训练缺失的 basemodel。这个参数适合训练中断后继续跑，避免重复覆盖已完成阶段。

## Q12: 为什么不能用 uv python train_sequential.py？

`uv python` 是管理 Python 版本的子命令，不用于运行脚本。

错误命令：

```bash
uv python train_sequential.py --config configs/config_cpu_smoke.yaml
```

正确方式：

```bash
python train_sequential.py --config configs/config_cpu_smoke.yaml
```

或者显式使用项目独立环境：

```bash
../.venv/bin/python train_sequential.py --config configs/config_cpu_smoke.yaml
```

如果使用 `uv` 运行脚本，应使用：

```bash
uv run python train_sequential.py --config configs/config_cpu_smoke.yaml
```

## Q13: CSV 数据集归一化窗口如何处理？

当前已修复为只使用历史 lookback 窗口计算均值和标准差：

```text
mean/std 只来自 lookback_window 历史段
```

这样可以避免把未来预测窗口纳入归一化统计，造成未来数据泄漏。

