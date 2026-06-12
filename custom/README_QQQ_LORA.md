# QQQ LoRA 微调

本目录包含一套基于 `docs/new_train.md` 方法论实现的独立 QQQ 定制训练链路。
它不会复用现有的 `trained_models` 输出，也不会复用现有的
`finetune_csv/configs` 训练配置。

## 训练内容

- 底座预测模型：`./models/Kronos-base`
- Tokenizer：`./models/Kronos-Tokenizer-base`
- 可训练权重：注入到 `q_proj` 和 `v_proj` 的 LoRA adapter，以及一个小型下一步收益率回归头
- 冻结权重：Kronos 底座模型和 tokenizer
- 默认数据：`./finetune_csv/data/QQQ_5m_databento.csv`
- 输出文件：`./custom/outputs/qqq_lora/adapter.pt`

## Dry Run 验证

```bash
bash custom/train_qqq_lora.sh --dry-run
```

## 完整训练

```bash
bash custom/train_qqq_lora.sh
```

默认配置会在 QQQ 5 分钟 CSV 数据上训练 3 个 epoch。如果只想做较短实验，
可以调整 `custom/configs/qqq_lora.yaml` 中的 `training.batch_size`、
`training.epochs` 和 `training.max_train_steps`。

## 预测

训练生成 `adapter.pt` 后，执行：

```bash
.venv/bin/python custom/predict_qqq_lora.py \
  --config custom/configs/qqq_lora.yaml \
  --calibrate
```

`--calibrate` 会在最近的回看窗口上评估配置中的 `temperature_grid` 和
`top_p_grid`，然后使用命中率最高的一组参数预测下一根 K 线。

## 连续回测

默认回测 `test` 段，即训练集和验证集之后的时间段，避免落入训练数据时间段：

```bash
bash custom/backtest_qqq_lora.sh
```

快速检查可以限制回测根数：

```bash
bash custom/backtest_qqq_lora.sh --max-steps 20
```

回测结果会输出到 `./custom/outputs/qqq_lora/backtests/`，包含逐根 K 线 CSV 和汇总 JSON。
