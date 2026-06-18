# 自动调优 + Generator 模块

## 功能概述

1. **数据采集**：在历史数据上做滑动窗口，记录每个窗口的特征和最优配置
2. **规则归纳**：使用LLM从采集数据中归纳出条件化规则（同时参考总体特征和局部特征）
3. **规则验证**：在验证集上对比原始值、固定最优值、动态规则的效果
4. **可视化**：生成对比图表和报告

## 快速开始

```bash
# 使用默认配置运行
python -m experiments.autotune.main

# 指定数据集运行
python -m experiments.autotune.main --dataset melbourne_temp --horizon 7

# 只生成可视化（需要已有结果）
python -m experiments.autotune.main --visualize