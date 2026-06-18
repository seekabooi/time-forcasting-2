 

## 一、目录结构及文件作用

```
futureTime/
├── run_benchmark.py                      # ★ 生产预测入口：单次固定起源预测
│                                         # 支持 --use_rules（规则参考模式，跳过LLM决策）
│                                         # 支持 --no_residual（禁用残差修正技能）
│
├── compare_sliding.py                    # ★ 双模式滑动窗口对比测试
│                                         # 对比：规则模式 vs 无规则模式（7项指标）
│
├── compare_three_modes.py                # ★ 三模式对比测试（带缓存）
│                                         # 对比：无规则 vs 第一步规则 vs 第二步规则
│
├── src/                                  # ★ 核心预测框架
│   ├── agents/
│   │   ├── llm_planner.py                # ★ 核心预测Agent
│   │   │                                 # 三阶段：预处理 → 递归LLM预测 → 后处理
│   │   │                                 # 规则模式：规则作为Prompt参考，不跳过LLM
│   │   │                                 # 固定策略模式：_predict_with_fixed_strategy
│   │   ├── llm_client.py                 # LLM客户端（GLM-4调用、重试、JSON解析）
│   │   └── llm_prompts.py                # Prompt构建（含规则策略参考占位）
│   ├── skills/                           # 28个预测技能
│   │   ├── base.py                       # 技能基类（状态卡、元数据）
│   │   ├── registry.py                   # 技能注册表
│   │   ├── data_profiler.py              # 特征提取器（统计特征、熵、偏度等）
│   │   ├── skill_matcher.py              # 技能匹配器（硬过滤+DTW+路由加成）
│   │   ├── preprocess_skills.py          # 6个预处理技能（填充、截断、标准化等）
│   │   ├── postprocess_skills.py         # 后处理技能（逆变换、残差AR等）
│   │   └── [naive, prophet, ...]         # 具体技能实现
│   ├── evaluation/
│   │   └── fixed_origin_evaluator.py     # 固定起源评估器
│   └── tasks/
│       └── instance.py                   # 任务数据模型
│
├── experiments/
│   └── autotune/                         # ★ 自动调优+规则生成+闭环优化
│       ├── main.py                       # ★ 入口1：采集 → 归纳 → 聚类 → generated_rules.json
│       ├── config.yaml                   # ★ 全局配置（固定参数、stop_ratio等）
│       ├── collector.py                  # 滑动窗口采集器（记录轨迹）
│       ├── inducer.py                    # ★ 策略生成器（候选策略→回测→选best）
│       ├── cluster.py                    # K-Means聚类（策略向量聚类）
│       ├── iterative_refiner.py          # ★ 入口2：多轮闭环优化（Stop/Merge/Patch）
│       ├── rule_quality_evaluator.py     # ★ LLM评估规则质量，决定下一步动作
│       ├── meta_cluster.py               # ★ Merge实现（基于特征聚类+泛化生成）
│       ├── hard_refiner.py               # ★ Patch实现（针对MASE最差3个窗口）
│       ├── performance_auditor.py        # 性能审计（识别困难样本）
│       ├── cache_manager.py              # 策略缓存管理
│       ├── rule_engine.py                # ★ 规则执行引擎（匹配特征→返回策略）
│       ├── validator.py                  # 规则验证器
│       ├── visualizer.py                 # 可视化（图表和报告）
│       └── utils.py                      # 工具函数（特征提取、序列化）
│
└── storage/                               # ★ 运行时生成数据
    ├── autotune_results/
    │   ├── collected_windows.csv          # 采集结果汇总
    │   ├── generated_rules.json           # ★ 第一步：3条初始规则
    │   ├── refined_rules.json             # ★ 第二步：优化后规则
    │   ├── cache/                         # 预测结果缓存
    │   ├── strategy_cache/                # 策略生成缓存
    │   ├── logs/                          # 详细日志
    │   └── window_data/                   # 窗口数据（pickle）
    ├── comparison_results.csv             # 对比测试结果
    └── eval_*.csv                         # 单次预测明细
```


## 二、处理逻辑流程图

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                           第一步：采集 + 初始规则生成 (main.py)                         │
│                                                                                        │
│  历史数据 → 滑动窗口（21个，步长150） → 每个窗口执行三阶段预测 → 记录轨迹              │
│         ↓                                                                              │
│  归纳阶段 (inducer.py)：每窗口生成3~5个候选策略 → 回测 → 选best策略                    │
│         ↓                                                                              │
│  聚类阶段 (cluster.py)：对21个best策略做K-Means聚类 → 压缩为3条初始规则               │
│         ↓                                                                              │
│  输出：generated_rules.json（MASE=1.0867，比纯LLM改善23.64%）                         │
└────────────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                      第二步：多轮闭环优化 (iterative_refiner.py)                        │
│                                     （最多3轮）                                        │
│                                                                                        │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐  │
│  │  每轮循环:                                                                       │  │
│  │    ① 性能审计 (PerformanceAuditor)                                               │  │
│  │       → 用当前规则在验证窗口预测 → 计算MASE → 标记困难窗口（>平均×1.2）           │  │
│  │       → 输出：平均MASE、困难窗口数、MASE最差3个窗口                              │  │
│  │                                                                                  │  │
│  │    ② LLM决策 (RuleQualityEvaluator)                                              │  │
│  │       → 将性能报告+规则摘要发给LLM → 返回动作                                     │  │
│  │                                                                                  │  │
│  │    ③ 执行动作:                                                                    │  │
│  │       ├── Stop：困难窗口占比 < 1% → 结束优化                                      │  │
│  │       ├── Merge：存在语义相似的规则 → MetaCluster.auto_merge_by_features()        │  │
│  │       │         → 基于特征聚类 → LLM泛化生成 → 精简规则                           │  │
│  │       └── Patch：存在困难样本 → HardRefiner.refine()                              │  │
│  │                   → 针对MASE最差3个窗口生成补丁规则 → 改善>5%才采纳               │  │
│  │                                                                                  │  │
│  │    ④ 验证改进：比较新规则与旧规则的MASE → 无改善则回滚                            │  │
│  └──────────────────────────────────────────────────────────────────────────────────┘  │
│         ↓                                                                              │
│  输出：refined_rules.json（MASE=0.8666，相比初始规则再改善16.7%）                     │
└────────────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                      第三步：滑动窗口对比测试 (compare_three_modes.py)                  │
│                                                                                        │
│  在21个窗口上分别运行三种模式（带缓存）：                                                │
│    ① 无规则模式（纯LLM预测）                                                          │
│    ② 第一步规则（generated_rules.json）                                               │
│    ③ 第二步规则（refined_rules.json）                                                 │
│         ↓                                                                              │
│  计算7项指标：RMSE、MAE、MAPE、sMAPE、MASE、RMSSE、OWA                                │
│         ↓                                                                              │
│  输出对比表格 + storage/three_modes_comparison.csv                                    │
└────────────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                         生产预测 (run_benchmark.py)                                    │
│                                                                                        │
│  --use_rules refined_rules.json → 规则匹配 → 毫秒级预测（跳过LLM）                    │
│  不加 --use_rules → 原始三阶段LLM递归预测（10~30秒）                                   │
└────────────────────────────────────────────────────────────────────────────────────────┘
```


## 三、主要创新点

### 1. 三阶段预测架构（预处理 → 递归预测 → 后处理）
- **预处理**（LLM选择6种方法）：缺失填充、异常截断、标准化、去趋势、Box-Cox，支持多步骤组合。
- **后处理**：强制逆变换（系统自动绑定，安全锁）+ 智能增强（LLM选择残差修正/分位数校准）。
- **逆变换自动绑定**：预处理选了 `zscore_normalize`，后处理自动绑定 `invert_zscore`，绝不由LLM选择，防止选错。

### 2. 规则作为LLM参考，而非替代
- 规则策略不跳过LLM，而是作为Prompt中的“参考建议”输入，LLM仍自主决策权重。
- 对比实验证明，这种“先验知识+动态调整”的模式在所有指标上均优于纯LLM。
- **第一阶段规则**：MASE=1.0867，相比纯LLM（1.4230）改善 **23.64%**。

### 3. LLM驱动的多轮闭环优化（Stop / Merge / Patch）
- **Stop**：困难窗口占比 < 1% → 自动停止优化。
- **Merge**：LLM判断存在语义相似规则 → 基于特征聚类 → 泛化生成精简规则。
- **Patch**：针对MASE最差3个窗口生成补丁规则 → 改善>5%才采纳 → 否则回滚。
- **实际效果**：MASE从1.0401 → 0.9262（Merge）→ 0.8666（Patch），整体改善 **16.7%**。

### 4. 基于窗口数字特征的语义合并（参考MMSkills Phase 2）
- 合并依据不再是“条件文本重叠”或“权重阈值”，而是**窗口的数字特征向量**（`trend_strength`、`seasonal_strength`、`period`、`adf_pvalue`等）。
- LLM自主决定簇数和分组 → 对每组泛化生成新规则 → 丢弃边缘技能（权重<0.05）。

### 5. 多阶段加权策略（非单技能选择）
- 每个策略由多个阶段组成，每个阶段指定预测步数和技能权重字典。
- 示例：前3步用 `chunk_ensemble:0.7 + calendar:0.3`，后4步用 `multi_resolution:0.6 + naive:0.4`。
- 比单一技能选择更灵活、更贴近真实LLM递归决策行为。

### 6. 完整的缓存机制
- 预测结果缓存（`cache/`）→ 避免重复LLM调用，二次运行秒级加载。
- 策略生成缓存（`strategy_cache/`）→ 基于配置哈希，大幅加速重复调优。

### 7. 三种模式对比体系
- 支持7项评价指标（RMSE、MAE、MAPE、sMAPE、MASE、RMSSE、OWA）的滑动窗口对比。
- 独立对比“无规则”、“第一步规则”、“第二步规则”，量化每一步优化的实际收益。

### 8. 技能状态卡 + 硬过滤
- 每个技能自带状态卡（`when_to_use` / `when_not_to_use`）和决策提示。
- `SkillMatcher` 基于特征硬过滤 + DTW相似度 + 路由加成，减少LLM候选池，提升决策可靠性。


## 四、运行指令

### 第一步：采集 + 生成初始规则
```bash
python -m experiments.autotune.main --dataset melbourne_temp --horizon 7 --verbose
```
*输出：`storage/autotune_results/generated_rules.json`（3条初始规则）*

### 第二步：多轮闭环优化（Stop / Merge / Patch）
```bash
python -m experiments.autotune.iterative_refiner --dataset melbourne_temp --horizon 7 --rounds 3 --verbose
```
*输出：`storage/autotune_results/refined_rules.json`（优化后规则，MASE=0.8666）*

### 第三步：三种模式对比测试
```bash
python compare_three_modes.py --dataset melbourne_temp
```
*输出：终端对比表格 + `storage/three_modes_comparison.csv`*

### 第四步：生产预测（使用优化后规则）
```bash
# 使用规则（毫秒级，跳过LLM）
python run_benchmark.py --dataset melbourne_temp --min_train_size 600 --horizon 7 --use_rules storage/autotune_results/refined_rules.json

# 不使用规则（原始LLM预测）
python run_benchmark.py --dataset melbourne_temp --min_train_size 600 --horizon 7
```

### 额外：强制打印回测（用于截图展示）
在 `experiments/autotune/inducer.py` 中设置 `force_backtest = True`，然后运行：
```bash
python -m experiments.autotune.main --dataset melbourne_temp --horizon 7 --verbose
```
*会逐窗口打印候选策略、MASE和相似度，适合截图汇报。*

### 清空缓存（重新生成）
```bash
rmdir /s /q storage\autotune_results\cache
rmdir /s /q storage\autotune_results\strategy_cache
```


## 五、性能表现总结

| 模式 | MASE | 相比纯LLM改善 | 说明 |
| :--- | :--- | :--- | :--- |
| 纯LLM（无规则） | 1.4230 | — | 原始三阶段递归预测 |
| 第一步规则（初始） | 1.0867 | **23.64%** | K-Means聚类生成的初始规则 |
| 第二步规则（优化后） | **0.8666** | **39.12%** | 经过Merge+Patch闭环优化 |
| Holt-Winters（单技能） | ~1.45 | — | 经典统计模型（对比基准） |

**最终结论**：第二步规则相比纯LLM改善了 **39.12%**，相比第一步规则再改善 **16.7%**，验证了闭环优化（Stop/Merge/Patch）的有效性。
