# experiments/autotune/main.py
#!/usr/bin/env python
import os
import sys
import argparse
import pandas as pd
import numpy as np   # ★ 添加
from datetime import datetime
from typing import Dict, List, Optional, Any   # ★ 添加必要的导入

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from experiments.autotune.utils import ProgressLogger, MemoryCache, load_config
from experiments.autotune.collector import WindowCollector
from experiments.autotune.inducer import RuleInducer
from experiments.autotune.validator import RuleValidator
from experiments.autotune.visualizer import ResultVisualizer


class AutoTuner:

    def __init__(self, config_path: str = None, verbose: bool = False):
        self.config = load_config(config_path)
        self.verbose = verbose

        output_dir = self.config.get('output_dir', 'storage/autotune_results')
        self.logger = ProgressLogger(
            log_dir=os.path.join(output_dir, 'logs'),
            verbose=verbose
        )
        self.logger.start_log("autotune")

        self.cache = MemoryCache(
            cache_dir=os.path.join(output_dir, 'cache')
        )
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.collector = WindowCollector(self.config, self.logger, self.cache)
        self.collector.set_verbose(verbose)

    def run(self, dataset_name: str = None, min_train: int = None, horizon: int = None,
            compare: bool = False):
        """运行调优流程

        Args:
            compare: 是否进行对比测试（使用规则 vs 不使用规则）
        """
        self.logger.log("=" * 70)
        self.logger.log("🚀 自动调优 + Generator 启动")
        self.logger.log(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.log("=" * 70)

        datasets = self.config.get('datasets', [])
        if dataset_name:
            datasets = [d for d in datasets if d.get('name') == dataset_name]
            if not datasets:
                self.logger.log(f"❌ 数据集 {dataset_name} 不在配置中")
                return

        if not datasets:
            self.logger.log("⚠️ 没有配置数据集")
            return

        all_results = {}

        for ds_config in datasets:
            name = ds_config.get('name')
            if dataset_name and name != dataset_name:
                continue

            self.logger.log(f"\n{'='*70}")
            self.logger.log(f"📊 处理数据集: {name}")
            self.logger.log(f"{'='*70}")

            # 读取参数
            window_sizes = ds_config.get('window_sizes', [600])
            if not isinstance(window_sizes, list):
                window_sizes = [window_sizes]

            step = ds_config.get('step_size', 50)
            horizon_val = horizon or ds_config.get('horizon', 7)
            max_train = ds_config.get('max_train_size')

            self.logger.log(f"   📋 窗口大小列表: {window_sizes}")
            self.logger.log(f"   📋 滑动步长: {step}")
            self.logger.log(f"   📋 预测步数: {horizon_val}")

            # 采集
            collected = self.collector.collect(
                dataset_name=name,
                window_sizes=window_sizes,
                horizon=horizon_val,
                step=step,
                max_train_size=max_train
            )

            if not collected:
                self.logger.log(f"⚠️ 采集失败: {name}")
                continue

            collection_file = self.collector.save_results(self.output_dir)
            if collection_file is None:
                self.logger.log(f"⚠️ 无有效数据: {name}")
                continue

            collected_df = pd.DataFrame(collected)

            if len(collected_df) < 3:
                self.logger.log(f"⚠️ 数据太少 ({len(collected_df)} 个窗口)，跳过归纳")
                continue

            # Generator
            self.logger.log("🧠 开始规则归纳...")
            self.inducer = RuleInducer(self.config, self.logger)
            rules = self.inducer.induce(collected_df)
            rules_file = self.inducer.save_rules(self.output_dir)

            # 验证
            self.validator = RuleValidator(self.config, self.logger)
            validation_results = self.validator.validate(
                rules, name, window_sizes[0], horizon_val
            )

            # 可视化
            self.visualizer = ResultVisualizer(self.config, self.logger)
            self.visualizer.visualize(validation_results, name, self.output_dir)
            self.visualizer.generate_report(validation_results, name, self.output_dir)

            # ★ 对比测试
            if compare:
                self.logger.log("\n" + "=" * 70)
                self.logger.log("📊 对比测试: 使用规则 vs 不使用规则")
                self.logger.log("=" * 70)
                self._run_comparison(name, rules, window_sizes[0], horizon_val)

            all_results[name] = {
                'collected': len(collected),
                'rules': rules,
                'validation': validation_results
            }

        self.logger.log("\n" + "=" * 70)
        self.logger.log("✅ 所有任务完成!")
        self.logger.log(f"📁 结果保存在: {self.output_dir}")
        self.logger.log("=" * 70)

        return all_results

    def _run_comparison(self, dataset_name: str, rules: Dict, window_size: int, horizon: int):
        """运行对比测试"""
        try:
            from src.dataset.registry import DatasetRegistry
            from src.dataset.loader import load_dataset
            from src.agents.llm_planner import LLMPlannerAgent
            from src.tasks.instance import TaskInstance
            from src.skills.registry import SkillRegistry
            from run_benchmark import build_full_registry
            from experiments.autotune.utils import extract_features, compute_mase, detect_period

            # 加载数据
            registry = DatasetRegistry()
            ds_config = registry.get(dataset_name)
            if not ds_config:
                self.logger.log("❌ 无法加载数据集")
                return

            df = load_dataset(ds_config)
            target_col = ds_config['target_column']
            series = df[target_col].values

            # 取最后一段作为测试
            test_size = 100
            if len(series) < window_size + test_size + horizon:
                self.logger.log("⚠️ 数据不足，跳过对比")
                return

            train = series[-window_size - test_size - horizon:-test_size - horizon]
            test = series[-test_size - horizon:-horizon]
            future = series[-horizon:]

            # 构建技能注册表
            full_registry, _ = build_full_registry()

            # 不使用规则
            agent_no_rules = LLMPlannerAgent(
                model="glm-4",
                skill_registry=full_registry,
                log_file=None,
                use_skills=True,
                rules_file=None
            )

            task = TaskInstance(
                id="compare_no_rules",
                dataset_id=dataset_name,
                template_id="fixed_origin",
                question=f"预测未来{horizon}步",
                question_type="numerical",
                history=train.tolist(),
                horizon=horizon,
                frequency=ds_config.get('frequency', 'daily'),
                prediction_target={},
                resolution_date=datetime.now(),
                difficulty_level=1,
                ground_truth_extractor="",
                dates=None,
                target_date=""
            )

            pred_no_rules = agent_no_rules.predict(task)

            # 使用规则
            from experiments.autotune.rule_engine import RuleEngine
            rule_engine = RuleEngine(rules)

            # 提取特征
            features = extract_features(train)

            # 获取策略
            strategy = rule_engine.get_strategy(features)

            if strategy:
                # 使用固定策略预测
                pred_with_rules = self._predict_with_strategy(train, horizon, strategy, full_registry)
            else:
                # 使用默认参数
                agent_with_rules = LLMPlannerAgent(
                    model="glm-4",
                    skill_registry=full_registry,
                    log_file=None,
                    use_skills=True,
                    rules_file=None
                )
                # 设置规则参数
                params = rule_engine.get_params(features)
                for key, value in params.items():
                    os.environ[f"TUNE_{key}"] = str(value)
                pred_with_rules = agent_with_rules.predict(task)
                for key in params.keys():
                    os.environ.pop(f"TUNE_{key}", None)

            # 计算MASE
            period = detect_period(train)
            mase_scale = self._compute_mase_scale(train, period)
            mase_no_rules = compute_mase(np.array(pred_no_rules), future, mase_scale)
            mase_with_rules = compute_mase(np.array(pred_with_rules), future, mase_scale)

            self.logger.log(f"\n📊 对比结果:")
            self.logger.log(f"   不使用规则 MASE: {mase_no_rules:.4f}")
            self.logger.log(f"   使用规则 MASE: {mase_with_rules:.4f}")
            improvement = (mase_no_rules - mase_with_rules) / mase_no_rules * 100 if mase_no_rules > 0 else 0
            self.logger.log(f"   改善: {improvement:.2f}%")

        except Exception as e:
            self.logger.log(f"⚠️ 对比测试失败: {e}")
            import traceback
            self.logger.log(traceback.format_exc())

    def _predict_with_strategy(self, train: np.ndarray, horizon: int,
                                strategy: Dict, registry) -> List:
        """使用固定策略预测"""
        try:
            stages = strategy.get('stages', [])
            if not stages:
                return None

            predictions = []
            current_hist = train.copy()

            for stage in stages:
                steps = stage.get('steps', 0)
                weights = stage.get('weights', {})

                for _ in range(steps):
                    pred_val = 0.0
                    total_weight = 0.0
                    for skill_name, weight in weights.items():
                        skill = registry.get(skill_name)
                        if skill and weight > 0:
                            try:
                                forecast = skill.execute(current_hist, 1)
                                if forecast is not None and len(forecast) > 0:
                                    pred_val += forecast[0] * weight
                                    total_weight += weight
                            except:
                                pass
                    if total_weight > 0:
                        pred_val = pred_val / total_weight
                    else:
                        pred_val = np.mean(current_hist[-5:]) if len(current_hist) >= 5 else np.mean(current_hist)

                    predictions.append(pred_val)
                    current_hist = np.append(current_hist, pred_val)

            return predictions[:horizon]
        except Exception as e:
            return None

    def _compute_mase_scale(self, series: np.ndarray, period: int) -> float:
        n = len(series)
        if n >= 2 * period:
            seasonal_errors = np.abs(series[period:] - series[:-period])
            return np.mean(seasonal_errors) if len(seasonal_errors) > 0 else 1.0
        else:
            naive_errors = np.abs(np.diff(series))
            return np.mean(naive_errors) if len(naive_errors) > 0 else 1.0


def main():
    parser = argparse.ArgumentParser(description="自动调优 + Generator")
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--min_train', type=int, default=None)
    parser.add_argument('--horizon', type=int, default=None)
    parser.add_argument('--verbose', action='store_true', help='输出详细日志到终端')
    parser.add_argument('--compare', action='store_true', help='进行使用规则 vs 不使用规则的对比测试')
    args = parser.parse_args()

    tuner = AutoTuner(args.config, verbose=args.verbose)
    tuner.run(args.dataset, args.min_train, args.horizon, compare=args.compare)


if __name__ == '__main__':
    main()