# experiments/autotune/validator.py
import os
import sys
import io
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from experiments.autotune.utils import ProgressLogger, compute_mase, compute_smape, compute_rmse, compute_owa
from experiments.autotune.rule_engine import RuleEngine


class RuleValidator:
    """
    规则验证器：在验证集上测试规则效果
    """

    def __init__(self, config: Dict, logger: ProgressLogger):
        self.config = config
        self.logger = logger

    def validate(self, rules: Dict, dataset_name: str, min_train: int, horizon: int) -> Dict:
        self.logger.log(f"\n🔬 验证规则: {dataset_name}")

        series, freq = self._load_dataset(dataset_name)
        if series is None:
            return {'error': '无法加载数据集'}

        split_idx = int(len(series) * 0.8)
        val_series = series[split_idx:]

        if len(val_series) < min_train + horizon:
            self.logger.log(f"⚠️ 验证数据不足: {len(val_series)} < {min_train + horizon}")
            return {'error': '数据不足'}

        self.logger.log(f"📊 验证数据长度: {len(val_series)}")

        results = {
            'original': [],
            'fixed_best': [],
            'dynamic_rules': []
        }

        default_params = rules.get('default', {
            'long_skill_force_threshold': 0.80,
            'route_bonus_long_skill': 0.25,
            'residual_acf_threshold': 0.30
        })

        fixed_best = self._find_fixed_best(default_params)
        rule_engine = RuleEngine(rules)

        windows = list(range(min_train, len(val_series) - horizon, 30))
        if not windows:
            windows = [min_train]

        self.logger.log(f"📋 验证窗口数: {len(windows)}")

        for origin in windows:
            train = val_series[:origin]
            test = val_series[origin:origin + horizon]

            from experiments.autotune.utils import extract_features, detect_period
            period = detect_period(train, freq)
            features = extract_features(train)
            features['period'] = period

            mase_orig = self._run_with_params(train, test, horizon, period, default_params)
            results['original'].append(mase_orig)

            mase_fixed = self._run_with_params(train, test, horizon, period, fixed_best)
            results['fixed_best'].append(mase_fixed)

            dynamic_params = rule_engine.get_params(features)
            mase_dynamic = self._run_with_params(train, test, horizon, period, dynamic_params)
            results['dynamic_rules'].append(mase_dynamic)

        summary = self._compute_summary(results)

        self.logger.log(f"\n📊 验证结果汇总:")
        self.logger.log(f"   原始值 MASE: {summary['original']['mean']:.4f} ± {summary['original']['std']:.4f}")
        self.logger.log(f"   固定最优 MASE: {summary['fixed_best']['mean']:.4f} ± {summary['fixed_best']['std']:.4f}")
        self.logger.log(
            f"   动态规则 MASE: {summary['dynamic_rules']['mean']:.4f} ± {summary['dynamic_rules']['std']:.4f}")

        improvement = (summary['original']['mean'] - summary['dynamic_rules']['mean']) / summary['original'][
            'mean'] * 100 if summary['original']['mean'] != 0 else 0
        self.logger.log(f"   🎯 相比原始值改善: {improvement:.2f}%")

        return summary

    def _load_dataset(self, dataset_name: str):
        try:
            from src.dataset.registry import DatasetRegistry
            from src.dataset.loader import load_dataset

            registry = DatasetRegistry()
            ds_config = registry.get(dataset_name)
            if not ds_config:
                return None, None

            df = load_dataset(ds_config)
            target_col = ds_config['target_column']
            series = df[target_col].values
            freq = ds_config.get('frequency', 'daily')

            return series, freq
        except Exception as e:
            self.logger.log(f"⚠️ 加载失败: {e}")
            return None, None

    def _run_with_params(self, train: np.ndarray, test: np.ndarray,
                         horizon: int, period: int, params: Dict) -> float:
        """使用指定参数运行预测，抑制所有输出"""
        # 用 StringIO 捕获输出
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = stdout_capture
        sys.stderr = stderr_capture

        try:
            for key, value in params.items():
                os.environ[f"TUNE_{key}"] = str(value)

            from src.agents.llm_planner import LLMPlannerAgent
            from src.skills.registry import SkillRegistry
            from src.tasks.instance import TaskInstance
            from run_benchmark import build_full_registry

            # 所有输出被捕获
            full_registry, _ = build_full_registry()
            agent = LLMPlannerAgent(
                model="glm-4",
                skill_registry=full_registry,
                log_file=None,
                use_skills=True
            )

            task = TaskInstance(
                id=f"val_{len(train)}_{len(test)}",
                dataset_id="autotune_validate",
                template_id="fixed_origin",
                question=f"预测未来{len(test)}步",
                question_type="numerical",
                history=train.tolist(),
                horizon=len(test),
                frequency="daily",
                prediction_target={},
                resolution_date=None,
                difficulty_level=1,
                ground_truth_extractor="",
                dates=None,
                target_date=""
            )

            pred = agent.predict(task)
            pred_array = np.array(pred)

            for key in params.keys():
                os.environ.pop(f"TUNE_{key}", None)

            # 恢复输出流
            sys.stdout = old_stdout
            sys.stderr = old_stderr

            if len(pred_array) != len(test):
                return float('inf')

            from experiments.autotune.utils import detect_period
            period_val = detect_period(train)
            mase_scale = self._compute_mase_scale(train, period_val)
            return compute_mase(pred_array, test, mase_scale)

        except Exception as e:
            for key in params.keys():
                os.environ.pop(f"TUNE_{key}", None)
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            return float('inf')

    def _compute_mase_scale(self, series: np.ndarray, period: int) -> float:
        n = len(series)
        if n >= 2 * period:
            seasonal_errors = np.abs(series[period:] - series[:-period])
            return np.mean(seasonal_errors) if len(seasonal_errors) > 0 else 1.0
        else:
            naive_errors = np.abs(np.diff(series))
            return np.mean(naive_errors) if len(naive_errors) > 0 else 1.0

    def _find_fixed_best(self, default_params: Dict) -> Dict:
        return default_params

    def _compute_summary(self, results: Dict) -> Dict:
        summary = {}
        for mode, mases in results.items():
            valid = [m for m in mases if m < float('inf')]
            summary[mode] = {
                'mean': np.mean(valid) if valid else float('inf'),
                'std': np.std(valid) if valid else float('inf'),
                'min': np.min(valid) if valid else float('inf'),
                'max': np.max(valid) if valid else float('inf'),
                'count': len(valid)
            }
        return summary