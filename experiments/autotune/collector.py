# experiments/autotune/collector.py
import os
import sys
import io
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from tqdm import tqdm

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from experiments.autotune.utils import (
    ProgressLogger, MemoryCache, load_config,
    extract_features, detect_period,
    compute_mase, compute_smape, compute_rmse,
    serialize_trajectory, save_window_data
)


class WindowCollector:

    def __init__(self, config: Dict, logger: ProgressLogger, cache: MemoryCache):
        self.config = config
        self.logger = logger
        self.cache = cache
        self.results = []
        self._silent = False
        self._verbose = False

    def set_verbose(self, verbose: bool):
        self._verbose = verbose

    def collect(self, dataset_name: str, window_sizes: List[int], horizon: int, step: int,
                max_train_size: Optional[int] = None) -> List[Dict]:

        if self._verbose:
            self.logger.log(f"\n📊 开始采集: {dataset_name}")
            self.logger.log(f"   窗口大小列表: {window_sizes}, 预测步数: {horizon}, 滑动步长: {step}")

        series, freq = self._load_dataset(dataset_name)
        if series is None:
            if self._verbose:
                self.logger.log(f"❌ 无法加载数据集: {dataset_name}")
            return []

        period = detect_period(series, freq)
        mase_scale = self._compute_mase_scale(series, period)

        if self._verbose:
            self.logger.log(f"   📈 数据长度: {len(series)}, 周期: {period}, MASE缩放: {mase_scale:.4f}")

        fixed_params = self.config.get('fixed_params', {})
        self.logger.log(f"   📌 固定参数: {fixed_params}")

        for key, value in fixed_params.items():
            os.environ[f"TUNE_{key}"] = str(value)

        local_window_sizes = self.config.get('local_window_sizes', [7, 30])

        total_windows = 0
        for w in window_sizes:
            if max_train_size is not None:
                max_start = min(max_train_size - w, len(series) - w - horizon)
            else:
                max_start = len(series) - w - horizon
            if max_start >= 0:
                total_windows += len(list(range(0, max_start + 1, step)))

        if self._verbose:
            self.logger.log(f"   📋 局部窗口大小: {local_window_sizes}")
            self.logger.log(f"   📊 总窗口数: {total_windows}")
            self.logger.log("=" * 70)

        all_window_results = []
        failed_total = 0
        success_total = 0
        window_counter = 0

        pbar = tqdm(
            total=total_windows,
            desc=f"采集 {dataset_name}",
            unit="窗口",
            ncols=100,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}'
        )

        for w_idx, window_size in enumerate(window_sizes):
            if max_train_size is not None:
                max_start = min(max_train_size - window_size, len(series) - window_size - horizon)
            else:
                max_start = len(series) - window_size - horizon

            if max_start < 0:
                self.logger.log(f"⚠️ 窗口大小 {window_size} 超过数据长度，跳过")
                continue

            start_points = list(range(0, max_start + 1, step))
            if self._verbose:
                self.logger.log(f"\n   🔹 窗口大小 {window_size}，起点: {start_points[0]} ~ {start_points[-1]}，共 {len(start_points)} 个窗口")

            for idx, origin in enumerate(start_points):
                window_counter += 1
                self.logger.log(f"🔄 窗口 {window_counter}/{total_windows} (大小={window_size}) 开始: origin={origin}")

                train = series[origin:origin + window_size]
                test = series[origin + window_size:origin + window_size + horizon]
                features = extract_features(train, local_window_sizes=local_window_sizes)
                features['window_size'] = window_size

                mase, trajectory = self._run_prediction(train, test, horizon, period, mase_scale)

                if mase == float('inf'):
                    failed_total += 1
                    self.logger.log(f"❌ 窗口 {window_counter} 预测失败，跳过")
                    pbar.update(1)
                    continue

                success_total += 1

                window_data_path = save_window_data(
                    train, test, period, mase_scale, features,
                    window_counter, dataset_name, horizon
                )

                window_record = {
                    'dataset': dataset_name,
                    'window_id': window_counter,
                    'origin': origin,
                    'window_size': window_size,
                    'train_size': len(train),
                    'test_size': len(test),
                    'horizon': horizon,
                    'period': period,
                    'mase_scale': mase_scale,
                    'window_data_path': window_data_path,
                    **features,
                    'best_trajectory': serialize_trajectory(trajectory),
                    'best_mase': mase,
                    'best_long_skill_force_threshold': float(fixed_params.get('long_skill_force_threshold', 0.70)),
                    'best_route_bonus_long_skill': float(fixed_params.get('route_bonus_long_skill', 0.15)),
                    'best_residual_acf_threshold': float(fixed_params.get('residual_acf_threshold', 0.20)),
                }

                all_window_results.append(window_record)

                self.logger.log(
                    f"✅ 窗口 {window_counter}/{total_windows} (大小={window_size}) 完成: "
                    f"origin={origin}, best_mase={mase:.4f}"
                )

                pbar.set_postfix({
                    'size': window_size,
                    'origin': origin,
                    'best': f'{mase:.4f}',
                    'ok': success_total,
                    'fail': failed_total
                })
                pbar.update(1)

        pbar.close()
        self._silent = False

        for key in fixed_params.keys():
            os.environ.pop(f"TUNE_{key}", None)

        if self._verbose:
            self.logger.log(f"\n   ✅ 成功: {success_total} 个窗口, 失败: {failed_total} 个窗口")
            self.logger.log(f"   📊 总窗口数: {len(all_window_results)}")
            self.logger.finish(f"采集完成: {len(all_window_results)} 个有效窗口")

        self.results.extend(all_window_results)
        return all_window_results

    def _load_dataset(self, dataset_name: str) -> Tuple[Optional[np.ndarray], Optional[str]]:
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
        except Exception:
            return None, None

    def _compute_mase_scale(self, series: np.ndarray, period: int) -> float:
        n = len(series)
        if n >= 2 * period:
            seasonal_errors = np.abs(series[period:] - series[:-period])
            return np.mean(seasonal_errors) if len(seasonal_errors) > 0 else 1.0
        else:
            naive_errors = np.abs(np.diff(series))
            return np.mean(naive_errors) if len(naive_errors) > 0 else 1.0

    def _run_prediction(self, train: np.ndarray, test: np.ndarray,
                        horizon: int, period: int, mase_scale: float,
                        verbose: bool = False) -> Tuple[float, List]:
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = stdout_capture
        sys.stderr = stderr_capture

        try:
            import hashlib
            import pickle
            from datetime import datetime

            train_bytes = pickle.dumps(train)
            train_hash = hashlib.md5(train_bytes).hexdigest()
            cache_key = f"pred_{train_hash}_{horizon}"

            if self.cache.exists(cache_key):
                cached = self.cache.get(cache_key)
                if cached and 'pred' in cached and 'trajectory' in cached:
                    pred = np.array(cached['pred'])
                    if len(pred) == len(test):
                        sys.stdout = old_stdout
                        sys.stderr = old_stderr
                        return compute_mase(pred, test, mase_scale), cached['trajectory']

            from src.agents.llm_planner import LLMPlannerAgent
            from src.tasks.instance import TaskInstance
            from run_benchmark import build_full_registry

            full_registry, _ = build_full_registry()
            agent = LLMPlannerAgent(
                model="glm-4",
                skill_registry=full_registry,
                log_file=None,
                use_skills=True
            )

            task = TaskInstance(
                id=f"collect_{len(train)}_{len(test)}",
                dataset_id="autotune",
                template_id="fixed_origin",
                question=f"预测未来{len(test)}步",
                question_type="numerical",
                history=train.tolist(),
                horizon=len(test),
                frequency="daily",
                prediction_target={},
                resolution_date=datetime.now(),
                difficulty_level=1,
                ground_truth_extractor="",
                dates=None,
                target_date=""
            )

            pred, trajectory = agent.predict_with_trajectory(task)
            pred_array = np.array(pred)

            self.cache.set(cache_key, {
                'pred': pred_array.tolist(),
                'train_size': len(train),
                'test_size': len(test),
                'trajectory': trajectory
            })

            sys.stdout = old_stdout
            sys.stderr = old_stderr

            if len(pred_array) != len(test):
                return float('inf'), trajectory

            mase = compute_mase(pred_array, test, mase_scale)
            return mase, trajectory

        except Exception as e:
            self.logger.log(f"⚠️ 预测异常: {e}")
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            return float('inf'), []

    def save_results(self, output_path: str):
        if not self.results:
            return None

        df = pd.DataFrame(self.results)
        output_file = os.path.join(output_path, "collected_windows.csv")
        os.makedirs(output_path, exist_ok=True)
        df.to_csv(output_file, index=False)
        self.logger.log(f"📁 采集结果已保存: {output_file}")
        return output_file