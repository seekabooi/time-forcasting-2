# experiments/autotune/utils.py
import os
import sys
import json
import yaml
import time
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from tqdm import tqdm

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def load_config(config_path: str = None) -> Dict:
    """加载YAML配置"""
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


class ProgressLogger:
    """进度日志器"""

    def __init__(self, log_dir: str = "storage/autotune_results/logs", verbose: bool = True):
        self.verbose = verbose
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self._start_time = None
        self._log_file = None
        self._step_count = 0
        self._total_steps = 0

    def start_log(self, name: str = "autotune"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._log_file = os.path.join(self.log_dir, f"{name}_{timestamp}.log")

    def log(self, message: str, level: str = "INFO"):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        full_msg = f"[{timestamp}] [{level}] {message}"
        if self.verbose:
            print(full_msg)
        if self._log_file:
            with open(self._log_file, "a", encoding="utf-8") as f:
                f.write(full_msg + "\n")

    def start(self, total_steps: int, message: str = ""):
        self._start_time = time.time()
        self._total_steps = total_steps
        self._step_count = 0
        self.log("=" * 70)
        if message:
            self.log(f"🚀 {message}")
        self.log(f"📊 总步数: {total_steps}")
        self.log("=" * 70)

    def step(self, message: str = "", sub_message: str = ""):
        self._step_count += 1
        elapsed = time.time() - self._start_time if self._start_time else 0
        remaining = (elapsed / self._step_count) * (
                    self._total_steps - self._step_count) if self._step_count > 0 and self._total_steps > 0 else 0
        progress = f"[{self._step_count}/{self._total_steps}]"
        time_info = f"⏱️ 已用: {elapsed:.1f}s | 预计剩余: {remaining:.1f}s" if self._total_steps > 0 else ""
        log_msg = f"{progress} {message}"
        if sub_message:
            log_msg += f" → {sub_message}"
        if time_info:
            log_msg += f"  {time_info}"
        self.log(log_msg)

    def finish(self, message: str = ""):
        if self._start_time:
            elapsed = time.time() - self._start_time
            self.log(f"✅ 完成! 总耗时: {elapsed:.1f}s")
        if message:
            self.log(message)
        self.log("=" * 70)

    def get_log_file(self) -> str:
        return self._log_file


class MemoryCache:
    def __init__(self, cache_dir: str = "storage/autotune_results/cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def get(self, key: str) -> Optional[Any]:
        cache_file = os.path.join(self.cache_dir, f"{key}.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return None
        return None

    def set(self, key: str, value: Any):
        cache_file = os.path.join(self.cache_dir, f"{key}.json")
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(value, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️ 缓存写入失败: {e}")

    def exists(self, key: str) -> bool:
        return os.path.exists(os.path.join(self.cache_dir, f"{key}.json"))


def compute_mase(pred: np.ndarray, actual: np.ndarray, scale: float = 1.0) -> float:
    if len(pred) == 0:
        return float('nan')
    errors = np.abs(pred - actual)
    mae = np.mean(errors)
    return mae / scale if scale > 0 else float('nan')


def compute_smape(pred: np.ndarray, actual: np.ndarray) -> float:
    if len(pred) == 0:
        return float('nan')
    errors = pred - actual
    denominator = np.abs(pred) + np.abs(actual) + 1e-8
    return np.mean(2.0 * np.abs(errors) / denominator) * 100


def compute_rmse(pred: np.ndarray, actual: np.ndarray) -> float:
    if len(pred) == 0:
        return float('nan')
    errors = pred - actual
    return np.sqrt(np.mean(errors ** 2))


def compute_owa(pred: np.ndarray, actual: np.ndarray) -> float:
    if len(pred) == 0:
        return float('nan')
    rmse = compute_rmse(pred, actual)
    mae = np.mean(np.abs(pred - actual))
    return (rmse + mae) / 2.0


def detect_period(series: np.ndarray, freq: str = None) -> int:
    from src.skills.data_profiler import DataProfiler
    return DataProfiler._auto_period(series, freq=freq)


def extract_features(series: np.ndarray, local_window_sizes: List[int] = None) -> Dict[str, float]:
    from src.skills.data_profiler import DataProfiler

    n = len(series)
    profile = DataProfiler.profile_selected(series, [
        'trend_strength', 'seasonal_strength', 'adf_pvalue',
        'period', 'data_length', 'skewness', 'cv'
    ])

    features = {
        'trend_strength': profile.get('trend_strength', 0.0),
        'seasonal_strength': profile.get('seasonal_strength', 0.0),
        'adf_pvalue': profile.get('adf_pvalue', 0.5),
        'period': profile.get('period', 12),
        'data_length': n,
        'skewness': profile.get('skewness', 0.0),
        'cv': profile.get('cv', 0.0),
    }

    if local_window_sizes is None:
        local_window_sizes = [7, 30]

    for local_window in local_window_sizes:
        local_n = min(local_window, n)
        if local_n <= 0:
            continue
        local_series = series[-local_n:]

        if len(local_series) >= 3:
            from scipy import stats
            x = np.arange(len(local_series))
            slope, _, _, _, _ = stats.linregress(x, local_series)
        else:
            slope = 0.0

        local_std = np.std(local_series) if len(local_series) > 1 else 0.0
        global_std = np.std(series) if len(series) > 1 else 1.0
        local_std_ratio = local_std / (global_std + 1e-8)

        if len(local_series) >= 2 and local_series[0] != 0:
            local_change_rate = (local_series[-1] - local_series[0]) / abs(local_series[0] + 1e-8)
        else:
            local_change_rate = 0.0

        local_mean = np.mean(local_series) if len(local_series) > 0 else 0.0
        global_mean = np.mean(series) if len(series) > 0 else 1.0
        local_mean_ratio = local_mean / (global_mean + 1e-8)

        features[f'local_slope_{local_window}'] = slope
        features[f'local_std_ratio_{local_window}'] = local_std_ratio
        features[f'local_change_rate_{local_window}'] = local_change_rate
        features[f'local_mean_ratio_{local_window}'] = local_mean_ratio

    return features


def serialize_trajectory(trajectory):
    import json
    return json.dumps(trajectory, ensure_ascii=False)


def deserialize_trajectory(traj_str):
    import json
    return json.loads(traj_str)


def save_window_data(train: np.ndarray, test: np.ndarray, period: int,
                     mase_scale: float, features: Dict, window_id: int,
                     dataset_name: str, horizon: int) -> str:
    """保存窗口数据供inducer回测使用"""
    import pickle
    data_dir = "storage/autotune_results/window_data"
    os.makedirs(data_dir, exist_ok=True)

    file_path = os.path.join(data_dir, f"{dataset_name}_window_{window_id}.pkl")
    data = {
        'train': train,
        'test': test,
        'period': period,
        'mase_scale': mase_scale,
        'features': features,
        'window_id': window_id,
        'horizon': horizon
    }
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
    return file_path


def load_window_data(file_path: str) -> Dict:
    import pickle
    with open(file_path, 'rb') as f:
        return pickle.load(f)