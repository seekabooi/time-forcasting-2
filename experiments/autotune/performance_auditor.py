# experiments/autotune/performance_auditor.py
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from experiments.autotune.utils import load_window_data, compute_mase, extract_features
from experiments.autotune.rule_engine import RuleEngine


class PerformanceAuditor:
    """审计规则在验证集上的表现，返回详细报告"""

    def __init__(self, config: Dict, logger):
        self.config = config
        self.logger = logger
        # ★ 可配置的阈值
        self.threshold_multiplier = config.get('audit_threshold_multiplier', 1.2)
        self.stop_ratio = config.get('stop_ratio', 0.01)  # ★ 改为 1%

    def audit(self, collected_data: pd.DataFrame, rules: Dict) -> Tuple[List[int], Dict]:
        """
        返回: (hard_window_ids, report)
        report 包含：avg_mase, std_mase, total_windows, hard_window_ids,
        worst_3_window_ids（★ 新增：MASE最差的3个窗口）
        """
        self.logger.log("🔍 开始性能审计...")
        rule_engine = RuleEngine(rules)

        mase_list = []  # (window_id, mase)
        hard_window_ids = []

        for idx, row in collected_data.iterrows():
            window_id = row['window_id']
            window_data_path = row['window_data_path']
            if not window_data_path:
                continue

            try:
                wdata = load_window_data(window_data_path)
                train = wdata['train']
                test = wdata['test']
                period = wdata.get('period', 365)
                mase_scale = wdata.get('mase_scale', 1.0)

                features = extract_features(train)
                strategy = rule_engine.get_strategy(features)
                if strategy is None:
                    continue

                pred = self._predict_with_strategy(train, len(test), period, strategy)
                if pred is None or len(pred) != len(test):
                    continue

                mase = compute_mase(pred, test, mase_scale)
                mase_list.append((window_id, mase))
            except Exception as e:
                self.logger.log(f"⚠️ 窗口 {window_id} 审计失败: {e}")

        if not mase_list:
            self.logger.log("⚠️ 无有效窗口数据")
            return [], {'avg_mase': float('inf'), 'std_mase': 0, 'total_windows': 0,
                        'hard_window_ids': [], 'worst_3_window_ids': []}

        mase_list.sort(key=lambda x: x[1])  # 按MASE升序
        all_mases = np.array([m for _, m in mase_list])
        avg_mase = np.mean(all_mases)
        std_mase = np.std(all_mases)
        threshold = avg_mase * self.threshold_multiplier
        hard_window_ids = [wid for wid, m in mase_list if m > threshold]

        # ★ 取 MASE 最差的 3 个（无论是否超过阈值）
        worst_3_ids = [wid for wid, _ in mase_list[-3:]] if len(mase_list) >= 3 else [wid for wid, _ in mase_list]

        report = {
            'avg_mase': avg_mase,
            'std_mase': std_mase,
            'total_windows': len(mase_list),
            'hard_window_ids': hard_window_ids,
            'hard_count': len(hard_window_ids),
            'worst_3_window_ids': worst_3_ids,  # ★ 新增
            'threshold': threshold
        }

        self.logger.log(f"  平均MASE: {avg_mase:.4f} ± {std_mase:.4f}")
        self.logger.log(f"  困难窗口数: {len(hard_window_ids)}/{len(mase_list)}")
        self.logger.log(f"  MASE最差3个窗口: {worst_3_ids}")
        return hard_window_ids, report

    def _predict_with_strategy(self, train, horizon, period, strategy):
        try:
            from src.skills.registry import SkillRegistry
            from run_benchmark import build_full_registry
            full_registry, _ = build_full_registry()
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
                    total_w = 0.0
                    for skill_name, weight in weights.items():
                        skill = full_registry.get(skill_name)
                        if skill and weight > 0:
                            try:
                                forecast = skill.execute(current_hist, 1, period=period)
                                if forecast is not None and len(forecast) > 0:
                                    pred_val += forecast[0] * weight
                                    total_w += weight
                            except:
                                pass
                    if total_w > 0:
                        pred_val /= total_w
                    else:
                        pred_val = np.mean(current_hist[-5:]) if len(current_hist) >= 5 else np.mean(current_hist)
                    predictions.append(pred_val)
                    current_hist = np.append(current_hist, pred_val)
            return np.array(predictions[:horizon])
        except Exception:
            return None