# experiments/autotune/hard_refiner.py
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from experiments.autotune.utils import load_window_data, extract_features, compute_mase
from experiments.autotune.inducer import RuleInducer


class HardRefiner:
    """针对指定的困难窗口生成补丁规则，并验证有效性"""

    def __init__(self, config: Dict, logger):
        self.config = config
        self.logger = logger
        self.inducer = RuleInducer(config, logger)

    def refine(self, collected_data: pd.DataFrame, target_window_ids: List[int],
               current_rules: List[Dict]) -> List[Dict]:
        """
        只针对 target_window_ids 中的窗口生成补丁规则
        target_window_ids 通常是 MASE 最差的 3 个窗口
        """
        if not target_window_ids:
            self.logger.log("⚠️ 无目标窗口，跳过重练")
            return []

        self.logger.log(f"🎯 针对指定 {len(target_window_ids)} 个窗口生成补丁规则: {target_window_ids}")
        target_rows = collected_data[collected_data['window_id'].isin(target_window_ids)]
        new_rules = []

        for idx, row in target_rows.iterrows():
            window_id = row['window_id']
            window_data_path = row['window_data_path']
            if not window_data_path:
                continue

            try:
                wdata = load_window_data(window_data_path)
                train = wdata['train']
                test = wdata['test']
                period = wdata.get('period', 365)
                features = extract_features(train)
                trajectory_str = row.get('best_trajectory', '[]')
                import json
                trajectory = json.loads(trajectory_str) if trajectory_str else []

                # 生成候选策略
                candidate_strategies = self.inducer._generate_candidate_strategies(
                    features, trajectory, window_id
                )
                if not candidate_strategies:
                    continue

                # 评估原始规则在该窗口的MASE
                original_mase = self._get_original_mase(window_id, current_rules, wdata)
                if original_mase is None:
                    continue

                best_strategy = None
                best_mase = float('inf')
                for strategy in candidate_strategies:
                    pred = self.inducer._predict_with_strategy(train, len(test), period, strategy)
                    if pred is not None and len(pred) == len(test):
                        mase = compute_mase(pred, test, wdata.get('mase_scale', 1.0))
                        if mase < best_mase:
                            best_mase = mase
                            best_strategy = strategy

                if best_strategy is None:
                    continue

                improvement = (original_mase - best_mase) / original_mase if original_mase > 0 else 0
                # 改善 > 5% 才采纳
                if improvement > 0.05:
                    condition = self._generate_condition_from_features(features)
                    rule = {
                        'condition': condition,
                        'params': {
                            'long_skill_force_threshold': 0.70,
                            'route_bonus_long_skill': 0.15,
                            'residual_acf_threshold': 0.20
                        },
                        'skill_strategy': best_strategy,
                        'is_patch': True,
                        'patch_for_window': window_id,
                        'patch_mase': best_mase,
                        'improvement': improvement
                    }
                    new_rules.append(rule)
                    self.logger.log(f"  窗口 {window_id} 生成补丁规则，MASE从 {original_mase:.4f} 降至 {best_mase:.4f} (改善 {improvement:.1%})")
                else:
                    self.logger.log(f"  窗口 {window_id} 补丁无显著改善 (改善 {improvement:.1%})，跳过")

            except Exception as e:
                self.logger.log(f"⚠️ 窗口 {window_id} 重练失败: {e}")

        self.logger.log(f"✅ 生成 {len(new_rules)} 条有效补丁规则")
        return new_rules

    def _get_original_mase(self, window_id: int, rules: List[Dict], wdata: Dict) -> Optional[float]:
        try:
            from experiments.autotune.rule_engine import RuleEngine
            rule_engine = RuleEngine({'rules': rules})
            train = wdata['train']
            test = wdata['test']
            period = wdata.get('period', 365)
            mase_scale = wdata.get('mase_scale', 1.0)
            features = extract_features(train)
            strategy = rule_engine.get_strategy(features)
            if strategy is None:
                return None
            pred = self.inducer._predict_with_strategy(train, len(test), period, strategy)
            if pred is None or len(pred) != len(test):
                return None
            return compute_mase(pred, test, mase_scale)
        except:
            return None

    def _generate_condition_from_features(self, features: Dict) -> str:
        conditions = []
        if 'period' in features:
            conditions.append(f"period == {int(features['period'])}")
        if 'trend_strength' in features:
            val = features['trend_strength']
            conditions.append(f"trend_strength > {val:.2f}" if val > 0.5 else f"trend_strength <= {val:.2f}")
        if 'seasonal_strength' in features:
            val = features['seasonal_strength']
            conditions.append(f"seasonal_strength > {val:.2f}" if val > 0.5 else f"seasonal_strength <= {val:.2f}")
        if not conditions:
            return 'True'
        return ' and '.join(conditions)