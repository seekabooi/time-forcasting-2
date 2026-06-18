# experiments/autotune/meta_cluster.py
import json
import re
import numpy as np
from typing import Dict, List, Optional
from experiments.autotune.utils import ProgressLogger


class MetaCluster:
    """LLM 驱动的语义合并（参考 MMSkills Phase 2）"""

    def __init__(self, config: Dict, logger: ProgressLogger):
        self.config = config
        self.logger = logger

    # ==================== 核心：LLM 主导的合并 ====================
    def auto_merge_by_features(self, rules: List[Dict]) -> List[Dict]:
        """
        ★ LLM 自主决定：
        1. 分成几组（簇数）
        2. 每组包含哪些规则
        3. 对每组泛化生成新规则
        """
        if len(rules) <= 1:
            return rules

        self.logger.log("🧠 启动 LLM 主导的语义合并（分组 → 泛化生成）...")

        # 1. 构建规则摘要（含特征）
        rule_summaries = []
        for idx, rule in enumerate(rules):
            strategy = rule.get('skill_strategy', {})
            features = strategy.get('_features', {})
            stages = strategy.get('stages', [])
            stage_desc = []
            for st in stages:
                steps = st.get('steps', 0)
                weights = st.get('weights', {})
                w_str = ', '.join([f"{k}:{v:.2f}" for k, v in weights.items()])
                stage_desc.append(f"{steps}步{{{w_str}}}")

            feat_str = ', '.join([
                f"trend={features.get('trend_strength', 0):.2f}",
                f"season={features.get('seasonal_strength', 0):.2f}",
                f"period={int(features.get('period', 0))}",
                f"adf={features.get('adf_pvalue', 0):.3f}"
            ])

            rule_summaries.append(
                f"规则{idx}: 特征=[{feat_str}] → 策略={' → '.join(stage_desc)}"
            )

        # 2. LLM 分组
        group_prompt = f"""你是一个时序预测规则优化专家。以下是当前所有规则，每条规则包含窗口特征和对应的多阶段预测策略。

{chr(10).join(rule_summaries)}

任务：分析这些规则，将解决【同一类问题】的规则分到一组。

分组原则：
1. 如果两条规则的特征相似（如周期、趋势强度、季节强度相近），它们解决的是同一类问题。
2. 如果两条规则虽然特征不同，但最终使用的策略结构相似（如都依赖 chunk_ensemble），也可考虑归为一组。
3. 每组至少2条规则，最多4条。如果某条规则与其他都不相似，则单独一组（不合并）。
4. 簇数不固定，由你根据实际特征分布决定。

输出 JSON 格式：
{{
  "groups": [
    {{"rule_indices": [0, 1, 2], "scenario": "强周期平稳场景"}},
    {{"rule_indices": [3, 4], "scenario": "高波动突变场景"}}
  ],
  "singletons": [5, 6]
}}
只输出JSON，不要解释。"""

        try:
            from src.agents.llm_client import LLMClient
            import sys, io
            old_out = sys.stdout
            sys.stdout = io.StringIO()
            client = LLMClient(model=self.config.get('llm', {}).get('model', 'glm-4'))
            resp = client.call_with_retry(group_prompt, max_retries=2)
            sys.stdout = old_out

            content = resp.choices[0].message.content
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if not json_match:
                self.logger.log("⚠️ 无法解析LLM分组响应，保留原规则")
                return rules

            result = json.loads(json_match.group())
            groups = result.get('groups', [])
            singletons = result.get('singletons', [])

            self.logger.log(f"   📊 LLM 分组结果: {len(groups)} 个组, {len(singletons)} 条独立规则")

            # 3. 对每组进行泛化生成
            new_rules = []
            used_indices = set()

            for group in groups:
                indices = group.get('rule_indices', [])
                scenario = group.get('scenario', '通用场景')
                if len(indices) < 2:
                    continue

                group_rules = [rules[i] for i in indices if i < len(rules)]
                if len(group_rules) < 2:
                    continue

                self.logger.log(f"   🔄 合并组 {indices} (场景: {scenario})")

                # ★ LLM 泛化生成新策略
                merged_strategy = self._llm_generate_strategy(group_rules, scenario)

                if merged_strategy and self._validate_strategy(merged_strategy):
                    # 生成合并后的规则
                    merged_rule = {
                        'condition': self._generate_condition_from_group(group_rules),
                        'params': {
                            'long_skill_force_threshold': 0.70,
                            'route_bonus_long_skill': 0.15,
                            'residual_acf_threshold': 0.20
                        },
                        'skill_strategy': merged_strategy,
                        'merged_from': indices,
                        'cluster_size': len(indices),
                        'scenario': scenario,
                        'avg_mase': np.mean([rules[i].get('avg_mase', 0) for i in indices])
                    }
                    new_rules.append(merged_rule)
                    for idx in indices:
                        used_indices.add(idx)
                    self.logger.log(f"      ✅ 泛化生成新规则成功")
                else:
                    self.logger.log(f"      ⚠️ 泛化生成失败，保留原规则")
                    for idx in indices:
                        if idx not in used_indices:
                            new_rules.append(rules[idx])
                            used_indices.add(idx)

            # 4. 处理独立规则
            for idx in singletons:
                if idx < len(rules) and idx not in used_indices:
                    new_rules.append(rules[idx])
                    used_indices.add(idx)

            # 5. 添加未处理的规则
            for idx, rule in enumerate(rules):
                if idx not in used_indices:
                    new_rules.append(rule)

            self.logger.log(f"📋 合并后规则数: {len(rules)} → {len(new_rules)}")
            return new_rules

        except Exception as e:
            self.logger.log(f"⚠️ 合并失败: {e}")
            return rules

    # ==================== LLM 泛化生成策略 ====================
    def _llm_generate_strategy(self, group_rules: List[Dict], scenario: str) -> Optional[Dict]:
        """让 LLM 综合多条规则，泛化生成一条精简的新策略"""
        # 提取规则摘要
        summaries = []
        for i, rule in enumerate(group_rules):
            strategy = rule.get('skill_strategy', {})
            stages = strategy.get('stages', [])
            stage_desc = []
            for st in stages:
                steps = st.get('steps', 0)
                weights = st.get('weights', {})
                w_str = ', '.join([f"{k}:{v:.2f}" for k, v in weights.items()])
                stage_desc.append(f"{steps}步{{{w_str}}}")
            summaries.append(f"规则{i}: {' → '.join(stage_desc)}")

        prompt = f"""你是一个时序预测策略精简专家。以下 {len(group_rules)} 条规则都用于解决同一个场景：{scenario}。

{chr(10).join(summaries)}

任务：综合这 {len(group_rules)} 条规则，生成一条更精简、更泛化的新策略。

要求：
1. 保持总步数 = 7。
2. 如果多条规则在相同步数使用了相似的技能组合，合并它们（权重取平均）。
3. 丢弃权重 < 0.05 的边缘技能。
4. 阶段数尽量精简（建议 2~3 个阶段）。
5. 输出 JSON 格式的 skill_strategy。

输出格式：
{{
  "stages": [
    {{"steps": 3, "weights": {{"chunk_ensemble": 0.70, "multi_resolution": 0.30}}}},
    {{"steps": 4, "weights": {{"chunk_ensemble": 0.60, "residual_correction_advanced": 0.40}}}}
  ]
}}
只输出JSON，不要解释。"""

        try:
            from src.agents.llm_client import LLMClient
            import sys, io
            old_out = sys.stdout
            sys.stdout = io.StringIO()
            client = LLMClient(model=self.config.get('llm', {}).get('model', 'glm-4'))
            resp = client.call_with_retry(prompt, max_retries=2)
            sys.stdout = old_out

            content = resp.choices[0].message.content
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return None
        except Exception as e:
            self.logger.log(f"⚠️ LLM 泛化生成失败: {e}")
            return None

    # ==================== 辅助方法 ====================
    def _validate_strategy(self, strategy: Dict) -> bool:
        if 'stages' not in strategy:
            return False
        if not isinstance(strategy['stages'], list) or len(strategy['stages']) == 0:
            return False
        total_steps = 0
        for stage in strategy['stages']:
            if 'steps' not in stage or 'weights' not in stage:
                return False
            if not isinstance(stage['steps'], int) or stage['steps'] <= 0:
                return False
            if not isinstance(stage['weights'], dict) or len(stage['weights']) == 0:
                return False
            if abs(sum(stage['weights'].values()) - 1.0) > 1e-6:
                return False
            total_steps += stage['steps']
        return total_steps == 7

    def _generate_condition_from_group(self, group_rules: List[Dict]) -> str:
        """从组内规则的特征生成 condition"""
        features_list = []
        for rule in group_rules:
            f = rule.get('skill_strategy', {}).get('_features', {})
            if f:
                features_list.append(f)

        if not features_list:
            return 'True'

        # 取特征均值
        avg_features = {}
        keys = ['period', 'trend_strength', 'seasonal_strength', 'adf_pvalue']
        for k in keys:
            vals = [f.get(k, 0) for f in features_list if k in f]
            if vals:
                avg_features[k] = np.mean(vals)

        conditions = []
        if 'period' in avg_features:
            conditions.append(f"period == {int(round(avg_features['period']))}")
        if 'trend_strength' in avg_features:
            v = avg_features['trend_strength']
            conditions.append(f"trend_strength > 0.5" if v > 0.5 else f"trend_strength <= 0.5")
        if 'seasonal_strength' in avg_features:
            v = avg_features['seasonal_strength']
            conditions.append(f"seasonal_strength > 0.5" if v > 0.5 else f"seasonal_strength <= 0.5")
        if 'adf_pvalue' in avg_features:
            v = avg_features['adf_pvalue']
            conditions.append(f"adf_pvalue < 0.05" if v < 0.05 else f"adf_pvalue >= 0.05")

        if not conditions:
            return 'True'
        return ' and '.join(conditions)