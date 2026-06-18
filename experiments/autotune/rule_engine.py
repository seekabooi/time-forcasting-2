# experiments/autotune/rule_engine.py
from typing import Dict, Any, List, Optional


class RuleEngine:
    def __init__(self, rules_config: Dict):
        self.rules = rules_config.get('rules', [])
        self.default = rules_config.get('default', {})
        self.strategies = rules_config.get('candidate_strategies', [])
        self.window_best = rules_config.get('window_best_strategies', [])

    def get_params(self, features: Dict[str, float]) -> Dict[str, float]:
        """根据特征匹配规则，返回参数"""
        params = self.default.copy()
        for rule in self.rules:
            condition = rule.get('condition', '')
            if self._evaluate_condition(condition, features):
                rule_params = rule.get('params', {})
                for key, value in rule_params.items():
                    params[key] = value
                break
        return params

    def get_strategy(self, features: Dict[str, float]) -> Optional[Dict]:
        """根据特征匹配规则，返回策略"""
        for rule in self.rules:
            condition = rule.get('condition', '')
            if self._evaluate_condition(condition, features):
                strategy = rule.get('skill_strategy')
                if strategy:
                    return strategy
        # 返回默认策略（从window_best中取第一个）
        if self.window_best:
            return self.window_best[0]
        return None

    def get_stage_weights(self, features: Dict[str, float], step: int) -> Optional[Dict[str, float]]:
        """获取指定步数的权重（根据策略和当前步数）"""
        strategy = self.get_strategy(features)
        if not strategy:
            return None

        stages = strategy.get('stages', [])
        cumulative = 0
        for stage in stages:
            steps = stage.get('steps', 0)
            if step < cumulative + steps:
                return stage.get('weights', {})
            cumulative += steps

        # 如果超出范围，返回最后一段的权重
        if stages:
            return stages[-1].get('weights', {})
        return None

    def _evaluate_condition(self, condition: str, features: Dict[str, float]) -> bool:
        """安全评估条件表达式"""
        try:
            # 替换常见的比较符
            safe_condition = condition.replace('==', '==').replace('>=', '>=').replace('<=', '<=')
            safe_condition = safe_condition.replace('>', '>').replace('<', '<')
            safe_condition = safe_condition.replace(' and ', ' and ').replace(' or ', ' or ')
            safe_condition = safe_condition.replace('True', 'True').replace('False', 'False')
            result = eval(safe_condition, {"__builtins__": {}}, features)
            return bool(result)
        except Exception:
            # 如果条件无法评估，返回True（默认匹配）
            return True

    def apply_rules_to_params(self, features: Dict[str, float], params: Dict[str, float]) -> Dict[str, float]:
        """应用规则更新参数"""
        rule_params = self.get_params(features)
        for key, value in rule_params.items():
            params[key] = value
        return params