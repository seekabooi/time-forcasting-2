# experiments/autotune/inducer.py
import os
import sys
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import re
from collections import defaultdict
from datetime import datetime

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from experiments.autotune.utils import (
    ProgressLogger, load_config, deserialize_trajectory,
    load_window_data, compute_mase
)
from experiments.autotune.cluster import StrategyCluster
from experiments.autotune.cache_manager import StrategyCache


class RuleInducer:
    def __init__(self, config: Dict, logger: ProgressLogger):
        self.config = config
        self.logger = logger
        self.rules = []
        self.default_threshold = 0.80
        self.strategies = []
        self.window_best_strategies = []
        self.cache = StrategyCache()

    def induce(self, collected_data: pd.DataFrame) -> Dict:
        self.logger.log("\n" + "=" * 70)
        self.logger.log("🧠 [Generator] 规则归纳阶段")
        self.logger.log("=" * 70)

        if collected_data.empty:
            self.logger.log("⚠️ 无数据，使用默认规则")
            return self._default_rules()

        self.logger.log(f"📊 数据量: {len(collected_data)} 个窗口")

        # ★ 强制回测模式（用于打印，截图后请注释掉或改为 False）
        force_backtest = True
        if force_backtest:
            self.logger.log("🔥 强制回测模式：将逐窗口生成策略并回测")
        else:
            self.logger.log("ℹ️ 正常模式：优先使用缓存")

        # 获取配置参数用于缓存键
        ds_config = self.config.get('datasets', [{}])[0]
        window_size = ds_config.get('window_sizes', [600])[0] if ds_config else 600
        step = ds_config.get('step_size', 150)
        horizon = ds_config.get('horizon', 7)
        fixed_params = self.config.get('fixed_params', {})
        dataset_name = collected_data.iloc[0].get('dataset', 'unknown')

        # ★ 仅在非强制回测模式下尝试从缓存加载
        if not force_backtest:
            cached_data = self.cache.get(dataset_name, window_size, step, horizon, fixed_params)
            if cached_data:
                self.logger.log("📦 从缓存加载策略结果")
                self.window_best_strategies = cached_data.get('best_strategies', [])
                if self.window_best_strategies:
                    clustered_strategies = self._cluster_strategies(self.window_best_strategies)
                    rules = self._generate_rules_from_clusters(clustered_strategies)
                    self.rules = rules
                    return {'rules': rules, 'default': self._default_rules().get('default', {})}
                else:
                    self.logger.log("⚠️ 缓存数据为空，重新生成")
        else:
            self.logger.log("⏭️ 跳过缓存，强制执行回测")

        # ★ 对每个窗口生成候选策略并选best（这里会打印回测）
        self._process_each_window(collected_data)

        if not self.window_best_strategies:
            self.logger.log("⚠️ 所有窗口策略生成失败，使用默认规则")
            return self._default_rules()

        # ★ 保存缓存（仅在非强制回测模式下保存）
        if not force_backtest:
            self.cache.save(dataset_name, window_size, step, horizon, fixed_params, {
                'best_strategies': self.window_best_strategies,
                'timestamp': datetime.now().isoformat()
            })
            self.logger.log("📦 策略结果已缓存")
        else:
            self.logger.log("⏭️ 强制回测模式下不保存缓存")

        # ★ 聚类所有窗口的best策略
        self.logger.log(f"\n📊 共收集 {len(self.window_best_strategies)} 个窗口的best策略")
        clustered_strategies = self._cluster_strategies(self.window_best_strategies)

        # ★ 生成最终规则
        rules = self._generate_rules_from_clusters(clustered_strategies)

        self.rules = rules
        return {'rules': rules, 'default': self._default_rules().get('default', {})}

    def _cluster_strategies(self, strategies: List[Dict]) -> List[Dict]:
        """聚类策略（独立方法便于复用）"""
        cluster = StrategyCluster(self.logger)
        return cluster.cluster(strategies)

    def _process_each_window(self, data: pd.DataFrame):
        """对每个窗口生成候选策略并选best"""
        self.logger.log("\n" + "=" * 70)
        self.logger.log("📋 逐窗口策略生成与评估")
        self.logger.log("=" * 70)

        for idx, row in data.iterrows():
            window_id = row.get('window_id', idx + 1)
            window_size = row.get('window_size', 600)
            origin = row.get('origin', 0)
            features = self._extract_features_dict(row)
            trajectory_str = row.get('best_trajectory', '[]')
            window_data_path = row.get('window_data_path', '')
            horizon_val = int(row.get('horizon', 7))

            self.logger.log(f"\n🔹 窗口 {window_id} (origin={origin}, size={window_size})")

            if not window_data_path or not os.path.exists(window_data_path):
                self.logger.log(f"   ⚠️ 窗口数据不存在，跳过")
                continue

            window_data = load_window_data(window_data_path)
            train = window_data['train']
            test = window_data['test']
            period = window_data.get('period', 365)
            mase_scale = window_data.get('mase_scale', 1.0)
            horizon_val = window_data.get('horizon', horizon_val)

            # 解析轨迹
            try:
                trajectory = deserialize_trajectory(trajectory_str)
            except:
                trajectory = []

            # ★ 让LLM生成候选策略
            candidate_strategies = self._generate_candidate_strategies(
                features, trajectory, window_id
            )

            if not candidate_strategies:
                self.logger.log(f"   ⚠️ 无候选策略，跳过")
                continue

            # ★ 打印候选策略
            self.logger.log(f"   📋 候选策略 ({len(candidate_strategies)} 个):")
            for si, strategy in enumerate(candidate_strategies):
                name = strategy.get('name', f'候选{si+1}')
                stages = strategy.get('stages', [])
                stage_desc = []
                total_steps = 0
                for st in stages:
                    steps = st.get('steps', 0)
                    total_steps += steps
                    weights = st.get('weights', {})
                    w_str = ', '.join([f"{k}:{v:.4f}" for k, v in weights.items()])
                    stage_desc.append(f"{steps}步{{{w_str}}}")
                self.logger.log(f"      {name}: {' → '.join(stage_desc)} (总步数={total_steps})")
                self.logger.log(f"        描述: {strategy.get('description', '')}")

            # ★ 评估每个候选策略，选best
            best_strategy = None
            best_mase = float('inf')
            best_similarity = -float('inf')

            for si, strategy in enumerate(candidate_strategies):
                try:
                    pred = self._predict_with_strategy(train, horizon_val, period, strategy)
                    if pred is not None and len(pred) == len(test):
                        mase = compute_mase(pred, test, mase_scale)
                        similarity = self._strategy_similarity(strategy, trajectory)

                        self.logger.log(f"      策略 {si+1}: MASE={mase:.4f}, 相似度={similarity:.3f}")

                        # 优先选MASE低的，如果MASE相同选相似度高的
                        if mase < best_mase or (mase == best_mase and similarity > best_similarity):
                            best_mase = mase
                            best_strategy = strategy
                            best_similarity = similarity
                except Exception as e:
                    self.logger.log(f"      策略 {si+1}: 评估失败 - {e}")

            if best_strategy is None:
                self.logger.log(f"   ⚠️ 无有效策略，跳过")
                continue

            # ★ 记录窗口的best策略
            best_strategy['_window_id'] = window_id
            best_strategy['_origin'] = origin
            best_strategy['_mase'] = best_mase
            best_strategy['_features'] = features
            self.window_best_strategies.append(best_strategy)

            self.logger.log(f"   🏆 窗口 {window_id} 最优策略: {best_strategy.get('name', '')}")
            self.logger.log(f"      最优MASE: {best_mase:.4f}")

    def _extract_features_dict(self, row: pd.Series) -> Dict:
        """从DataFrame行提取特征字典"""
        features = {}
        for col in row.index:
            if col not in ['dataset', 'window_id', 'origin', 'train_size', 'test_size',
                           'period', 'mase_scale', 'best_config_name', 'best_mase',
                           'best_trajectory', 'window_data_path', 'best_long_skill_force_threshold',
                           'best_route_bonus_long_skill', 'best_residual_acf_threshold',
                           'horizon', 'window_size']:
                try:
                    features[col] = float(row[col])
                except:
                    pass
        return features

    def _generate_candidate_strategies(self, features: Dict, trajectory: List,
                                        window_id: int) -> List[Dict]:
        """让LLM生成候选策略"""
        prompt = self._build_strategy_prompt(features, trajectory, window_id)

        try:
            result = self._call_llm(prompt)
            strategies = result.get('candidate_strategies', [])
            for i, s in enumerate(strategies):
                if not s.get('name'):
                    s['name'] = f"策略{chr(65+i)}"
            return strategies
        except Exception as e:
            self.logger.log(f"   ⚠️ 策略生成失败: {e}")
            return []

    def _build_strategy_prompt(self, features: Dict, trajectory: List, window_id: int) -> str:
        """构建策略生成Prompt"""
        feat_desc = []
        for k, v in features.items():
            if isinstance(v, (int, float)):
                feat_desc.append(f"{k}: {v:.3f}")

        traj_desc = []
        for step_info in trajectory[:10]:
            step = step_info.get('step', 0)
            weights = step_info.get('weights', {})
            interval = step_info.get('interval', 1)
            w_str = ', '.join([f"{k}: {v:.4f}" for k, v in weights.items()])
            traj_desc.append(f"第{step}步: {{{w_str}}} (间隔={interval})")
        traj_summary = '\n'.join(traj_desc)

        prompt = f"""你是一个时序预测策略专家。给定一个窗口的数据特征和预测轨迹，请生成3-5个可复用的"分段加权预测策略"。

窗口特征：
{', '.join(feat_desc)}

原始预测轨迹（每一步的技能权重组合）：
{traj_summary}

策略格式：每个策略包含多个阶段(stages)，每个阶段指定：
- steps: 该阶段预测的步数（总步数应为7）
- weights: 技能名称到权重的字典（权重之和为1）

要求：
1. 生成3-5个不同的策略，每个策略要有独特的名称
2. 策略要基于轨迹中出现的技能组合进行提炼
3. 每个阶段的权重之和必须为1
4. 权重保留8位小数
5. 各阶段步数之和必须等于7

输出 JSON 格式：
{{
  "candidate_strategies": [
    {{
      "name": "策略名称",
      "stages": [
        {{"steps": 3, "weights": {{"chunk_ensemble": 0.70000000, "multi_resolution": 0.30000000}}}},
        {{"steps": 4, "weights": {{"chunk_ensemble": 0.60000000, "calendar": 0.40000000}}}}
      ],
      "description": "策略描述"
    }}
  ]
}}

只输出JSON，不要任何解释。"""
        return prompt

    def _call_llm(self, prompt: str) -> Dict:
        try:
            from src.agents.llm_client import LLMClient
            llm_config = self.config.get('llm', {})
            model = llm_config.get('model', 'glm-4')
            max_tokens = llm_config.get('max_tokens', 4000)

            import sys, io
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = io.StringIO()
            sys.stderr = io.StringIO()
            try:
                client = LLMClient(model=model, log_file=None)
                resp = client.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=llm_config.get('temperature', 0.0),
                    max_tokens=max_tokens,
                    timeout=30
                )
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr

            content = resp.choices[0].message.content
            self.logger.log(f"   📝 LLM响应 (长度{len(content)})")

            result = self._extract_json(content)
            if result is None:
                self.logger.log("   ⚠️ JSON解析失败")
                return {}
            return result
        except Exception as e:
            self.logger.log(f"   ⚠️ LLM调用失败: {e}")
            return {}

    def _extract_json(self, content: str) -> Optional[Dict]:
        content = re.sub(r'```json\s*', '', content)
        content = re.sub(r'```\s*', '', content)
        content = content.strip()

        try:
            return json.loads(content)
        except:
            pass

        start = content.find('{')
        end = content.rfind('}')
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(content[start:end+1])
            except:
                pass
        return None

    def _predict_with_strategy(self, train: np.ndarray, horizon: int,
                                period: int, strategy: Dict) -> Optional[np.ndarray]:
        """使用固定策略预测"""
        try:
            from src.skills.registry import SkillRegistry
            from run_benchmark import build_full_registry

            full_registry, _ = build_full_registry()
            stages = strategy.get('stages', [])

            if not stages:
                return None

            # 计算总步数并调整
            total_steps = sum(s.get('steps', 0) for s in stages)
            if total_steps != horizon:
                # 调整最后一步的步数
                last_stage = stages[-1]
                diff = horizon - total_steps
                if diff > 0:
                    last_stage['steps'] = last_stage.get('steps', 0) + diff
                elif diff < 0:
                    # 截断策略
                    pass

            predictions = []
            current_hist = train.copy()

            for stage in stages:
                steps = stage.get('steps', 0)
                weights = stage.get('weights', {})

                for _ in range(steps):
                    pred_val = 0.0
                    total_weight = 0.0
                    for skill_name, weight in weights.items():
                        skill = full_registry.get(skill_name)
                        if skill and weight > 0:
                            try:
                                forecast = skill.execute(current_hist, 1, period=period)
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

            return np.array(predictions[:horizon])
        except Exception as e:
            return None

    def _strategy_similarity(self, strategy: Dict, trajectory: List) -> float:
        """计算策略与轨迹的相似度"""
        try:
            strategy_weights = {}
            for stage in strategy.get('stages', []):
                for skill, weight in stage.get('weights', {}).items():
                    if skill not in strategy_weights:
                        strategy_weights[skill] = []
                    strategy_weights[skill].append(weight)

            avg_weights = {k: np.mean(v) for k, v in strategy_weights.items()}

            traj_weights = {}
            for step_info in trajectory:
                for skill, weight in step_info.get('weights', {}).items():
                    if skill not in traj_weights:
                        traj_weights[skill] = []
                    traj_weights[skill].append(weight)

            traj_avg = {k: np.mean(v) for k, v in traj_weights.items()}

            all_skills = set(avg_weights.keys()) | set(traj_avg.keys())
            if not all_skills:
                return 0.0

            v1 = np.array([avg_weights.get(s, 0) for s in all_skills])
            v2 = np.array([traj_avg.get(s, 0) for s in all_skills])

            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return float(np.dot(v1, v2) / (norm1 * norm2))
        except:
            return 0.0

    def _generate_rules_from_clusters(self, clustered_strategies: List) -> List:
        """从聚类结果生成规则"""
        rules = []
        for cluster in clustered_strategies:
            strategies = cluster.get('strategies', [])
            avg_mase = cluster.get('avg_mase', 0)

            if strategies:
                representative = strategies[0]
                # 生成条件
                feature_conditions = self._generate_condition_from_cluster(cluster)
                rules.append({
                    'condition': feature_conditions,
                    'params': {
                        'long_skill_force_threshold': 0.70,
                        'route_bonus_long_skill': 0.15,
                        'residual_acf_threshold': 0.20
                    },
                    'skill_strategy': representative,
                    'cluster_size': len(strategies),
                    'avg_mase': avg_mase
                })

        if not rules:
            rules.append({
                'condition': 'True',
                'params': {
                    'long_skill_force_threshold': 0.70,
                    'route_bonus_long_skill': 0.15,
                    'residual_acf_threshold': 0.20
                },
                'skill_strategy': self.window_best_strategies[0] if self.window_best_strategies else None,
                'cluster_size': 0,
                'avg_mase': 0
            })

        return rules

    def _generate_condition_from_cluster(self, cluster: Dict) -> str:
        """从聚类生成条件表达式"""
        centroid = cluster.get('centroid', {})
        if not centroid:
            return 'True'

        conditions = []
        key_features = ['period', 'trend_strength', 'seasonal_strength', 'adf_pvalue']
        for feat in key_features:
            if feat in centroid:
                val = centroid[feat]
                if feat == 'period':
                    conditions.append(f"period == {int(val)}")
                elif feat == 'trend_strength':
                    conditions.append(f"trend_strength > 0.5" if val > 0.5 else f"trend_strength <= 0.5")
                elif feat == 'seasonal_strength':
                    conditions.append(f"seasonal_strength > 0.5" if val > 0.5 else f"seasonal_strength <= 0.5")
                elif feat == 'adf_pvalue':
                    conditions.append(f"adf_pvalue < 0.05" if val < 0.05 else f"adf_pvalue >= 0.05")

        if not conditions:
            return 'True'
        return ' and '.join(conditions)

    def _default_rules(self) -> Dict:
        return {
            'rules': [
                {
                    'condition': 'True',
                    'params': {
                        'long_skill_force_threshold': 0.70,
                        'route_bonus_long_skill': 0.15,
                        'residual_acf_threshold': 0.20
                    }
                }
            ],
            'default': {
                'long_skill_force_threshold': 0.70,
                'route_bonus_long_skill': 0.15,
                'residual_acf_threshold': 0.20
            }
        }

    def save_rules(self, output_path: str):
        output_file = os.path.join(output_path, "generated_rules.json")
        os.makedirs(output_path, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'rules': self.rules,
                'default': {
                    'long_skill_force_threshold': 0.70,
                    'route_bonus_long_skill': 0.15,
                    'residual_acf_threshold': 0.20
                },
                'window_best_strategies': self.window_best_strategies
            }, f, ensure_ascii=False, indent=2)
        self.logger.log(f"📁 规则已保存: {output_file}")
        return output_file