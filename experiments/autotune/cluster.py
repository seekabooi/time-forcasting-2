# experiments/autotune/cluster.py
import numpy as np
from typing import Dict, List, Any
from collections import defaultdict
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')


class StrategyCluster:
    """策略聚类器"""

    def __init__(self, logger):
        self.logger = logger

    def cluster(self, strategies: List[Dict], n_clusters: int = 3) -> List[Dict]:
        """对策略进行聚类"""
        if len(strategies) <= 1:
            self.logger.log("   ⚠️ 策略数量太少，不进行聚类")
            return [{'centroid': {}, 'strategies': strategies, 'avg_mase': 0}]

        self.logger.log(f"   🔬 对 {len(strategies)} 个策略进行聚类 (目标簇数={n_clusters})")

        # ★ 收集所有策略中出现的技能
        all_skills = set()
        for s in strategies:
            for stage in s.get('stages', []):
                for skill in stage.get('weights', {}).keys():
                    all_skills.add(skill)
        all_skills = sorted(list(all_skills))  # 固定顺序
        skill_to_idx = {skill: i for i, skill in enumerate(all_skills)}

        # ★ 提取固定长度的向量
        vectors = []
        max_stages = 5  # 最多取前5个阶段

        for s in strategies:
            # 技能权重向量
            skill_vec = np.zeros(len(all_skills))
            stages = s.get('stages', [])
            # 取所有阶段的权重累加并归一化（简单表征整体技能偏好）
            total_weights = defaultdict(float)
            for stage in stages:
                for skill, weight in stage.get('weights', {}).items():
                    total_weights[skill] += weight
            if total_weights:
                norm = sum(total_weights.values())
                for skill in total_weights:
                    total_weights[skill] /= norm
            for skill, weight in total_weights.items():
                if skill in skill_to_idx:
                    skill_vec[skill_to_idx[skill]] = weight

            # 阶段步数向量（固定长度）
            steps_vec = np.zeros(max_stages)
            for i, stage in enumerate(stages[:max_stages]):
                steps_vec[i] = stage.get('steps', 0)

            # 合并为最终向量
            vec = np.concatenate([skill_vec, steps_vec])
            vectors.append(vec)

        if len(vectors) == 0:
            return [{'centroid': {}, 'strategies': strategies, 'avg_mase': 0}]

        X = np.array(vectors)  # 现在形状一致

        n_samples = len(X)
        actual_clusters = min(n_clusters, n_samples)

        if actual_clusters <= 1:
            return [{'centroid': {}, 'strategies': strategies, 'avg_mase': 0}]

        kmeans = KMeans(n_clusters=actual_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)

        # 按聚类分组
        clusters = defaultdict(list)
        for i, label in enumerate(labels):
            clusters[label].append(strategies[i])

        # 计算每个聚类的中心
        result = []
        for label, cluster_strategies in clusters.items():
            mases = [s.get('_mase', 0) for s in cluster_strategies]
            avg_mase = np.mean(mases) if mases else 0

            # 找聚类中心（最接近平均的策略）
            centroid_idx = np.argmin([abs(s.get('_mase', 0) - avg_mase) for s in cluster_strategies])
            centroid = cluster_strategies[centroid_idx] if centroid_idx < len(cluster_strategies) else cluster_strategies[0]

            result.append({
                'centroid': centroid,
                'strategies': cluster_strategies,
                'avg_mase': avg_mase,
                'size': len(cluster_strategies)
            })

            self.logger.log(f"      簇 {label+1}: {len(cluster_strategies)} 个策略, 平均MASE={avg_mase:.4f}")

        return result