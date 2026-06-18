# experiments/autotune/iterative_refiner.py
import os
import sys
import json
import argparse
import pandas as pd
import shutil
from datetime import datetime
from experiments.autotune.utils import ProgressLogger, load_config
from experiments.autotune.performance_auditor import PerformanceAuditor
from experiments.autotune.rule_quality_evaluator import RuleQualityEvaluator
from experiments.autotune.meta_cluster import MetaCluster
from experiments.autotune.hard_refiner import HardRefiner


class IterativeRefiner:
    def __init__(self, config_path: str = None, verbose: bool = False):
        print("🔧 初始化优化器...")
        self.config = load_config(config_path)
        if not self.config:
            print("❌ 加载配置失败")
            sys.exit(1)
        self.verbose = verbose
        self.logger = ProgressLogger(
            log_dir=self.config.get('output_dir', 'storage/autotune_results') + '/logs',
            verbose=verbose
        )
        self.logger.start_log("iterative_refiner")
        self.output_dir = self.config.get('output_dir', 'storage/autotune_results')
        self.stop_ratio = self.config.get('stop_ratio', 0.01)
        self.max_rules = self.config.get('max_rules', 5)
        self.max_rounds = 3
        print(f"📁 输出目录: {self.output_dir}")

    def run(self, dataset_name: str, horizon: int, rounds: int = 3):
        self.logger.log("=" * 70)
        self.logger.log("🔄 启动多轮闭环优化（LLM驱动决策）")
        self.logger.log(f"📅 {datetime.now()}")
        self.logger.log("=" * 70)
        print("🔄 开始多轮闭环优化（LLM驱动决策）...")

        csv_path = os.path.join(self.output_dir, "collected_windows.csv")
        if not os.path.exists(csv_path):
            msg = f"❌ 未找到采集数据: {csv_path}"
            self.logger.log(msg)
            print(msg)
            return

        collected_df = pd.read_csv(csv_path)
        rules_path = os.path.join(self.output_dir, "generated_rules.json")
        if not os.path.exists(rules_path):
            msg = f"❌ 未找到规则文件: {rules_path}"
            self.logger.log(msg)
            print(msg)
            return

        with open(rules_path, 'r', encoding='utf-8') as f:
            rules_data = json.load(f)
        rules = rules_data.get('rules', [])
        window_best = rules_data.get('window_best_strategies', [])
        print(f"📋 初始规则数: {len(rules)}")

        auditor = PerformanceAuditor(self.config, self.logger)
        evaluator = RuleQualityEvaluator(self.config, self.logger)
        meta = MetaCluster(self.config, self.logger)
        refiner = HardRefiner(self.config, self.logger)

        previous_avg_mase = float('inf')
        no_improvement_count = 0

        for round_num in range(1, rounds + 1):
            print(f"\n🔄 第 {round_num} 轮")
            self.logger.log(f"\n{'='*50}")
            self.logger.log(f"🔄 第 {round_num} 轮")
            self.logger.log(f"{'='*50}")

            # 1. 审计
            hard_ids, report = auditor.audit(collected_df, {'rules': rules})
            avg_mase = report['avg_mase']
            hard_count = report['hard_count']
            total_windows = report['total_windows']
            hard_ratio = hard_count / max(1, total_windows)

            # 2. 检查是否满足停止条件
            if hard_ratio < self.stop_ratio:
                self.logger.log(f"✅ 困难窗口占比 {hard_ratio:.2%} < {self.stop_ratio:.0%}，自动停止")
                print(f"✅ 规则已足够好，停止优化")
                break

            # 3. LLM 决策
            action, info = evaluator.decide_action(rules, report)
            if action == 'stop':
                print("✅ LLM建议停止优化")
                break

            elif action == 'merge':
                # ★ 使用基于特征聚类的自动合并（LLM主导）
                self.logger.log("🧠 执行基于特征聚类的自动合并（LLM主导）...")
                # ★ 修正方法名：auto_merge_by_features
                new_rules = meta.auto_merge_by_features(rules)
                if len(new_rules) < len(rules):
                    # 验证合并后的效果
                    _, new_report = auditor.audit(collected_df, {'rules': new_rules})
                    new_avg = new_report['avg_mase']
                    if new_avg < avg_mase:
                        rules = new_rules
                        print(f"📈 合并后MASE: {avg_mase:.4f} → {new_avg:.4f}")
                    else:
                        print(f"⚠️ 合并无改善，回滚")
                else:
                    print("⚠️ 合并无变化，保留原规则")

            elif action == 'patch':
                targets = info.get('targets', [])
                if not targets:
                    targets = report.get('worst_3_window_ids', [])
                if targets:
                    patch_rules = refiner.refine(collected_df, targets, rules)
                    if patch_rules:
                        combined_rules = patch_rules + rules
                        _, new_report = auditor.audit(collected_df, {'rules': combined_rules})
                        new_avg = new_report['avg_mase']
                        if new_avg < avg_mase and len(combined_rules) <= self.max_rules:
                            rules = combined_rules
                            print(f"📈 补丁后MASE: {avg_mase:.4f} → {new_avg:.4f}")
                        else:
                            print(f"⚠️ 补丁无改善或超限，回滚")
                else:
                    print("⚠️ 无目标窗口，跳过补丁")

            # 检查改进
            if avg_mase < previous_avg_mase:
                previous_avg_mase = avg_mase
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                if no_improvement_count >= 2:
                    print("⚠️ 连续两轮无改善，停止优化")
                    break

            # 保存中间结果
            temp_path = os.path.join(self.output_dir, f"rules_round_{round_num}.json")
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump({'rules': rules, 'window_best_strategies': window_best}, f, indent=2)

        final_path = os.path.join(self.output_dir, "refined_rules.json")
        with open(final_path, 'w', encoding='utf-8') as f:
            json.dump({'rules': rules, 'window_best_strategies': window_best}, f, indent=2)
        shutil.copy(final_path, rules_path)
        print(f"📁 最终规则已保存: {final_path}")
        self.logger.log("✅ 多轮优化完成")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='melbourne_temp')
    parser.add_argument('--horizon', type=int, default=7)
    parser.add_argument('--rounds', type=int, default=3, help='最大优化轮数')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    print("🚀 启动迭代优化器...")
    refiner = IterativeRefiner(verbose=args.verbose)
    refiner.run(args.dataset, args.horizon, args.rounds)


if __name__ == '__main__':
    main()