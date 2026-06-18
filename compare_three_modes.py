# compare_three_modes.py
import os
import sys
import json
import hashlib
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime

from src.dataset.registry import DatasetRegistry
from src.dataset.loader import load_dataset
from src.agents.llm_planner import LLMPlannerAgent
from src.tasks.instance import TaskInstance
from run_benchmark import build_full_registry
from src.skills.data_profiler import DataProfiler
from experiments.autotune.utils import load_config


def load_data(dataset_name: str):
    registry = DatasetRegistry()
    ds_config = registry.get(dataset_name)
    if not ds_config:
        raise ValueError(f"Dataset {dataset_name} not found")
    df = load_dataset(ds_config)
    target_col = ds_config['target_column']
    series = df[target_col].values
    freq = ds_config.get('frequency', 'daily')
    return series, freq


def compute_metrics(pred, true, mase_scale):
    errors = pred - true
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))
    mape = np.mean(np.abs(errors / (true + 1e-8))) * 100
    smape = np.mean(2.0 * np.abs(errors) / (np.abs(pred) + np.abs(true) + 1e-8)) * 100
    mase = mae / mase_scale if mase_scale != 0 else np.nan
    rmsse = rmse / mase_scale if mase_scale != 0 else np.nan
    owa = (mase + rmsse) / 2 if not np.isnan(mase) and not np.isnan(rmsse) else np.nan
    return {'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'sMAPE': smape, 'MASE': mase, 'RMSSE': rmsse, 'OWA': owa}


def get_cache_key(start_pos, window_size, horizon, mode):
    """★ 修复：使用 mode 标识，而非文件内容哈希"""
    return f"{start_pos}_{window_size}_{horizon}_{mode}"


def predict_window(agent, train, horizon, freq, start_pos, window_size, mode, cache_dir):
    """带缓存的预测，mode = 'no_rule' | 'gen_rule' | 'ref_rule'"""
    cache_key = get_cache_key(start_pos, window_size, horizon, mode)
    cache_path = os.path.join(cache_dir, f"{cache_key}.npy")
    if os.path.exists(cache_path):
        return np.load(cache_path)

    task = TaskInstance(
        id="compare",
        dataset_id="compare",
        template_id="fixed_origin",
        question="",
        question_type="numerical",
        history=train.tolist(),
        horizon=horizon,
        frequency=freq,
        prediction_target={},
        resolution_date=datetime.now(),
        difficulty_level=1,
        ground_truth_extractor="",
        dates=None,
        target_date=""
    )
    pred = agent.predict(task)
    pred_array = np.array(pred)
    os.makedirs(cache_dir, exist_ok=True)
    np.save(cache_path, pred_array)
    return pred_array


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='melbourne_temp')
    parser.add_argument('--config', type=str, default='experiments/autotune/config.yaml')
    parser.add_argument('--output', type=str, default='storage/three_modes_comparison.csv')
    args = parser.parse_args()

    config = load_config(args.config)
    comp_cfg = config.get('comparison', {})
    window_size = comp_cfg.get('window_size', 600)
    horizon = comp_cfg.get('horizon', 7)
    step = comp_cfg.get('step', 150)          # ★ 确保与采集步长一致
    start = comp_cfg.get('start', 0)
    end = comp_cfg.get('end', None)

    # ★ 检查规则文件是否存在
    gen_rules_file = "storage/autotune_results/generated_rules.json"
    ref_rules_file = "storage/autotune_results/refined_rules.json"

    if not os.path.exists(ref_rules_file):
        print(f"⚠️ 警告: {ref_rules_file} 不存在，将使用 generated_rules.json 作为替代")
        ref_rules_file = gen_rules_file

    cache_dir = "storage/cache_compare"

    print("加载数据...")
    series, freq = load_data(args.dataset)
    n = len(series)
    max_start = n - window_size - horizon
    if end is not None and end < max_start:
        max_start = min(end, max_start)
    if start > max_start:
        start = max_start
    starts = list(range(start, max_start + 1, step))
    if not starts:
        print("数据太短，无法生成任何窗口。")
        return
    print(f"生成 {len(starts)} 个滑动窗口 (大小={window_size}, 步长={step})")

    full_registry, _ = build_full_registry()

    # ★ 创建三种 Agent，并确保规则文件存在
    agent_no_rule = LLMPlannerAgent(
        model="glm-4",
        skill_registry=full_registry,
        log_file=None,
        use_skills=True,
        verbose=False,
        rules_file=None
    )

    agent_gen_rule = LLMPlannerAgent(
        model="glm-4",
        skill_registry=full_registry,
        log_file=None,
        use_skills=True,
        verbose=False,
        rules_file=gen_rules_file
    )

    agent_ref_rule = LLMPlannerAgent(
        model="glm-4",
        skill_registry=full_registry,
        log_file=None,
        use_skills=True,
        verbose=False,
        rules_file=ref_rules_file
    )

    results_no_rule = []
    results_gen_rule = []
    results_ref_rule = []

    for idx, start_pos in enumerate(tqdm(starts, desc="处理窗口")):
        train = series[start_pos:start_pos + window_size]
        test = series[start_pos + window_size:start_pos + window_size + horizon]

        period = DataProfiler._auto_period(train, freq=freq)
        if period > 1 and len(train) >= 2 * period:
            seasonal_errors = np.abs(train[period:] - train[:-period])
            mase_scale = np.mean(seasonal_errors) if len(seasonal_errors) > 0 else 1.0
        else:
            naive_errors = np.abs(np.diff(train))
            mase_scale = np.mean(naive_errors) if len(naive_errors) > 0 else 1.0

        # ★ 三种模式预测（使用 mode 标识区分缓存）
        pred_no = predict_window(agent_no_rule, train, horizon, freq, start_pos, window_size, "no_rule", cache_dir)
        pred_gen = predict_window(agent_gen_rule, train, horizon, freq, start_pos, window_size, "gen_rule", cache_dir)
        pred_ref = predict_window(agent_ref_rule, train, horizon, freq, start_pos, window_size, "ref_rule", cache_dir)

        results_no_rule.append(compute_metrics(pred_no, test, mase_scale))
        results_gen_rule.append(compute_metrics(pred_gen, test, mase_scale))
        results_ref_rule.append(compute_metrics(pred_ref, test, mase_scale))

    # 汇总
    df_no = pd.DataFrame(results_no_rule)
    df_gen = pd.DataFrame(results_gen_rule)
    df_ref = pd.DataFrame(results_ref_rule)

    summary_no = df_no.agg(['mean', 'std']).round(4)
    summary_gen = df_gen.agg(['mean', 'std']).round(4)
    summary_ref = df_ref.agg(['mean', 'std']).round(4)

    compare = pd.DataFrame({
        'NoRule_Mean': summary_no.loc['mean'],
        'NoRule_Std': summary_no.loc['std'],
        'GenRule_Mean': summary_gen.loc['mean'],
        'GenRule_Std': summary_gen.loc['std'],
        'RefRule_Mean': summary_ref.loc['mean'],
        'RefRule_Std': summary_ref.loc['std'],
        'Gen_vs_No_%': ((summary_no.loc['mean'] - summary_gen.loc['mean']) / summary_no.loc['mean'] * 100).round(2),
        'Ref_vs_No_%': ((summary_no.loc['mean'] - summary_ref.loc['mean']) / summary_no.loc['mean'] * 100).round(2),
        'Ref_vs_Gen_%': ((summary_gen.loc['mean'] - summary_ref.loc['mean']) / summary_gen.loc['mean'] * 100).round(2)
    })

    print("\n" + "=" * 100)
    print("📊 三种模式对比结果（无规则 vs 第一步规则 vs 第二步规则）")
    print(f"窗口数量: {len(starts)}")
    print("=" * 100)
    print(compare.to_string())
    print("=" * 100)

    compare.to_csv(args.output)
    print(f"📁 结果已保存至: {args.output}")
    print(f"📁 预测缓存保存在: {cache_dir}")


if __name__ == "__main__":
    main()