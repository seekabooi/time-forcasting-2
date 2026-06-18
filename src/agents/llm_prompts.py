import numpy as np


def build_prompt(profile, history, candidates, local_errors, LONG_SKILLS, step_counter):
    seas = profile.get('seasonal_strength', 0)
    trend = profile.get('trend_strength', 0)
    period = profile.get('period', 12)
    local_slope = profile.get('local_slope', 0.0)
    change_detected = profile.get('change_point_detected', False)
    data_len = len(history)
    adf_pvalue = profile.get('adf_pvalue', 0.5)
    missing_rate = profile.get('missing_rate', 0.0)
    recent_volatility = profile.get('recent_volatility', 1.0)
    acf_peak_lag = profile.get('acf_peak_lag', 0)
    diff_adf_pvalue = profile.get('diff_adf_pvalue', 0.5)
    sample_entropy = profile.get('sample_entropy', 0.0)
    spectral_entropy = profile.get('spectral_entropy', 0.0)
    fft_peak_freq = profile.get('fft_peak_freq', 0.0)
    acf_365 = profile.get('acf_365', 0.0)

    month = profile.get('month_of_year', 0)
    year = profile.get('year', 0)
    quarter = profile.get('quarter', 0)
    is_month_end = profile.get('is_month_end', False)
    days_from_start = profile.get('days_from_start', data_len)

    recent_points = list(history[-10:])
    recent_str = ", ".join([f"{v:.1f}" for v in recent_points])
    if len(history) >= 5:
        win_mean = np.mean(history[-5:])
        win_std = np.std(history[-5:])
        win_trend = "上升" if history[-1] > history[-5] else "下降"
    else:
        win_mean, win_std, win_trend = 0.0, 0.0, "未知"

    skill_info_lines = []
    error_list = []
    for c in candidates:
        sk = c['skill']
        mae = local_errors.get(sk.name)
        mae_str = f"{mae:.10f}" if mae is not None else "未计算"
        error_list.append((sk.name, mae))
        min_data = sk.min_data_points
        full_hist = "是" if sk.requires_full_history else "否"
        tags = ", ".join(sk.strength_tags) if sk.strength_tags else "通用"
        hint = sk.decision_hint if sk.decision_hint else "无特殊建议"
        skill_info_lines.append(
            f"- {sk.name}: MAE={mae_str} | 最少数据:{min_data} | 需全历史:{full_hist} | 擅长:{tags}\n  使用建议: {hint}"
        )
    skill_info = "\n".join(skill_info_lines)

    valid_errors = [(n, e) for n, e in error_list if e is not None]
    error_hint = ""
    if len(valid_errors) >= 2:
        best_name, best_mae = min(valid_errors, key=lambda x: x[1])
        worst_name, worst_mae = max(valid_errors, key=lambda x: x[1])
        error_hint = f"最低MAE技能: {best_name} ({best_mae:.10f}), 最高MAE技能: {worst_name} ({worst_mae:.10f})。\n"

    season_hint = ""
    if seas > 0.5:
        season_hint = f"序列季节性较强（{seas:.10f}）。建议选择适合季节性的技能，并可考虑多个技能加权组合。\n"

    calendar_hint = ""
    if seas > 0.5 and data_len >= 24 and profile.get('has_dates', False):
        calendar_hint = "日历技能（calendar）可作为辅助，与其他技能组合。\n"

    precision_hint = "请输出精确到十位小数的权重（如 0.7234567890, 0.1867234567）。"

    performance_hint = ""

    date_info = ""
    if profile.get('has_dates', False):
        date_info = f"- 当前时间点：{year}年{month}月 (Q{quarter})，{'月末' if is_month_end else '非月末'}，距起始 {days_from_start} 天\n"

    long_hint = ""
    if data_len > 400:
        rec_names = ", ".join(LONG_SKILLS)
        long_hint = f"💡 提示：对于长度>400的长序列，{rec_names} 等技能在多步预测上表现优异，可以分配较高权重（如0.5~0.8）。\n"

    # ★ 从 profile 中获取规则策略参考
    rule_strategy = profile.get('rule_strategy', None)
    rule_hint = ""
    if rule_strategy:
        rule_hint = f"\n📋 参考策略（由离线规则库提供，可作为决策参考）：{rule_strategy}\n"

    prompt = f"""你是时间序列预测专家。请根据以下特征决定最佳预测方案，使用 1~3 个技能加权组合 。

序列特征：
- 长度:{data_len}，季节强度:{seas:.10f}，趋势强度:{trend:.10f}，周期:{period}
- 平稳性(ADF p-value):{adf_pvalue:.6f} (越小越平稳)
- 差分后平稳性(1阶差分 ADF):{diff_adf_pvalue:.6f}
- 缺失率:{missing_rate:.4f}
- 近期波动比(最近5点/整体):{recent_volatility:.4f}
- 局部斜率:{local_slope:.10f}，突变:{change_detected}
- 自相关峰值 lag:{acf_peak_lag}，年自相关(365):{acf_365:.4f}
- 样本熵(复杂度):{sample_entropy:.4f}，频谱熵:{spectral_entropy:.4f}
- 主频(FFT峰值):{fft_peak_freq:.4f} (0~0.5，高频表示短期波动)
- 近期5点均值:{win_mean:.10f}，波动:{win_std:.10f}，走势:{win_trend}
- 最近10点: {recent_str}
{date_info}
{rule_hint}
候选技能对比：
{skill_info}
{error_hint}{season_hint}{calendar_hint}{precision_hint}{performance_hint}
{long_hint}

要求：
1. 输出 JSON 格式，包含两个字段：
   - "skill_weights": 技能名称到权重的字典，权重十位小数，总和为1。
   - "replan_interval": 整数 (1~5)，表示多少步后需要强制重新决策（基于当前序列的稳定性，步数越短重决策越频繁）。
2. 输出示例：{{"skill_weights": {{"multi_resolution": 0.7738936947, "calendar": 0.3287439022}}, "replan_interval": 3}}
3. 不要输出任何解释。"""
    return prompt


def build_preprocess_prompt(profile: dict, history: np.ndarray) -> str:
    """构建预处理选择的Prompt"""
    prompt = f"""你是时序数据预处理专家。请根据以下数据特征，选择最合适的预处理方法。

数据特征：
- 偏度 (Skewness): {profile.get('skewness', 0):.3f} (|>1| 表示严重偏斜，建议 Box-Cox)
- 变异系数 (CV = std/mean): {profile.get('cv', 0):.3f} (>0.5 表示量级差异大，建议 Z-score)
- 趋势强度: {profile.get('trend_strength', 0):.3f} (>0.6 建议线性去趋势)
- 最小值: {np.min(history):.3f} (<=0 则不能使用 Box-Cox)
- ADF p-value: {profile.get('adf_pvalue', 0.5):.3f} (>0.05 非平稳)

可选预处理方法（只选一个）：
1. identity_pre: 原始数据直接输入
2. zscore_normalize: 标准化 (x-μ)/σ
3. linear_detrend: 线性去趋势
4. boxcox_transform: Box-Cox幂变换 (需数据全为正)

输出 JSON 格式: {{"preprocess": "方法名"}}
不要输出任何解释。"""
    return prompt


def build_post_enhance_prompt(profile: dict, residual_stats: dict, horizon: int) -> str:
    """构建后处理增强选择的Prompt（残差修正/校准）"""
    acf1 = residual_stats.get('acf_lag1', 0.0)
    var_ratio = residual_stats.get('var_ratio', 1.0)

    prompt = f"""你是时序预测后处理专家。当前主干模型已生成预测，请决定是否需要增强修正。

残差诊断指标（基于验证集）：
- 残差 lag-1 自相关: {acf1:.3f} (若 >0.3，说明残差有规律，建议启用残差修正)
- 近期残差方差比 (最近10点/全部): {var_ratio:.3f} (若 >2.0，近期误差增大，建议启用)
- 预测长度: {horizon} (若 >50，不建议长期残差外推)

可选后处理增强（只选一个）：
1. enhance_identity: 不做任何修正 (推荐用于残差为白噪声)
2. residual_ar: AR(1)残差修正 (推荐用于残差有短期自相关)
3. quantile_calibration: 分位数校准 (推荐用于存在系统偏差)

输出 JSON 格式: {{"enhance_skill": "增强方法名"}}
不要输出任何解释。"""
    return prompt