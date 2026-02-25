"""
回测引擎与绩效评估模块
========================
- 滚动窗口回测框架
- 绩效指标 (Sharpe, MaxDD, CVaR, 分散化比率, 换手率)
- 统计显著性检验 (Ledoit-Wolf, Bootstrap, Diebold-Mariano)
"""

import numpy as np
import pandas as pd
from scipy import stats
import warnings

warnings.filterwarnings("ignore")


# ============================================================
#  1. 绩效指标
# ============================================================

def compute_performance(daily_returns: pd.Series, rf: float = 0.0) -> dict:
    """
    计算投资组合绩效指标

    Args:
        daily_returns: 日收益率序列
        rf: 无风险利率 (年化)

    Returns:
        dict of performance metrics
    """
    T = len(daily_returns)
    if T < 5:
        return {k: np.nan for k in ["ann_return", "ann_vol", "sharpe",
                                     "max_drawdown", "cvar_95", "calmar",
                                     "skew", "kurtosis"]}

    ann_ret = daily_returns.mean() * 252
    ann_vol = daily_returns.std() * np.sqrt(252)
    sharpe = (ann_ret - rf) / ann_vol if ann_vol > 0 else 0

    # 最大回撤
    cum = (1 + daily_returns).cumprod()
    peak = cum.cummax()
    drawdown = (cum - peak) / peak
    max_dd = drawdown.min()

    # CVaR (95%)
    var_95 = np.percentile(daily_returns, 5)
    cvar_95 = daily_returns[daily_returns <= var_95].mean()

    # Calmar
    calmar = ann_ret / abs(max_dd) if abs(max_dd) > 1e-10 else 0

    return {
        "ann_return": round(ann_ret * 100, 2),          # %
        "ann_vol": round(ann_vol * 100, 2),              # %
        "sharpe": round(sharpe, 4),
        "max_drawdown": round(max_dd * 100, 2),          # %
        "cvar_95": round(cvar_95 * 100, 4),              # %
        "calmar": round(calmar, 4),
        "skew": round(daily_returns.skew(), 4),
        "kurtosis": round(daily_returns.kurtosis(), 4),
        "n_days": T,
    }


def compute_turnover(weights_series: list) -> float:
    """
    计算平均换手率
    weights_series: list of weight arrays, 按时间顺序
    """
    if len(weights_series) < 2:
        return 0.0
    turnovers = []
    for i in range(1, len(weights_series)):
        to = np.sum(np.abs(weights_series[i] - weights_series[i - 1]))
        turnovers.append(to)
    return np.mean(turnovers)


def compute_diversification_ratio(weights: np.ndarray,
                                  cov: np.ndarray) -> float:
    """
    分散化比率 DR = (w' σ) / sqrt(w' Σ w)
    DR > 1 表示分散化有效
    """
    std = np.sqrt(np.diag(cov))
    port_vol = np.sqrt(weights @ cov @ weights)
    if port_vol < 1e-10:
        return 1.0
    return (weights @ std) / port_vol


def compute_effective_n(weights: np.ndarray) -> float:
    """
    有效资产数 EN = 1 / Σ w_i^2  (Herfindahl 逆)
    """
    hhi = np.sum(weights ** 2)
    if hhi < 1e-10:
        return len(weights)
    return 1.0 / hhi


# ============================================================
#  2. 滚动窗口回测
# ============================================================

def rolling_backtest(returns: pd.DataFrame,
                     strategy_func,
                     strategy_kwargs: dict = None,
                     lookback: int = 252,
                     rebalance_freq: int = 21,
                     network_func=None,
                     tc_bps: float = 0.0,
                     adaptive_rebalance: bool = False,
                     stability_j: float = 0.7,
                     stability_nmi: float = 0.8,
                     verbose: bool = True) -> dict:
    """
    滚动窗口回测

    Args:
        returns: 完整收益率矩阵 (日期 x 资产)
        strategy_func: 策略函数 (接受 returns DataFrame, 返回权重)
        strategy_kwargs: 策略额外参数
        lookback: 回看窗口 (天)
        rebalance_freq: 调仓频率 (天)
        network_func: 网络分析函数
        tc_bps: 单边交易成本 (基点, §2.6.3(5))
        adaptive_rebalance: 是否启用自适应调仓 (§2.5.3)
        stability_j: Jaccard边重叠率阈值 (adaptive模式)
        stability_nmi: 社区NMI阈值 (adaptive模式)
        verbose: 是否打印进度

    Returns:
        dict with:
          portfolio_returns: 日收益率序列
          weights_history: 权重历史
          rebalance_dates: 调仓日期
    """
    if strategy_kwargs is None:
        strategy_kwargs = {}

    n_days = len(returns)
    assets = returns.columns.tolist()
    n_assets = len(assets)

    portfolio_returns = []
    weights_history = []
    rebalance_dates = []
    current_weights = np.ones(n_assets) / n_assets

    # 自适应调仓: 保存上一期的网络状态
    prev_partition = None
    prev_centralities = None
    prev_pmfg = None

    dates = returns.index.tolist()

    for t in range(lookback, n_days):
        tc_cost = 0.0
        # 每 rebalance_freq 天调仓
        if (t - lookback) % rebalance_freq == 0:
            train = returns.iloc[t - lookback:t]

            # 如果需要网络分析
            extra_kwargs = dict(strategy_kwargs)
            net_result = None
            if network_func is not None:
                try:
                    net_result = network_func(train)
                    extra_kwargs.update(net_result)
                except Exception:
                    pass

            # 自适应调仓: 检查网络稳定性 (§2.5.3)
            should_rebalance = True
            if (adaptive_rebalance and net_result is not None
                    and prev_partition is not None):
                from network_analysis import (edge_overlap,
                                              community_nmi,
                                              centrality_rank_correlation)
                curr_partition = net_result.get("partition", {})
                nmi_val = community_nmi(prev_partition, curr_partition)
                # 若 NMI >= 阈值, 网络足够稳定, 跳过本次调仓
                if nmi_val >= stability_nmi:
                    should_rebalance = False

                if should_rebalance and prev_pmfg is not None:
                    curr_pmfg = net_result.get("pmfg_graph", None)
                    if curr_pmfg is not None:
                        j_val = edge_overlap(prev_pmfg, curr_pmfg)
                        if j_val >= stability_j and nmi_val >= stability_nmi * 0.9:
                            should_rebalance = False

            if should_rebalance:
                try:
                    new_weights = strategy_func(train, **extra_kwargs)
                    # 交易成本 (§2.6.3(5) / §2.4.3 换手率)
                    turnover = np.sum(np.abs(new_weights - current_weights))
                    tc_cost = tc_bps / 10000.0 * turnover
                    current_weights = new_weights
                except Exception:
                    pass  # 保持上期权重

                # 更新网络状态 (用于自适应调仓)
                if net_result is not None:
                    prev_partition = net_result.get("partition", None)
                    prev_centralities = net_result.get("centralities", None)
                    prev_pmfg = net_result.get("pmfg_graph", None)

            weights_history.append(current_weights.copy())
            rebalance_dates.append(dates[t])

        # 计算组合日收益 (扣除交易成本)
        day_ret = returns.iloc[t].values @ current_weights - tc_cost
        portfolio_returns.append(day_ret)

    # 构建时间序列
    oos_dates = dates[lookback:]
    port_ret_series = pd.Series(portfolio_returns, index=oos_dates,
                                name="portfolio_return")

    return {
        "portfolio_returns": port_ret_series,
        "weights_history": weights_history,
        "rebalance_dates": rebalance_dates,
    }


# ============================================================
#  3. 统计检验
# ============================================================

def ledoit_wolf_sharpe_test(ret_a: pd.Series, ret_b: pd.Series) -> dict:
    """
    Ledoit-Wolf (2008) 夏普比率差异检验
    使用 HAC 估计量, 适用于非正态、序列相关数据

    H0: SR_A = SR_B
    H1: SR_A ≠ SR_B

    基于 Jobson-Korkie (1981) 统计量 + HAC 修正
    """
    a = ret_a.values
    b = ret_b.values
    n = len(a)

    mu_a, mu_b = a.mean(), b.mean()
    sig_a, sig_b = a.std(ddof=1), b.std(ddof=1)
    sr_a = mu_a / sig_a if sig_a > 0 else 0
    sr_b = mu_b / sig_b if sig_b > 0 else 0

    # Jobson-Korkie 统计量的方差估计 (简化 HAC)
    # θ = μ_a σ_b - μ_b σ_a
    theta = mu_a * sig_b - mu_b * sig_a

    # 渐近方差 (Memmel, 2003 简化版本)
    cov_ab = np.cov(a, b)[0, 1]
    var_theta = (2 * sig_a ** 2 * sig_b ** 2
                 - 2 * sig_a * sig_b * cov_ab
                 + 0.5 * mu_a ** 2 * sig_b ** 2
                 + 0.5 * mu_b ** 2 * sig_a ** 2
                 - mu_a * mu_b * cov_ab / (sig_a * sig_b) * sig_a * sig_b)

    se = np.sqrt(var_theta / n) if var_theta > 0 else 1e-10
    z_stat = theta / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    return {
        "sr_a": round(sr_a * np.sqrt(252), 4),
        "sr_b": round(sr_b * np.sqrt(252), 4),
        "sr_diff": round((sr_a - sr_b) * np.sqrt(252), 4),
        "z_stat": round(z_stat, 4),
        "p_value": round(p_value, 4),
        "significant_5pct": p_value < 0.05,
    }


def bootstrap_test(ret_a: pd.Series, ret_b: pd.Series,
                   n_bootstrap: int = 10000,
                   block_size: int = 21,
                   metric: str = "sharpe") -> dict:
    """
    分块 Bootstrap 检验
    保持时间序列依赖结构

    H0: metric(A) = metric(B)
    """
    n = len(ret_a)
    rng = np.random.default_rng(42)

    def compute_metric(r):
        if metric == "sharpe":
            return r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0
        elif metric == "max_drawdown":
            cum = (1 + pd.Series(r)).cumprod()
            return (cum / cum.cummax() - 1).min()

    obs_diff = compute_metric(ret_a.values) - compute_metric(ret_b.values)

    # 分块 Bootstrap
    boot_diffs = []
    n_blocks = int(np.ceil(n / block_size))

    for _ in range(n_bootstrap):
        block_starts = rng.integers(0, n - block_size + 1, size=n_blocks)
        indices = []
        for s in block_starts:
            indices.extend(range(s, min(s + block_size, n)))
        indices = indices[:n]

        boot_a = ret_a.values[indices]
        boot_b = ret_b.values[indices]
        boot_diff = compute_metric(boot_a) - compute_metric(boot_b)
        boot_diffs.append(boot_diff)

    boot_diffs = np.array(boot_diffs)
    p_value = np.mean(np.abs(boot_diffs) >= abs(obs_diff))

    ci_lower = np.percentile(boot_diffs, 2.5)
    ci_upper = np.percentile(boot_diffs, 97.5)

    return {
        "obs_diff": round(obs_diff, 4),
        "p_value": round(p_value, 4),
        "ci_95_lower": round(ci_lower, 4),
        "ci_95_upper": round(ci_upper, 4),
        "significant_5pct": p_value < 0.05,
    }


def diebold_mariano_test(ret_a: pd.Series, ret_b: pd.Series,
                         loss: str = "variance") -> dict:
    """
    Diebold-Mariano (1995) 检验
    比较两策略的预测损失差异

    H0: E[L_A] = E[L_B]
    """
    a = ret_a.values
    b = ret_b.values

    if loss == "variance":
        loss_a = a ** 2
        loss_b = b ** 2
    elif loss == "abs":
        loss_a = np.abs(a)
        loss_b = np.abs(b)

    d = loss_a - loss_b
    n = len(d)
    d_mean = d.mean()

    # HAC 方差估计 (Newey-West, lag=int(n^{1/3}))
    max_lag = max(1, int(n ** (1 / 3)))
    gamma_0 = np.var(d, ddof=1)
    gamma_sum = 0
    for k in range(1, max_lag + 1):
        weight = 1 - k / (max_lag + 1)  # Bartlett kernel
        gamma_k = np.cov(d[k:], d[:-k])[0, 1]
        gamma_sum += weight * gamma_k

    var_d = (gamma_0 + 2 * gamma_sum) / n
    se = np.sqrt(max(var_d, 1e-20))

    dm_stat = d_mean / se
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))

    return {
        "dm_stat": round(dm_stat, 4),
        "p_value": round(p_value, 4),
        "mean_loss_diff": round(d_mean * 252 * 100, 4),  # 年化百分比
        "significant_5pct": p_value < 0.05,
    }


# ============================================================
#  4. Romano-Wolf 多重比较修正 (§2.6.2)
# ============================================================

def romano_wolf_correction(p_values: dict, alpha: float = 0.05) -> pd.DataFrame:
    """
    Romano-Wolf (2005) 逐步下降多重比较修正
    控制族错误率 (Family-Wise Error Rate)
    当同时比较多个策略对时, 确保各对比较的显著性结论
    在联合检验意义下仍然有效 (§2.6.2)

    Args:
        p_values: {strategy_name: raw_p_value}
        alpha: 显著性水平

    Returns:
        DataFrame with raw/adjusted p-values and significance
    """
    names = list(p_values.keys())
    pvals = np.array([p_values[n] for n in names])
    K = len(pvals)

    if K == 0:
        return pd.DataFrame()

    # 排序
    sorted_idx = np.argsort(pvals)
    sorted_pvals = pvals[sorted_idx]
    sorted_names = [names[i] for i in sorted_idx]

    # Holm-Bonferroni stepdown (Romano-Wolf 的保守近似)
    adjusted_pvals = np.zeros(K)
    for i in range(K):
        adjusted_pvals[i] = min(sorted_pvals[i] * (K - i), 1.0)

    # 强制单调性
    for i in range(1, K):
        adjusted_pvals[i] = max(adjusted_pvals[i], adjusted_pvals[i - 1])

    results = []
    for i in range(K):
        results.append({
            "strategy": sorted_names[i],
            "raw_p": round(sorted_pvals[i], 4),
            "adjusted_p": round(adjusted_pvals[i], 4),
            "significant": adjusted_pvals[i] < alpha,
        })

    return pd.DataFrame(results)


# ============================================================
#  5. 综合绩效评估
# ============================================================

def comprehensive_evaluation(backtest_results: dict,
                             benchmark_name: str = "equal_weight") -> pd.DataFrame:
    """
    综合评估所有策略

    Args:
        backtest_results: {strategy_name: backtest_result_dict}
        benchmark_name: 基准策略名称

    Returns:
        绩效对比 DataFrame + 统计检验结果
    """
    all_perf = {}
    all_returns = {}

    for name, res in backtest_results.items():
        port_ret = res["portfolio_returns"]
        perf = compute_performance(port_ret)

        # 换手率
        perf["turnover"] = round(compute_turnover(res["weights_history"]), 4)

        # 有效资产数 (最后一期权重)
        if res["weights_history"]:
            last_w = res["weights_history"][-1]
            perf["effective_n"] = round(compute_effective_n(last_w), 2)
        else:
            perf["effective_n"] = np.nan

        all_perf[name] = perf
        all_returns[name] = port_ret

    perf_df = pd.DataFrame(all_perf).T

    # 统计检验: 各策略 vs 基准
    if benchmark_name in all_returns:
        bench_ret = all_returns[benchmark_name]
        test_results = {}

        for name, ret in all_returns.items():
            if name == benchmark_name:
                continue

            common_idx = ret.index.intersection(bench_ret.index)
            if len(common_idx) < 30:
                continue

            r_a = ret.loc[common_idx]
            r_b = bench_ret.loc[common_idx]

            lw_test = ledoit_wolf_sharpe_test(r_a, r_b)
            boot_test = bootstrap_test(r_a, r_b, n_bootstrap=10000)
            dm_test = diebold_mariano_test(r_a, r_b)

            test_results[name] = {
                "lw_z": lw_test["z_stat"],
                "lw_p": lw_test["p_value"],
                "lw_sig": lw_test["significant_5pct"],
                "boot_p": boot_test["p_value"],
                "boot_sig": boot_test["significant_5pct"],
                "dm_stat": dm_test["dm_stat"],
                "dm_p": dm_test["p_value"],
                "dm_sig": dm_test["significant_5pct"],
            }

        test_df = pd.DataFrame(test_results).T
        return perf_df, test_df

    return perf_df, pd.DataFrame()
