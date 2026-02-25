"""
主控脚本: 基于图模型的资产配置策略 - 完整分析
==============================================
串联所有模块, 生成论文所需的全部表格和图表

输出目录: results/
  ├── tables/           # 论文表格 (CSV)
  ├── figures/          # 论文图表 (PNG)
  └── logs/             # 运行日志

运行方式:
    python run_full_analysis.py
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rcParams
import warnings

warnings.filterwarnings("ignore")

# 中文字体
rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

# 导入自定义模块
from network_analysis import (
    compute_correlation, corr_to_distance, corr_to_distance_mi,
    build_mst, build_pmfg, build_glasso_network,
    compute_centralities, detect_communities, compute_modularity,
    community_summary, rmt_test, random_network_test,
    degree_preserving_rewiring_test,
    edge_overlap, centrality_rank_correlation, community_nmi,
    full_network_analysis,
)
from portfolio_strategies import (
    equal_weight, mean_variance, minimum_variance, max_diversification,
    risk_parity, hrp,
    black_litterman_no_views, black_litterman_momentum,
    inverse_centrality, direct_centrality, pagerank_weight,
    network_regularized_rp,
    graph_enhanced_bl, denoised_rp, community_bl_rp,
    graph_laplacian_mv, pmfg_filtered_rp, graph_smoothed_bl,
    network_constrained_minvar, community_minvar,
    select_gamma_cv, STRATEGY_REGISTRY, compute_weights,
)
from backtest_engine import (
    compute_performance, compute_turnover,
    compute_diversification_ratio, compute_effective_n,
    rolling_backtest, comprehensive_evaluation,
    ledoit_wolf_sharpe_test, bootstrap_test, diebold_mariano_test,
    romano_wolf_correction,
)


# ============================================================
#  配置
# ============================================================

DATA_ROOT = "data"
RESULTS_DIR = "results"
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
YEARS = ["2023", "2024", "2025"]

# 回测参数
LOOKBACK = 252        # 滚动窗口1年
REBALANCE_FREQ = 21   # 每月调仓

for d in [RESULTS_DIR, TABLES_DIR, FIGURES_DIR]:
    os.makedirs(d, exist_ok=True)


# ============================================================
#  辅助函数
# ============================================================

def load_returns(year: str) -> pd.DataFrame:
    """加载某年收益率数据"""
    path = os.path.join(DATA_ROOT, year, f"returns_{year}.csv")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df


def load_prices(year: str) -> pd.DataFrame:
    """加载某年价格数据"""
    path = os.path.join(DATA_ROOT, year, "prices.csv")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df


def load_all_returns() -> pd.DataFrame:
    """加载三年合并收益率"""
    path = os.path.join(DATA_ROOT, "all", "returns_all.csv")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df


def load_meta() -> pd.DataFrame:
    """加载资产元信息"""
    path = os.path.join(DATA_ROOT, "2023", "meta.csv")
    return pd.read_csv(path)


def quick_network(returns: pd.DataFrame) -> dict:
    """
    快速网络分析 (用于回测内部每期调仓)
    仅计算 PMFG + 中心性 + 社区, 不做显著性检验
    返回 pmfg_graph 供 PageRank 策略使用 (§2.2.2 规则3)
    """
    corr = compute_correlation(returns, method="ledoit_wolf")
    pmfg = build_pmfg(corr)
    centralities = compute_centralities(pmfg)
    partition = detect_communities(pmfg)
    return {
        "centralities": centralities,
        "partition": partition,
        "pmfg_graph": pmfg,
    }


def quick_network_mst(returns: pd.DataFrame) -> dict:
    """基于 MST 的网络分析 (稳健性对比, §2.6.3(1))"""
    corr = compute_correlation(returns, method="ledoit_wolf")
    dist = corr_to_distance(corr)
    mst = build_mst(dist)
    centralities = compute_centralities(mst)
    partition = detect_communities(mst)
    return {
        "centralities": centralities,
        "partition": partition,
    }


def quick_network_glasso(returns: pd.DataFrame) -> dict:
    """基于 Graphical LASSO 的网络分析 (稳健性对比, §2.6.3(1))"""
    try:
        glasso_net = build_glasso_network(returns)
        if glasso_net.number_of_edges() < 3:
            raise ValueError("GLASSO 网络边数过少")
        centralities = compute_centralities(glasso_net)
        partition = detect_communities(glasso_net)
        return {
            "centralities": centralities,
            "partition": partition,
        }
    except Exception:
        # Fallback to PMFG
        return quick_network(returns)


# ============================================================
#  Part 1: 分年网络分析
# ============================================================

def part1_network_analysis():
    """
    Part 1: 分年网络分析
    输出:
      - 表1: 三年网络拓扑统计对比
      - 表2: 各年中心性排名 Top10
      - 表3: 社区划分
      - 表4: RMT + 随机网络显著性检验
      - 图1: MST 网络图
      - 图2: 中心性热力图
    """
    print("\n" + "█" * 60)
    print("  Part 1: 分年网络分析")
    print("█" * 60)

    net_results = {}
    topo_stats = []

    for year in YEARS:
        returns = load_returns(year)

        # 排除宽基指数(仅保留个股)
        idx_cols = [c for c in returns.columns if "指数" in c]
        returns_stocks = returns.drop(columns=idx_cols, errors="ignore")

        result = full_network_analysis(
            returns_stocks, year_label=year,
            output_dir=os.path.join(RESULTS_DIR, year)
        )
        net_results[year] = result

        # 拓扑统计
        mst = result["mst"]
        pmfg = result["pmfg"]
        topo_stats.append({
            "年份": year,
            "MST_节点": mst.number_of_nodes(),
            "MST_边": mst.number_of_edges(),
            "MST_平均路径": round(
                __import__("networkx").average_shortest_path_length(mst), 3),
            "PMFG_边": pmfg.number_of_edges(),
            "PMFG_平均度": round(
                2 * pmfg.number_of_edges() / pmfg.number_of_nodes(), 2),
            "PMFG_模块度": round(result["modularity_pmfg"], 4),
            "社区数": len(set(result["partition_pmfg"].values())),
            "RMT_信号特征值": result["rmt_result"]["n_signal_eigenvalues"],
            "RMT_信号方差占比": round(
                result["rmt_result"]["signal_var_explained"] * 100, 1),
            "随机网络_模块度p值": result["random_test"]["p_value_modularity"],
        })

    # 表1: 拓扑统计
    topo_df = pd.DataFrame(topo_stats)
    topo_df.to_csv(os.path.join(TABLES_DIR, "table1_topology.csv"), index=False)
    print(f"\n  表1: 三年网络拓扑统计对比")
    print(topo_df.to_string(index=False))

    # 表2: 中心性排名 Top10
    for year in YEARS:
        cent = net_results[year]["centralities_pmfg"]
        top10 = cent.sort_values("betweenness", ascending=False).head(10)
        top10.to_csv(os.path.join(TABLES_DIR, f"table2_centrality_{year}.csv"))

    # 表3: 社区划分
    for year in YEARS:
        comm = community_summary(net_results[year]["partition_pmfg"])
        comm.to_csv(os.path.join(TABLES_DIR, f"table3_communities_{year}.csv"))

    # 度分布保持的重连检验 (§2.3.4 检验方法三)
    print(f"\n  ---- 度分布保持重连检验 (Maslov-Sneppen) ----")
    for year in YEARS:
        pmfg = net_results[year]["pmfg"]
        print(f"    {year}年 PMFG 重连检验...", end="")
        rw_test = degree_preserving_rewiring_test(pmfg,
                                                   n_rewires=500,
                                                   n_simulations=100)
        net_results[year]["rewiring_test"] = rw_test
        print(f" 模块度p={rw_test['p_value_modularity']:.4f}, "
              f"Top5保留率={rw_test['top5_node_preservation']:.2f}")

    # 表4: 显著性检验汇总 (含三种检验方法)
    sig_data = []
    for year in YEARS:
        r = net_results[year]
        row = {
            "年份": year,
            "RMT_信号特征值数": r["rmt_result"]["n_signal_eigenvalues"],
            "最大特征值": r["rmt_result"]["largest_eigenvalue"],
            "MP上界": r["rmt_result"]["mp_lambda_plus"],
            "随机_模块度_p": r["random_test"]["p_value_modularity"],
            "随机_路径_p": r["random_test"]["p_value_path"],
            "随机_介数_p": r["random_test"]["p_value_betweenness"],
        }
        if "rewiring_test" in r:
            row["重连_模块度_p"] = r["rewiring_test"]["p_value_modularity"]
            row["重连_Top5保留"] = r["rewiring_test"]["top5_node_preservation"]
        sig_data.append(row)

    sig_df = pd.DataFrame(sig_data)
    sig_df.to_csv(os.path.join(TABLES_DIR, "table4_significance.csv"), index=False)
    print(f"\n  表4: 网络显著性检验 (含三种方法)")
    print(sig_df.to_string(index=False))

    # 图1: MST 网络图 (每年)
    _plot_mst(net_results)

    # 图2: 中心性对比热力图
    _plot_centrality_heatmap(net_results)

    # 网络时变性分析
    _analyze_time_variation(net_results)

    return net_results


def _plot_mst(net_results: dict):
    """绘制三年 MST 对比图"""
    import networkx as nx

    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    for idx, year in enumerate(YEARS):
        ax = axes[idx]
        mst = net_results[year]["mst"]
        partition = net_results[year]["partition_pmfg"]

        pos = nx.spring_layout(mst, seed=42, k=2)

        # 社区颜色
        communities = set(partition.values())
        cmap = plt.cm.Set3
        colors = {c: cmap(i / max(len(communities), 1))
                  for i, c in enumerate(sorted(communities))}
        node_colors = [colors.get(partition.get(n, 0), "gray")
                       for n in mst.nodes()]

        # 节点大小按介数中心性
        cent = net_results[year]["centralities_mst"]
        sizes = [300 + 3000 * cent.loc[n, "betweenness"]
                 for n in mst.nodes()]

        nx.draw_networkx(mst, pos, ax=ax, node_color=node_colors,
                         node_size=sizes, font_size=6,
                         width=0.8, alpha=0.9, font_family="SimHei")
        ax.set_title(f"{year}年 MST\n"
                     f"(节点大小=介数中心性, 颜色=社区)",
                     fontsize=12)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig1_mst_networks.png"),
                dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  图1已保存: {FIGURES_DIR}/fig1_mst_networks.png")


def _plot_centrality_heatmap(net_results: dict):
    """中心性对比热力图"""
    fig, axes = plt.subplots(1, 3, figsize=(20, 10))

    for idx, year in enumerate(YEARS):
        ax = axes[idx]
        cent = net_results[year]["centralities_pmfg"]
        # 排序: 按介数中心性
        cent_sorted = cent.sort_values("betweenness", ascending=False)
        im = ax.imshow(cent_sorted.values, aspect="auto", cmap="YlOrRd")
        ax.set_yticks(range(len(cent_sorted)))
        ax.set_yticklabels(cent_sorted.index, fontsize=7)
        ax.set_xticks(range(4))
        ax.set_xticklabels(["度", "介数", "接近", "特征向量"], fontsize=9)
        ax.set_title(f"{year}年 中心性指标", fontsize=12)
        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig2_centrality_heatmap.png"),
                dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  图2已保存: {FIGURES_DIR}/fig2_centrality_heatmap.png")


def _analyze_time_variation(net_results: dict):
    """网络时变性分析"""
    print(f"\n  ---- 网络时变性分析 ----")

    pairs = [("2023", "2024"), ("2024", "2025"), ("2023", "2025")]
    tv_data = []

    for y1, y2 in pairs:
        eo_mst = edge_overlap(net_results[y1]["mst"], net_results[y2]["mst"])
        eo_pmfg = edge_overlap(net_results[y1]["pmfg"], net_results[y2]["pmfg"])

        cent1 = net_results[y1]["centralities_pmfg"]["betweenness"]
        cent2 = net_results[y2]["centralities_pmfg"]["betweenness"]
        rank_corr = centrality_rank_correlation(cent1, cent2)

        nmi = community_nmi(net_results[y1]["partition_pmfg"],
                           net_results[y2]["partition_pmfg"])

        tv_data.append({
            "对比": f"{y1} vs {y2}",
            "MST边重叠率": round(eo_mst, 4),
            "PMFG边重叠率": round(eo_pmfg, 4),
            "介数秩相关": round(rank_corr, 4),
            "社区NMI": round(nmi, 4),
        })
        print(f"    {y1} vs {y2}: MST重叠={eo_mst:.3f}, "
              f"PMFG重叠={eo_pmfg:.3f}, 秩相关={rank_corr:.3f}, NMI={nmi:.3f}")

    tv_df = pd.DataFrame(tv_data)
    tv_df.to_csv(os.path.join(TABLES_DIR, "table5_time_variation.csv"), index=False)


# ============================================================
#  Part 2: γ 参数选择
# ============================================================

def part2_gamma_selection(net_results: dict):
    """
    Part 2: 使用2023年数据选择最优 γ, 在2024年验证
    """
    print("\n" + "█" * 60)
    print("  Part 2: 网络正则化参数 γ 选择")
    print("█" * 60)

    returns_2023 = load_returns("2023")
    returns_2024 = load_returns("2024")

    # 排除指数
    idx_cols = [c for c in returns_2023.columns if "指数" in c]
    returns_2023_s = returns_2023.drop(columns=idx_cols, errors="ignore")
    returns_2024_s = returns_2024.drop(columns=idx_cols, errors="ignore")

    centralities = net_results["2023"]["centralities_pmfg"]
    partition = net_results["2023"]["partition_pmfg"]

    gamma_grid = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]

    result = select_gamma_cv(
        returns_2023_s, returns_2024_s,
        centralities=centralities,
        partition=partition,
        gamma_grid=gamma_grid,
        metric="sharpe"
    )

    print(f"\n  γ 交叉验证结果:")
    print(result["results_df"].to_string(index=False))

    # γ选择策略: 若最优γ=0则使用1-SE规则选取最大γ使得Sharpe在最优值1个标准误内
    best_gamma = result["best_gamma"]
    if best_gamma == 0.0:
        df_cv = result["results_df"]
        max_sharpe = df_cv["sharpe"].max()
        # 使用经验SE估计 (约0.1)
        se_threshold = max_sharpe * 0.95  # 5%容忍度
        valid = df_cv[df_cv["sharpe"] >= se_threshold]
        if len(valid) > 1:
            best_gamma = valid["gamma"].max()  # 取最大γ
            print(f"\n  1-SE规则: 最优Sharpe={max_sharpe:.4f}, "
                  f"95%阈值={se_threshold:.4f}")
            print(f"  选取范围内最大γ={best_gamma} "
                  f"(Sharpe={valid.loc[valid['gamma']==best_gamma, 'sharpe'].values[0]:.4f})")

    print(f"\n  最终选定 γ = {best_gamma}")

    result["results_df"].to_csv(
        os.path.join(TABLES_DIR, "table6_gamma_cv.csv"), index=False)

    # 图3: γ vs Sharpe
    fig, ax = plt.subplots(figsize=(8, 5))
    df = result["results_df"]
    ax.plot(df["gamma"], df["sharpe"], "o-", color="steelblue", linewidth=2)
    ax.axvline(x=best_gamma, color="red", linestyle="--",
               label=f"选定γ={best_gamma}")
    ax.set_xlabel("γ (网络正则化强度)", fontsize=12)
    ax.set_ylabel("样本外夏普比率", fontsize=12)
    ax.set_title("网络正则化参数 γ 交叉验证", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig3_gamma_cv.png"),
                dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  图3已保存: {FIGURES_DIR}/fig3_gamma_cv.png")

    return best_gamma


# ============================================================
#  Part 3: 滚动回测
# ============================================================

def part3_backtest(net_results: dict, best_gamma: float):
    """
    Part 3: 20种策略滚动回测
      基准 (1种): 等权
      传统 (7种): MV, MinVar, MaxDiv, RP, HRP, BL无观点, BL动量
      图模型简单规则 (3种): 反中心性, 直接中心性, PageRank
      融合方法 (9种): NRRP, Graph-BL, Denoised-RP, Community-BL-RP,
                      Laplacian-MV, PMFG-Filtered-RP, Graph-Smoothed-BL,
                      Net-Constrained-MinVar, Community-MinVar
    """
    print("\n" + "█" * 60)
    print("  Part 3: 20种策略滚动回测")
    print("█" * 60)

    # 加载三年合并数据
    all_returns = load_all_returns()

    # 排除宽基指数
    idx_cols = [c for c in all_returns.columns if "指数" in c]
    returns_stocks = all_returns.drop(columns=idx_cols, errors="ignore")

    print(f"  数据: {len(returns_stocks)}天 × {returns_stocks.shape[1]}个资产")
    print(f"  回测窗口: {LOOKBACK}天, 调仓频率: 每{REBALANCE_FREQ}天")
    print(f"  样本外起始: {returns_stocks.index[LOOKBACK].strftime('%Y-%m-%d')}")

    backtest_results = {}
    N_STRAT = 20

    # ---- 基准 ----
    print(f"\n  [ 1/{N_STRAT}] 等权 (1/N)...", end="")
    t0 = time.time()
    res = rolling_backtest(returns_stocks, equal_weight,
                           lookback=LOOKBACK, rebalance_freq=REBALANCE_FREQ,
                           verbose=False)
    backtest_results["equal_weight"] = res
    print(f" 完成 ({time.time()-t0:.1f}s)")

    # ---- 传统策略 ----
    print(f"  [ 2/{N_STRAT}] 均值方差 (MV, β_max=10%)...", end="")
    t0 = time.time()
    res = rolling_backtest(returns_stocks, mean_variance,
                           lookback=LOOKBACK, rebalance_freq=REBALANCE_FREQ,
                           verbose=False)
    backtest_results["mean_variance"] = res
    print(f" 完成 ({time.time()-t0:.1f}s)")

    print(f"  [ 3/{N_STRAT}] 风险平价 (RP, β_max=10%)...", end="")
    t0 = time.time()
    res = rolling_backtest(returns_stocks, risk_parity,
                           lookback=LOOKBACK, rebalance_freq=REBALANCE_FREQ,
                           verbose=False)
    backtest_results["risk_parity"] = res
    print(f" 完成 ({time.time()-t0:.1f}s)")

    print(f"  [ 4/{N_STRAT}] 层次化RP (HRP)...", end="")
    t0 = time.time()
    res = rolling_backtest(returns_stocks, hrp,
                           lookback=LOOKBACK, rebalance_freq=REBALANCE_FREQ,
                           verbose=False)
    backtest_results["hrp"] = res
    print(f" 完成 ({time.time()-t0:.1f}s)")

    print(f"  [ 5/{N_STRAT}] BL无观点 (逆优化均衡收益)...", end="")
    t0 = time.time()
    res = rolling_backtest(returns_stocks, black_litterman_no_views,
                           lookback=LOOKBACK, rebalance_freq=REBALANCE_FREQ,
                           verbose=False)
    backtest_results["bl_no_views"] = res
    print(f" 完成 ({time.time()-t0:.1f}s)")

    print(f"  [ 6/{N_STRAT}] BL动量观点 (60天动量)...", end="")
    t0 = time.time()
    res = rolling_backtest(returns_stocks, black_litterman_momentum,
                           lookback=LOOKBACK, rebalance_freq=REBALANCE_FREQ,
                           verbose=False)
    backtest_results["bl_momentum"] = res
    print(f" 完成 ({time.time()-t0:.1f}s)")

    # ---- 图模型简单规则 ----
    print(f"  [ 7/{N_STRAT}] 反中心性 (规则1)...", end="")
    t0 = time.time()
    res = rolling_backtest(
        returns_stocks, inverse_centrality,
        lookback=LOOKBACK, rebalance_freq=REBALANCE_FREQ,
        network_func=quick_network, verbose=False)
    backtest_results["inv_centrality"] = res
    print(f" 完成 ({time.time()-t0:.1f}s)")

    print(f"  [ 8/{N_STRAT}] 直接中心性 (规则2)...", end="")
    t0 = time.time()
    res = rolling_backtest(
        returns_stocks, direct_centrality,
        lookback=LOOKBACK, rebalance_freq=REBALANCE_FREQ,
        network_func=quick_network, verbose=False)
    backtest_results["direct_centrality"] = res
    print(f" 完成 ({time.time()-t0:.1f}s)")

    print(f"  [ 9/{N_STRAT}] PageRank (规则3)...", end="")
    t0 = time.time()
    res = rolling_backtest(
        returns_stocks, pagerank_weight,
        lookback=LOOKBACK, rebalance_freq=REBALANCE_FREQ,
        network_func=quick_network, verbose=False)
    backtest_results["pagerank"] = res
    print(f" 完成 ({time.time()-t0:.1f}s)")

    # ---- 融合方法 (本文贡献) ----
    print(f"  [10/{N_STRAT}] 网络正则化RP (γ={best_gamma})...", end="")
    t0 = time.time()

    def nrrp_with_gamma(returns, centralities=None, partition=None, **kw):
        return network_regularized_rp(
            returns, centralities=centralities,
            partition=partition, gamma=best_gamma)

    res = rolling_backtest(
        returns_stocks, nrrp_with_gamma,
        lookback=LOOKBACK, rebalance_freq=REBALANCE_FREQ,
        network_func=quick_network, verbose=False)
    backtest_results["network_reg_rp"] = res
    print(f" 完成 ({time.time()-t0:.1f}s)")

    print(f"  [11/{N_STRAT}] ★ 网络增强BL (Graph-Enhanced BL)...", end="")
    t0 = time.time()
    res = rolling_backtest(
        returns_stocks, graph_enhanced_bl,
        lookback=LOOKBACK, rebalance_freq=REBALANCE_FREQ,
        network_func=quick_network, verbose=False)
    backtest_results["graph_bl"] = res
    print(f" 完成 ({time.time()-t0:.1f}s)")

    print(f"  [12/{N_STRAT}] ★ 去噪风险平价 (RMT-Denoised RP)...", end="")
    t0 = time.time()
    res = rolling_backtest(
        returns_stocks, denoised_rp,
        lookback=LOOKBACK, rebalance_freq=REBALANCE_FREQ,
        verbose=False)
    backtest_results["denoised_rp"] = res
    print(f" 完成 ({time.time()-t0:.1f}s)")

    print(f"  [13/{N_STRAT}] ★ 社区BL-RP混合 (Community BL-RP)...", end="")
    t0 = time.time()
    res = rolling_backtest(
        returns_stocks, community_bl_rp,
        lookback=LOOKBACK, rebalance_freq=REBALANCE_FREQ,
        network_func=quick_network, verbose=False)
    backtest_results["community_bl_rp"] = res
    print(f" 完成 ({time.time()-t0:.1f}s)")

    # ---- 前沿新增策略 ----
    print(f"  [14/{N_STRAT}] ◆ 最小方差 (MinVar)...", end="")
    t0 = time.time()
    res = rolling_backtest(returns_stocks, minimum_variance,
                           lookback=LOOKBACK, rebalance_freq=REBALANCE_FREQ,
                           verbose=False)
    backtest_results["min_variance"] = res
    print(f" 完成 ({time.time()-t0:.1f}s)")

    print(f"  [15/{N_STRAT}] ◆ 最大分散化 (MaxDiv)...", end="")
    t0 = time.time()
    res = rolling_backtest(returns_stocks, max_diversification,
                           lookback=LOOKBACK, rebalance_freq=REBALANCE_FREQ,
                           verbose=False)
    backtest_results["max_div"] = res
    print(f" 完成 ({time.time()-t0:.1f}s)")

    print(f"  [16/{N_STRAT}] ◆ 图拉普拉斯MV (Laplacian-MV)...", end="")
    t0 = time.time()
    res = rolling_backtest(
        returns_stocks, graph_laplacian_mv,
        lookback=LOOKBACK, rebalance_freq=REBALANCE_FREQ,
        network_func=quick_network, verbose=False)
    backtest_results["laplacian_mv"] = res
    print(f" 完成 ({time.time()-t0:.1f}s)")

    print(f"  [17/{N_STRAT}] ◆ 图过滤RP (PMFG-Filtered RP)...", end="")
    t0 = time.time()
    res = rolling_backtest(
        returns_stocks, pmfg_filtered_rp,
        lookback=LOOKBACK, rebalance_freq=REBALANCE_FREQ,
        network_func=quick_network, verbose=False)
    backtest_results["pmfg_filtered_rp"] = res
    print(f" 完成 ({time.time()-t0:.1f}s)")

    print(f"  [18/{N_STRAT}] ◆ 图平滑动量BL (Graph-Smoothed BL)...", end="")
    t0 = time.time()
    res = rolling_backtest(
        returns_stocks, graph_smoothed_bl,
        lookback=LOOKBACK, rebalance_freq=REBALANCE_FREQ,
        network_func=quick_network, verbose=False)
    backtest_results["graph_smooth_bl"] = res
    print(f" 完成 ({time.time()-t0:.1f}s)")

    # ---- 图模型+MinVar融合 (v4核心创新) ----
    print(f"  [19/{N_STRAT}] ◆ 网络约束MinVar (Net-Constrained MinVar)...", end="")
    t0 = time.time()
    res = rolling_backtest(
        returns_stocks, network_constrained_minvar,
        lookback=LOOKBACK, rebalance_freq=REBALANCE_FREQ,
        network_func=quick_network, verbose=False)
    backtest_results["net_minvar"] = res
    print(f" 完成 ({time.time()-t0:.1f}s)")

    print(f"  [20/{N_STRAT}] ◆ 社区MinVar (Community-MinVar)...", end="")
    t0 = time.time()
    res = rolling_backtest(
        returns_stocks, community_minvar,
        lookback=LOOKBACK, rebalance_freq=REBALANCE_FREQ,
        network_func=quick_network, verbose=False)
    backtest_results["comm_minvar"] = res
    print(f" 完成 ({time.time()-t0:.1f}s)")

    return backtest_results


# ============================================================
#  Part 4: 绩效评估与统计检验
# ============================================================

def part4_evaluation(backtest_results: dict):
    """
    Part 4: 综合绩效评估、统计检验、可视化
    """
    print("\n" + "█" * 60)
    print("  Part 4: 综合绩效评估与统计检验")
    print("█" * 60)

    # 综合评估
    perf_df, test_df = comprehensive_evaluation(
        backtest_results, benchmark_name="equal_weight")

    # 添加中文标签
    label_map = {
        "equal_weight": "等权(1/N)",
        "mean_variance": "均值方差(MV)",
        "min_variance": "◆最小方差(MinVar)",
        "max_div": "◆最大分散化(MaxDiv)",
        "risk_parity": "风险平价(RP)",
        "hrp": "层次化RP(HRP)",
        "bl_no_views": "BL无观点",
        "bl_momentum": "BL动量观点",
        "inv_centrality": "反中心性(规则1)",
        "direct_centrality": "直接中心性(规则2)",
        "pagerank": "PageRank(规则3)",
        "network_reg_rp": "网络正则化RP",
        "graph_bl": "★网络增强BL",
        "denoised_rp": "★去噪风险平价",
        "community_bl_rp": "★社区BL-RP",
        "laplacian_mv": "◆拉普拉斯MV",
        "pmfg_filtered_rp": "◆图过滤RP",
        "graph_smooth_bl": "◆图平滑动量BL",
        "net_minvar": "◆网络约束MinVar",
        "comm_minvar": "◆社区MinVar",
    }
    perf_df.index = [label_map.get(n, n) for n in perf_df.index]
    if not test_df.empty:
        test_df.index = [label_map.get(n, n) for n in test_df.index]

    # 表7: 绩效对比
    print(f"\n  ===== 表7: 策略绩效对比 =====")
    print(perf_df.to_string())
    perf_df.to_csv(os.path.join(TABLES_DIR, "table7_performance.csv"))

    # 表8: 统计检验 (vs 等权)
    if not test_df.empty:
        print(f"\n  ===== 表8: 统计检验 (vs 等权, Bootstrap n=10000) =====")
        print(test_df.to_string())
        test_df.to_csv(os.path.join(TABLES_DIR, "table8_statistical_tests.csv"))

        # Romano-Wolf 多重比较修正 (§2.6.2)
        boot_p_values = {name: row["boot_p"]
                         for name, row in test_df.iterrows()}
        rw_df = romano_wolf_correction(boot_p_values)
        if not rw_df.empty:
            print(f"\n  ===== 表8b: Romano-Wolf 多重比较修正 (§2.6.2) =====")
            print(rw_df.to_string(index=False))
            rw_df.to_csv(os.path.join(TABLES_DIR,
                                      "table8b_romano_wolf.csv"), index=False)

    # 分年绩效
    _yearly_performance(backtest_results, label_map)

    # 图4: 累计收益曲线
    _plot_cumulative_returns(backtest_results, label_map)

    # 图5: 回撤曲线
    _plot_drawdowns(backtest_results, label_map)

    # 图6: 权重分布
    _plot_weight_distribution(backtest_results, label_map)

    # 图7: 风险收益散点图
    _plot_risk_return_scatter(perf_df)

    return perf_df, test_df


def _yearly_performance(backtest_results: dict, label_map: dict):
    """分年绩效统计"""
    print(f"\n  ---- 分年绩效 ----")

    for year in ["2024", "2025"]:
        print(f"\n  {year}年:")
        year_perf = {}
        for name, res in backtest_results.items():
            ret = res["portfolio_returns"]
            ret_year = ret[ret.index.year == int(year)]
            if len(ret_year) > 0:
                perf = compute_performance(ret_year)
                year_perf[label_map.get(name, name)] = perf

        year_df = pd.DataFrame(year_perf).T
        print(year_df[["ann_return", "ann_vol", "sharpe",
                       "max_drawdown"]].to_string())
        year_df.to_csv(os.path.join(TABLES_DIR, f"table9_perf_{year}.csv"))


def _plot_cumulative_returns(backtest_results: dict, label_map: dict):
    """累计收益曲线"""
    fig, ax = plt.subplots(figsize=(14, 7))

    colors = {
        "equal_weight": "#888888", "mean_variance": "#2196F3",
        "min_variance": "#1565C0", "max_div": "#0D47A1",
        "risk_parity": "#4CAF50", "hrp": "#FF9800",
        "bl_no_views": "#00BCD4", "bl_momentum": "#3F51B5",
        "inv_centrality": "#9C27B0", "direct_centrality": "#795548",
        "pagerank": "#607D8B",
        "network_reg_rp": "#E91E63", "graph_bl": "#F44336",
        "denoised_rp": "#009688", "community_bl_rp": "#FF5722",
        "laplacian_mv": "#D32F2F", "pmfg_filtered_rp": "#00695C",
        "graph_smooth_bl": "#BF360C",
    }
    linewidths = {
        "equal_weight": 1.0, "mean_variance": 0.8,
        "min_variance": 1.0, "max_div": 1.0,
        "risk_parity": 0.8, "hrp": 1.0,
        "bl_no_views": 0.8, "bl_momentum": 0.8,
        "inv_centrality": 0.6, "direct_centrality": 0.6,
        "pagerank": 0.6,
        "network_reg_rp": 1.2, "graph_bl": 2.5,
        "denoised_rp": 1.5, "community_bl_rp": 1.5,
        "laplacian_mv": 2.0, "pmfg_filtered_rp": 2.0,
        "graph_smooth_bl": 2.0,
    }

    for name, res in backtest_results.items():
        ret = res["portfolio_returns"]
        cum = (1 + ret).cumprod()
        ax.plot(cum.index, cum.values,
                label=label_map.get(name, name),
                color=colors.get(name, "black"),
                linewidth=linewidths.get(name, 1.0),
                alpha=0.85)

    ax.set_xlabel("日期", fontsize=12)
    ax.set_ylabel("累计净值", fontsize=12)
    ax.set_title("二十种资产配置策略累计净值对比", fontsize=14)
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1.0, color="black", linestyle="-", linewidth=0.5, alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig4_cumulative_returns.png"),
                dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  图4已保存: {FIGURES_DIR}/fig4_cumulative_returns.png")


def _plot_drawdowns(backtest_results: dict, label_map: dict):
    """回撤曲线"""
    fig, ax = plt.subplots(figsize=(14, 5))

    colors = {
        "equal_weight": "#888888",
        "risk_parity": "#4CAF50",
        "graph_bl": "#F44336",
        "community_bl_rp": "#FF5722",
        "denoised_rp": "#009688",
    }
    # 画5个主要策略 (含3个融合方法), 重点对比
    for name in ["equal_weight", "risk_parity", "graph_bl",
                  "denoised_rp", "community_bl_rp"]:
        if name not in backtest_results:
            continue
        ret = backtest_results[name]["portfolio_returns"]
        cum = (1 + ret).cumprod()
        dd = cum / cum.cummax() - 1
        ax.fill_between(dd.index, dd.values, 0,
                        alpha=0.3, color=colors.get(name, "gray"),
                        label=label_map.get(name, name))
        ax.plot(dd.index, dd.values, color=colors.get(name, "gray"),
                linewidth=0.8)

    ax.set_xlabel("日期", fontsize=12)
    ax.set_ylabel("回撤", fontsize=12)
    ax.set_title("策略回撤对比 (等权 vs 风险平价 vs 融合方法)",
                 fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig5_drawdowns.png"),
                dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  图5已保存: {FIGURES_DIR}/fig5_drawdowns.png")


def _plot_weight_distribution(backtest_results: dict, label_map: dict):
    """权重分布箱线图 (最后一期)"""
    n_strats = len(backtest_results)
    n_rows = (n_strats + 3) // 4
    fig, axes = plt.subplots(n_rows, 4, figsize=(22, 5 * n_rows))

    axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    for idx, (name, res) in enumerate(backtest_results.items()):
        if idx >= len(axes_flat):
            break
        ax = axes_flat[idx]
        if not res["weights_history"]:
            continue
        last_w = res["weights_history"][-1]
        ax.bar(range(len(last_w)), np.sort(last_w)[::-1],
               color="steelblue", alpha=0.7)
        ax.set_title(label_map.get(name, name), fontsize=11)
        ax.set_xlabel("资产排序", fontsize=9)
        ax.set_ylabel("权重", fontsize=9)
        eff_n = compute_effective_n(last_w)
        ax.text(0.95, 0.95, f"EN={eff_n:.1f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=10,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.suptitle("各策略权重分布 (最后一期调仓)", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig6_weight_dist.png"),
                dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  图6已保存: {FIGURES_DIR}/fig6_weight_dist.png")


def _plot_risk_return_scatter(perf_df: pd.DataFrame):
    """风险-收益散点图"""
    fig, ax = plt.subplots(figsize=(9, 7))

    colors = ["#888888", "#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336"]

    for i, (idx, row) in enumerate(perf_df.iterrows()):
        c = colors[i % len(colors)]
        ax.scatter(row["ann_vol"], row["ann_return"],
                   s=200, c=c, zorder=5, edgecolors="white", linewidth=1.5)
        ax.annotate(idx, (row["ann_vol"], row["ann_return"]),
                    textcoords="offset points", xytext=(10, 5),
                    fontsize=9, color=c)

    ax.set_xlabel("年化波动率 (%)", fontsize=12)
    ax.set_ylabel("年化收益率 (%)", fontsize=12)
    ax.set_title("风险-收益散点图", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="black", linewidth=0.5, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig7_risk_return.png"),
                dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  图7已保存: {FIGURES_DIR}/fig7_risk_return.png")


# ============================================================
#  Part 5: 稳健性分析
# ============================================================

def part5_robustness(backtest_results: dict, net_results: dict,
                     best_gamma: float):
    """
    Part 5: 完整稳健性检验 (§2.6.3)
      5.1 不同中心性指标对比 (§2.2.3)
      5.2 网络过滤方法稳健性: MST vs PMFG vs GLASSO (§2.6.3(1))
      5.3 窗口长度敏感性: {126, 189, 252, 378, 504} (§2.6.3(2))
      5.4 样本期稳健性: 牛市/熊市/震荡市 (§2.6.3(3))
      5.5 交易成本敏感性: {0, 5, 10, 20}bp (§2.6.3(5))
      5.6 替代距离函数稳健性 (§2.3.1)
      5.7 自适应调仓 vs 固定调仓 (§2.5.3)
    """
    print("\n" + "█" * 60)
    print("  Part 5: 完整稳健性分析 (§2.6.3)")
    print("█" * 60)

    all_returns = load_all_returns()
    idx_cols = [c for c in all_returns.columns if "指数" in c]
    returns_stocks = all_returns.drop(columns=idx_cols, errors="ignore")

    # ================================================================
    # 5.1 不同中心性指标对比 (§2.2.3)
    # ================================================================
    print(f"\n  [5.1] 不同中心性指标对比 (§2.2.3)...")
    centrality_types = ["degree", "betweenness", "closeness", "eigenvector"]
    cent_perf = {}

    for ct in centrality_types:
        _ct = ct  # 闭包捕获

        def nrrp_ct(returns, centralities=None, partition=None,
                    _ctype=_ct, **kw):
            return network_regularized_rp(
                returns, centralities=centralities,
                partition=partition, gamma=best_gamma,
                centrality_type=_ctype)

        res = rolling_backtest(
            returns_stocks, nrrp_ct,
            lookback=LOOKBACK, rebalance_freq=REBALANCE_FREQ,
            network_func=quick_network, verbose=False)

        perf = compute_performance(res["portfolio_returns"])
        cent_perf[ct] = perf

    cent_df = pd.DataFrame(cent_perf).T
    cent_df.to_csv(os.path.join(TABLES_DIR, "table10_centrality_robustness.csv"))
    print(f"    结果:")
    print(cent_df[["ann_return", "ann_vol", "sharpe", "max_drawdown"]].to_string())

    # ================================================================
    # 5.2 网络过滤方法稳健性: MST vs PMFG vs GLASSO (§2.6.3(1))
    # ================================================================
    print(f"\n  [5.2] 网络过滤方法稳健性 (§2.6.3(1))...")
    net_method_perf = {}

    net_funcs = {
        "PMFG": quick_network,
        "MST": quick_network_mst,
        "GLASSO": quick_network_glasso,
    }

    for method_name, net_func in net_funcs.items():
        print(f"    {method_name}...", end="")
        t0 = time.time()

        def nrrp_method(returns, centralities=None, partition=None, **kw):
            return network_regularized_rp(
                returns, centralities=centralities,
                partition=partition, gamma=best_gamma)

        res = rolling_backtest(
            returns_stocks, nrrp_method,
            lookback=LOOKBACK, rebalance_freq=REBALANCE_FREQ,
            network_func=net_func, verbose=False)

        perf = compute_performance(res["portfolio_returns"])
        net_method_perf[method_name] = perf
        print(f" 完成 ({time.time()-t0:.1f}s)")

    net_method_df = pd.DataFrame(net_method_perf).T
    net_method_df.to_csv(os.path.join(TABLES_DIR,
                                      "table12_network_method_robustness.csv"))
    print(f"    结果:")
    print(net_method_df[["ann_return", "ann_vol", "sharpe",
                         "max_drawdown"]].to_string())

    # ================================================================
    # 5.3 窗口长度敏感性: τ ∈ {126, 189, 252, 378, 504} (§2.6.3(2))
    # ================================================================
    print(f"\n  [5.3] 窗口长度敏感性 (§2.6.3(2))...")
    window_perf = {}
    n_total = len(returns_stocks)

    for lb in [126, 189, 252, 378, 504]:
        if lb >= n_total - 60:
            print(f"    τ={lb}天: 跳过 (数据不足, 需至少60天OOS)")
            continue
        print(f"    τ={lb}天...", end="")
        t0 = time.time()

        def nrrp_w(returns, centralities=None, partition=None, **kw):
            return network_regularized_rp(
                returns, centralities=centralities,
                partition=partition, gamma=best_gamma)

        res = rolling_backtest(
            returns_stocks, nrrp_w,
            lookback=lb, rebalance_freq=REBALANCE_FREQ,
            network_func=quick_network, verbose=False)

        perf = compute_performance(res["portfolio_returns"])
        window_perf[f"τ={lb}"] = perf
        print(f" 完成 ({time.time()-t0:.1f}s, "
              f"OOS={len(res['portfolio_returns'])}天)")

    window_df = pd.DataFrame(window_perf).T
    window_df.to_csv(os.path.join(TABLES_DIR, "table11_window_robustness.csv"))
    print(f"    结果:")
    print(window_df[["ann_return", "ann_vol", "sharpe", "max_drawdown"]].to_string())

    # ================================================================
    # 5.4 样本期稳健性: 牛市/熊市/震荡市 (§2.6.3(3))
    # ================================================================
    print(f"\n  [5.4] 样本期稳健性 - 市场状态分析 (§2.6.3(3))...")

    # 使用等权组合的21日滚动收益划分市场状态
    ew_ret = backtest_results["equal_weight"]["portfolio_returns"]
    rolling_21d = ew_ret.rolling(21).sum()

    bull_mask = rolling_21d > 0.02
    bear_mask = rolling_21d < -0.02
    sideways_mask = ~bull_mask & ~bear_mask

    regime_perf = {}
    regime_map = {"牛市": bull_mask, "熊市": bear_mask, "震荡市": sideways_mask}

    for regime_name, mask in regime_map.items():
        valid_dates = mask[mask].index
        if len(valid_dates) < 10:
            continue
        regime_perf[regime_name] = {}
        for strat_name, res in backtest_results.items():
            ret = res["portfolio_returns"]
            regime_ret = ret[ret.index.isin(valid_dates)]
            if len(regime_ret) > 5:
                perf = compute_performance(regime_ret)
                regime_perf[regime_name][strat_name] = perf

    for regime_name, perfs in regime_perf.items():
        rdf = pd.DataFrame(perfs).T
        label_map_local = {
            "equal_weight": "等权(1/N)",
            "mean_variance": "均值方差(MV)",
            "min_variance": "◆最小方差",
            "max_div": "◆最大分散化",
            "risk_parity": "风险平价(RP)",
            "hrp": "层次化RP(HRP)",
            "bl_no_views": "BL无观点",
            "bl_momentum": "BL动量观点",
            "inv_centrality": "反中心性(规则1)",
            "direct_centrality": "直接中心性(规则2)",
            "pagerank": "PageRank(规则3)",
            "network_reg_rp": "网络正则化RP",
            "graph_bl": "★网络增强BL",
            "denoised_rp": "★去噪风险平价",
            "community_bl_rp": "★社区BL-RP",
            "laplacian_mv": "◆拉普拉斯MV",
            "pmfg_filtered_rp": "◆图过滤RP",
            "graph_smooth_bl": "◆图平滑动量BL",
            "net_minvar": "◆网络约束MinVar",
            "comm_minvar": "◆社区MinVar",
        }
        rdf.index = [label_map_local.get(n, n) for n in rdf.index]
        rdf.to_csv(os.path.join(TABLES_DIR,
                                f"table13_regime_{regime_name}.csv"))
        n_regime_days = int(regime_map[regime_name].sum())
        print(f"    {regime_name} ({n_regime_days}天):")
        print(rdf[["ann_return", "ann_vol", "sharpe",
                    "max_drawdown"]].to_string())

    # ================================================================
    # 5.5 交易成本敏感性: {0, 5, 10, 20} bp (§2.6.3(5))
    # ================================================================
    print(f"\n  [5.5] 交易成本敏感性 (§2.6.3(5))...")
    tc_perf = {}

    for tc in [0, 5, 10, 20]:
        print(f"    TC={tc}bp...", end="")
        t0 = time.time()

        def nrrp_tc(returns, centralities=None, partition=None, **kw):
            return network_regularized_rp(
                returns, centralities=centralities,
                partition=partition, gamma=best_gamma)

        res = rolling_backtest(
            returns_stocks, nrrp_tc,
            lookback=LOOKBACK, rebalance_freq=REBALANCE_FREQ,
            network_func=quick_network,
            tc_bps=float(tc),
            verbose=False)

        perf = compute_performance(res["portfolio_returns"])
        tc_perf[f"{tc}bp"] = perf
        print(f" 完成 ({time.time()-t0:.1f}s)")

    tc_df = pd.DataFrame(tc_perf).T
    tc_df.to_csv(os.path.join(TABLES_DIR, "table14_transaction_cost.csv"))
    print(f"    结果:")
    print(tc_df[["ann_return", "ann_vol", "sharpe", "max_drawdown"]].to_string())

    # ================================================================
    # 5.6 替代距离函数稳健性 (§2.3.1)
    # ================================================================
    print(f"\n  [5.6] 替代距离函数稳健性 (§2.3.1)...")
    # 使用 d'_ij = sqrt(1-ρ²) 构建 MST, 对比中心性排序
    for year in YEARS:
        returns_y = load_returns(year)
        idx_cols_y = [c for c in returns_y.columns if "指数" in c]
        returns_y = returns_y.drop(columns=idx_cols_y, errors="ignore")

        corr = compute_correlation(returns_y, method="ledoit_wolf")
        dist_std = corr_to_distance(corr)
        dist_mi = corr_to_distance_mi(corr)

        mst_std = build_mst(dist_std)
        mst_mi = build_mst(dist_mi)

        cent_std = compute_centralities(mst_std)
        cent_mi = compute_centralities(mst_mi)

        from scipy.stats import spearmanr
        rho, _ = spearmanr(cent_std["betweenness"], cent_mi["betweenness"])
        eo = edge_overlap(mst_std, mst_mi)
        print(f"    {year}: 介数秩相关={rho:.3f}, MST边重叠={eo:.3f}")

    # ================================================================
    # 5.7 自适应调仓 vs 固定调仓 (§2.5.3)
    # ================================================================
    print(f"\n  [5.7] 自适应调仓 vs 固定调仓 (§2.5.3)...")
    print(f"    固定调仓 (已有)...", end="")
    fixed_perf = compute_performance(
        backtest_results["network_reg_rp"]["portfolio_returns"])
    print(f" Sharpe={fixed_perf['sharpe']:.4f}")

    print(f"    自适应调仓 (J<0.7/NMI<0.8)...", end="")
    t0 = time.time()

    def nrrp_adaptive(returns, centralities=None, partition=None, **kw):
        return network_regularized_rp(
            returns, centralities=centralities,
            partition=partition, gamma=best_gamma)

    res_adaptive = rolling_backtest(
        returns_stocks, nrrp_adaptive,
        lookback=LOOKBACK, rebalance_freq=REBALANCE_FREQ,
        network_func=quick_network,
        adaptive_rebalance=True,
        stability_j=0.7, stability_nmi=0.8,
        verbose=False)

    adaptive_perf = compute_performance(res_adaptive["portfolio_returns"])
    print(f" Sharpe={adaptive_perf['sharpe']:.4f}, "
          f"实际调仓次数={len(res_adaptive['rebalance_dates'])}"
          f" ({time.time()-t0:.1f}s)")

    adapt_cmp = pd.DataFrame({
        "固定调仓": fixed_perf,
        "自适应调仓": adaptive_perf,
    }).T
    adapt_cmp.to_csv(os.path.join(TABLES_DIR, "table15_adaptive_rebalance.csv"))
    print(adapt_cmp[["ann_return", "ann_vol", "sharpe",
                     "max_drawdown", "turnover"]].to_string()
          if "turnover" in adapt_cmp.columns else
          adapt_cmp[["ann_return", "ann_vol", "sharpe",
                     "max_drawdown"]].to_string())

    return cent_df, window_df


# ============================================================
#  MAIN
# ============================================================

if __name__ == "__main__":
    total_start = time.time()

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  基于图模型的资产配置策略 - 完整分析 (v4增强版)             ║")
    print("║  20种策略: 基准(1)+传统(7)+图规则(3)+融合(9)               ║")
    print("║  新增: Net-MinVar, Comm-MinVar 等图模型+MinVar融合策略      ║")
    print("║  覆盖: 从经典到前沿的完整策略谱系                          ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    # Part 1: 网络分析
    net_results = part1_network_analysis()

    # Part 2: γ 选择
    best_gamma = part2_gamma_selection(net_results)

    # Part 3: 滚动回测
    backtest_results = part3_backtest(net_results, best_gamma)

    # Part 4: 绩效评估
    perf_df, test_df = part4_evaluation(backtest_results)

    # Part 5: 稳健性分析
    cent_df, window_df = part5_robustness(
        backtest_results, net_results, best_gamma)

    # 总结
    total_time = time.time() - total_start
    print("\n" + "█" * 60)
    print(f"  全部分析完成! 总耗时: {total_time/60:.1f}分钟")
    print("█" * 60)
    print(f"\n  输出目录: {RESULTS_DIR}/")
    print(f"  表格: {TABLES_DIR}/ ({len(os.listdir(TABLES_DIR))} 个文件)")
    print(f"  图表: {FIGURES_DIR}/ ({len(os.listdir(FIGURES_DIR))} 个文件)")
    print(f"\n  论文表格清单:")
    for f in sorted(os.listdir(TABLES_DIR)):
        print(f"    {f}")
    print(f"\n  论文图表清单:")
    for f in sorted(os.listdir(FIGURES_DIR)):
        print(f"    {f}")
