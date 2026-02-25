"""
网络分析模块
===========
MST/PMFG 构建、中心性计算、社区检测、显著性检验、可视化
"""

import numpy as np
import pandas as pd
import networkx as nx
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.covariance import GraphicalLassoCV, LedoitWolf
import community as community_louvain
import warnings

warnings.filterwarnings("ignore")


# ============================================================
#  1. 相关性 → 距离矩阵
# ============================================================

def corr_to_distance(corr: pd.DataFrame) -> pd.DataFrame:
    """相关系数 → 欧几里得距离 d_ij = sqrt(2(1-rho_ij))"""
    dist = np.sqrt(2.0 * (1.0 - corr))
    np.fill_diagonal(dist.values, 0.0)
    return dist


def corr_to_distance_mi(corr: pd.DataFrame) -> pd.DataFrame:
    """替代距离函数 (基于互信息近似): d'_ij = sqrt(1 - rho_ij^2)
    用于稳健性检验, 评估线性相关假设对结果的影响 (§2.2.1)
    """
    dist = np.sqrt(1.0 - np.clip(corr.values ** 2, 0, 1))
    np.fill_diagonal(dist, 0.0)
    return pd.DataFrame(dist, index=corr.index, columns=corr.columns)


def compute_correlation(returns: pd.DataFrame,
                        method: str = "ledoit_wolf") -> pd.DataFrame:
    """
    计算相关系数矩阵（支持多种估计方法）
    method: 'pearson' | 'ledoit_wolf'
    """
    if method == "ledoit_wolf":
        lw = LedoitWolf().fit(returns.values)
        cov = lw.covariance_
        std = np.sqrt(np.diag(cov))
        corr = cov / np.outer(std, std)
        np.fill_diagonal(corr, 1.0)
        corr = np.clip(corr, -1.0, 1.0)
        return pd.DataFrame(corr, index=returns.columns, columns=returns.columns)
    else:
        return returns.corr()


# ============================================================
#  2. MST 构建
# ============================================================

def build_mst(dist: pd.DataFrame) -> nx.Graph:
    """
    从距离矩阵构建最小生成树 (Kruskal 算法)
    """
    assets = dist.columns.tolist()
    n = len(assets)

    # 构建完全加权图
    G_full = nx.Graph()
    for i in range(n):
        for j in range(i + 1, n):
            G_full.add_edge(assets[i], assets[j], weight=dist.iloc[i, j])

    # MST
    mst = nx.minimum_spanning_tree(G_full, algorithm="kruskal")
    return mst


# ============================================================
#  3. PMFG 构建 (平面最大过滤图)
# ============================================================

def build_pmfg(corr: pd.DataFrame) -> nx.Graph:
    """
    构建平面最大过滤图 (Planar Maximally Filtered Graph)
    贪心算法: 按相关系数从高到低依次添加边, 保持图的平面性
    保留 3(n-2) 条边
    """
    assets = corr.columns.tolist()
    n = len(assets)
    target_edges = 3 * (n - 2)

    # 所有边按相关系数降序排列
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((assets[i], assets[j], corr.iloc[i, j]))
    edges.sort(key=lambda x: -x[2])  # 降序

    G = nx.Graph()
    G.add_nodes_from(assets)

    for u, v, w in edges:
        if G.number_of_edges() >= target_edges:
            break
        G.add_edge(u, v, weight=w)
        # 检查平面性
        is_planar, _ = nx.check_planarity(G)
        if not is_planar:
            G.remove_edge(u, v)

    return G


# ============================================================
#  4. Graphical LASSO 条件依赖网络
# ============================================================

def build_glasso_network(returns: pd.DataFrame,
                         alpha_range: int = 10) -> nx.Graph:
    """
    基于 Graphical LASSO 构建条件依赖网络
    精度矩阵非零项 → 边
    """
    model = GraphicalLassoCV(cv=5, n_jobs=-1)
    model.fit(returns.values)

    precision = model.precision_
    assets = returns.columns.tolist()
    n = len(assets)

    G = nx.Graph()
    G.add_nodes_from(assets)

    for i in range(n):
        for j in range(i + 1, n):
            if abs(precision[i, j]) > 1e-6:
                G.add_edge(assets[i], assets[j],
                           weight=abs(precision[i, j]))
    return G


# ============================================================
#  5. 中心性指标计算
# ============================================================

def compute_centralities(G: nx.Graph) -> pd.DataFrame:
    """
    计算四种中心性指标
    返回 DataFrame: index=资产, columns=[degree, betweenness, closeness, eigenvector]
    """
    nodes = list(G.nodes())

    degree = nx.degree_centrality(G)
    betweenness = nx.betweenness_centrality(G, normalized=True)
    closeness = nx.closeness_centrality(G)

    try:
        eigenvector = nx.eigenvector_centrality(G, max_iter=1000, tol=1e-06)
    except nx.PowerIterationFailedConvergence:
        eigenvector = {n: 0.0 for n in nodes}

    df = pd.DataFrame({
        "degree": degree,
        "betweenness": betweenness,
        "closeness": closeness,
        "eigenvector": eigenvector,
    })
    return df.loc[nodes]


# ============================================================
#  6. 社区检测 (Louvain)
# ============================================================

def detect_communities(G: nx.Graph) -> dict:
    """
    Louvain 算法社区检测
    返回 {node: community_id}
    """
    partition = community_louvain.best_partition(G, random_state=42)
    return partition


def community_summary(partition: dict, meta_df: pd.DataFrame = None) -> pd.DataFrame:
    """社区划分汇总"""
    comm_df = pd.DataFrame.from_dict(partition, orient="index", columns=["community"])
    comm_df.index.name = "asset"

    summary = comm_df.groupby("community").apply(
        lambda g: pd.Series({
            "n_assets": len(g),
            "members": ", ".join(g.index.tolist()),
        })
    )
    return summary


def compute_modularity(G: nx.Graph, partition: dict) -> float:
    """计算模块度 Q"""
    return community_louvain.modularity(partition, G)


# ============================================================
#  7. 网络显著性检验
# ============================================================

def rmt_test(corr: pd.DataFrame, T: int) -> dict:
    """
    随机矩阵理论 (RMT) 检验
    比较特征值分布与 Marchenko-Pastur 极限
    """
    n = corr.shape[0]
    q = n / T  # 维度/样本比

    eigenvalues = np.linalg.eigvalsh(corr.values)
    eigenvalues = np.sort(eigenvalues)[::-1]

    # Marchenko-Pastur 边界
    lambda_plus = (1 + np.sqrt(q)) ** 2
    lambda_minus = (1 - np.sqrt(q)) ** 2

    n_signal = np.sum(eigenvalues > lambda_plus)
    n_noise = np.sum((eigenvalues >= lambda_minus) & (eigenvalues <= lambda_plus))
    n_below = np.sum(eigenvalues < lambda_minus)

    # 最大特征值占比 (解释方差)
    var_explained_signal = eigenvalues[:n_signal].sum() / eigenvalues.sum() if n_signal > 0 else 0

    return {
        "n_assets": n,
        "n_observations": T,
        "q_ratio": round(q, 4),
        "mp_lambda_plus": round(lambda_plus, 4),
        "mp_lambda_minus": round(lambda_minus, 4),
        "n_signal_eigenvalues": n_signal,
        "n_noise_eigenvalues": n_noise,
        "largest_eigenvalue": round(eigenvalues[0], 4),
        "signal_var_explained": round(var_explained_signal, 4),
        "eigenvalues": eigenvalues,
    }


def random_network_test(corr: pd.DataFrame, T: int,
                        n_simulations: int = 1000,
                        network_builder=None) -> dict:
    """
    随机网络对比检验
    打乱相关结构, 检验观测网络指标是否显著偏离随机水平

    Args:
        corr: 观测相关矩阵
        T: 时间序列长度
        n_simulations: 模拟次数
        network_builder: 网络构建函数 (接受距离矩阵, 返回 nx.Graph)
    """
    if network_builder is None:
        network_builder = build_mst

    n = corr.shape[0]

    # 观测网络指标
    dist_obs = corr_to_distance(corr)
    G_obs = network_builder(dist_obs) if network_builder == build_mst else network_builder(corr)
    obs_avg_path = nx.average_shortest_path_length(G_obs) if nx.is_connected(G_obs) else float('inf')
    obs_betweenness = np.mean(list(nx.betweenness_centrality(G_obs).values()))

    # 尝试计算观测网络模块度
    partition_obs = detect_communities(G_obs)
    obs_modularity = compute_modularity(G_obs, partition_obs)

    # 随机模拟
    random_paths = []
    random_betweenness = []
    random_modularity = []

    rng = np.random.default_rng(42)
    for _ in range(n_simulations):
        # 生成随机数据 (独立正态)
        random_returns = rng.standard_normal((T, n))
        random_corr_mat = np.corrcoef(random_returns.T)
        np.fill_diagonal(random_corr_mat, 1.0)

        random_corr_df = pd.DataFrame(random_corr_mat,
                                       index=corr.index, columns=corr.columns)
        random_dist = corr_to_distance(random_corr_df)

        if network_builder == build_mst:
            G_rand = build_mst(random_dist)
        else:
            G_rand = network_builder(random_corr_df)

        if nx.is_connected(G_rand):
            random_paths.append(nx.average_shortest_path_length(G_rand))
        random_betweenness.append(
            np.mean(list(nx.betweenness_centrality(G_rand).values())))

        part_rand = detect_communities(G_rand)
        random_modularity.append(compute_modularity(G_rand, part_rand))

    # p 值计算
    random_paths = np.array(random_paths)
    random_betweenness = np.array(random_betweenness)
    random_modularity = np.array(random_modularity)

    p_path = np.mean(random_paths <= obs_avg_path) if len(random_paths) > 0 else np.nan
    p_betweenness = np.mean(random_betweenness >= obs_betweenness)
    p_modularity = np.mean(random_modularity >= obs_modularity)

    return {
        "obs_avg_path_length": round(obs_avg_path, 4),
        "rand_avg_path_mean": round(np.mean(random_paths), 4) if len(random_paths) > 0 else np.nan,
        "p_value_path": round(p_path, 4),
        "obs_avg_betweenness": round(obs_betweenness, 6),
        "rand_avg_betweenness_mean": round(np.mean(random_betweenness), 6),
        "p_value_betweenness": round(p_betweenness, 4),
        "obs_modularity": round(obs_modularity, 4),
        "rand_modularity_mean": round(np.mean(random_modularity), 4),
        "p_value_modularity": round(p_modularity, 4),
    }


# ============================================================
#  8. 网络时变性指标
# ============================================================

def edge_overlap(G1: nx.Graph, G2: nx.Graph) -> float:
    """Jaccard 边重叠率"""
    e1 = set(G1.edges())
    e2 = set(G2.edges())
    if len(e1 | e2) == 0:
        return 0.0
    return len(e1 & e2) / len(e1 | e2)


def centrality_rank_correlation(cent1: pd.Series, cent2: pd.Series) -> float:
    """中心性秩相关 (Spearman)"""
    from scipy.stats import spearmanr
    common = cent1.index.intersection(cent2.index)
    if len(common) < 3:
        return np.nan
    rho, _ = spearmanr(cent1[common], cent2[common])
    return rho


def community_nmi(partition1: dict, partition2: dict) -> float:
    """
    归一化互信息 (NMI) 度量社区划分一致性
    """
    from sklearn.metrics import normalized_mutual_info_score
    common_nodes = sorted(set(partition1.keys()) & set(partition2.keys()))
    if len(common_nodes) < 3:
        return np.nan
    labels1 = [partition1[n] for n in common_nodes]
    labels2 = [partition2[n] for n in common_nodes]
    return normalized_mutual_info_score(labels1, labels2)


# ============================================================
#  9. 度分布保持的重连检验 (Maslov & Sneppen, 2002)  §2.2.4
# ============================================================

def degree_preserving_rewiring_test(G: nx.Graph,
                                     n_rewires: int = 1000,
                                     n_simulations: int = 100) -> dict:
    """
    度分布保持的重连检验 (Maslov & Sneppen, 2002)
    在保持原始网络度分布不变的前提下随机重连边,
    检验社区结构和核心节点是否仅由度分布决定,
    还是反映了更深层的连接模式 (§2.2.4 检验方法三)
    """
    partition_obs = detect_communities(G)
    modularity_obs = compute_modularity(G, partition_obs)
    betweenness_obs = nx.betweenness_centrality(G, normalized=True)

    # 观测网络介数中心性 Top-5
    top_nodes_obs = sorted(betweenness_obs.keys(),
                           key=lambda x: betweenness_obs[x],
                           reverse=True)[:5]

    random_modularities = []
    top_node_preserved = []

    rng = np.random.default_rng(42)
    for _ in range(n_simulations):
        G_rw = G.copy()
        edges = list(G_rw.edges())
        for _ in range(n_rewires):
            if len(edges) < 2:
                break
            idx = rng.choice(len(edges), 2, replace=False)
            u1, v1 = edges[idx[0]]
            u2, v2 = edges[idx[1]]
            # 确保4个节点互不相同且不产生重边
            if (u1 != u2 and u1 != v2 and v1 != u2 and v1 != v2
                    and not G_rw.has_edge(u1, u2) and not G_rw.has_edge(v1, v2)):
                G_rw.remove_edge(u1, v1)
                G_rw.remove_edge(u2, v2)
                G_rw.add_edge(u1, u2)
                G_rw.add_edge(v1, v2)
                edges = list(G_rw.edges())

        part_rw = detect_communities(G_rw)
        mod_rw = compute_modularity(G_rw, part_rw)
        random_modularities.append(mod_rw)

        bet_rw = nx.betweenness_centrality(G_rw, normalized=True)
        top_rw = sorted(bet_rw.keys(),
                        key=lambda x: bet_rw[x], reverse=True)[:5]
        overlap = len(set(top_nodes_obs) & set(top_rw))
        top_node_preserved.append(overlap / 5.0)

    random_modularities = np.array(random_modularities)
    p_modularity = np.mean(random_modularities >= modularity_obs)

    return {
        "obs_modularity": round(modularity_obs, 4),
        "rewired_modularity_mean": round(np.mean(random_modularities), 4),
        "rewired_modularity_std": round(np.std(random_modularities), 4),
        "p_value_modularity": round(p_modularity, 4),
        "top5_node_preservation": round(np.mean(top_node_preserved), 4),
        "significant_community": p_modularity < 0.05,
    }


# ============================================================
#  10. 综合网络分析 (一键运行)
# ============================================================

def full_network_analysis(returns: pd.DataFrame, year_label: str = "",
                          output_dir: str = "results") -> dict:
    """
    完整网络分析流程:
      1. 计算相关/距离矩阵 (Ledoit-Wolf 收缩)
      2. 构建 MST + PMFG + GLASSO 网络
      3. 计算中心性
      4. 社区检测
      5. 显著性检验 (RMT + 随机网络)
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"网络分析 {year_label}")
    print(f"{'='*60}")

    n_days, n_assets = returns.shape
    print(f"  数据: {n_days}天 × {n_assets}个资产")

    # 1. 相关性与距离
    print("  [1/6] 计算相关/距离矩阵 (Ledoit-Wolf 收缩)...")
    corr = compute_correlation(returns, method="ledoit_wolf")
    corr_pearson = returns.corr()
    dist = corr_to_distance(corr)

    # 2. MST
    print("  [2/6] 构建 MST...")
    mst = build_mst(dist)
    print(f"         节点={mst.number_of_nodes()}, 边={mst.number_of_edges()}")

    # 3. PMFG
    print("  [3/6] 构建 PMFG (可能需要几秒)...")
    pmfg = build_pmfg(corr)
    print(f"         节点={pmfg.number_of_nodes()}, 边={pmfg.number_of_edges()}")

    # 4. GLASSO
    print("  [4/6] 构建 Graphical LASSO 网络...")
    try:
        glasso_net = build_glasso_network(returns)
        print(f"         节点={glasso_net.number_of_nodes()}, 边={glasso_net.number_of_edges()}")
    except Exception as e:
        print(f"         GLASSO 失败: {e}, 跳过")
        glasso_net = None

    # 5. 中心性 (在 PMFG 上计算)
    print("  [5/6] 计算中心性指标 (PMFG)...")
    centralities_pmfg = compute_centralities(pmfg)
    centralities_mst = compute_centralities(mst)

    # 社区检测 (在 PMFG 上)
    partition_pmfg = detect_communities(pmfg)
    modularity_pmfg = compute_modularity(pmfg, partition_pmfg)
    comm_summary = community_summary(partition_pmfg)

    print(f"         模块度 Q = {modularity_pmfg:.4f}")
    print(f"         社区数 = {len(set(partition_pmfg.values()))}")
    for _, row in comm_summary.iterrows():
        print(f"           [{row.name}] {row['n_assets']}个: {row['members']}")

    # 6. 显著性检验
    print("  [6/6] 网络显著性检验...")
    rmt_result = rmt_test(corr_pearson, T=n_days)
    print(f"         RMT: {rmt_result['n_signal_eigenvalues']}个信号特征值 "
          f"(解释{rmt_result['signal_var_explained']*100:.1f}%方差)")

    print(f"         随机网络对比 (1000次模拟)...")
    random_test = random_network_test(corr_pearson, T=n_days,
                                       n_simulations=1000,
                                       network_builder=build_mst)
    print(f"         模块度 p={random_test['p_value_modularity']:.4f}, "
          f"平均路径 p={random_test['p_value_path']:.4f}")

    # 保存结果
    suffix = f"_{year_label}" if year_label else ""
    corr.to_csv(os.path.join(output_dir, f"corr_lw{suffix}.csv"))
    centralities_pmfg.to_csv(os.path.join(output_dir, f"centralities_pmfg{suffix}.csv"))
    centralities_mst.to_csv(os.path.join(output_dir, f"centralities_mst{suffix}.csv"))
    comm_summary.to_csv(os.path.join(output_dir, f"communities{suffix}.csv"))

    return {
        "corr": corr,
        "corr_pearson": corr_pearson,
        "dist": dist,
        "mst": mst,
        "pmfg": pmfg,
        "glasso_net": glasso_net,
        "centralities_pmfg": centralities_pmfg,
        "centralities_mst": centralities_mst,
        "partition_pmfg": partition_pmfg,
        "modularity_pmfg": modularity_pmfg,
        "rmt_result": rmt_result,
        "random_test": random_test,
    }
