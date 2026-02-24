"""
投资组合策略模块
================
10种资产配置策略实现:
  传统 / 基准:
    1. Equal Weight (1/N)
    2. Mean-Variance (Markowitz + Ledoit-Wolf 收缩)
    3. Risk Parity (标准风险平价)
    4. HRP (层次化风险平价)
    5. Black-Litterman 无观点 (逆优化隐含均衡收益)
    6. Black-Litterman 动量观点 (量化信号自动生成观点)
  图模型:
    7. Inverse Centrality (反中心性, 原论文方法)
    8. Direct Centrality (中心性直接加权)
    9. PageRank Weighting
   10. Network-Regularized Risk Parity (本文改进方法)
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage, leaves_list
from sklearn.covariance import LedoitWolf
import warnings

warnings.filterwarnings("ignore")


# ============================================================
#  1. Equal Weight (等权)
# ============================================================

def equal_weight(returns: pd.DataFrame, **kwargs) -> np.ndarray:
    """等权配置: w_i = 1/n"""
    n = returns.shape[1]
    return np.ones(n) / n


# ============================================================
#  2. Mean-Variance (均值方差 + Ledoit-Wolf 收缩)
# ============================================================

def mean_variance(returns: pd.DataFrame, risk_aversion: float = 2.0,
                  max_weight: float = 0.10, **kwargs) -> np.ndarray:
    """
    Markowitz 均值方差优化
    使用 Ledoit-Wolf 收缩估计协方差矩阵, 提高数值稳定性
    max w'μ - (λ/2) w'Σw  s.t. sum(w)=1, w>=0, w_i<=β_max
    含单一标的集中度约束 (§2.4.3)
    """
    n = returns.shape[1]
    mu = returns.mean().values * 252  # 年化
    lw = LedoitWolf().fit(returns.values)
    cov = lw.covariance_ * 252  # 年化

    def neg_utility(w):
        return -(w @ mu - (risk_aversion / 2) * w @ cov @ w)

    def neg_utility_grad(w):
        return -(mu - risk_aversion * cov @ w)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, max_weight)] * n
    w0 = np.ones(n) / n

    result = minimize(neg_utility, w0, jac=neg_utility_grad,
                      method="SLSQP", bounds=bounds,
                      constraints=constraints,
                      options={"maxiter": 1000, "ftol": 1e-12})

    if result.success:
        w = result.x
        w = np.maximum(w, 0)
        w /= w.sum()
        return w
    else:
        return np.ones(n) / n  # fallback


# ============================================================
#  3. Risk Parity (风险平价)
# ============================================================

def risk_parity(returns: pd.DataFrame, max_weight: float = 0.10,
                **kwargs) -> np.ndarray:
    """
    标准风险平价: 每项资产对组合风险的边际贡献相等
    min Σ_i Σ_j (RC_i - RC_j)^2
    RC_i = w_i * (Σw)_i / σ_p
    含单一标的集中度约束 (§2.4.3)
    """
    n = returns.shape[1]
    lw = LedoitWolf().fit(returns.values)
    cov = lw.covariance_ * 252

    def risk_contrib_obj(w):
        sigma_p = np.sqrt(w @ cov @ w)
        if sigma_p < 1e-10:
            return 1e10
        mrc = cov @ w / sigma_p  # marginal risk contribution
        rc = w * mrc  # risk contribution
        # 目标: 所有 RC 相等
        target_rc = sigma_p / n
        return np.sum((rc - target_rc) ** 2)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(1e-6, max_weight)] * n
    w0 = np.ones(n) / n

    result = minimize(risk_contrib_obj, w0, method="SLSQP",
                      bounds=bounds, constraints=constraints,
                      options={"maxiter": 2000, "ftol": 1e-14})

    if result.success:
        w = result.x
        w = np.maximum(w, 0)
        w /= w.sum()
        return w
    else:
        return np.ones(n) / n


# ============================================================
#  4. HRP (层次化风险平价)
# ============================================================

def hrp(returns: pd.DataFrame, **kwargs) -> np.ndarray:
    """
    Hierarchical Risk Parity (López de Prado, 2016)
    步骤:
      1. 基于相关距离层次聚类
      2. 根据聚类树重排协方差矩阵
      3. 沿树结构递归反方差分配
    """
    corr = returns.corr()
    cov = LedoitWolf().fit(returns.values).covariance_ * 252
    n = corr.shape[0]

    # 1. 距离矩阵
    dist = np.sqrt(0.5 * (1.0 - corr.values))
    np.fill_diagonal(dist, 0.0)

    # 压缩距离矩阵
    from scipy.spatial.distance import squareform
    dist_condensed = squareform(dist, checks=False)

    # 2. 层次聚类
    link = linkage(dist_condensed, method="single")
    sorted_indices = leaves_list(link).tolist()

    # 3. 递归二分反方差分配
    def _get_cluster_var(cov, indices):
        """聚类的方差 (反方差加权)"""
        sub_cov = cov[np.ix_(indices, indices)]
        inv_diag = 1.0 / np.diag(sub_cov)
        w = inv_diag / inv_diag.sum()
        return w @ sub_cov @ w

    def _recursive_bisection(cov, sorted_idx):
        """递归二分"""
        w = np.zeros(cov.shape[0])
        items = [sorted_idx]

        while items:
            new_items = []
            for cluster in items:
                if len(cluster) == 1:
                    w[cluster[0]] = 1.0
                    continue
                mid = len(cluster) // 2
                left = cluster[:mid]
                right = cluster[mid:]

                var_left = _get_cluster_var(cov, left)
                var_right = _get_cluster_var(cov, right)

                alpha = 1.0 - var_left / (var_left + var_right)

                for i in left:
                    w[i] *= alpha
                for i in right:
                    w[i] *= (1.0 - alpha)

                new_items.append(left)
                new_items.append(right)

            items = new_items

        return w

    # 初始化权重为 1
    w = np.ones(n)
    items = [sorted_indices]

    while items:
        new_items = []
        for cluster in items:
            if len(cluster) <= 1:
                continue
            mid = len(cluster) // 2
            left = cluster[:mid]
            right = cluster[mid:]

            var_left = _get_cluster_var(cov, left)
            var_right = _get_cluster_var(cov, right)

            alpha = 1.0 - var_left / (var_left + var_right)

            for i in left:
                w[i] *= alpha
            for i in right:
                w[i] *= (1.0 - alpha)

            new_items.append(left)
            new_items.append(right)

        items = new_items

    w = np.maximum(w, 0)
    w /= w.sum()
    return w


# ============================================================
#  5. Black-Litterman 无观点 (逆优化隐含均衡收益)
# ============================================================

def black_litterman_no_views(returns: pd.DataFrame,
                              risk_aversion: float = 2.5,
                              tau: float = 0.05,
                              max_weight: float = 0.10,
                              **kwargs) -> np.ndarray:
    """
    无观点 Black-Litterman 模型 (Black & Litterman, 1992)

    核心思想:
      不使用不稳定的样本均值作为期望收益, 而是通过逆优化从
      "市场组合"反推出隐含均衡收益 π = δΣw_mkt, 再以此做MV优化。
      由于无法获取市值权重, 使用反方差组合作为市场组合代理。

    步骤:
      1. 估计协方差矩阵 Σ (Ledoit-Wolf 收缩)
      2. 构造市场组合代理: w_mkt ∝ 1/σ_i²
      3. 隐含均衡收益: π = δΣw_mkt
      4. 后验协方差: Σ_BL = (1+τ)Σ
      5. MV优化: max w'π - (δ/2)w'Σ_BL*w  s.t. Σw=1, w≥0, w≤β_max

    优点: 期望收益估计稳定, 避免样本均值的噪声
    """
    n = returns.shape[1]
    lw = LedoitWolf().fit(returns.values)
    cov = lw.covariance_ * 252  # 年化协方差

    # 市场组合代理: 反方差加权 (比等权更合理的代理)
    inv_var = 1.0 / np.maximum(np.diag(cov), 1e-10)
    w_mkt = inv_var / inv_var.sum()

    # 隐含均衡收益 (§2.1 BL模型)
    pi = risk_aversion * cov @ w_mkt

    # 后验协方差 (无观点时)
    sigma_bl = (1.0 + tau) * cov

    # MV优化: 使用稳定的均衡收益
    def neg_utility(w):
        return -(w @ pi - (risk_aversion / 2) * w @ sigma_bl @ w)

    def neg_utility_grad(w):
        return -(pi - risk_aversion * sigma_bl @ w)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, max_weight)] * n
    w0 = np.ones(n) / n

    result = minimize(neg_utility, w0, jac=neg_utility_grad,
                      method="SLSQP", bounds=bounds,
                      constraints=constraints,
                      options={"maxiter": 1000, "ftol": 1e-12})

    if result.success:
        w = result.x
        w = np.maximum(w, 0)
        w /= w.sum()
        return w
    else:
        return np.ones(n) / n  # fallback


# ============================================================
#  6. Black-Litterman 动量观点 (量化信号自动生成观点)
# ============================================================

def black_litterman_momentum(returns: pd.DataFrame,
                              risk_aversion: float = 2.5,
                              tau: float = 0.05,
                              momentum_window: int = 60,
                              confidence: float = 0.5,
                              max_weight: float = 0.10,
                              **kwargs) -> np.ndarray:
    """
    动量信号 Black-Litterman 模型

    核心思想:
      在无观点BL的基础上, 使用动量信号自动生成投资者观点,
      通过BL贝叶斯框架将先验(均衡收益)与观点(动量信号)融合。

    步骤:
      1. 先验: π = δΣw_mkt (同无观点BL)
      2. 动量信号: 过去 momentum_window 天的年化收益率作为观点
      3. 观点矩阵: P = I (每个资产一个绝对观点)
      4. 观点向量: q = momentum (年化动量信号)
      5. 观点不确定性: Ω = (1/c) * τ * diag(PΣP')
         其中 c 为观点置信度, c越大则越信任动量信号
      6. 后验收益: μ_BL = [(τΣ)^{-1} + P'Ω^{-1}P]^{-1}
                          × [(τΣ)^{-1}π + P'Ω^{-1}q]
      7. 后验协方差: Σ_BL = [(τΣ)^{-1} + P'Ω^{-1}P]^{-1} + Σ
      8. MV优化: max w'μ_BL - (δ/2)w'Σ_BL*w

    参数:
      momentum_window: 动量信号回看天数 (默认60天, 约3个月)
      confidence: 观点置信度 ∈ (0,1], 越高越信任动量 (默认0.5)
    """
    n = returns.shape[1]
    T = len(returns)
    lw = LedoitWolf().fit(returns.values)
    cov = lw.covariance_ * 252  # 年化

    # 市场组合代理: 反方差加权
    inv_var = 1.0 / np.maximum(np.diag(cov), 1e-10)
    w_mkt = inv_var / inv_var.sum()

    # 隐含均衡收益
    pi = risk_aversion * cov @ w_mkt

    # --- 动量信号生成观点 ---
    mom_window = min(momentum_window, T)
    recent = returns.iloc[-mom_window:]
    momentum = recent.mean().values * 252  # 年化动量

    # 对极端动量值进行截断 (防止数值不稳定)
    # 限制在 [-50%, +50%] 年化范围内
    momentum = np.clip(momentum, -0.5, 0.5)

    # 观点矩阵 P = I (绝对观点: 每个资产一个)
    P = np.eye(n)
    q = momentum  # 观点向量

    # 观点不确定性 Ω (对角矩阵)
    # Ω_ii = (1/confidence) * τ * (P Σ P')_ii
    # confidence ∈ (0,1]: 越高越信任动量信号
    conf = np.clip(confidence, 0.01, 1.0)
    omega_diag = (1.0 / conf) * tau * np.diag(P @ cov @ P.T)
    omega_diag = np.maximum(omega_diag, 1e-10)  # 防止除零

    # BL 后验公式
    tau_cov = tau * cov
    # 正则化以确保可逆
    tau_cov_reg = tau_cov + np.eye(n) * 1e-8
    try:
        tau_cov_inv = np.linalg.inv(tau_cov_reg)
    except np.linalg.LinAlgError:
        tau_cov_inv = np.linalg.pinv(tau_cov_reg)

    omega_inv = np.diag(1.0 / omega_diag)

    # 后验精度矩阵
    post_precision = tau_cov_inv + P.T @ omega_inv @ P
    try:
        post_cov_mu = np.linalg.inv(post_precision)
    except np.linalg.LinAlgError:
        post_cov_mu = np.linalg.pinv(post_precision)

    # 后验期望收益
    mu_bl = post_cov_mu @ (tau_cov_inv @ pi + P.T @ omega_inv @ q)

    # 后验协方差
    sigma_bl = post_cov_mu + cov

    # MV优化
    def neg_utility(w):
        return -(w @ mu_bl - (risk_aversion / 2) * w @ sigma_bl @ w)

    def neg_utility_grad(w):
        return -(mu_bl - risk_aversion * sigma_bl @ w)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, max_weight)] * n
    w0 = np.ones(n) / n

    result = minimize(neg_utility, w0, jac=neg_utility_grad,
                      method="SLSQP", bounds=bounds,
                      constraints=constraints,
                      options={"maxiter": 1000, "ftol": 1e-12})

    if result.success:
        w = result.x
        w = np.maximum(w, 0)
        w /= w.sum()
        return w
    else:
        return np.ones(n) / n  # fallback


# ============================================================
#  7. Inverse Centrality (反中心性加权, 原论文方法)
# ============================================================

def inverse_centrality(returns: pd.DataFrame,
                       centralities: pd.DataFrame = None,
                       centrality_type: str = "betweenness",
                       **kwargs) -> np.ndarray:
    """
    反中心性加权: w_i ∝ (1 - c_i)
    c_i 为归一化中心性指标
    """
    if centralities is None:
        raise ValueError("需要提供 centralities DataFrame")

    assets = returns.columns.tolist()
    c = centralities.loc[assets, centrality_type].values

    # 归一化到 [0, 1]
    c_min, c_max = c.min(), c.max()
    if c_max > c_min:
        c_norm = (c - c_min) / (c_max - c_min)
    else:
        c_norm = np.zeros_like(c)

    w = 1.0 - c_norm
    w = np.maximum(w, 1e-6)
    w /= w.sum()
    return w


# ============================================================
#  6. Network-Regularized Risk Parity (本文方法)
# ============================================================

def network_regularized_rp(returns: pd.DataFrame,
                           centralities: pd.DataFrame = None,
                           partition: dict = None,
                           gamma: float = 1.0,
                           centrality_type: str = "betweenness",
                           max_weight: float = 0.10,
                           **kwargs) -> np.ndarray:
    """
    网络正则化风险平价 (本文核心创新)

    两层框架:
      第一层: 跨社区风险平价 (若提供 partition)
      第二层: 社区内部网络正则化风险平价

    目标函数:
      min Σ(RC_i - RC_j)^2 + γ * Σ C_B(i) * w_i^2
      s.t. Σw = 1, w >= 0
    """
    n = returns.shape[1]
    assets = returns.columns.tolist()
    lw = LedoitWolf().fit(returns.values)
    cov = lw.covariance_ * 252

    if centralities is None:
        return risk_parity(returns)

    c = centralities.loc[assets, centrality_type].values
    c_min, c_max = c.min(), c.max()
    if c_max > c_min:
        c_norm = (c - c_min) / (c_max - c_min)
    else:
        c_norm = np.zeros_like(c)

    # --- 无社区划分: 单层网络正则化 RP ---
    if partition is None:
        return _nrrp_single_layer(cov, c_norm, n, gamma, max_weight=max_weight)

    # --- 有社区划分: 两层框架 ---
    # 整理社区
    communities = {}
    for asset in assets:
        comm_id = partition.get(asset, 0)
        if comm_id not in communities:
            communities[comm_id] = []
        communities[comm_id].append(assets.index(asset))

    comm_ids = sorted(communities.keys())
    K = len(comm_ids)

    if K <= 1:
        return _nrrp_single_layer(cov, c_norm, n, gamma, max_weight=max_weight)

    # 第一层: 跨社区风险平价
    comm_returns = []
    for cid in comm_ids:
        idx = communities[cid]
        # 社区等权平均收益
        comm_ret = returns.iloc[:, idx].mean(axis=1)
        comm_returns.append(comm_ret)
    comm_returns_df = pd.DataFrame(comm_returns).T

    lw_comm = LedoitWolf().fit(comm_returns_df.values)
    cov_comm = lw_comm.covariance_ * 252

    # 社区层面风险平价
    def comm_rp_obj(W):
        sigma = np.sqrt(W @ cov_comm @ W)
        if sigma < 1e-10:
            return 1e10
        mrc = cov_comm @ W / sigma
        rc = W * mrc
        target = sigma / K
        return np.sum((rc - target) ** 2)

    W0 = np.ones(K) / K
    res_comm = minimize(comm_rp_obj, W0, method="SLSQP",
                        bounds=[(1e-6, 1.0)] * K,
                        constraints=[{"type": "eq", "fun": lambda W: np.sum(W) - 1.0}],
                        options={"maxiter": 2000, "ftol": 1e-14})

    W_comm = res_comm.x if res_comm.success else np.ones(K) / K
    W_comm = np.maximum(W_comm, 0)
    W_comm /= W_comm.sum()

    # 第二层: 社区内部网络正则化 RP
    w_final = np.zeros(n)
    for k, cid in enumerate(comm_ids):
        idx = communities[cid]
        n_k = len(idx)

        if n_k == 1:
            w_final[idx[0]] = W_comm[k]
            continue

        cov_k = cov[np.ix_(idx, idx)]
        c_k = c_norm[idx]

        w_k = _nrrp_single_layer(cov_k, c_k, n_k, gamma, max_weight=1.0)
        for j, i in enumerate(idx):
            w_final[i] = W_comm[k] * w_k[j]

    w_final = np.maximum(w_final, 0)
    if w_final.sum() > 0:
        w_final /= w_final.sum()
    else:
        w_final = np.ones(n) / n

    return w_final


def _nrrp_single_layer(cov: np.ndarray, c_norm: np.ndarray,
                       n: int, gamma: float,
                       max_weight: float = 0.10) -> np.ndarray:
    """
    单层网络正则化风险平价优化
    min Σ(RC_i - RC_j)^2 + γ * Σ c_i * w_i^2
    s.t. Σw=1, w>=0, w_i<=β_max  (§2.4.3)
    """
    def objective(w):
        sigma_p = np.sqrt(w @ cov @ w)
        if sigma_p < 1e-10:
            return 1e10

        mrc = cov @ w / sigma_p
        rc = w * mrc
        target_rc = sigma_p / n

        # 风险平价项
        rp_loss = np.sum((rc - target_rc) ** 2)

        # 网络正则化项
        network_penalty = gamma * np.sum(c_norm * w ** 2)

        return rp_loss + network_penalty

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(1e-6, max_weight)] * n
    w0 = np.ones(n) / n

    result = minimize(objective, w0, method="SLSQP",
                      bounds=bounds, constraints=constraints,
                      options={"maxiter": 3000, "ftol": 1e-14})

    if result.success:
        w = result.x
        w = np.maximum(w, 0)
        w /= w.sum()
        return w
    else:
        return np.ones(n) / n


# ============================================================
#  8. Direct Centrality (中心性直接加权, §2.2.2 规则2)
# ============================================================

def direct_centrality(returns: pd.DataFrame,
                      centralities: pd.DataFrame = None,
                      centrality_type: str = "betweenness",
                      **kwargs) -> np.ndarray:
    """
    中心性直接加权: w_i ∝ C(i)
    假设高中心性资产具有更高的信息效率或流动性溢价
    (Výrost et al., 2019) (§2.2.2 规则2)
    """
    if centralities is None:
        raise ValueError("需要提供 centralities DataFrame")

    assets = returns.columns.tolist()
    c = centralities.loc[assets, centrality_type].values
    c = np.maximum(c, 1e-6)
    w = c / c.sum()
    return w


# ============================================================
#  9. PageRank Weighting (PageRank加权, §2.2.2 规则3)
# ============================================================

def pagerank_weight(returns: pd.DataFrame,
                    centralities: pd.DataFrame = None,
                    partition: dict = None,
                    pmfg_graph=None,
                    **kwargs) -> np.ndarray:
    """
    PageRank 加权: w_i ∝ PR(i)
    综合考虑直接连接数目和邻居重要性 (§2.2.2 规则3)
    """
    import networkx as _nx
    from network_analysis import compute_correlation, build_pmfg

    assets = returns.columns.tolist()

    if pmfg_graph is not None:
        G = pmfg_graph
    else:
        corr = compute_correlation(returns, method="ledoit_wolf")
        G = build_pmfg(corr)

    pr = _nx.pagerank(G, alpha=0.85)
    pr_values = np.array([pr.get(a, 1.0 / len(assets)) for a in assets])
    w = pr_values / pr_values.sum()
    return w


# ============================================================
#  10. Graph-Enhanced BL (网络增强BL, 本文融合方法)
# ============================================================

def graph_enhanced_bl(returns: pd.DataFrame,
                       centralities: pd.DataFrame = None,
                       partition: dict = None,
                       risk_aversion: float = 2.5,
                       tau: float = 0.05,
                       view_intensity: float = 0.3,
                       max_weight: float = 0.10,
                       **kwargs) -> np.ndarray:
    """
    Graph-Enhanced Black-Litterman (网络增强BL)

    核心创新: 将网络中心性信息转化为Black-Litterman投资者观点

    理论逻辑:
      高中心性(介数)资产位于风险传导网络的核心位置,
      在市场压力时期更容易受到/传播系统性风险传染 (Billio et al., 2012),
      因此应获得 "风险折价" (预期收益低于均衡水平)。

      低中心性资产位于网络外围, 更能规避系统性风险传染,
      因此应获得 "安全溢价" (预期收益高于均衡水平)。

    实现步骤:
      1. 先验: π = δΣw_mkt (逆优化均衡收益)
      2. 网络观点: q_i = π_i - α·z(c_i)·σ_i
         z(c_i) = (c_i - c̄)/s_c 为中心性z-score
         α 为观点强度, σ_i 为资产年化波动率
      3. 观点不确定性: Ω_ii ∝ 1/|z(c_i)| (极端中心性→更确信)
      4. BL后验融合先验与网络观点
      5. MV优化

    优势: 结合BL的期望收益稳定性与网络的风险拓扑信息
    """
    n = returns.shape[1]
    assets = returns.columns.tolist()
    lw = LedoitWolf().fit(returns.values)
    cov = lw.covariance_ * 252

    # 市场组合代理: 反方差加权
    inv_var = 1.0 / np.maximum(np.diag(cov), 1e-10)
    w_mkt = inv_var / inv_var.sum()

    # 隐含均衡收益
    pi = risk_aversion * cov @ w_mkt

    # --- 无中心性: 退化为BL无观点 ---
    if centralities is None:
        return black_litterman_no_views(returns, risk_aversion=risk_aversion,
                                         tau=tau, max_weight=max_weight)

    # --- 网络中心性 → BL观点 ---
    c = centralities.loc[assets, "betweenness"].values
    c_std = c.std()
    if c_std < 1e-10:
        return black_litterman_no_views(returns, risk_aversion=risk_aversion,
                                         tau=tau, max_weight=max_weight)

    z_c = (c - c.mean()) / c_std  # 中心性 z-score
    asset_vol = np.sqrt(np.diag(cov))  # 年化波动率

    # 观点: 高中心性 → 降低期望收益, 低中心性 → 提高期望收益
    # q_i = π_i - view_intensity * z(c_i) * σ_i
    view_adj = -view_intensity * z_c * asset_vol
    q = pi + view_adj

    # 观点矩阵 P=I (绝对观点)
    P = np.eye(n)

    # 观点不确定性: 极端中心性→更确信, 中等中心性→不确信
    z_abs = np.abs(z_c)
    z_max = z_abs.max()
    if z_max > 0:
        conf_scale = np.clip(z_abs / z_max, 0.1, 1.0)
    else:
        conf_scale = np.ones(n) * 0.5

    omega_diag = (1.0 / conf_scale) * tau * np.diag(cov)
    omega_diag = np.maximum(omega_diag, 1e-10)

    # BL后验
    tau_cov = tau * cov + np.eye(n) * 1e-8
    try:
        tau_cov_inv = np.linalg.inv(tau_cov)
    except np.linalg.LinAlgError:
        tau_cov_inv = np.linalg.pinv(tau_cov)

    omega_inv = np.diag(1.0 / omega_diag)

    post_precision = tau_cov_inv + P.T @ omega_inv @ P
    try:
        post_cov_mu = np.linalg.inv(post_precision + np.eye(n) * 1e-10)
    except np.linalg.LinAlgError:
        post_cov_mu = np.linalg.pinv(post_precision)

    mu_bl = post_cov_mu @ (tau_cov_inv @ pi + P.T @ omega_inv @ q)
    sigma_bl = post_cov_mu + cov

    # MV优化
    def neg_utility(w):
        return -(w @ mu_bl - (risk_aversion / 2) * w @ sigma_bl @ w)

    def neg_utility_grad(w):
        return -(mu_bl - risk_aversion * sigma_bl @ w)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, max_weight)] * n
    w0 = np.ones(n) / n

    result = minimize(neg_utility, w0, jac=neg_utility_grad,
                      method="SLSQP", bounds=bounds,
                      constraints=constraints,
                      options={"maxiter": 1000, "ftol": 1e-12})

    if result.success:
        w = result.x
        w = np.maximum(w, 0)
        w /= w.sum()
        return w
    else:
        return black_litterman_no_views(returns, risk_aversion=risk_aversion,
                                         tau=tau, max_weight=max_weight)


# ============================================================
#  11. RMT-Denoised Risk Parity (去噪风险平价)
# ============================================================

def denoised_rp(returns: pd.DataFrame,
                max_weight: float = 0.10,
                **kwargs) -> np.ndarray:
    """
    RMT去噪风险平价

    核心创新: 使用随机矩阵理论(RMT)对协方差矩阵去噪后执行风险平价

    理论基础 (Laloux et al., 1999; López de Prado, 2020):
      Marchenko-Pastur定律: 对于 T×n 的随机矩阵, 样本相关矩阵的
      特征值分布有确定上界 λ_+ = (1+√(n/T))²。
      超过此界的特征值包含真实信号 (市场因子/行业结构),
      低于此界的特征值主要是估计噪声。

      通过将噪声特征值收缩为均值 (保持矩阵迹不变),
      重构的协方差矩阵更准确地反映真实风险结构,
      从而显著提升风险平价策略的权重配置质量。

    步骤:
      1. 计算样本相关矩阵并特征分解
      2. Marchenko-Pastur定律确定噪声上界 λ_+
      3. 收缩噪声特征值 → 均值
      4. 重构去噪协方差矩阵
      5. 基于去噪Σ执行标准风险平价
    """
    n = returns.shape[1]
    T = len(returns)

    if T < n + 10:
        return risk_parity(returns, max_weight=max_weight)

    # Step 1: 样本相关矩阵
    corr = returns.corr().values
    eigenvalues, eigenvectors = np.linalg.eigh(corr)
    # 降序排列
    idx_sort = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx_sort]
    eigenvectors = eigenvectors[:, idx_sort]

    # Step 2: Marchenko-Pastur上界
    q = n / T
    lambda_plus = (1.0 + np.sqrt(q)) ** 2

    # Step 3: 去噪
    denoised_vals = eigenvalues.copy()
    noise_mask = eigenvalues <= lambda_plus
    n_signal = int((~noise_mask).sum())

    if noise_mask.any() and n_signal > 0:
        # 保持迹不变: 噪声特征值替换为均值
        noise_mean = denoised_vals[noise_mask].mean()
        denoised_vals[noise_mask] = noise_mean

    denoised_vals = np.maximum(denoised_vals, 1e-8)

    # Step 4: 重构相关矩阵
    denoised_corr = eigenvectors @ np.diag(denoised_vals) @ eigenvectors.T
    d = np.sqrt(np.diag(denoised_corr))
    d = np.maximum(d, 1e-8)
    denoised_corr = denoised_corr / np.outer(d, d)
    np.fill_diagonal(denoised_corr, 1.0)

    # Step 5: 转为年化协方差
    std_annual = returns.std().values * np.sqrt(252)
    denoised_cov = denoised_corr * np.outer(std_annual, std_annual)

    # Step 6: 风险平价 (去噪Σ)
    def risk_contrib_obj(w):
        sigma_p = np.sqrt(w @ denoised_cov @ w)
        if sigma_p < 1e-10:
            return 1e10
        mrc = denoised_cov @ w / sigma_p
        rc = w * mrc
        target_rc = sigma_p / n
        return np.sum((rc - target_rc) ** 2)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(1e-6, max_weight)] * n
    w0 = np.ones(n) / n

    result = minimize(risk_contrib_obj, w0, method="SLSQP",
                      bounds=bounds, constraints=constraints,
                      options={"maxiter": 2000, "ftol": 1e-14})

    if result.success:
        w = result.x
        w = np.maximum(w, 0)
        w /= w.sum()
        return w
    else:
        return risk_parity(returns, max_weight=max_weight)


# ============================================================
#  12. Community BL-RP Hybrid (社区层级BL-RP混合, 本文融合方法)
# ============================================================

def community_bl_rp(returns: pd.DataFrame,
                     centralities: pd.DataFrame = None,
                     partition: dict = None,
                     risk_aversion: float = 2.5,
                     tau: float = 0.05,
                     max_weight: float = 0.10,
                     **kwargs) -> np.ndarray:
    """
    社区层级 BL-RP 混合策略

    核心创新: 三层融合框架, 结合图模型、BL、风险平价三者优势

    Layer 1: 跨社区风险平价
      使用PMFG社区检测结果将资产分为K个社区。
      社区间采用风险平价配置, 确保系统性分散化,
      比HRP的二叉树分割更具经济学意义。

    Layer 2: 社区内BL均衡优化
      每个社区内部使用BL无观点模型 (逆优化均衡收益),
      期望收益比样本均值稳定, 降低估计误差,
      同时充分利用社区内部的协方差结构。

    Layer 3: 中心性自适应权重约束
      高中心性资产的权重上限进一步收紧:
      β_max(i) = β_base * (1 - 0.5 * c_norm(i))
      确保风险传导核心节点不获得过高权重。

    理论优势:
      - 社区结构: 基于PMFG的经济学意义分组 (vs HRP纯数学聚类)
      - BL收益: 逆优化均衡收益的稳定性 (vs 样本均值的噪声)
      - 风险平价: 跨社区风险贡献均衡 (vs 等权的忽视风险)
      - 中心性约束: 系统性风险传导的拓扑保护 (纯BL/RP不具备)
    """
    n = returns.shape[1]
    assets = returns.columns.tolist()
    lw = LedoitWolf().fit(returns.values)
    cov = lw.covariance_ * 252

    # --- 无社区信息: 退化为BL无观点 ---
    if partition is None or centralities is None:
        return black_litterman_no_views(returns, risk_aversion=risk_aversion,
                                         tau=tau, max_weight=max_weight)

    # --- 社区分组 ---
    communities = {}
    for asset in assets:
        cid = partition.get(asset, 0)
        if cid not in communities:
            communities[cid] = []
        communities[cid].append(assets.index(asset))

    comm_ids = sorted(communities.keys())
    K = len(comm_ids)

    if K <= 1:
        return black_litterman_no_views(returns, risk_aversion=risk_aversion,
                                         tau=tau, max_weight=max_weight)

    # --- Layer 1: 跨社区风险平价 ---
    comm_returns_list = []
    for cid in comm_ids:
        idx = communities[cid]
        comm_ret = returns.iloc[:, idx].mean(axis=1)
        comm_returns_list.append(comm_ret)
    comm_returns_df = pd.DataFrame(comm_returns_list).T

    lw_comm = LedoitWolf().fit(comm_returns_df.values)
    cov_comm = lw_comm.covariance_ * 252

    def comm_rp_obj(W):
        sigma = np.sqrt(W @ cov_comm @ W)
        if sigma < 1e-10:
            return 1e10
        mrc = cov_comm @ W / sigma
        rc = W * mrc
        target = sigma / K
        return np.sum((rc - target) ** 2)

    W0 = np.ones(K) / K
    res_comm = minimize(comm_rp_obj, W0, method="SLSQP",
                        bounds=[(1e-6, 1.0)] * K,
                        constraints=[{"type": "eq",
                                     "fun": lambda W: np.sum(W) - 1.0}],
                        options={"maxiter": 2000, "ftol": 1e-14})

    W_comm = res_comm.x if res_comm.success else np.ones(K) / K
    W_comm = np.maximum(W_comm, 0)
    W_comm /= W_comm.sum()

    # --- 中心性归一化 (Layer 3 用) ---
    c = centralities.loc[assets, "betweenness"].values
    c_min, c_max = c.min(), c.max()
    if c_max > c_min:
        c_norm = (c - c_min) / (c_max - c_min)
    else:
        c_norm = np.zeros(n)

    # --- Layer 2 + 3: 社区内 BL + 中心性约束 ---
    w_final = np.zeros(n)

    for k, cid in enumerate(comm_ids):
        idx = communities[cid]
        n_k = len(idx)

        if n_k == 1:
            w_final[idx[0]] = W_comm[k]
            continue

        # 社区子数据
        ret_k = returns.iloc[:, idx]
        lw_k = LedoitWolf().fit(ret_k.values)
        cov_k = lw_k.covariance_ * 252

        # BL均衡收益
        inv_var_k = 1.0 / np.maximum(np.diag(cov_k), 1e-10)
        w_mkt_k = inv_var_k / inv_var_k.sum()
        pi_k = risk_aversion * cov_k @ w_mkt_k
        sigma_bl_k = (1 + tau) * cov_k

        # Layer 3: 中心性自适应权重上限
        c_k = c_norm[idx]
        max_w_k = np.array([max(max_weight * (1.0 - 0.5 * c_k[j]), 0.02)
                            for j in range(n_k)])

        # BL均衡收益优化
        def neg_utility_k(w, _pi=pi_k, _sig=sigma_bl_k, _ra=risk_aversion):
            return -(w @ _pi - (_ra / 2) * w @ _sig @ w)

        def neg_utility_k_grad(w, _pi=pi_k, _sig=sigma_bl_k,
                                _ra=risk_aversion):
            return -(_pi - _ra * _sig @ w)

        constraints_k = [{"type": "eq",
                          "fun": lambda w: np.sum(w) - 1.0}]
        bounds_k = [(0.0, float(max_w_k[j])) for j in range(n_k)]
        w0_k = np.ones(n_k) / n_k

        result_k = minimize(neg_utility_k, w0_k, jac=neg_utility_k_grad,
                            method="SLSQP", bounds=bounds_k,
                            constraints=constraints_k,
                            options={"maxiter": 1000, "ftol": 1e-12})

        if result_k.success:
            w_k = result_k.x
            w_k = np.maximum(w_k, 0)
            w_k /= w_k.sum()
        else:
            w_k = np.ones(n_k) / n_k

        for j, i in enumerate(idx):
            w_final[i] = W_comm[k] * w_k[j]

    w_final = np.maximum(w_final, 0)
    if w_final.sum() > 0:
        w_final /= w_final.sum()
    else:
        w_final = np.ones(n) / n

    return w_final


# ============================================================
#  γ 参数交叉验证选择
# ============================================================

def select_gamma_cv(returns_train: pd.DataFrame,
                    returns_val: pd.DataFrame,
                    centralities: pd.DataFrame,
                    partition: dict = None,
                    gamma_grid: list = None,
                    metric: str = "sharpe") -> dict:
    """
    通过时序交叉验证选择最优 γ

    Args:
        returns_train: 训练集收益率
        returns_val: 验证集收益率
        centralities: PMFG 中心性
        partition: 社区划分
        gamma_grid: γ 候选值
        metric: 优化目标 'sharpe' | 'neg_cvar'

    Returns:
        {best_gamma, results_df}
    """
    if gamma_grid is None:
        gamma_grid = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

    results = []
    for g in gamma_grid:
        w = network_regularized_rp(returns_train, centralities=centralities,
                                   partition=partition, gamma=g)
        # 在验证集上评估
        port_ret = returns_val.values @ w
        ann_ret = port_ret.mean() * 252
        ann_vol = port_ret.std() * np.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
        cvar_95 = -np.percentile(port_ret, 5)

        results.append({
            "gamma": g,
            "ann_return": round(ann_ret * 100, 2),
            "ann_volatility": round(ann_vol * 100, 2),
            "sharpe": round(sharpe, 4),
            "cvar_95": round(cvar_95 * 100, 4),
        })

    results_df = pd.DataFrame(results)

    if metric == "sharpe":
        best_idx = results_df["sharpe"].idxmax()
    else:
        best_idx = results_df["cvar_95"].idxmin()

    best_gamma = results_df.loc[best_idx, "gamma"]
    return {"best_gamma": best_gamma, "results_df": results_df}


# ============================================================
#  策略工厂: 统一接口
# ============================================================

STRATEGY_REGISTRY = {
    # 基准
    "equal_weight":      {"func": equal_weight,      "label": "等权(1/N)",          "type": "基准"},
    # 传统
    "mean_variance":     {"func": mean_variance,     "label": "均值方差(MV)",       "type": "传统"},
    "risk_parity":       {"func": risk_parity,       "label": "风险平价(RP)",       "type": "传统"},
    "hrp":               {"func": hrp,               "label": "层次化RP(HRP)",      "type": "传统"},
    "bl_no_views":       {"func": black_litterman_no_views,  "label": "BL无观点",    "type": "传统"},
    "bl_momentum":       {"func": black_litterman_momentum,  "label": "BL动量观点",  "type": "传统"},
    # 图模型 (简单规则)
    "inv_centrality":    {"func": inverse_centrality, "label": "反中心性(规则1)",    "type": "图模型"},
    "direct_centrality": {"func": direct_centrality,  "label": "直接中心性(规则2)",  "type": "图模型"},
    "pagerank":          {"func": pagerank_weight,    "label": "PageRank(规则3)",    "type": "图模型"},
    # 图模型 (融合方法, 本文贡献)
    "network_reg_rp":    {"func": network_regularized_rp,  "label": "网络正则化RP",      "type": "融合"},
    "graph_bl":          {"func": graph_enhanced_bl,       "label": "网络增强BL",        "type": "融合"},
    "denoised_rp":       {"func": denoised_rp,             "label": "去噪风险平价",      "type": "融合"},
    "community_bl_rp":   {"func": community_bl_rp,         "label": "社区BL-RP混合",     "type": "融合"},
}


def compute_weights(strategy_name: str, returns: pd.DataFrame,
                    **kwargs) -> np.ndarray:
    """统一策略接口"""
    if strategy_name not in STRATEGY_REGISTRY:
        raise ValueError(f"未知策略: {strategy_name}")
    func = STRATEGY_REGISTRY[strategy_name]["func"]
    return func(returns, **kwargs)
