"""
投资组合策略模块
================
20种资产配置策略实现 (与 theoretical_framework_v2 §2.3.2 对应):
  基准 (1种):
    1. Equal Weight (1/N)
  传统 (7种):
    2. Mean-Variance (Markowitz + Ledoit-Wolf 收缩)
    3. Risk Parity (标准风险平价)
    4. HRP (层次化风险平价)
    5. Black-Litterman 无观点 (逆优化隐含均衡收益)
    6. Black-Litterman 动量观点 (量化信号自动生成观点)
    7. Minimum Variance (最小方差, Jagannathan & Ma 2003)
    8. Maximum Diversification (最大分散化, Choueifaty 2008)
  图模型简单规则 (3种):
    9. Inverse Centrality (反中心性, §2.3.2 规则1)
   10. Direct Centrality (中心性直接加权, §2.3.2 规则2)
   11. PageRank Weighting (§2.3.2 规则3)
  融合方法 (9种, 含本文贡献):
   12. Network-Regularized Risk Parity (§2.4.2 NRRP)
   13. Graph-Enhanced Black-Litterman (§2.4.3 网络增强BL)
   14. RMT-Denoised Risk Parity (§2.4.4 去噪风险平价)
   15. Community BL-RP Hybrid (§2.4.5 社区BL-RP混合)
   16. Graph Laplacian MV (图拉普拉斯正则化)
   17. PMFG-Filtered RP (图结构协方差过滤RP)
   18. Graph-Smoothed Momentum BL (图平滑动量BL)
  ★ 图模型+MinVar融合 (2种, v3核心创新):
   19. Network-Constrained MinVar (网络约束最小方差, §2.4.10)
   20. Community-MinVar (社区平衡最小方差, §2.4.11)
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
    含单一标的集中度约束 (§2.3.4)
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
    含单一标的集中度约束 (§2.3.4)
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
#  10. Network-Regularized Risk Parity (§2.4.2 NRRP)
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
    lw = LedoitWolf().fit(returns.to_numpy())
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

    lw_comm = LedoitWolf().fit(comm_returns_df.to_numpy())
    cov_comm = lw_comm.covariance_ * 252

    # 社区层面风险平价
    def comm_rp_obj(w):
        sigma = np.sqrt(w @ cov_comm @ w)
        if sigma < 1e-10:
            return 1e10
        mrc = cov_comm @ w / sigma
        rc = w * mrc
        target = sigma / K
        return np.sum((rc - target) ** 2)

    w0 = np.ones(K) / K
    res_comm = minimize(comm_rp_obj, w0, method="SLSQP",
                        bounds=[(1e-6, 1.0)] * K,
                        constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}],
                        options={"maxiter": 2000, "ftol": 1e-14})

    w_comm = res_comm.x if res_comm.success else np.ones(K) / K
    w_comm = np.maximum(w_comm, 0)
    w_comm /= w_comm.sum()

    # 第二层: 社区内部网络正则化 RP
    w_final = np.zeros(n)
    for k, cid in enumerate(comm_ids):
        idx = communities[cid]
        n_k = len(idx)

        if n_k == 1:
            w_final[idx[0]] = w_comm[k]
            continue

        cov_k = cov[np.ix_(idx, idx)]
        c_k = c_norm[idx]

        w_k = _nrrp_single_layer(cov_k, c_k, n_k, gamma, max_weight=1.0)
        for j, i in enumerate(idx):
            w_final[i] = w_comm[k] * w_k[j]

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
    s.t. Σw=1, w>=0, w_i<=β_max  (§2.3.4)
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
#  8. Direct Centrality (中心性直接加权, §2.3.2 规则2)
# ============================================================

def direct_centrality(returns: pd.DataFrame,
                      centralities: pd.DataFrame = None,
                      centrality_type: str = "betweenness",
                      **kwargs) -> np.ndarray:
    """
    中心性直接加权: w_i ∝ C(i)
    假设高中心性资产具有更高的信息效率或流动性溢价
    (Výrost et al., 2019) (§2.3.2 规则2)
    """
    if centralities is None:
        raise ValueError("需要提供 centralities DataFrame")

    assets = returns.columns.tolist()
    c = centralities.loc[assets, centrality_type].values
    c = np.maximum(c, 1e-6)
    w = c / c.sum()
    return w


# ============================================================
#  9. PageRank Weighting (PageRank加权, §2.3.2 规则3)
# ============================================================

def pagerank_weight(returns: pd.DataFrame,
                    centralities: pd.DataFrame = None,
                    partition: dict = None,
                    pmfg_graph=None,
                    **kwargs) -> np.ndarray:
    """
    PageRank 加权: w_i ∝ PR(i)
    综合考虑直接连接数目和邻居重要性 (§2.3.2 规则3)
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

    lw_comm = LedoitWolf().fit(comm_returns_df.to_numpy())
    cov_comm = lw_comm.covariance_ * 252

    def comm_rp_obj(w):
        sigma = np.sqrt(w @ cov_comm @ w)
        if sigma < 1e-10:
            return 1e10
        mrc = cov_comm @ w / sigma
        rc = w * mrc
        target = sigma / K
        return np.sum((rc - target) ** 2)

    w0 = np.ones(K) / K
    res_comm = minimize(comm_rp_obj, w0, method="SLSQP",
                        bounds=[(1e-6, 1.0)] * K,
                        constraints=[{"type": "eq",
                                     "fun": lambda w: np.sum(w) - 1.0}],
                        options={"maxiter": 2000, "ftol": 1e-14})

    w_comm = res_comm.x if res_comm.success else np.ones(K) / K
    w_comm = np.maximum(w_comm, 0)
    w_comm /= w_comm.sum()

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
            w_final[idx[0]] = w_comm[k]
            continue

        # 社区子数据
        ret_k = returns.iloc[:, idx]
        lw_k = LedoitWolf().fit(ret_k.to_numpy())
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
            w_final[i] = w_comm[k] * w_k[j]

    w_final = np.maximum(w_final, 0)
    if w_final.sum() > 0:
        w_final /= w_final.sum()
    else:
        w_final = np.ones(n) / n

    return w_final


# ============================================================
#  13. Minimum Variance (最小方差组合, 前沿基准)
# ============================================================

def minimum_variance(returns: pd.DataFrame,
                      max_weight: float = 0.10,
                      **kwargs) -> np.ndarray:
    """
    最小方差组合 (Minimum Variance Portfolio)

    min  w'Σw   s.t.  Σw=1, w≥0, w_i≤β_max

    理论基础 (Jagannathan & Ma, 2003):
      不需要期望收益估计, 仅依赖协方差矩阵。
      当真实Sharpe比率差异较小时, 最小方差组合在
      样本外表现优于均值方差组合。
      加入上界约束等价于对协方差矩阵施加收缩。

    优势: 完全避免μ估计误差; 在高波动市场中表现优异
    """
    n = returns.shape[1]
    lw = LedoitWolf().fit(returns.values)
    cov = lw.covariance_ * 252

    def portfolio_var(w):
        return w @ cov @ w

    def portfolio_var_grad(w):
        return 2 * cov @ w

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, max_weight)] * n
    w0 = np.ones(n) / n

    result = minimize(portfolio_var, w0, jac=portfolio_var_grad,
                      method="SLSQP", bounds=bounds,
                      constraints=constraints,
                      options={"maxiter": 1000, "ftol": 1e-12})

    if result.success:
        w = result.x
        w = np.maximum(w, 0)
        w /= w.sum()
        return w
    else:
        return np.ones(n) / n


# ============================================================
#  14. Maximum Diversification (最大分散化组合)
# ============================================================

def max_diversification(returns: pd.DataFrame,
                         max_weight: float = 0.10,
                         **kwargs) -> np.ndarray:
    """
    最大分散化组合 (Choueifaty & Coignard, 2008)

    max  DR(w) = (w'σ) / √(w'Σw)

    等价优化:
      min  w'Σw / (w'σ)²
      即: min  w'Cw  s.t. w'1=1, w≥0  其中 C = 相关矩阵

    理论基础:
      DR衡量组合的分散化程度。DR=1表示完全不分散(单一资产),
      DR越大表示分散化越充分。最大化DR等价于在相关矩阵上
      做最小方差优化, 不需要期望收益估计。
      (Choueifaty, Froidure & Reynier, 2013)

    优势: 显式最大化分散化; 对估计误差稳健; 理论上在
          相关性结构稳定时表现优于等权和最小方差
    """
    n = returns.shape[1]
    lw = LedoitWolf().fit(returns.values)
    cov = lw.covariance_ * 252

    # 相关矩阵
    std = np.sqrt(np.diag(cov))
    std = np.maximum(std, 1e-10)
    corr = cov / np.outer(std, std)

    # 在相关矩阵上做最小方差
    def corr_var(w):
        return w @ corr @ w

    def corr_var_grad(w):
        return 2 * corr @ w

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, max_weight)] * n
    w0 = np.ones(n) / n

    result = minimize(corr_var, w0, jac=corr_var_grad,
                      method="SLSQP", bounds=bounds,
                      constraints=constraints,
                      options={"maxiter": 1000, "ftol": 1e-12})

    if result.success:
        w = result.x
        w = np.maximum(w, 0)
        w /= w.sum()
        return w
    else:
        return np.ones(n) / n


# ============================================================
#  15. Graph Laplacian Regularized MV (图拉普拉斯正则化)
# ============================================================

def graph_laplacian_mv(returns: pd.DataFrame,
                        centralities: pd.DataFrame = None,
                        partition: dict = None,
                        pmfg_graph=None,
                        lambda_lap: float = 0.1,
                        risk_aversion: float = 2.0,
                        max_weight: float = 0.10,
                        **kwargs) -> np.ndarray:
    """
    图拉普拉斯正则化均值方差 (Graph Laplacian Regularized MV)

    min  (λ/2)w'Σw + λ_L·w'Lw - w'μ
    s.t. Σw=1, w≥0, w_i≤β_max

    其中 L = D - A 为PMFG网络的图拉普拉斯矩阵,
    D 为度矩阵, A 为(相关性加权)邻接矩阵。

    理论基础 (Cardoso et al., 2021; Peralta & Zareei, 2016):
      拉普拉斯正则化项 w'Lw = Σ_{(i,j)∈E} A_ij(w_i-w_j)²
      惩罚网络中相连资产之间的权重差异。

      经济学含义: 在PMFG中相连的资产具有强相关性,
      若它们的配置权重差异过大, 则组合对该"相关性桥梁"
      的暴露不均衡, 在冲击传导时会产生风险集中。
      拉普拉斯正则化迫使相连资产的权重趋近,
      实现"沿网络结构的平滑配置"。

    vs NRRP:
      NRRP惩罚高中心性节点 → 减少核心节点权重
      Laplacian惩罚相连节点的权重差异 → 权重沿网络结构平滑
      两者互补: NRRP是节点级正则化, Laplacian是边级正则化
    """
    import networkx as _nx

    n = returns.shape[1]
    assets = returns.columns.tolist()
    lw = LedoitWolf().fit(returns.values)
    cov = lw.covariance_ * 252
    mu = returns.mean().values * 252

    # 构建图拉普拉斯矩阵
    if pmfg_graph is not None:
        G = pmfg_graph
    else:
        from network_analysis import compute_correlation, build_pmfg
        corr = compute_correlation(returns, method="ledoit_wolf")
        G = build_pmfg(corr)

    # 邻接矩阵 (相关性加权)
    A = np.zeros((n, n))
    for u, v, data in G.edges(data=True):
        i = assets.index(u) if u in assets else -1
        j = assets.index(v) if v in assets else -1
        if i >= 0 and j >= 0:
            w_edge = max(data.get("weight", 0.5), 0.0)
            A[i, j] = w_edge
            A[j, i] = w_edge

    # 图拉普拉斯 L = D - A
    D = np.diag(A.sum(axis=1))
    L = D - A

    # 复合目标矩阵: (λ/2)Σ + λ_L·L
    Q = risk_aversion * cov + 2 * lambda_lap * L

    def objective(w):
        return 0.5 * w @ Q @ w - w @ mu

    def objective_grad(w):
        return Q @ w - mu

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, max_weight)] * n
    w0 = np.ones(n) / n

    result = minimize(objective, w0, jac=objective_grad,
                      method="SLSQP", bounds=bounds,
                      constraints=constraints,
                      options={"maxiter": 1000, "ftol": 1e-12})

    if result.success:
        w = result.x
        w = np.maximum(w, 0)
        w /= w.sum()
        return w
    else:
        return np.ones(n) / n


# ============================================================
#  16. PMFG-Filtered Covariance RP (图结构协方差过滤风险平价)
# ============================================================

def pmfg_filtered_rp(returns: pd.DataFrame,
                      centralities: pd.DataFrame = None,
                      partition: dict = None,
                      pmfg_graph=None,
                      shrinkage: float = 0.8,
                      max_weight: float = 0.10,
                      **kwargs) -> np.ndarray:
    """
    PMFG图结构协方差过滤风险平价

    核心创新: 利用PMFG的边结构作为协方差矩阵的稀疏性先验,
    对非连接资产对的相关性施加更强的收缩, 保留连接资产对的
    相关性信号。

    过滤公式:
      Σ_filtered_ij = Σ_ij,                            若(i,j)∈E_PMFG
      Σ_filtered_ij = (1-shrinkage) · Σ_ij,            若(i,j)∉E_PMFG
      Σ_filtered_ii = Σ_ii                              (对角线不变)

    理论基础 (Aste & Di Matteo, 2017; Millington & Niranjan, 2020):
      PMFG保留了资产间最强的3(n-2)个边际相关关系。
      非连接资产对的相关性更可能是由共同因子间接驱动的
      "伪相关"或估计噪声。将其收缩可以:
      1. 降低协方差矩阵的有效秩, 提高数值稳定性
      2. 保留真实的强依赖信号, 消除弱/伪信号
      3. 使风险平价的风险贡献计算更准确

    vs RMT去噪:
      RMT在特征值空间去噪 (谱域), 保留因子结构
      PMFG过滤在相关性空间去噪 (空间域), 保留图结构
      两者互补: RMT更适合因子驱动的市场,
               PMFG过滤更适合网络结构清晰的市场
    """
    n = returns.shape[1]
    assets = returns.columns.tolist()
    lw = LedoitWolf().fit(returns.values)
    cov = lw.covariance_ * 252

    # 构建PMFG连接掩码
    if pmfg_graph is not None:
        G = pmfg_graph
    else:
        from network_analysis import compute_correlation, build_pmfg
        corr = compute_correlation(returns, method="ledoit_wolf")
        G = build_pmfg(corr)

    # PMFG连接矩阵
    connected = np.zeros((n, n), dtype=bool)
    for u, v in G.edges():
        i = assets.index(u) if u in assets else -1
        j = assets.index(v) if v in assets else -1
        if i >= 0 and j >= 0:
            connected[i, j] = True
            connected[j, i] = True

    # 过滤协方差矩阵
    filtered_cov = cov.copy()
    for i in range(n):
        for j in range(n):
            if i != j and not connected[i, j]:
                filtered_cov[i, j] *= (1.0 - shrinkage)

    # 确保正定性 (特征值截断)
    eigenvalues, eigenvectors = np.linalg.eigh(filtered_cov)
    eigenvalues = np.maximum(eigenvalues, 1e-8)
    filtered_cov = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

    # 标准风险平价 (基于过滤后Σ)
    def risk_contrib_obj(w):
        sigma_p = np.sqrt(w @ filtered_cov @ w)
        if sigma_p < 1e-10:
            return 1e10
        mrc = filtered_cov @ w / sigma_p
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
#  17. Graph-Smoothed Momentum BL (图平滑动量BL)
# ============================================================

def graph_smoothed_bl(returns: pd.DataFrame,
                       centralities: pd.DataFrame = None,
                       partition: dict = None,
                       pmfg_graph=None,
                       risk_aversion: float = 2.5,
                       tau: float = 0.05,
                       momentum_window: int = 60,
                       smoothing_alpha: float = 0.5,
                       confidence: float = 0.5,
                       max_weight: float = 0.10,
                       **kwargs) -> np.ndarray:
    """
    图平滑动量BL (Graph-Smoothed Momentum BL)

    核心创新: 使用PMFG网络结构对动量信号进行图平滑,
    然后将平滑后的信号作为BL投资者观点。

    图平滑公式:
      mom_smooth(i) = (1-α)·mom(i) + α·Σ_{j∈N(i)} w_ij·mom(j)

      其中 N(i) 为PMFG中i的邻居集, w_ij = corr(i,j)/Σ_k corr(i,k)
      α ∈ [0,1] 为平滑强度

    理论基础 (Marti et al., 2021; Shuman et al., 2013):
      1. 图信号处理: 动量信号可视为定义在资产网络上的"图信号",
         图平滑等价于低通滤波, 去除高频噪声保留趋势信号。
      2. 信息聚合: 单资产动量受特异性噪声影响大,
         与网络邻居聚合后捕获的是行业/风格级别的趋势,
         信噪比更高 (类似因子动量 vs 个股动量)。
      3. 修复BL动量的失败: 原始BL动量(Sharpe 0.84)失败的原因是
         个股动量太噪声; 图平滑后的动量信噪比提高, 应显著改善。

    vs 原始BL动量:
      BL动量直接用个股60天收益率 → 噪声大, 换手高
      图平滑BL用网络邻居聚合后的趋势 → 更稳定, 信噪比高
    """
    import networkx as _nx

    n = returns.shape[1]
    T = len(returns)
    assets = returns.columns.tolist()
    lw = LedoitWolf().fit(returns.values)
    cov = lw.covariance_ * 252

    # 市场组合代理
    inv_var = 1.0 / np.maximum(np.diag(cov), 1e-10)
    w_mkt = inv_var / inv_var.sum()
    pi = risk_aversion * cov @ w_mkt

    # --- 原始动量信号 ---
    mom_win = min(momentum_window, T)
    raw_mom = returns.iloc[-mom_win:].mean().values * 252
    raw_mom = np.clip(raw_mom, -0.5, 0.5)

    # --- 图平滑 ---
    if pmfg_graph is not None:
        G = pmfg_graph
    else:
        from network_analysis import compute_correlation, build_pmfg
        corr_mat = compute_correlation(returns, method="ledoit_wolf")
        G = build_pmfg(corr_mat)

    smoothed_mom = raw_mom.copy()
    for i, asset in enumerate(assets):
        if asset not in G:
            continue
        neighbors = list(G.neighbors(asset))
        if not neighbors:
            continue
        # 邻居权重 = 边权(相关性)归一化
        neighbor_weights = []
        neighbor_moms = []
        for nb in neighbors:
            if nb in assets:
                j = assets.index(nb)
                edge_w = G[asset][nb].get("weight", 0.5)
                neighbor_weights.append(max(edge_w, 0.01))
                neighbor_moms.append(raw_mom[j])

        if neighbor_weights:
            total_w = sum(neighbor_weights)
            weighted_nb_mom = sum(
                w * m for w, m in zip(neighbor_weights, neighbor_moms)
            ) / total_w
            smoothed_mom[i] = ((1 - smoothing_alpha) * raw_mom[i]
                               + smoothing_alpha * weighted_nb_mom)

    # --- BL融合 ---
    P = np.eye(n)
    q = smoothed_mom

    conf = np.clip(confidence, 0.01, 1.0)
    omega_diag = (1.0 / conf) * tau * np.diag(cov)
    omega_diag = np.maximum(omega_diag, 1e-10)

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
#  19. Network-Constrained MinVar (网络约束最小方差)
# ============================================================

def network_constrained_minvar(returns: pd.DataFrame,
                               centralities: pd.DataFrame = None,
                               max_weight: float = 0.10,
                               centrality_shrink: float = 0.5,
                               **kwargs) -> np.ndarray:
    """
    网络约束最小方差 (Network-Constrained MinVar)

    核心思想:
      MinVar 仅依赖 Σ, 完全避免 μ 估计误差。
      其瓶颈在于: 可能过度集中于低波动但系统重要性高的资产
      (如银行股、宽基指数), 在系统性风险爆发时集中暴露。

      本策略利用网络介数中心性(衡量系统性风险传导桥梁角色)
      对 MinVar 的权重上界施加 **自适应约束**:
        β_max(i) = β_base × (1 - shrink × normalized_centrality(i))

      高中心性资产(风险传导枢纽) → 更低的权重上界
      低中心性资产(网络外围)     → 保持原始上界

    理论依据:
      - Jagannathan & Ma (2003): 约束隐式等价于协方差收缩
      - 命题1 (中心性-风险传染假说): 高中心性资产的系统性风险更高
      - 自适应约束 ≠ 改变目标函数, 保留了MinVar的最优性结构

    参数:
      centrality_shrink: 最大收缩比例 (默认0.5, 即最高中心性资产
                         的上界收缩到 β_base × 0.5)
    """
    n = returns.shape[1]
    lw = LedoitWolf().fit(returns.values)
    cov = lw.covariance_ * 252

    # 自适应权重上界
    if centralities is not None and 'betweenness' in centralities.columns:
        assets = returns.columns.tolist()
        betw = np.array([
            centralities.loc[a, 'betweenness']
            if a in centralities.index else 0.0
            for a in assets
        ])
        # Min-Max归一化到 [0, 1]
        b_min, b_max = betw.min(), betw.max()
        if b_max > b_min:
            norm_betw = (betw - b_min) / (b_max - b_min)
        else:
            norm_betw = np.zeros(n)

        # 自适应上界: 高中心性 → 更低的上界
        adaptive_ub = max_weight * (1.0 - centrality_shrink * norm_betw)
        # 确保下界不低于 max_weight 的 20%
        adaptive_ub = np.maximum(adaptive_ub, max_weight * 0.2)
    else:
        adaptive_ub = np.full(n, max_weight)

    def portfolio_var(w):
        return w @ cov @ w

    def portfolio_var_grad(w):
        return 2 * cov @ w

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, float(adaptive_ub[i])) for i in range(n)]
    w0 = np.ones(n) / n

    result = minimize(portfolio_var, w0, jac=portfolio_var_grad,
                      method="SLSQP", bounds=bounds,
                      constraints=constraints,
                      options={"maxiter": 1000, "ftol": 1e-12})

    if result.success:
        w = result.x
        w = np.maximum(w, 0)
        w /= w.sum()
        return w
    else:
        return minimum_variance(returns, max_weight=max_weight)


# ============================================================
#  20. Community-MinVar (社区平衡最小方差)
# ============================================================

def community_minvar(returns: pd.DataFrame,
                     centralities: pd.DataFrame = None,
                     partition: dict = None,
                     max_weight: float = 0.10,
                     centrality_shrink: float = 0.3,
                     **kwargs) -> np.ndarray:
    """
    社区平衡最小方差 (Community-Balanced MinVar)

    核心思想 (两层框架):
      Layer 1: 跨社区风险平价 — 确保各网络社区的风险贡献均衡
      Layer 2: 社区内最小方差 — 在每个社区内执行MinVar,
              同时用中心性自适应约束防止集中于系统核心资产

    理论创新:
      标准MinVar的问题: 可能将资金集中在少数低波动社区(如银行社区),
      导致社区间分散化不足。
      本策略融合了:
        - MinVar的低波动优势 (每个社区内部最小化方差)
        - 社区结构的分散化 (跨社区风险平价, 命题2)
        - 中心性风险控制 (自适应约束, 命题1)

    对比:
      vs 标准MinVar:   增加了跨社区分散化
      vs Community-BL-RP: 用MinVar替代BL(不需μ估计)
      vs HRP:          用PMFG社区替代二叉树聚类(经济学含义更强)
    """
    n = returns.shape[1]
    assets = returns.columns.tolist()

    # --- 获取社区划分 ---
    if partition is None:
        # 无社区信息, 退化为网络约束MinVar
        return network_constrained_minvar(
            returns, centralities=centralities,
            max_weight=max_weight, centrality_shrink=centrality_shrink
        )

    # 社区映射
    communities = {}
    for asset in assets:
        c = partition.get(asset, 0)
        if c not in communities:
            communities[c] = []
        communities[c].append(asset)

    # 至少2个社区才有分层意义
    if len(communities) < 2:
        return network_constrained_minvar(
            returns, centralities=centralities,
            max_weight=max_weight, centrality_shrink=centrality_shrink
        )

    # --- Layer 1: 跨社区风险平价 ---
    lw_full = LedoitWolf().fit(returns.to_numpy())
    cov_full = lw_full.covariance_ * 252

    # 社区等权收益率
    comm_ids = sorted(communities.keys())
    K = len(comm_ids)
    comm_returns = pd.DataFrame()
    for c_id in comm_ids:
        members = communities[c_id]
        cols = [a for a in members if a in returns.columns]
        if len(cols) > 0:
            comm_returns[f'comm_{c_id}'] = returns[cols].mean(axis=1)

    if comm_returns.shape[1] < 2:
        return network_constrained_minvar(
            returns, centralities=centralities,
            max_weight=max_weight, centrality_shrink=centrality_shrink
        )

    # 社区间风险平价
    lw_comm = LedoitWolf().fit(comm_returns.to_numpy())
    cov_comm = lw_comm.covariance_ * 252

    def rp_obj(w_c):
        sigma_p = np.sqrt(w_c @ cov_comm @ w_c + 1e-12)
        mrc = cov_comm @ w_c / sigma_p
        rc = w_c * mrc
        target = sigma_p / len(w_c)
        return np.sum((rc - target) ** 2)

    cons_c = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bnds_c = [(0.01, 1.0)] * K
    w0_c = np.ones(K) / K

    res_c = minimize(rp_obj, w0_c, method="SLSQP",
                     bounds=bnds_c, constraints=cons_c,
                     options={"maxiter": 500, "ftol": 1e-10})

    if res_c.success:
        w_comm = np.maximum(res_c.x, 0)
        w_comm /= w_comm.sum()
    else:
        w_comm = np.ones(K) / K

    # --- Layer 2: 社区内 MinVar + 中心性约束 ---
    w_final = np.zeros(n)

    for idx, c_id in enumerate(comm_ids):
        members = communities[c_id]
        cols = [a for a in members if a in returns.columns]
        if len(cols) == 0:
            continue

        member_idx = [assets.index(a) for a in cols]
        nc = len(cols)

        if nc == 1:
            w_final[member_idx[0]] = w_comm[idx]
            continue

        # 社区内协方差
        cov_sub = cov_full[np.ix_(member_idx, member_idx)]

        # 中心性自适应约束
        if centralities is not None and 'betweenness' in centralities.columns:
            betw_sub = np.array([
                centralities.loc[a, 'betweenness']
                if a in centralities.index else 0.0
                for a in cols
            ])
            b_min, b_max = betw_sub.min(), betw_sub.max()
            if b_max > b_min:
                norm_b = (betw_sub - b_min) / (b_max - b_min)
            else:
                norm_b = np.zeros(nc)
            # 社区内上界 (相对于社区权重)
            relative_ub = (1.0 - centrality_shrink * norm_b)
            relative_ub = np.maximum(relative_ub, 0.2)
        else:
            relative_ub = np.ones(nc)

        # 社区内 MinVar
        def sub_var(w, cov_sub=cov_sub):
            return w @ cov_sub @ w

        def sub_var_grad(w, cov_sub=cov_sub):
            return 2 * cov_sub @ w

        cons_sub = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bnds_sub = [(0.0, float(relative_ub[j])) for j in range(nc)]
        w0_sub = np.ones(nc) / nc

        res_sub = minimize(sub_var, w0_sub, jac=sub_var_grad,
                           method="SLSQP", bounds=bnds_sub,
                           constraints=cons_sub,
                           options={"maxiter": 500, "ftol": 1e-12})

        if res_sub.success:
            w_sub = np.maximum(res_sub.x, 0)
            w_sub /= w_sub.sum()
        else:
            w_sub = np.ones(nc) / nc

        for j, mi in enumerate(member_idx):
            w_final[mi] = w_comm[idx] * w_sub[j]

    # 全局最大权重约束 & 归一化
    w_final = np.minimum(w_final, max_weight)
    if w_final.sum() > 0:
        w_final /= w_final.sum()
    else:
        w_final = np.ones(n) / n

    return w_final


# ============================================================
#  策略工厂: 统一接口
# ============================================================

STRATEGY_REGISTRY = {
    # 基准 (1)
    "equal_weight":      {"func": equal_weight,      "label": "等权(1/N)",          "type": "基准"},
    # 传统 (7)
    "mean_variance":     {"func": mean_variance,     "label": "均值方差(MV)",       "type": "传统"},
    "min_variance":      {"func": minimum_variance,  "label": "最小方差(MinVar)",   "type": "传统"},
    "max_div":           {"func": max_diversification, "label": "最大分散化(MaxDiv)", "type": "传统"},
    "risk_parity":       {"func": risk_parity,       "label": "风险平价(RP)",       "type": "传统"},
    "hrp":               {"func": hrp,               "label": "层次化RP(HRP)",      "type": "传统"},
    "bl_no_views":       {"func": black_litterman_no_views,  "label": "BL无观点",    "type": "传统"},
    "bl_momentum":       {"func": black_litterman_momentum,  "label": "BL动量观点",  "type": "传统"},
    # 图模型简单规则 (3)
    "inv_centrality":    {"func": inverse_centrality, "label": "反中心性(规则1)",    "type": "图模型"},
    "direct_centrality": {"func": direct_centrality,  "label": "直接中心性(规则2)",  "type": "图模型"},
    "pagerank":          {"func": pagerank_weight,    "label": "PageRank(规则3)",    "type": "图模型"},
    # 融合方法 (7, 含本文贡献)
    "network_reg_rp":    {"func": network_regularized_rp,  "label": "网络正则化RP",      "type": "融合"},
    "graph_bl":          {"func": graph_enhanced_bl,       "label": "网络增强BL",        "type": "融合"},
    "denoised_rp":       {"func": denoised_rp,             "label": "去噪风险平价",      "type": "融合"},
    "community_bl_rp":   {"func": community_bl_rp,         "label": "社区BL-RP混合",     "type": "融合"},
    "laplacian_mv":      {"func": graph_laplacian_mv,      "label": "拉普拉斯MV",        "type": "融合"},
    "pmfg_filtered_rp":  {"func": pmfg_filtered_rp,        "label": "图过滤RP",          "type": "融合"},
    "graph_smooth_bl":   {"func": graph_smoothed_bl,       "label": "图平滑动量BL",      "type": "融合"},
    "net_minvar":        {"func": network_constrained_minvar, "label": "◆网络约束MinVar",  "type": "融合"},
    "comm_minvar":       {"func": community_minvar,          "label": "◆社区MinVar",       "type": "融合"},
}


def compute_weights(strategy_name: str, returns: pd.DataFrame,
                    **kwargs) -> np.ndarray:
    """统一策略接口"""
    if strategy_name not in STRATEGY_REGISTRY:
        raise ValueError(f"未知策略: {strategy_name}")
    func = STRATEGY_REGISTRY[strategy_name]["func"]
    return func(returns, **kwargs)
