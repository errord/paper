"""
基于图模型的资产配置策略研究 - 数据获取与预处理
==============================================
完全免费，无需 Wind / Choice 等付费终端

数据源:
  - BaoStock (baostock.com): 个股/指数日频数据, 无速率限制, 无需token
  - AKShare (akshare.xyz): 国债收益率等补充数据

提供两套方案:
  方案A（推荐）: 沪深300代表性个股(30只) + 国债收益率
    → 30个网络节点, 覆盖主要行业, 社区结构丰富
  方案B（备选）: ETF多资产组合(20只)
    → 覆盖固收/权益/商品三大类, 贴合华西证券自营结构

使用方法:
    python data_fetch.py              # 默认方案A, 分年获取2023/2024/2025
    python data_fetch.py --plan A     # 方案A: 个股
    python data_fetch.py --plan B     # 方案B: ETF
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ============================================================
#  方案A: 沪深300代表性个股 (推荐)
#  30只个股覆盖10个行业, 网络分析社区结构丰富
#  数据源: BaoStock (完全免费, 无限制)
# ============================================================

# 资产池: 沪深300成分股中各行业龙头
STOCK_ASSETS = {
    # ---- 银行 (6只, 代表固收敏感型资产) ----
    "sh.601398": {"name": "工商银行", "sector": "银行"},
    "sh.601939": {"name": "建设银行", "sector": "银行"},
    "sh.601288": {"name": "农业银行", "sector": "银行"},
    "sh.600036": {"name": "招商银行", "sector": "银行"},
    "sh.601166": {"name": "兴业银行", "sector": "银行"},
    "sz.000001": {"name": "平安银行", "sector": "银行"},

    # ---- 非银金融 (3只) ----
    "sh.601318": {"name": "中国平安", "sector": "非银金融"},
    "sh.600030": {"name": "中信证券", "sector": "非银金融"},
    "sh.601628": {"name": "中国人寿", "sector": "非银金融"},

    # ---- 房地产 (2只) ----
    "sz.000002": {"name": "万科A",   "sector": "房地产"},
    "sh.600048": {"name": "保利发展", "sector": "房地产"},

    # ---- 食品饮料 (4只) ----
    "sh.600519": {"name": "贵州茅台", "sector": "食品饮料"},
    "sz.000858": {"name": "五粮液",   "sector": "食品饮料"},
    "sh.600887": {"name": "伊利股份", "sector": "食品饮料"},
    "sh.603288": {"name": "海天味业", "sector": "食品饮料"},

    # ---- 医药生物 (3只) ----
    "sh.600276": {"name": "恒瑞医药", "sector": "医药生物"},
    "sh.603259": {"name": "药明康德", "sector": "医药生物"},
    "sz.300760": {"name": "迈瑞医疗", "sector": "医药生物"},

    # ---- 电子/科技 (4只) ----
    "sz.300750": {"name": "宁德时代", "sector": "科技制造"},
    "sz.002594": {"name": "比亚迪",   "sector": "科技制造"},
    "sz.002415": {"name": "海康威视", "sector": "科技制造"},
    "sz.002475": {"name": "立讯精密", "sector": "科技制造"},

    # ---- 能源/材料 (3只) ----
    "sh.601857": {"name": "中国石油", "sector": "能源材料"},
    "sh.601088": {"name": "中国神华", "sector": "能源材料"},
    "sh.601899": {"name": "紫金矿业", "sector": "能源材料"},

    # ---- 工业/制造 (3只) ----
    "sh.600031": {"name": "三一重工", "sector": "工业制造"},
    "sh.601888": {"name": "中国中免", "sector": "工业制造"},
    "sz.000333": {"name": "美的集团", "sector": "工业制造"},

    # ---- 通信/公用事业 (2只) ----
    "sh.600941": {"name": "中国移动", "sector": "通信"},
    "sz.000063": {"name": "中兴通讯", "sector": "通信"},
}


def fetch_baostock_single(bs_session, code: str, start_date: str,
                          end_date: str) -> pd.DataFrame:
    """通过BaoStock获取单只股票/指数的日频数据"""
    rs = bs_session.query_history_k_data_plus(
        code,
        "date,close,volume,pctChg",
        start_date=start_date,
        end_date=end_date,
        frequency="d",
        adjustflag="2"  # 前复权
    )
    rows = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=rs.fields)
    df["date"] = pd.to_datetime(df["date"])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    return df[["date", "close", "volume"]].dropna(subset=["close"])


def fetch_plan_a(start_date="2023-01-01", end_date="2024-12-31",
                 output_dir="data"):
    """
    方案A: 获取30只沪深300代表性个股 + 宽基指数
    数据源: BaoStock (免费, 无限制)
    """
    import baostock as bs

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("方案A: 沪深300代表性个股 (BaoStock)")
    print("=" * 60)

    lg = bs.login()
    if lg.error_code != "0":
        print(f"BaoStock登录失败: {lg.error_msg}")
        return pd.DataFrame(), pd.DataFrame()
    print(f"  BaoStock登录成功\n")

    all_prices = {}
    meta_records = []
    failed = []

    # 1. 获取个股数据
    for code, info in STOCK_ASSETS.items():
        label = f"{info['name']}"
        print(f"  [{info['sector']}] {code} {label}...", end="")
        df = fetch_baostock_single(bs, code, start_date, end_date)

        if len(df) >= 100:
            all_prices[label] = df.set_index("date")["close"]
            meta_records.append({
                "code": code, "name": info["name"],
                "sector": info["sector"], "trading_days": len(df),
            })
            print(f" ✓ {len(df)}天")
        else:
            failed.append(f"{code}({info['name']})")
            print(f" ✗ 仅{len(df)}天")

    # 2. 补充宽基指数(作为基准)
    benchmarks = {
        "sh.000300": "沪深300指数",
        "sh.000905": "中证500指数",
    }
    for code, name in benchmarks.items():
        print(f"  [基准] {code} {name}...", end="")
        rs = bs.query_history_k_data_plus(code, "date,close",
                                          start_date=start_date,
                                          end_date=end_date,
                                          frequency="d")
        rows = []
        while rs.error_code == "0" and rs.next():
            rows.append(rs.get_row_data())
        if rows:
            df = pd.DataFrame(rows, columns=rs.fields)
            df["date"] = pd.to_datetime(df["date"])
            df["close"] = pd.to_numeric(df["close"], errors="coerce")
            all_prices[name] = df.set_index("date")["close"]
            meta_records.append({
                "code": code, "name": name,
                "sector": "宽基指数", "trading_days": len(df),
            })
            print(f" ✓ {len(df)}天")

    bs.logout()

    # 合并
    prices_df = pd.DataFrame(all_prices).sort_index()
    prices_df.index = pd.to_datetime(prices_df.index)
    prices_df = prices_df.dropna(how="any")  # 取日期交集

    meta_df = pd.DataFrame(meta_records)

    print(f"\n{'='*40}")
    print(f"获取成功: {len(meta_records)} 个资产, {len(prices_df)} 个交易日")
    print(f"行业分布:")
    for sector, group in meta_df.groupby("sector"):
        print(f"  {sector}: {len(group)}个 ({', '.join(group['name'])})")
    if failed:
        print(f"获取失败: {failed}")

    # 保存
    prices_df.to_csv(os.path.join(output_dir, "prices_plan_a.csv"))
    meta_df.to_csv(os.path.join(output_dir, "meta_plan_a.csv"), index=False)
    print(f"\n数据已保存: {output_dir}/prices_plan_a.csv")

    return prices_df, meta_df


# ============================================================
#  方案B: ETF多资产组合
#  覆盖固收/权益/商品, 贴合华西证券自营投资结构
#  数据源: AKShare (免费, 需控制请求频率)
# ============================================================

ETF_ASSETS = {
    # 固定收益 (对应华西 ~65% 固收)
    "511010": {"name": "国债ETF",     "category": "固定收益"},
    "511260": {"name": "十年国债ETF", "category": "固定收益"},
    "511220": {"name": "城投债ETF",   "category": "固定收益"},
    "511060": {"name": "五年地债ETF", "category": "固定收益"},
    "511030": {"name": "公司债ETF",   "category": "固定收益"},
    # 权益 (对应华西 ~20% 权益)
    "510300": {"name": "沪深300ETF",  "category": "权益"},
    "510500": {"name": "中证500ETF",  "category": "权益"},
    "159915": {"name": "创业板ETF",   "category": "权益"},
    "510050": {"name": "上证50ETF",   "category": "权益"},
    "512800": {"name": "银行ETF",     "category": "权益"},
    "512880": {"name": "证券ETF",     "category": "权益"},
    "512010": {"name": "医药ETF",     "category": "权益"},
    "512690": {"name": "酒ETF",       "category": "权益"},
    "512480": {"name": "半导体ETF",   "category": "权益"},
    "510880": {"name": "红利ETF",     "category": "权益"},
    # 商品/另类 (对应华西 ~15% 衍生品)
    "518880": {"name": "黄金ETF",     "category": "商品"},
    "159981": {"name": "能源化工ETF", "category": "商品"},
    "159985": {"name": "豆粕ETF",     "category": "商品"},
}


def fetch_etf_with_retry(symbol: str, start_date: str, end_date: str,
                         max_retries: int = 3, delay: float = 5.0) -> pd.DataFrame:
    """带重试的ETF数据获取"""
    import akshare as ak

    for attempt in range(max_retries):
        try:
            df = ak.fund_etf_hist_em(
                symbol=symbol, period="daily",
                start_date=start_date, end_date=end_date,
                adjust="qfq"
            )
            if df is not None and len(df) > 0:
                df = df.rename(columns={"日期": "date", "收盘": "close",
                                        "成交量": "volume"})
                df["date"] = pd.to_datetime(df["date"])
                return df[["date", "close", "volume"]].sort_values("date").reset_index(drop=True)
        except Exception as e:
            if attempt < max_retries - 1:
                wait = delay * (attempt + 1)
                print(f" [重试{attempt+1}, 等待{wait}s]", end="")
                time.sleep(wait)
            else:
                return pd.DataFrame()

    return pd.DataFrame()


def fetch_plan_b(start_date="20230101", end_date="20241231",
                 output_dir="data"):
    """
    方案B: 获取ETF多资产组合数据
    数据源: AKShare (免费, 需控制请求频率3-5秒/次)
    """
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("方案B: ETF多资产组合 (AKShare)")
    print("  注意: 需控制请求频率, 获取速度较慢(约2分钟)")
    print("=" * 60)

    all_prices = {}
    meta_records = []
    failed = []

    for code, info in ETF_ASSETS.items():
        print(f"  {code} {info['name']} ({info['category']})...", end="")
        df = fetch_etf_with_retry(code, start_date, end_date,
                                  max_retries=3, delay=5.0)
        if len(df) >= 100:
            label = f"{info['name']}"
            all_prices[label] = df.set_index("date")["close"]
            meta_records.append({
                "code": code, "name": info["name"],
                "category": info["category"],
                "trading_days": len(df),
            })
            print(f" ✓ {len(df)}天")
        else:
            failed.append(f"{code}({info['name']})")
            print(f" ✗")

        time.sleep(3)  # 每次请求间隔3秒

    prices_df = pd.DataFrame(all_prices).sort_index()
    prices_df.index = pd.to_datetime(prices_df.index)
    prices_df = prices_df.dropna(how="any")

    meta_df = pd.DataFrame(meta_records)

    print(f"\n获取成功: {len(meta_records)} 个ETF, {len(prices_df)} 个交易日")
    if failed:
        print(f"获取失败: {failed}")
        print("  提示: 若AKShare被限流, 请等待几分钟后重试, 或改用方案A")

    prices_df.to_csv(os.path.join(output_dir, "prices_plan_b.csv"))
    meta_df.to_csv(os.path.join(output_dir, "meta_plan_b.csv"), index=False)
    print(f"数据已保存: {output_dir}/prices_plan_b.csv")

    return prices_df, meta_df


# ============================================================
#  补充数据: 国债收益率 (两个方案通用)
# ============================================================

def fetch_bond_yields(year: str, output_dir: str = "data"):
    """获取指定年份的中国国债收益率曲线 (AKShare)"""
    os.makedirs(output_dir, exist_ok=True)

    start_date = f"{year}0101"
    end_date_str = f"{year}-12-31"

    try:
        import akshare as ak
        df = ak.bond_zh_us_rate(start_date=start_date)
        if df is not None and len(df) > 0:
            df = df.rename(columns={"日期": "date"})
            df["date"] = pd.to_datetime(df["date"])
            cn_cols = ["date"] + [c for c in df.columns if "中国" in str(c)]
            df_cn = df[cn_cols].set_index("date").sort_index()
            df_cn = df_cn[(df_cn.index >= f"{year}-01-01") &
                          (df_cn.index <= end_date_str)]
            out_path = os.path.join(output_dir, f"bond_yields_{year}.csv")
            df_cn.to_csv(out_path)
            print(f"  ✓ {year}年: {len(df_cn)}个交易日 → {out_path}")
            return df_cn
    except Exception as e:
        print(f"  ✗ {year}年: {e}")
        print("  提示: 国债收益率为补充数据, 不影响主要分析")
    return pd.DataFrame()


# ============================================================
#  数据预处理: 收益率计算 / 异常值处理 / 质量报告
# ============================================================

def preprocess(prices_df: pd.DataFrame, meta_df: pd.DataFrame,
               output_dir: str = "data", year: str = "") -> pd.DataFrame:
    """
    数据预处理:
      1. 计算对数日收益率
      2. Winsorize异常值(±5σ)
      3. 输出质量报告与相关性统计
    """
    print("\n" + "=" * 60)
    print("数据预处理")
    print("=" * 60)

    # 对数收益率
    returns_df = np.log(prices_df / prices_df.shift(1)).dropna()
    n_days, n_assets = returns_df.shape
    print(f"  收益率矩阵: {n_days}天 × {n_assets}个资产")

    # Winsorize
    n_outliers = 0
    for col in returns_df.columns:
        mu, sigma = returns_df[col].mean(), returns_df[col].std()
        hi = returns_df[col] > mu + 5 * sigma
        lo = returns_df[col] < mu - 5 * sigma
        n_out = hi.sum() + lo.sum()
        if n_out > 0:
            returns_df.loc[hi, col] = mu + 5 * sigma
            returns_df.loc[lo, col] = mu - 5 * sigma
            n_outliers += n_out
    print(f"  Winsorize: {n_outliers}个极端值")

    # 质量报告
    stats = pd.DataFrame({
        "年化收益(%)": (returns_df.mean() * 252 * 100).round(2),
        "年化波动(%)": (returns_df.std() * np.sqrt(252) * 100).round(2),
        "偏度": returns_df.skew().round(3),
        "峰度": returns_df.kurtosis().round(3),
    })
    print(f"\n  ---- 描述性统计 ----")
    print(stats.to_string())

    # 相关系数矩阵
    corr = returns_df.corr()
    upper = corr.values[np.triu_indices_from(corr.values, k=1)]
    print(f"\n  相关系数: 均值={upper.mean():.4f}, "
          f"范围=[{upper.min():.4f}, {upper.max():.4f}]")

    # 距离矩阵 (用于网络构建)
    dist = np.sqrt(2 * (1 - corr))

    # 保存 (带年份标识)
    suffix = f"_{year}" if year else ""
    returns_df.to_csv(os.path.join(output_dir, f"returns{suffix}.csv"))
    corr.to_csv(os.path.join(output_dir, f"correlation_matrix{suffix}.csv"))
    dist.to_csv(os.path.join(output_dir, f"distance_matrix{suffix}.csv"))
    stats.to_csv(os.path.join(output_dir, f"descriptive_stats{suffix}.csv"))

    print(f"\n  已保存:")
    print(f"    {output_dir}/returns{suffix}.csv             (日收益率)")
    print(f"    {output_dir}/correlation_matrix{suffix}.csv  (相关系数矩阵)")
    print(f"    {output_dir}/distance_matrix{suffix}.csv     (距离矩阵)")
    print(f"    {output_dir}/descriptive_stats{suffix}.csv   (描述性统计)")

    return returns_df


# ============================================================
#  三年数据分别获取与处理 (核心逻辑)
# ============================================================

# 三个完整年份
YEARS = ["2023", "2024", "2025"]


def run_all_years(plan: str = "A", output_root: str = "data"):
    """
    分年获取 2023/2024/2025 三年数据, 每年独立保存
    目录结构:
        data/
        ├── 2023/           # 2023年完整数据
        │   ├── prices.csv
        │   ├── meta.csv
        │   ├── returns_2023.csv
        │   ├── correlation_matrix_2023.csv
        │   ├── distance_matrix_2023.csv
        │   ├── descriptive_stats_2023.csv
        │   └── bond_yields_2023.csv
        ├── 2024/           # 2024年完整数据
        │   └── ...
        ├── 2025/           # 2025年完整数据
        │   └── ...
        └── all/            # 三年合并数据
            └── ...
    """
    all_year_prices = {}  # {year: prices_df}
    all_year_meta = {}

    for year in YEARS:
        start = f"{year}-01-01"
        end = f"{year}-12-31"
        year_dir = os.path.join(output_root, year)
        os.makedirs(year_dir, exist_ok=True)

        print("\n")
        print("╔" + "═" * 58 + "╗")
        print(f"║  {year}年 完整数据获取" + " " * (58 - len(f"  {year}年 完整数据获取")) + "║")
        print("║  " + f"{start} ~ {end}" + " " * (58 - len(f"  {start} ~ {end}")) + "║")
        print("╚" + "═" * 58 + "╝")

        if plan == "A":
            prices, meta = fetch_plan_a(start, end, year_dir)
        else:
            s = start.replace("-", "")
            e = end.replace("-", "")
            prices, meta = fetch_plan_b(s, e, year_dir)

        if len(prices) > 0:
            # 重命名保存的原始文件
            src_price = os.path.join(year_dir,
                                     "prices_plan_a.csv" if plan == "A" else "prices_plan_b.csv")
            dst_price = os.path.join(year_dir, "prices.csv")
            if os.path.exists(src_price):
                os.rename(src_price, dst_price)

            src_meta = os.path.join(year_dir,
                                    "meta_plan_a.csv" if plan == "A" else "meta_plan_b.csv")
            dst_meta = os.path.join(year_dir, "meta.csv")
            if os.path.exists(src_meta):
                os.rename(src_meta, dst_meta)

            # 预处理
            preprocess(prices, meta, year_dir, year=year)

            all_year_prices[year] = prices
            all_year_meta[year] = meta
        else:
            print(f"\n  ✗ {year}年数据获取失败!")

    # 获取国债收益率 (每年分别获取)
    print("\n" + "=" * 60)
    print("补充数据: 中国国债收益率 (分年)")
    print("=" * 60)
    for year in YEARS:
        year_dir = os.path.join(output_root, year)
        fetch_bond_yields(year, year_dir)
        time.sleep(1)

    # 合并三年数据 → data/all/
    if len(all_year_prices) == len(YEARS):
        all_dir = os.path.join(output_root, "all")
        os.makedirs(all_dir, exist_ok=True)

        print("\n" + "=" * 60)
        print("合并三年数据 → data/all/")
        print("=" * 60)

        combined_prices = pd.concat(all_year_prices.values()).sort_index()
        combined_prices = combined_prices[~combined_prices.index.duplicated(keep="first")]
        combined_prices.to_csv(os.path.join(all_dir, "prices.csv"))

        # 合并 meta (取任意一年, 结构相同)
        first_meta = list(all_year_meta.values())[0]
        first_meta.to_csv(os.path.join(all_dir, "meta.csv"), index=False)

        # 预处理合并数据
        preprocess(combined_prices, first_meta, all_dir, year="all")

        # 合并国债收益率
        bond_parts = []
        for year in YEARS:
            bp = os.path.join(output_root, year, f"bond_yields_{year}.csv")
            if os.path.exists(bp):
                bond_parts.append(pd.read_csv(bp, index_col=0, parse_dates=True))
        if bond_parts:
            combined_bonds = pd.concat(bond_parts).sort_index()
            combined_bonds = combined_bonds[~combined_bonds.index.duplicated(keep="first")]
            combined_bonds.to_csv(os.path.join(all_dir, "bond_yields_all.csv"))
            print(f"  国债收益率合并: {len(combined_bonds)}个交易日 → {all_dir}/bond_yields_all.csv")

        print(f"\n  三年合并: {len(combined_prices)}个交易日 → {all_dir}/prices.csv")

    # 最终汇总
    print("\n" + "=" * 60)
    print("✓ 全部数据获取与预处理完成!")
    print("=" * 60)
    print(f"\n目录结构:")
    for year in YEARS:
        year_dir = os.path.join(output_root, year)
        if year in all_year_prices:
            n = len(all_year_prices[year])
            print(f"  {year_dir}/  ({n}个交易日)")
    if len(all_year_prices) == len(YEARS):
        print(f"  {os.path.join(output_root, 'all')}/  (三年合并)")

    print(f"\n各年份独立文件:")
    for year in YEARS:
        print(f"  data/{year}/returns_{year}.csv")
        print(f"  data/{year}/correlation_matrix_{year}.csv")
        print(f"  data/{year}/distance_matrix_{year}.csv")


# ============================================================
#  主程序
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="论文数据获取 (完全免费, 无需Wind)")
    parser.add_argument("--plan", type=str, default="A",
                        choices=["A", "B"],
                        help="A=个股30只(BaoStock,推荐), B=ETF多资产(AKShare)")
    parser.add_argument("--output", type=str, default="data",
                        help="输出根目录 (默认 data)")
    args = parser.parse_args()

    print("╔════════════════════════════════════════════════════╗")
    print("║  基于图模型的资产配置策略 - 数据获取               ║")
    print("║  免费数据源: BaoStock + AKShare                   ║")
    print("║  时间范围: 2023 / 2024 / 2025 三个完整年度         ║")
    print("╚════════════════════════════════════════════════════╝")
    print(f"  方案: {'A (个股30只, BaoStock)' if args.plan == 'A' else 'B (ETF多资产, AKShare)'}")
    print(f"  输出: {args.output}/\n")

    run_all_years(plan=args.plan, output_root=args.output)
