"""
基于图模型的资产配置策略研究 v3 - 多资产数据获取与预处理
============================================================
免费数据源:
  BaoStock  - 个股 (无限制, 快速)
  AKShare stock_zh_index_daily - ETF (网易API, 无限流)
  AKShare futures_zh_daily_sina - 期货 (新浪API, 无限流)
  AKShare bond_zh_us_rate - 国债收益率

资产池 (30个):
  固定收益 (8): 5只债券ETF + 1只债券指数 + 2只国债期货
  权益     (16): 8只权益ETF + 8只个股
  衍生品/商品 (6): 3只股指期货 + 3只商品ETF

使用: python data_fetch.py
"""

import os
import time
import warnings
from datetime import date
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ============================================================
#  资产池定义
# ============================================================

# I. 固定收益 (8)
BOND_ETFS = {
    "sh511010": {"name": "国债ETF",       "cat": "固定收益"},
    "sh511260": {"name": "十年国债ETF",   "cat": "固定收益"},
    "sh511270": {"name": "短债ETF",       "cat": "固定收益"},
    "sh511060": {"name": "五年地债ETF",   "cat": "固定收益"},
    "sh511020": {"name": "活跃国债ETF",   "cat": "固定收益"},
}
BOND_FUTURES = {
    "T0":  {"name": "十年国债期货", "cat": "固定收益"},
    "TF0": {"name": "五年国债期货", "cat": "固定收益"},
}

# II. 权益 (16)
EQUITY_ETFS = {
    "sh510300": {"name": "沪深300ETF",  "cat": "权益"},
    "sh510500": {"name": "中证500ETF",  "cat": "权益"},
    "sz159915": {"name": "创业板ETF",   "cat": "权益"},
    "sh510050": {"name": "上证50ETF",   "cat": "权益"},
    "sh512800": {"name": "银行ETF",     "cat": "权益"},
    "sh512880": {"name": "证券ETF",     "cat": "权益"},
    "sh512010": {"name": "医药ETF",     "cat": "权益"},
    "sh510880": {"name": "红利ETF",     "cat": "权益"},
}
STOCKS = {
    "sh.601398": {"name": "工商银行", "cat": "权益"},
    "sh.600036": {"name": "招商银行", "cat": "权益"},
    "sh.601318": {"name": "中国平安", "cat": "权益"},
    "sh.600519": {"name": "贵州茅台", "cat": "权益"},
    "sz.300750": {"name": "宁德时代", "cat": "权益"},
    "sz.002594": {"name": "比亚迪",   "cat": "权益"},
    "sh.601857": {"name": "中国石油", "cat": "权益"},
    "sh.601899": {"name": "紫金矿业", "cat": "权益"},
}

# 企业债指数 via BaoStock (补充固定收益类)
BOND_INDEX = {
    "sh.000013": {"name": "企业债指数", "cat": "固定收益"},
}

# III. 衍生品及商品 (6)
INDEX_FUTURES = {
    "IF0": {"name": "沪深300股指期货", "cat": "衍生品"},
    "IC0": {"name": "中证500股指期货", "cat": "衍生品"},
    "IH0": {"name": "上证50股指期货",  "cat": "衍生品"},
}
COMMODITY_ETFS = {
    "sh518880": {"name": "黄金ETF",     "cat": "商品"},
    "sz159981": {"name": "能源化工ETF", "cat": "商品"},
    "sz159985": {"name": "豆粕ETF",     "cat": "商品"},
}

ALL_ETF_CODES = {**BOND_ETFS, **EQUITY_ETFS, **COMMODITY_ETFS}
ALL_FUTURES_CODES = {**BOND_FUTURES, **INDEX_FUTURES}
ALL_BAOSTOCK = {**STOCKS, **BOND_INDEX}

YEARS = ["2023", "2024", "2025"]
START_DATE = date(2023, 1, 1)
END_DATE = date(2025, 12, 31)


# ============================================================
#  获取函数
# ============================================================

def fetch_stocks() -> dict:
    """BaoStock: 获取个股+债券指数 2023-2025 (快速, 无限制)"""
    import baostock as bs
    lg = bs.login()
    if lg.error_code != "0":
        print(f"  BaoStock登录失败: {lg.error_msg}")
        return {}

    result = {}
    for code, info in ALL_BAOSTOCK.items():
        name = info["name"]
        print(f"  {code} {name}...", end="", flush=True)
        rs = bs.query_history_k_data_plus(
            code, "date,close",
            start_date="2023-01-01", end_date="2025-12-31",
            frequency="d", adjustflag="2"
        )
        rows = []
        while rs.error_code == "0" and rs.next():
            rows.append(rs.get_row_data())
        if len(rows) >= 100:
            df = pd.DataFrame(rows, columns=rs.fields)
            df["date"] = pd.to_datetime(df["date"])
            df["close"] = pd.to_numeric(df["close"], errors="coerce")
            result[name] = df.dropna(subset=["close"]).set_index("date")["close"]
            print(f" {len(rows)}天")
        else:
            print(f" FAIL ({len(rows)}天)")

    bs.logout()
    print(f"  BaoStock: {len(result)}/{len(ALL_BAOSTOCK)}")
    return result


def fetch_etfs() -> dict:
    """AKShare stock_zh_index_daily (网易API): 获取17只ETF 2023-2025"""
    import akshare as ak

    result = {}
    total = len(ALL_ETF_CODES)
    for i, (code, info) in enumerate(ALL_ETF_CODES.items()):
        name = info["name"]
        cat = info["cat"]
        print(f"  [{i+1}/{total}] {code} {name} ({cat})...",
              end="", flush=True)
        try:
            df = ak.stock_zh_index_daily(symbol=code)
            df_sub = df[(df["date"] >= START_DATE) &
                        (df["date"] <= END_DATE)].copy()
            if len(df_sub) >= 100:
                df_sub["date"] = pd.to_datetime(df_sub["date"])
                df_sub["close"] = pd.to_numeric(
                    df_sub["close"], errors="coerce")
                result[name] = df_sub.dropna(
                    subset=["close"]).set_index("date")["close"]
                print(f" {len(df_sub)}天")
            else:
                print(f" FAIL ({len(df_sub)}天)")
        except Exception as e:
            print(f" ERROR: {str(e)[:60]}")
        time.sleep(0.5)

    print(f"  ETF: {len(result)}/{total}")
    return result


def fetch_futures() -> dict:
    """AKShare futures_zh_daily_sina (新浪API): 获取5只期货 2023-2025"""
    import akshare as ak

    result = {}
    for symbol, info in ALL_FUTURES_CODES.items():
        name = info["name"]
        cat = info["cat"]
        print(f"  {symbol} {name} ({cat})...", end="", flush=True)
        try:
            df = ak.futures_zh_daily_sina(symbol=symbol)
            if df is not None and len(df) > 0:
                df["date"] = pd.to_datetime(df["date"])
                df_sub = df[(df["date"] >= str(START_DATE)) &
                            (df["date"] <= str(END_DATE))].copy()
                df_sub["close"] = pd.to_numeric(
                    df_sub["close"], errors="coerce")
                df_sub = df_sub.dropna(subset=["close"]).sort_values("date")
                if len(df_sub) >= 100:
                    result[name] = df_sub.set_index("date")["close"]
                    print(f" {len(df_sub)}天")
                else:
                    print(f" FAIL ({len(df_sub)}天)")
            else:
                print(" FAIL")
        except Exception as e:
            print(f" ERROR: {str(e)[:60]}")
        time.sleep(1)

    print(f"  期货: {len(result)}/{len(ALL_FUTURES_CODES)}")
    return result


def fetch_bond_yields(output_root: str = "data"):
    """AKShare: 中国国债收益率 (补充数据)"""
    try:
        import akshare as ak
        df = ak.bond_zh_us_rate(start_date="20230101")
        if df is None or len(df) == 0:
            print("  获取失败")
            return
        df = df.rename(columns={"日期": "date"})
        df["date"] = pd.to_datetime(df["date"])
        cn_cols = ["date"] + [c for c in df.columns if "中国" in str(c)]
        df_cn = df[cn_cols].set_index("date").sort_index()

        for year in YEARS:
            ydir = os.path.join(output_root, year)
            os.makedirs(ydir, exist_ok=True)
            mask = (df_cn.index >= f"{year}-01-01") & \
                   (df_cn.index <= f"{year}-12-31")
            sub = df_cn[mask]
            sub.to_csv(os.path.join(ydir, f"bond_yields_{year}.csv"))
            print(f"  {year}: {len(sub)}天")

        adir = os.path.join(output_root, "all")
        os.makedirs(adir, exist_ok=True)
        full = df_cn[(df_cn.index >= "2023-01-01") &
                     (df_cn.index <= "2025-12-31")]
        full.to_csv(os.path.join(adir, "bond_yields_all.csv"))
        print(f"  合并: {len(full)}天")
    except Exception as e:
        print(f"  失败: {e}")


# ============================================================
#  预处理
# ============================================================

def preprocess(prices_df: pd.DataFrame, meta_df: pd.DataFrame,
               output_dir: str, year: str = "") -> pd.DataFrame:
    """对数收益率 -> Winsorize(+-5sigma) -> 统计"""
    returns_df = np.log(prices_df / prices_df.shift(1)).dropna()
    n_days, n_assets = returns_df.shape

    n_outliers = 0
    for col in returns_df.columns:
        mu, sigma = returns_df[col].mean(), returns_df[col].std()
        hi = returns_df[col] > mu + 5 * sigma
        lo = returns_df[col] < mu - 5 * sigma
        cnt = hi.sum() + lo.sum()
        if cnt > 0:
            returns_df.loc[hi, col] = mu + 5 * sigma
            returns_df.loc[lo, col] = mu - 5 * sigma
            n_outliers += cnt

    meta_lookup = dict(zip(meta_df["name"], meta_df["category"]))
    stats = pd.DataFrame({
        "类别": [meta_lookup.get(c, "未知") for c in returns_df.columns],
        "年化收益(%)": (returns_df.mean() * 252 * 100).round(2),
        "年化波动(%)": (returns_df.std() * np.sqrt(252) * 100).round(2),
        "偏度": returns_df.skew().round(3),
        "峰度": returns_df.kurtosis().round(3),
    })

    corr = returns_df.corr()
    dist = np.sqrt(2 * (1 - corr))

    suffix = f"_{year}" if year else ""
    os.makedirs(output_dir, exist_ok=True)
    returns_df.to_csv(os.path.join(output_dir, f"returns{suffix}.csv"))
    corr.to_csv(os.path.join(output_dir, f"correlation_matrix{suffix}.csv"))
    dist.to_csv(os.path.join(output_dir, f"distance_matrix{suffix}.csv"))
    stats.to_csv(os.path.join(output_dir, f"descriptive_stats{suffix}.csv"))

    print(f"  [{year or 'all'}] {n_days}天 x {n_assets}资产, "
          f"Winsorize {n_outliers}个")
    return returns_df


# ============================================================
#  主流程
# ============================================================

def run():
    output_root = "data"

    print("\n" + "=" * 60)
    print("Phase 1: 数据获取 (2023-01-01 ~ 2025-12-31)")
    print("=" * 60)

    print("\n[1/3] 个股 (BaoStock):")
    stock_data = fetch_stocks()

    print("\n[2/3] ETF (AKShare/网易):")
    etf_data = fetch_etfs()

    print("\n[3/3] 期货 (AKShare/新浪):")
    futures_data = fetch_futures()

    # 合并
    all_data = {}
    all_data.update(stock_data)
    all_data.update(etf_data)
    all_data.update(futures_data)

    n_got = len(all_data)
    print(f"\n获取到 {n_got}/30 个资产")
    if n_got < 20:
        print("ERROR: 资产数不足, 请检查网络后重试")
        return

    # 构建 meta
    all_asset_defs = {}
    for d in [BOND_ETFS, BOND_FUTURES, EQUITY_ETFS, STOCKS, BOND_INDEX,
              INDEX_FUTURES, COMMODITY_ETFS]:
        for code, info in d.items():
            all_asset_defs[info["name"]] = {"code": code, **info}

    meta_records = []
    for name in all_data.keys():
        if name in all_asset_defs:
            info = all_asset_defs[name]
            meta_records.append({"code": info["code"], "name": name,
                                 "category": info["cat"]})
    meta_df = pd.DataFrame(meta_records)

    # 完整数据 (日期交集)
    full_prices = pd.DataFrame(all_data).sort_index()
    full_prices.index = pd.to_datetime(full_prices.index)
    n_before = len(full_prices)
    full_prices = full_prices.dropna(how="any")

    print(f"\n{'='*60}")
    print("数据汇总:")
    print(f"  资产: {len(full_prices.columns)}个")
    print(f"  交易日: {len(full_prices)} (取交集前{n_before})")
    for cat in ["固定收益", "权益", "衍生品", "商品"]:
        grp = meta_df[meta_df["category"] == cat]
        if len(grp) > 0:
            print(f"  {cat} ({len(grp)}): "
                  f"{', '.join(grp['name'].tolist())}")

    # Phase 2: 分年 + 预处理
    print(f"\n{'='*60}")
    print("Phase 2: 分年切分与预处理")
    print("=" * 60)

    for year in YEARS:
        ydir = os.path.join(output_root, year)
        os.makedirs(ydir, exist_ok=True)
        mask = (full_prices.index >= f"{year}-01-01") & \
               (full_prices.index <= f"{year}-12-31")
        yp = full_prices[mask].dropna(how="any")
        yp.to_csv(os.path.join(ydir, "prices.csv"))
        meta_df.to_csv(os.path.join(ydir, "meta.csv"), index=False)
        preprocess(yp, meta_df, ydir, year=year)

    adir = os.path.join(output_root, "all")
    os.makedirs(adir, exist_ok=True)
    full_prices.to_csv(os.path.join(adir, "prices.csv"))
    meta_df.to_csv(os.path.join(adir, "meta.csv"), index=False)
    preprocess(full_prices, meta_df, adir, year="all")

    # Phase 3: 国债收益率
    print(f"\n{'='*60}")
    print("Phase 3: 国债收益率")
    print("=" * 60)
    fetch_bond_yields(output_root)

    # 完成
    print(f"\n{'='*60}")
    print("全部完成!")
    print("=" * 60)
    print(f"  {len(full_prices.columns)}个资产, {len(full_prices)}个交易日")
    for year in YEARS:
        yp = os.path.join(output_root, year, "prices.csv")
        if os.path.exists(yp):
            n = len(pd.read_csv(yp))
            print(f"  {year}: {n}天")
    print(f"  合并: {len(full_prices)}天")
    print(f"\n输出: data/2023/ data/2024/ data/2025/ data/all/")


if __name__ == "__main__":
    print("+" + "=" * 56 + "+")
    print("|  基于图模型的资产配置策略研究 v3                     |")
    print("|  多资产数据获取 (免费: BaoStock + AKShare)           |")
    print("|  固收(8) + 权益(16) + 衍生品/商品(6) = 30           |")
    print("|  2023 / 2024 / 2025                                 |")
    print("+" + "=" * 56 + "+")
    run()
