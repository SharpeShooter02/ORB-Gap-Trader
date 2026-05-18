"""
_production_run.py
==================
Final production config. All decisions locked in.

When imported: exposes universe/sigma/filter definitions only (safe for live system).
When run directly: also executes full backtest, computes stats, saves equity chart.
"""
import sys, io, contextlib
import numpy as np
import pandas as pd
sys.path.insert(0, ".")
from orb_backtester import StrategyConfig, run_backtest
from reference.config import INSTRUMENTS

# ── Active trading universe ───────────────────────────────────────────────────
# Explicit whitelist — grouped by sector. Everything in INSTRUMENTS not listed
# here is implicitly cut. Class A = independent catalyst (2× quiet, excl filter,
# no purity skip). Class B = market-driven (purity-skipped when total_n ≤ 3).
# Direction filters and special rules noted inline.

ACTIVE = [
    # ── Broad Market — Class B ────────────────────────────────────────────────
    "TQQQ",   # QQQ  3x bull
    "SQQQ",   # QQQ  3x bear
    "UPRO",   # SPY  3x bull
    "SPXS",   # SPY  3x bear
    "URTY",   # IWM  3x bull
    "TZA",    # IWM  3x bear  (SRTY cut: weaker, 95%+ co-fire with TZA)
    "UDOW",   # DIA  3x bull
    "SDOW",   # DIA  3x bear
    "FNGD",   # FANG 3x bear  (FNGU cut: duplicate QQQ exposure)

    # ── Technology Sector — Class B ───────────────────────────────────────────
    "TECL",   # XLK  3x bull
    "TECS",   # XLK  3x bear
    "WEBL",   # XLC  3x bull  (communication services)
    "WEBS",   # XLC  3x bear

    # ── Semiconductors — Class B ──────────────────────────────────────────────
    "SOXL",   # SOXX 3x bull
    "SOXS",   # SOXX 3x bear

    # ── Financials — Class B ──────────────────────────────────────────────────
    "FAZ",    # XLF  3x bear  (FAS cut: 47.1% WR weak sister)

    # ── Healthcare — Class B ──────────────────────────────────────────────────
    "CURE",   # XLV  3x bull

    # ── Volatility — Class B ──────────────────────────────────────────────────
    "UVIX",   # VIX  2x bull

    # ── Aerospace & Defense — Class B ─────────────────────────────────────────
    "DFEN",   # ITA  3x bull

    # ── Real Estate — Class B ─────────────────────────────────────────────────
    "URE",    # IYR  2x bull

    # ── Utilities — Class B ───────────────────────────────────────────────────
    "UTSL",   # XLU  3x bull

    # ── Gold Miners — Class B ─────────────────────────────────────────────────
    "NUGT",   # GDX  2x bull
    "DUST",   # GDX  2x bear  (JDST cut: confirmed loser)
    "JNUG",   # GDXJ 2x bull

    # ── EM / International — Class A ──────────────────────────────────────────
    "EDC",    # EEM  3x bull  [direction: up gaps only]  [flood loser: sized 0×]
    "EDZ",    # EEM  3x bear  [direction: down gaps only]
    "KORU",   # EWY  3x bull  (South Korea)
    "MEXX",   # EWW  3x bull  (Mexico)
    "CHAU",   # ASHR 2x bull  (China A shares)  [direction: up gaps only]
    "INDL",   # INDY 3x bull  (India)

    # ── Biotech — Class A ─────────────────────────────────────────────────────
    "LABU",   # IBB  3x bull  [direction: up gaps only]
    "LABD",   # IBB  3x bear  [direction: down gaps only / short trade]

    # ── Regional Banking — Class A ────────────────────────────────────────────
    "DPST",   # KRE  3x bull

    # ── Nat Gas — Class A (drift instrument) ──────────────────────────────────
    "KOLD",   # UNG  2x bear  [gap excl exemption: big gaps allowed through]

    # ── Single Stock — Class A ────────────────────────────────────────────────
    "NVDU",   # NVDA 2x bull
    "TSLL",   # TSLA 2x bull
    "AMDL",   # AMD  2x bull

    # ── Crypto (BTC) — Class A ────────────────────────────────────────────────
    "BITU",   # BTC  2x bull
    "BTCL",   # BTC  2x bull (Rex — alternate provider)
    "BTCZ",   # BTC  2x bear  [direction: down gaps only / short trade]
    "SBIT",   # BTC  2x bear  [direction: down gaps only / short trade]

    # ── Crypto (ETH) — Class A ────────────────────────────────────────────────
    "ETHU",   # ETH  2x bull  [gap filter: 2% flat]  [Mon excluded: 65h weekend gap]
    "ETHD",   # ETH  2x bear  [gap filter: 2% flat]  [Mon excluded]
    "ETU",    # ETH  2x bull (ProShares — alternate provider)

    # ── Crypto (SOL / XRP) — Class A ──────────────────────────────────────────
    "SOLT",   # SOL  2x bull  (thin: ~2024+)
    "XRPT",   # XRP  2x bull  (thin: ~2025+)
    "UXRP",   # XRP  2x bull  (thin: ~2025+)
]

# Key cuts: BITX (low-quality exclusive days vs BITU), BOIL (thin margins),
# FNGU (QQQ duplication), FAS/TNA/SRTY/ETHT/DRN (correlation cuts),
# all 2x broad-market (no edge over 3x), rates (unvalidated),
# crypto 1x/spot (no gap depletion mechanism).

UNIVERSE = {s: INSTRUMENTS[s] for s in ACTIVE if s in INSTRUMENTS}
ETH_SYMS = {sym for sym, info in UNIVERSE.items() if info["underlying"] == "ETH"}
GAP_FILTER = {
    sym: 0.02 if sym in ETH_SYMS else round(info["leverage"] * 0.02, 4)
    for sym, info in UNIVERSE.items()
}
DOW_EXCL = {sym: {0} for sym in ("ETHU", "ETHD") if sym in UNIVERSE}
SYMS = list(UNIVERSE.keys())
SIGMA = {
    "BTC":0.0264,"ETH":0.0329,"SOL":0.0457,"XRP":0.0436,
    "QQQ":0.0102,"SPY":0.0095,"IWM":0.0121,"DIA":0.0087,
    "XLF":0.0152,"XLE":0.0141,"IBB":0.0102,"IHE":0.0135,
    "GDX":0.0182,"GDXJ":0.0513,"SOXX":0.0136,
    "EEM":0.0177,"VWO":0.0141,"FXI":0.0178,"EWY":0.0150,
    "XLK":0.0133,"XLC":0.0101,"XLV":0.0082,
    "KRE":0.0156,"XRT":0.0138,"ITB":0.0161,"INDY":0.0102,"KWEB":0.0171,
    "GLD":0.0081,"SLV":0.0152,"USO":0.0167,"UNG":0.0208,
    "MSTR":0.1645,"COIN":0.0366,"TSM":0.0192,
    "NVDA":0.0226,"TSLA":0.0287,"SMCI":0.0390,"AMD":0.0348,
    "TLT":0.0060,"IEF":0.0029,"VIX":0.0603,
}
def ps_filter(sym, info, k=1.25):
    ul, sig = info["underlying"], SIGMA.get(info["underlying"])
    if sig is None: return None
    return (ul, sig*k, True) if info["inverse"] else (ul, sig*k)
PS_FILTERS = {s: ps_filter(s, i) for s, i in UNIVERSE.items() if ps_filter(s, i)}

# Class A — independent catalyst; gap exclusion applied to Class A only
_CLASS_A_UL   = {"BTC","ETH","SOL","XRP","NVDA","TSLA","SMCI","AMD","UNG"}
_CLASS_A_SYMS = {"KORU","CHAU","MEXX","INDL","LABU","LABD","DPST","KOLD"}
_EXCL_EXCEPTIONS = {"KOLD"}   # drift instrument — let big gaps through
CLASS_A_EXCL_SYMS = tuple(
    s for s, i in UNIVERSE.items()
    if s not in _EXCL_EXCEPTIONS
    and (s in _CLASS_A_SYMS or i["underlying"] in _CLASS_A_UL)
)

# ── Two-class instrument taxonomy (needed by live sizing logic) ───────────────
CRYPTO_UL          = {"BTC","ETH","SOL","XRP"}
SINGLESTK_UL       = {"NVDA","TSLA","SMCI","AMD"}
NATGAS_UL          = {"UNG"}
INTERNATIONAL_SYMS = {"KORU","CHAU","MEXX","INDL"}
BIOTECH_SYMS       = {"LABU","LABD"}
BANKING_SYMS       = {"DPST"}
DRIFT_SYMS         = {"KOLD"}
CLASS_A_UL         = CRYPTO_UL | SINGLESTK_UL | NATGAS_UL
CLASS_A_SYMS       = INTERNATIONAL_SYMS | BIOTECH_SYMS | BANKING_SYMS | DRIFT_SYMS
FLOOD_LOSERS       = {"EDC"}

def _is_class_a(sym, ul):
    return sym in CLASS_A_SYMS or ul in CLASS_A_UL

# ── Production sizing parameters ──────────────────────────────────────────────
BASELINE_PCT = 15.0
CAP_UNITS    = float("inf")

def apply_sizing(df):
    df = df.copy()
    def mult(row):
        if row["purity_skip"]:   return 0.0
        if row["equity_flood"] and row["crypto_cluster"]:
            if row["is_flood_loser"]: return 0.0
            if row["is_crypto"]:      return 1.5
            return 1.0
        if row["equity_flood"]:
            return 0.0 if row["is_flood_loser"] else 1.0
        if row["crypto_cluster"]:
            return 1.5 if row["is_crypto"] else 1.0
        if row["quiet"]:
            return 2.0 if row["is_class_a"] else 1.0
        return 1.0
    df["mult"] = df.apply(mult, axis=1)
    day_mult_sum = df.groupby("date")["mult"].sum().rename("day_mult_sum")
    df = df.merge(day_mult_sum, on="date")
    df["cap_factor"]  = (CAP_UNITS / df["day_mult_sum"]).clip(upper=1.0)
    df["capped_mult"] = df["mult"] * df["cap_factor"]
    df["sized_pnl"]   = df["dollar_pnl"] * df["capped_mult"] * (BASELINE_PCT / 10.0)
    return df

# ── StrategyConfig for this production universe ───────────────────────────────
cfg = StrategyConfig(
    symbols=SYMS, instrument_gap_filters=GAP_FILTER,
    prior_session_filters=PS_FILTERS, day_of_week_exclusions=DOW_EXCL,
    direction_filters={"SBIT":-1,"BTCZ":-1,"LABD":-1,"EDZ":-1,"LABU":+1,"CHAU":+1,"EDC":+1},
    use_rtg_scaling=False,
    rtg_gap_exclusion=True, rtg_gap_exclusion_threshold=0.08,
    rtg_gap_exclusion_symbols=CLASS_A_EXCL_SYMS,
    min_increment_pct=0.0, min_profit_pct=0.0,
    start_date="2020-01-01", initial_equity=100_000.0, daily_risk_pct=0.20,
    eod_exit_hour=16, eod_exit_minute=0,
)

# ── Backtest execution — only when run as script ──────────────────────────────
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.patches import Patch

    print("Running backtest...")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        trades = run_backtest(cfg)

    trades = trades.copy()
    trades["dollar_pnl"] = trades["pnl_pct"] * 1_000.0
    trades["date"]       = pd.to_datetime(trades["date"])
    trades["underlying"] = trades["symbol"].map({s: i["underlying"] for s, i in UNIVERSE.items()})
    trades["year"]       = trades["date"].dt.year

    trades["is_crypto"]       = trades["underlying"].isin(CRYPTO_UL)
    trades["is_class_a"]      = trades.apply(lambda r: _is_class_a(r["symbol"], r["underlying"]), axis=1)
    trades["is_flood_loser"]  = trades["symbol"].isin(FLOOD_LOSERS)

    day_stats = trades.groupby("date").agg(
        total_n      = ("symbol", "count"),
        n_crypto_uls = ("underlying", lambda x: x[trades.loc[x.index, "is_crypto"]].nunique()),
    ).reset_index()
    day_stats["equity_flood"]   = day_stats["total_n"] >= 10
    day_stats["crypto_cluster"] = day_stats["n_crypto_uls"] >= 3
    day_stats["quiet"]          = ~day_stats["equity_flood"] & ~day_stats["crypto_cluster"]
    trades = trades.merge(
        day_stats[["date","total_n","equity_flood","crypto_cluster","quiet"]], on="date"
    )

    PURITY_THRESH = 3
    trades["purity_skip"] = ~trades["is_class_a"] & (trades["total_n"] <= PURITY_THRESH)

    trades = apply_sizing(trades)
    baseline_daily = trades.groupby("date")["dollar_pnl"].sum() * (BASELINE_PCT / 10.0)
    sized_daily    = trades.groupby("date")["sized_pnl"].sum()

    rf_daily = 0.043 / 365
    START    = 10_000.0
    cal      = pd.date_range(trades["date"].min(), trades["date"].max(), freq="D")

    def compound_equity(daily):
        filled = daily.reindex(cal, fill_value=0.0)
        return START * (1 + filled / START).cumprod()

    def stats(daily):
        filled = daily.reindex(cal, fill_value=0.0)
        r      = filled / 1000.0
        sh     = (r.mean()-rf_daily)/r.std()*np.sqrt(365) if r.std()>0 else np.nan
        eq     = compound_equity(daily)
        peak   = eq.cummax()
        dd_pct = (eq - peak) / peak
        maxdd  = dd_pct.min()
        years  = (cal[-1] - cal[0]).days / 365
        ann_ret = (eq.iloc[-1] / START) ** (1/years) - 1
        calmar = ann_ret / abs(maxdd) if maxdd != 0 else np.nan
        return sh, maxdd, calmar

    sh_b, dd_b, cal_b = stats(baseline_daily)
    sh_s, dd_s, cal_s = stats(sized_daily)

    print(f"\n{'='*60}")
    print("PRODUCTION CONFIG — FINAL RESULTS")
    print(f"{'='*60}")
    n_purity   = trades["purity_skip"].sum()
    n_flood    = (~trades["purity_skip"] & trades["mult"].eq(0)).sum()
    print(f"  Trades:          {len(trades):>6}  ({n_purity} purity-skipped  {n_flood} flood/other-skipped)")
    print(f"  Date range:    {trades['date'].min().date()} → {trades['date'].max().date()}")
    print(f"\n  {'Metric':<14}  {'Baseline':>10}  {'Production':>10}")
    print(f"  {'─'*38}")
    print(f"  {'Sharpe':<14}  {sh_b:>10.3f}  {sh_s:>10.3f}")
    print(f"  {'Max DD':<14}  {dd_b:>10.1%}  {dd_s:>10.1%}")
    print(f"  {'Calmar':<14}  {cal_b:>10.3f}  {cal_s:>10.3f}")
    print(f"  {'Total P&L':<14}  ${baseline_daily.sum():>9,.0f}  ${sized_daily.sum():>9,.0f}")
    print(f"  {'Avg/day':<14}  ${baseline_daily.mean():>9.2f}  ${sized_daily.mean():>9.2f}")

    print(f"\n  Year-by-year:")
    print(f"  {'Year':>5}  {'Base Sh':>8}  {'Prod Sh':>8}  {'Delta':>7}  {'Prod P&L':>10}  {'WR':>6}")
    for yr in sorted(trades["year"].unique()):
        t_yr = trades[trades["year"]==yr]
        db   = t_yr.groupby("date")["dollar_pnl"].sum()
        ds   = t_yr.groupby("date")["sized_pnl"].sum()
        yr_cal = pd.date_range(db.index.min(), db.index.max(), freq="D")
        def sh_yr(d):
            r = d.reindex(yr_cal, fill_value=0.0) / 1000.0
            return (r.mean()-rf_daily)/r.std()*np.sqrt(365) if r.std()>0 else np.nan
        shb = sh_yr(db); shs = sh_yr(ds)
        wr  = (t_yr[t_yr["mult"]>0]["sized_pnl"] > 0).mean()
        flag = " ▲" if shs > shb+0.05 else (" ▼" if shs < shb-0.05 else "")
        print(f"  {yr:>5}  {shb:>8.3f}  {shs:>8.3f}  {shs-shb:>+7.3f}{flag}  "
              f"${ds.sum():>9,.0f}  {wr:>6.1%}")

    eq_base = compound_equity(baseline_daily)
    eq_prod = compound_equity(sized_daily)
    dd_base = (eq_base - eq_base.cummax()) / eq_base.cummax() * 100
    dd_prod = (eq_prod - eq_prod.cummax()) / eq_prod.cummax() * 100
    max_dd_base = dd_base.min()
    max_dd_prod = dd_prod.min()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8),
                                    gridspec_kw={"height_ratios": [3, 1]},
                                    sharex=True)
    fig.patch.set_facecolor("#0f0f0f")
    for ax in (ax1, ax2):
        ax.set_facecolor("#0f0f0f")
        ax.tick_params(colors="#aaaaaa", labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333333")

    flood_dates   = day_stats[day_stats["equity_flood"]]["date"]
    cluster_dates = day_stats[day_stats["crypto_cluster"] & ~day_stats["equity_flood"]]["date"]
    for d in flood_dates:
        ax1.axvspan(d, d + pd.Timedelta(days=1), alpha=0.15, color="#ff6b35", lw=0)
    for d in cluster_dates:
        ax1.axvspan(d, d + pd.Timedelta(days=1), alpha=0.15, color="#7b68ee", lw=0)

    ax1.plot(eq_base.index, eq_base.values, color="#555555", lw=1.2, alpha=0.8,
             label=f"Flat $1k/trade  (Sh={sh_b:.2f}  MaxDD={max_dd_base:.1f}%)")
    ax1.plot(eq_prod.index, eq_prod.values, color="#00d4aa", lw=1.8,
             label=f"Production sizing  (Sh={sh_s:.2f}  MaxDD={max_dd_prod:.1f}%)")
    ax1.axhline(START, color="#444444", lw=0.8, ls="--")
    ax1.set_ylabel("Portfolio Value ($)", color="#aaaaaa", fontsize=10)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax1.legend(loc="upper left", framealpha=0.2, fontsize=9,
               labelcolor="#cccccc", facecolor="#1a1a1a",
               handles=[
                   plt.Line2D([],[],color="#555555",lw=1.2,
                              label=f"Flat $1k/trade  (Sh={sh_b:.2f}  MaxDD={max_dd_base:.1f}%)"),
                   plt.Line2D([],[],color="#00d4aa",lw=1.8,
                              label=f"Production sizing  (Sh={sh_s:.2f}  MaxDD={max_dd_prod:.1f}%)"),
                   Patch(color="#ff6b35",alpha=0.4,label="Equity flood days"),
                   Patch(color="#7b68ee",alpha=0.4,label="Crypto cluster days"),
               ])
    ax1.set_title("ORB Strategy — Production Config  ($10k account, 10% baseline per trade)",
                  color="#eeeeee", fontsize=12, pad=10)
    ax1.grid(axis="y", color="#222222", lw=0.5)
    for yr in range(2021, 2027):
        ax1.axvline(pd.Timestamp(f"{yr}-01-01"), color="#333333", lw=0.8, ls=":")

    ax2.fill_between(dd_base.index, dd_base.values, 0,
                     color="#555555", alpha=0.4, label="Baseline DD")
    ax2.fill_between(dd_prod.index, dd_prod.values, 0,
                     color="#00d4aa", alpha=0.35, label="Production DD")
    ax2.set_ylabel("Drawdown %", color="#aaaaaa", fontsize=9)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax2.legend(loc="lower left", framealpha=0.2, fontsize=8,
               labelcolor="#cccccc", facecolor="#1a1a1a")
    ax2.grid(axis="y", color="#222222", lw=0.5)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.tick_params(axis="x", colors="#aaaaaa")

    plt.tight_layout(h_pad=0.5)
    out_path = "production_equity_curve.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\nChart saved → {out_path}")
