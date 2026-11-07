"""Shareable, self-contained daily equity-pairs research pipeline.

This single file contains the Yahoo Finance download, rolling cointegration
screen, fixed-watchlist and chronological Z* walk-forward backtests, and all
visualization routines. It does not import the project's other Python files.

Install dependencies once:
    python -m pip install yfinance pandas numpy pyarrow statsmodels matplotlib

Run from any folder:
    python shareable_equity_pairs_research.py
    python shareable_equity_pairs_research.py --download
    python shareable_equity_pairs_research.py --download --start 2006-01-01 --end 2026-10-01

If the adjacent data/yfinance_daily.parquet file exists, it is reused unless
--download is supplied. Outputs are written beside this script under outputs/.
"""
from __future__ import annotations

import argparse
import sys

import json

from datetime import datetime, timezone

from pathlib import Path

import pandas as pd

import yfinance as yf

import warnings

import numpy as np

from statsmodels.stats.multitest import multipletests

from statsmodels.tsa.stattools import adfuller, coint

import matplotlib

import matplotlib.pyplot as plt

from types import SimpleNamespace
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent


# ===== Consolidated from download_yfinance_daily.py =====

DOWNLOAD_OUTPUT = ROOT / "data" / "yfinance_daily.parquet"

TICKERS = ["KO", "PEP", "XOM", "CVX", "HD", "LOW", "V", "MA", "UPS", "FDX", "GLD", "SLV", "INTC", "AMD"]

def download(start: str, end: str | None) -> pd.DataFrame:
    DOWNLOAD_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    # Use a project-local cache when supported; yfinance's cache API varies by
    # version, so don't make downloads depend on optional configuration hooks.
    set_cache_location = getattr(yf, "set_tz_cache_location", None)
    if callable(set_cache_location):
        try:
            set_cache_location(str(DOWNLOAD_OUTPUT.parent / ".yfinance_cache"))
        except Exception as exc:
            print(f"Note: using yfinance's default cache ({type(exc).__name__}: {exc})", flush=True)
    chunks = []
    failures = []
    for ticker in TICKERS:
        try:
            frame = yf.Ticker(ticker).history(
                start=start,
                end=end,
                interval="1d",
                auto_adjust=True,
                actions=True,
            )
        except Exception as exc:  # provider/network errors need symbol context
            failures.append(f"{ticker}: {type(exc).__name__}: {exc}")
            continue
        if frame.empty or "Close" not in frame or "Open" not in frame:
            failures.append(f"{ticker}: provider returned no usable daily bars")
            continue
        frame = frame.copy()
        index = pd.DatetimeIndex(frame.index)
        if index.tz is not None:
            index = index.tz_localize(None)
        frame.index = index.normalize()
        frame.index.name = "date"
        keep = [column for column in ("Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits") if column in frame]
        tidy = frame[keep].rename(columns={
            "Open": "open", "High": "high", "Low": "low", "Close": "close",
            "Volume": "volume", "Dividends": "dividends", "Stock Splits": "stock_splits",
        }).reset_index()
        tidy["ticker"] = ticker
        chunks.append(tidy)
        print(f"{ticker}: {len(tidy):,} daily rows ({tidy.date.min().date()} to {tidy.date.max().date()})", flush=True)
    if failures:
        raise RuntimeError("Download incomplete; no output written:\n" + "\n".join(failures))
    result = pd.concat(chunks, ignore_index=True).sort_values(["ticker", "date"])
    result = result.drop_duplicates(["ticker", "date"], keep="last")
    return result

def run_downloader() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2006-01-01", help="Inclusive first date")
    parser.add_argument("--end", default=None, help="Exclusive end date; default latest available")
    parser.add_argument("--output", type=Path, default=DOWNLOAD_OUTPUT)
    args = parser.parse_args()

    prices = download(args.start, args.end)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(args.output, index=False)
    metadata = {
        "provider": "yfinance history() wrapper",
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "interval": "1d",
        "auto_adjust": True,
        "start_inclusive": args.start,
        "end_exclusive": args.end,
        "tickers": TICKERS,
        "rows": len(prices),
        "data_file": str(args.output.resolve()),
        "price_note": "OHLC adjusted by yfinance for splits/dividends; dividend and split action columns are retained when returned.",
    }
    args.output.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote {len(prices):,} rows to {args.output}", flush=True)


# ===== Consolidated from rolling_daily_pairs.py =====

ROLLING_DATA_FILE = ROOT / "data" / "yfinance_daily.parquet"

ROLLING_OUT_DIR = ROOT / "outputs" / "rolling_daily"

PAIRS = [
    ("KO", "PEP", "Coca-Cola and PepsiCo: beverage companies"),
    ("XOM", "CVX", "Exxon Mobil and Chevron: integrated oil and gas majors"),
    ("HD", "LOW", "Home Depot and Lowe's: home improvement retail"),
    ("V", "MA", "Visa and Mastercard: payment networks"),
    ("UPS", "FDX", "UPS and FedEx: parcel delivery and logistics"),
    ("GLD", "SLV", "Gold and silver ETF proxies: precious metals relative value"),
    ("INTC", "AMD", "Intel and AMD: semiconductor designers/manufacturers with overlapping end markets"),
]

def ols_log(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    beta, alpha = np.polyfit(x, y, 1)
    return float(alpha), float(beta)

def screen_pair(left: str, right: str, rationale: str, window: pd.DataFrame, coverage_min: float, maxlag: int) -> dict:
    row = dict(left=left, right=right, economic_rationale=rationale, n=0,
               coverage=0.0, i1_support=False, p_value=1.0, t_stat=np.nan,
               alpha=np.nan, beta=np.nan, beta_half_drift=np.nan,
               half_mean_shift_sd=np.nan, half_vol_ratio=np.nan,
               spread_adf_p=np.nan, spread_stability_pass=False,
               eligible=False, selected=False, failure_reason="")
    data = window[[left, right]].dropna()
    row.update(n=len(data), coverage=len(data) / max(len(window), 1))
    if row["coverage"] < coverage_min or len(data) < max(100, maxlag + 20):
        row["failure_reason"] = "insufficient_pair_coverage"
        return row

    y = np.log(data[left].to_numpy(float))
    x = np.log(data[right].to_numpy(float))
    if not np.isfinite(x).all() or not np.isfinite(y).all() or (x <= 0).any() or (y <= 0).any():
        row["failure_reason"] = "invalid_log_prices"
        return row

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y_level = adfuller(y, maxlag=maxlag, autolag=None)[1]
            x_level = adfuller(x, maxlag=maxlag, autolag=None)[1]
            y_diff = adfuller(np.diff(y), maxlag=maxlag, autolag=None)[1]
            x_diff = adfuller(np.diff(x), maxlag=maxlag, autolag=None)[1]
            i1 = y_level >= 0.05 and x_level >= 0.05 and y_diff < 0.05 and x_diff < 0.05
            alpha, beta = ols_log(x, y)
            spread = y - alpha - beta * x
            coint_t, p_value, _ = coint(y, x, trend="c", maxlag=maxlag, autolag=None)
            residual_adf_p = adfuller(spread, maxlag=maxlag, autolag=None)[1]

        mid = len(data) // 2
        alpha_1, beta_1 = ols_log(x[:mid], y[:mid])
        alpha_2, beta_2 = ols_log(x[mid:], y[mid:])
        del alpha_1, alpha_2
        spread_sd = float(np.std(spread, ddof=1))
        beta_drift = abs(beta_1 - beta_2) / max(abs(beta), 1e-8)
        mean_shift = abs(float(np.mean(spread[:mid]) - np.mean(spread[mid:]))) / max(spread_sd, 1e-12)
        sd_1 = float(np.std(spread[:mid], ddof=1))
        sd_2 = float(np.std(spread[mid:], ddof=1))
        vol_ratio = max(sd_1, sd_2) / max(min(sd_1, sd_2), 1e-12)
        stable = beta > 0 and beta_drift <= 0.25 and mean_shift <= 1.0 and vol_ratio <= 2.0
        row.update(y_level_adf_p=y_level, x_level_adf_p=x_level,
                   y_diff_adf_p=y_diff, x_diff_adf_p=x_diff,
                   i1_support=bool(i1), t_stat=float(coint_t), p_value=float(p_value),
                   alpha=alpha, beta=beta, spread_sd=spread_sd,
                   beta_half_1=beta_1, beta_half_2=beta_2,
                   beta_half_drift=float(beta_drift),
                   half_mean_shift_sd=float(mean_shift),
                   half_vol_ratio=float(vol_ratio), spread_adf_p=float(residual_adf_p),
                   spread_stability_pass=bool(stable),
                   failure_reason="" if stable else "spread_stability_gate_failed")
    except (ValueError, np.linalg.LinAlgError, ZeroDivisionError) as exc:
        row["failure_reason"] = f"test_error:{type(exc).__name__}"
    return row

def form_daily_selections(close: pd.DataFrame, lookback: int, maxlag: int, coverage_min: float, max_pairs: int):
    screen_frames = []
    selected_at: dict[pd.Timestamp, list[tuple[str, str]]] = {}
    screen_at: dict[pd.Timestamp, pd.DataFrame] = {}
    dates = pd.DatetimeIndex(close.index)
    if len(dates) <= lookback:
        raise ValueError(f"Need more than {lookback} daily rows; got {len(dates)}")

    for number, idx in enumerate(range(lookback - 1, len(dates)), start=1):
        date = dates[idx]
        window = close.iloc[idx - lookback + 1:idx + 1]
        rows = [screen_pair(a, b, rationale, window, coverage_min, maxlag) for a, b, rationale in PAIRS]
        frame = pd.DataFrame(rows)
        frame["q_by"] = multipletests(frame.p_value.fillna(1.0), method="fdr_by")[1]
        frame["cointegration_pass"] = frame.p_value.lt(0.05)
        frame["eligible"] = (frame.i1_support & frame.cointegration_pass & frame.q_by.le(0.05)
                             & frame.spread_stability_pass & frame.coverage.ge(coverage_min))
        frame["formation_date"] = date.strftime("%Y-%m-%d")

        ranked = frame.loc[frame.eligible].sort_values(["q_by", "p_value"])
        selected = []
        used = set()
        for item in ranked.itertuples(index=False):
            if item.left in used or item.right in used:
                continue
            selected.append((item.left, item.right))
            used.update((item.left, item.right))
            if len(selected) >= max_pairs:
                break
        frame["selected"] = frame.apply(lambda r: (r.left, r.right) in selected, axis=1)
        selected_at[date] = selected
        screen_at[date] = frame
        screen_frames.append(frame)
        if number % 500 == 0:
            print(f"{date.date()}: {number}/{len(dates) - lookback + 1} daily formation screens", flush=True)
    return pd.concat(screen_frames, ignore_index=True), selected_at, screen_at

def fit_signal_model(log_close: pd.DataFrame, left: str, right: str, signal_idx: int,
                     beta_lookback: int, coverage_min: float):
    start = signal_idx - beta_lookback
    if start < 0:
        return None
    train = log_close.iloc[start:signal_idx][[left, right]].dropna()
    if len(train) < int(beta_lookback * coverage_min):
        return None
    x = train[right].to_numpy(float)
    y = train[left].to_numpy(float)
    if len(train) < 3 or not np.isfinite(x).all() or not np.isfinite(y).all():
        return None
    alpha, beta = ols_log(x, y)
    residual = y - alpha - beta * x
    sd = float(np.std(residual, ddof=2))
    if not np.isfinite(sd) or sd <= 1e-12:
        return None
    signal = log_close.iloc[signal_idx]
    if pd.isna(signal[left]) or pd.isna(signal[right]):
        return None
    z = (float(signal[left]) - alpha - beta * float(signal[right])) / sd
    return alpha, beta, sd, z, len(train)

def run_daily_backtest(open_px: pd.DataFrame, close_px: pd.DataFrame,
                       selected_at: dict[pd.Timestamp, list[tuple[str, str]]],
                       beta_lookback: int, coverage_min: float, entry_z: float, stop_z: float):
    dates = pd.DatetimeIndex(close_px.index)
    log_close = np.log(close_px.where(close_px > 0))
    positions: dict[tuple[str, str], dict] = {}
    disarmed: set[tuple[str, str]] = set()
    trades = []
    daily_pair_rows = []
    daily_total: dict[pd.Timestamp, dict] = {}

    def book_pnl(day, pair, pnl):
        target = daily_total.setdefault(day, {"gross_pnl": 0.0, "turnover": 0.0})
        target["gross_pnl"] += float(pnl)
        pair_rows[(day, pair)]["gross_pnl"] += float(pnl)

    pair_rows: dict[tuple[pd.Timestamp, tuple[str, str]], dict] = {}

    for idx in range(max(beta_lookback + 1, 1), len(dates)):
        trade_date = dates[idx]
        signal_date = dates[idx - 1]
        selected = set(selected_at.get(signal_date, []))
        for left, right, _ in PAIRS:
            pair = (left, right)
            is_selected = pair in selected
            existing = positions.get(pair)
            if existing is None and not is_selected:
                continue
            model = fit_signal_model(log_close, left, right, idx - 1, beta_lookback, coverage_min)
            px_open_l, px_open_r = open_px.at[trade_date, left], open_px.at[trade_date, right]
            px_close_l, px_close_r = close_px.at[trade_date, left], close_px.at[trade_date, right]
            prices_valid = all(pd.notna(v) and np.isfinite(v) and v > 0 for v in (px_open_l, px_open_r, px_close_l, px_close_r))
            if not prices_valid:
                continue

            daily_key = (trade_date, pair)
            pair_rows[daily_key] = {"date": trade_date, "pair": f"{left}/{right}", "selected_asof": signal_date,
                                    "selected": is_selected, "z_at_signal": model[3] if model else np.nan,
                                    "gross_pnl": 0.0, "turnover": 0.0}

            if existing is not None:
                gap_pnl = (existing["left_shares"] * (float(px_open_l) - existing["mark_left"])
                           + existing["right_shares"] * (float(px_open_r) - existing["mark_right"]))
                book_pnl(trade_date, pair, gap_pnl)
                existing["gross_pnl"] += gap_pnl
                z = model[3] if model else np.nan
                reason = None
                if not is_selected:
                    reason = "pair_deselected"
                elif model is None:
                    reason = "signal_model_unavailable"
                elif existing["direction"] == 1 and (z >= 0 or z <= -stop_z):
                    reason = "mean_reversion" if z >= 0 else "hard_stop"
                elif existing["direction"] == -1 and (z <= 0 or z >= stop_z):
                    reason = "mean_reversion" if z <= 0 else "hard_stop"
                if reason:
                    exit_turnover = abs(existing["left_shares"] * float(px_open_l)) + abs(existing["right_shares"] * float(px_open_r))
                    existing["turnover"] += exit_turnover
                    pair_rows[daily_key]["turnover"] += exit_turnover
                    daily_total[trade_date]["turnover"] += exit_turnover
                    existing.update(exit_date=trade_date, exit_reason=reason,
                                    exit_z=z if np.isfinite(z) else np.nan,
                                    holding_sessions=(trade_date - existing["entry_date"]).days)
                    trades.append(existing)
                    del positions[pair]
                    if reason == "hard_stop":
                        disarmed.add(pair)
                    existing = None
                else:
                    intraday_pnl = (existing["left_shares"] * (float(px_close_l) - float(px_open_l))
                                     + existing["right_shares"] * (float(px_close_r) - float(px_open_r)))
                    book_pnl(trade_date, pair, intraday_pnl)
                    existing["gross_pnl"] += intraday_pnl
                    existing["mark_left"], existing["mark_right"] = float(px_close_l), float(px_close_r)
                    existing["holding_sessions"] += 1
                    continue

            if existing is None and is_selected and model is not None:
                alpha, beta, sd, z, n_fit = model
                if pair in disarmed:
                    if abs(z) < entry_z:
                        disarmed.remove(pair)
                    else:
                        continue
                if z <= -entry_z:
                    direction = 1
                elif z >= entry_z:
                    direction = -1
                else:
                    continue
                abs_beta = abs(beta)
                left_weight = 1.0 / (1.0 + abs_beta)
                right_weight = abs_beta / (1.0 + abs_beta)
                left_shares = direction * left_weight / float(px_open_l)
                right_shares = -direction * right_weight / float(px_open_r)
                entry_turnover = abs(left_shares * float(px_open_l)) + abs(right_shares * float(px_open_r))
                position = dict(pair=f"{left}/{right}", left=left, right=right,
                                direction=direction, entry_date=trade_date,
                                selection_formation_date=signal_date,
                                entry_z=float(z), beta=float(beta), alpha=float(alpha),
                                spread_sd=float(sd), formation_observations=int(n_fit),
                                left_shares=float(left_shares), right_shares=float(right_shares),
                                gross_pnl=0.0, turnover=float(entry_turnover),
                                holding_sessions=0, mark_left=float(px_open_l), mark_right=float(px_open_r),
                                exit_date=pd.NaT, exit_reason="", exit_z=np.nan)
                pair_rows[daily_key]["turnover"] += entry_turnover
                daily_total.setdefault(trade_date, {"gross_pnl": 0.0, "turnover": 0.0})["turnover"] += entry_turnover
                intraday_pnl = (left_shares * (float(px_close_l) - float(px_open_l))
                                + right_shares * (float(px_close_r) - float(px_open_r)))
                book_pnl(trade_date, pair, intraday_pnl)
                position["gross_pnl"] += intraday_pnl
                position["mark_left"], position["mark_right"] = float(px_close_l), float(px_close_r)
                position["holding_sessions"] = 1
                positions[pair] = position

    # Mark any remaining positions to the last close and record liquidation turnover.
    if len(dates):
        last_date = dates[-1]
        for pair, position in list(positions.items()):
            left, right = pair
            pl, pr = close_px.at[last_date, left], close_px.at[last_date, right]
            if pd.notna(pl) and pd.notna(pr):
                turnover = abs(position["left_shares"] * float(pl)) + abs(position["right_shares"] * float(pr))
                position["turnover"] += turnover
                daily_total.setdefault(last_date, {"gross_pnl": 0.0, "turnover": 0.0})["turnover"] += turnover
                position.update(exit_date=last_date, exit_reason="end_of_data", exit_z=np.nan)
                trades.append(position)

    daily_rows = []
    for day, values in sorted(daily_total.items()):
        daily_rows.append({"date": day, **values, "gross_pnl_bps": values["gross_pnl"] * 10000.0})
    pair_daily = list(pair_rows.values())
    return pd.DataFrame(trades), pd.DataFrame(daily_rows), pd.DataFrame(pair_daily)

def run_rolling_screen() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROLLING_DATA_FILE)
    parser.add_argument("--output", type=Path, default=ROLLING_OUT_DIR)
    parser.add_argument("--selection-lookback", type=int, default=252)
    parser.add_argument("--beta-lookback", type=int, default=60)
    parser.add_argument("--maxlag", type=int, default=5)
    parser.add_argument("--min-coverage", type=float, default=0.95)
    parser.add_argument("--max-pairs", type=int, default=5)
    parser.add_argument("--entry-z", type=float, default=2.0)
    parser.add_argument("--stop-z", type=float, default=4.0)
    parser.add_argument("--costs-bps", default="0,1,5,10")
    args = parser.parse_args()
    if not args.data.exists():
        parser.error(f"Daily data not found: {args.data}. Run download_yfinance_daily.py first.")
    if args.selection_lookback < 120 or args.beta_lookback < 20:
        parser.error("Selection lookback must be >=120 and beta lookback >=20 sessions.")
    if not 0 < args.min_coverage <= 1 or args.entry_z <= 0 or args.stop_z <= args.entry_z:
        parser.error("Invalid coverage or z-score thresholds.")

    frame = pd.read_parquet(args.data)
    frame["date"] = pd.to_datetime(frame.date).dt.tz_localize(None).dt.normalize()
    close = frame.pivot(index="date", columns="ticker", values="close").sort_index()
    open_px = frame.pivot(index="date", columns="ticker", values="open").reindex(close.index)
    required = {ticker for left, right, _ in PAIRS for ticker in (left, right)}
    missing = sorted(required - set(close.columns))
    if missing:
        parser.error(f"Input data is missing required tickers: {missing}")
    close = close.sort_index()
    open_px = open_px.reindex(index=close.index, columns=close.columns)

    screens, selected_at, _ = form_daily_selections(
        close, args.selection_lookback, args.maxlag, args.min_coverage, args.max_pairs)
    trades, daily, pair_daily = run_daily_backtest(
        open_px, close, selected_at, args.beta_lookback, args.min_coverage,
        args.entry_z, args.stop_z)

    costs = [float(x.strip()) for x in args.costs_bps.split(",") if x.strip()]
    active_pairs = sorted(trades.pair.unique()) if not trades.empty else []
    performance = []
    for pair in active_pairs:
        subset = trades.loc[trades.pair.eq(pair)]
        gross = float(subset.gross_pnl.sum())
        turnover = float(subset.turnover.sum())
        for cost in costs:
            net = gross - cost / 10000.0 * turnover
            net_trade = subset.gross_pnl - cost / 10000.0 * subset.turnover
            performance.append(dict(pair=pair, trades=len(subset), gross_pnl_bps=gross * 10000,
                                    turnover_units=turnover, cost_bps_per_turnover=cost,
                                    cost_paid_bps=cost * turnover, net_pnl_bps=net * 10000,
                                    profitable_trades=int((net_trade > 0).sum()),
                                    win_rate=float((net_trade > 0).mean())))
    if not trades.empty:
        gross = float(trades.gross_pnl.sum())
        turnover = float(trades.turnover.sum())
        for cost in costs:
            net_trade = trades.gross_pnl - cost / 10000.0 * trades.turnover
            performance.append(dict(pair="ALL", trades=len(trades), gross_pnl_bps=gross * 10000,
                                    turnover_units=turnover, cost_bps_per_turnover=cost,
                                    cost_paid_bps=cost * turnover,
                                    net_pnl_bps=(gross - cost / 10000.0 * turnover) * 10000,
                                    profitable_trades=int((net_trade > 0).sum()),
                                    win_rate=float((net_trade > 0).mean())))

    args.output.mkdir(parents=True, exist_ok=True)
    screens.to_csv(args.output / "rolling_pair_screens.csv", index=False)
    pd.DataFrame([{"formation_date": d.strftime("%Y-%m-%d"), "left": a, "right": b}
                  for d, pairs in selected_at.items() for a, b in pairs]).to_csv(
                      args.output / "selected_pairs_by_date.csv", index=False)
    trades.to_csv(args.output / "trades.csv", index=False)
    daily.to_csv(args.output / "daily_pnl.csv", index=False)
    pair_daily.to_csv(args.output / "pair_daily_diagnostics.csv", index=False)
    pd.DataFrame(performance).to_csv(args.output / "performance_summary.csv", index=False)
    metadata = {
        "process": "Daily rolling selection on prior 252 closes; next-session open execution with rolling 60-session daily beta model.",
        "data_file": str(args.data.resolve()), "source_prices": "yfinance auto_adjust=True daily OHLC",
        "tickers": sorted(required), "candidate_pairs": [list(p[:2]) for p in PAIRS],
        "first_source_date": str(close.index.min().date()), "last_source_date": str(close.index.max().date()),
        "formation_dates": int(screens.formation_date.nunique()),
        "selected_pair_dates": int(screens.selected.sum()),
        "selection_lookback_sessions": args.selection_lookback,
        "beta_lookback_sessions": args.beta_lookback,
        "min_coverage": args.min_coverage, "maxlag": args.maxlag,
        "entry_z": args.entry_z, "exit_z": 0.0, "stop_z": args.stop_z,
        "costs_bps_per_turnover": costs,
        "pnl_normalization": "Each trade is sized to one unit of gross pair notional at entry. P&L and turnover are summed in those trade units without compounding or a portfolio-capital denominator; reported bps are the summed dollar P&L multiplied by 10,000.",
        "limitations": [
            "Daily-bar variant: entries/exits execute at next session open; holdings may span multiple sessions.",
            "The 60-session beta fit uses adjusted daily closes, not intraday bars.",
            "BY correction is per daily family of seven pairs, not across repeated overlapping daily screens.",
            "Pair universe was specified in advance but has not been tested on an untouched holdout.",
            "Adjusted prices account for split/dividend adjustments supplied by yfinance; these are not executable quote prices.",
            "No bid/ask spread, borrow, financing, market impact, or quote-size modeling.",
        ],
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(pd.DataFrame(performance).to_string(index=False), flush=True)

base = SimpleNamespace(PAIRS=PAIRS, fit_signal_model=fit_signal_model)


# ===== Consolidated from fixed_watchlist_backtest.py =====

FIXED_DATA_FILE = ROOT / "data" / "yfinance_daily.parquet"

BASELINE_TRADES_FILE = ROOT / "outputs" / "rolling_daily" / "trades.csv"

DESIGN_OUT_DIR = ROOT / "outputs" / "design_test"

def coherent_backtest(open_px: pd.DataFrame, close_px: pd.DataFrame,
                      eligible_at: dict[pd.Timestamp, set[tuple[str, str]]],
                      beta_lookback: int, coverage: float, entry_z: float,
                      stop_z: float, deselect_failures: int,
                      dynamic_rehedge: bool = False,
                      entry_z_by_date: dict[pd.Timestamp, float] | None = None,
                      signal_models: dict[tuple[str, str], list[tuple | None]] | None = None):
    """Test frozen entry parameters or a daily-updated, explicitly rehedged model."""
    dates = pd.DatetimeIndex(close_px.index)
    log_close = np.log(close_px.where(close_px > 0))
    positions: dict[tuple[str, str], dict] = {}
    failure_streak = {(left, right): 0 for left, right, _ in base.PAIRS}
    disarmed: set[tuple[str, str]] = set()
    trades: list[dict] = []
    day_book: dict[pd.Timestamp, dict[str, float]] = {}
    pair_book: dict[tuple[pd.Timestamp, tuple[str, str]], dict] = {}

    def record_pnl(day, pair, pnl):
        day_row = day_book.setdefault(day, {"gross_pnl": 0.0, "turnover": 0.0})
        day_row["gross_pnl"] += float(pnl)
        pair_book[(day, pair)]["gross_pnl"] += float(pnl)

    for idx in range(max(beta_lookback + 1, 1), len(dates)):
        trade_date, signal_date = dates[idx], dates[idx - 1]
        qualified = eligible_at.get(signal_date, set())

        for left, right, _ in base.PAIRS:
            pair = (left, right)
            is_qualified = pair in qualified
            failure_streak[pair] = 0 if is_qualified else failure_streak[pair] + 1
            existing = positions.get(pair)
            if existing is None and not is_qualified:
                continue

            px_open_l, px_open_r = open_px.at[trade_date, left], open_px.at[trade_date, right]
            px_close_l, px_close_r = close_px.at[trade_date, left], close_px.at[trade_date, right]
            prices = (px_open_l, px_open_r, px_close_l, px_close_r)
            if not all(pd.notna(v) and np.isfinite(v) and v > 0 for v in prices):
                continue

            # The signal model uses only observations before signal_date; signal_date's
            # close is evaluated out of sample, then execution occurs next session.
            entry_model = (signal_models[pair][idx - 1] if signal_models is not None
                           else base.fit_signal_model(
                               log_close, left, right, idx - 1, beta_lookback, coverage))
            if existing is not None:
                daily_key = (trade_date, pair)
                pair_book.setdefault(daily_key, {
                    "date": trade_date, "pair": f"{left}/{right}",
                    "selected_asof": signal_date, "selected": is_qualified,
                    "z_at_signal": np.nan, "gross_pnl": 0.0,
                    "turnover": 0.0,
                })
                frozen_z = np.nan
                signal_l, signal_r = log_close.at[signal_date, left], log_close.at[signal_date, right]
                if pd.notna(signal_l) and pd.notna(signal_r):
                    frozen_z = (float(signal_l) - existing["alpha"]
                                - existing["beta"] * float(signal_r)) / existing["spread_sd"]
                signal_z = (entry_model[3] if dynamic_rehedge and entry_model is not None
                            else frozen_z)

                gap_pnl = (existing["left_shares"] * (float(px_open_l) - existing["mark_left"])
                           + existing["right_shares"] * (float(px_open_r) - existing["mark_right"]))
                record_pnl(trade_date, pair, gap_pnl)
                existing["gross_pnl"] += gap_pnl

                reason = None
                if dynamic_rehedge and entry_model is not None and entry_model[1] <= 0:
                    reason = "hedge_sign_break"
                elif np.isfinite(signal_z) and existing["direction"] == 1:
                    if signal_z <= -stop_z:
                        reason = "hard_stop"
                    elif signal_z >= 0:
                        reason = "mean_reversion"
                elif np.isfinite(signal_z) and existing["direction"] == -1:
                    if signal_z >= stop_z:
                        reason = "hard_stop"
                    elif signal_z <= 0:
                        reason = "mean_reversion"
                if (reason is None and deselect_failures
                        and failure_streak[pair] >= deselect_failures):
                    reason = "persistent_deselection"

                pair_book[daily_key]["z_at_signal"] = signal_z

                if reason:
                    exit_turnover = (abs(existing["left_shares"] * float(px_open_l))
                                     + abs(existing["right_shares"] * float(px_open_r)))
                    existing["turnover"] += exit_turnover
                    pair_book[daily_key]["turnover"] += exit_turnover
                    day_book[trade_date]["turnover"] += exit_turnover
                    existing.update(exit_date=trade_date, exit_reason=reason,
                                    exit_z=signal_z,
                                    holding_sessions=idx - existing["entry_index"])
                    trades.append(existing)
                    del positions[pair]
                    if reason == "hard_stop":
                        disarmed.add(pair)
                    existing = None
                else:
                    if dynamic_rehedge and entry_model is not None:
                        beta = entry_model[1]
                        abs_beta = abs(beta)
                        left_weight = 1.0 / (1.0 + abs_beta)
                        right_weight = abs_beta / (1.0 + abs_beta)
                        target_left = existing["direction"] * left_weight / float(px_open_l)
                        target_right = -existing["direction"] * right_weight / float(px_open_r)
                        rebalance_turnover = (abs(target_left - existing["left_shares"]) * float(px_open_l)
                                              + abs(target_right - existing["right_shares"]) * float(px_open_r))
                        existing["turnover"] += rebalance_turnover
                        pair_book[daily_key]["turnover"] += rebalance_turnover
                        day_book[trade_date]["turnover"] += rebalance_turnover
                        existing["left_shares"], existing["right_shares"] = target_left, target_right
                        existing["alpha"], existing["beta"], existing["spread_sd"] = entry_model[:3]
                    intraday_pnl = (existing["left_shares"] * (float(px_close_l) - float(px_open_l))
                                    + existing["right_shares"] * (float(px_close_r) - float(px_open_r)))
                    record_pnl(trade_date, pair, intraday_pnl)
                    existing["gross_pnl"] += intraday_pnl
                    existing["mark_left"], existing["mark_right"] = float(px_close_l), float(px_close_r)
                    existing["holding_sessions"] += 1
                    continue

            if existing is None and is_qualified and entry_model is not None:
                alpha, beta, sd, z, n_fit = entry_model[:5]
                threshold = (entry_z_by_date.get(signal_date, np.inf)
                             if entry_z_by_date is not None else entry_z)
                if pair in disarmed:
                    if abs(z) < threshold:
                        disarmed.remove(pair)
                    else:
                        continue
                if z <= -threshold:
                    direction = 1
                elif z >= threshold:
                    direction = -1
                else:
                    continue

                abs_beta = abs(beta)
                left_weight = 1.0 / (1.0 + abs_beta)
                right_weight = abs_beta / (1.0 + abs_beta)
                left_shares = direction * left_weight / float(px_open_l)
                right_shares = -direction * right_weight / float(px_open_r)
                entry_turnover = abs(left_shares * float(px_open_l)) + abs(right_shares * float(px_open_r))
                position = {
                    "pair": f"{left}/{right}", "left": left, "right": right,
                    "direction": direction, "entry_date": trade_date,
                    "entry_index": idx,
                    "selection_formation_date": signal_date, "entry_z": float(z),
                    "beta": float(beta), "alpha": float(alpha), "spread_sd": float(sd),
                    "formation_observations": int(n_fit),
                    "left_shares": float(left_shares), "right_shares": float(right_shares),
                    "gross_pnl": 0.0, "turnover": float(entry_turnover),
                    "holding_sessions": 1, "mark_left": float(px_open_l),
                    "mark_right": float(px_open_r), "exit_date": pd.NaT,
                    "exit_reason": "", "exit_z": np.nan,
                }
                daily_key = (trade_date, pair)
                pair_book.setdefault(daily_key, {
                    "date": trade_date, "pair": f"{left}/{right}",
                    "selected_asof": signal_date, "selected": is_qualified,
                    "z_at_signal": float(z), "gross_pnl": 0.0,
                    "turnover": 0.0,
                })
                pair_book[daily_key]["turnover"] += entry_turnover
                day_book.setdefault(trade_date, {"gross_pnl": 0.0, "turnover": 0.0})["turnover"] += entry_turnover
                intraday_pnl = (left_shares * (float(px_close_l) - float(px_open_l))
                                + right_shares * (float(px_close_r) - float(px_open_r)))
                record_pnl(trade_date, pair, intraday_pnl)
                position["gross_pnl"] += intraday_pnl
                position["mark_left"], position["mark_right"] = float(px_close_l), float(px_close_r)
                positions[pair] = position

        # Mark end-of-session gross exposure for a portfolio-level ledger.
        exposure = 0.0
        net_exposure = 0.0
        for (left, right), position in positions.items():
            px_l, px_r = close_px.at[trade_date, left], close_px.at[trade_date, right]
            if pd.notna(px_l) and pd.notna(px_r):
                left_value = position["left_shares"] * float(px_l)
                right_value = position["right_shares"] * float(px_r)
                exposure += abs(left_value) + abs(right_value)
                net_exposure += left_value + right_value
        exposure_row = day_book.setdefault(trade_date, {"gross_pnl": 0.0, "turnover": 0.0})
        exposure_row["gross_exposure_units"] = exposure
        exposure_row["net_exposure_units"] = net_exposure

    if len(dates):
        last_date = dates[-1]
        for pair, position in list(positions.items()):
            left, right = pair
            pl, pr = close_px.at[last_date, left], close_px.at[last_date, right]
            if pd.notna(pl) and pd.notna(pr):
                liquidation = abs(position["left_shares"] * float(pl)) + abs(position["right_shares"] * float(pr))
                position["turnover"] += liquidation
                final_row = day_book.setdefault(last_date, {"gross_pnl": 0.0, "turnover": 0.0})
                final_row["turnover"] += liquidation
                final_row["gross_exposure_units"] = 0.0
                final_row["net_exposure_units"] = 0.0
                position.update(exit_date=last_date, exit_reason="end_of_data", exit_z=np.nan)
                position["holding_sessions"] = len(dates) - 1 - position["entry_index"]
                trades.append(position)

    daily = pd.DataFrame([{"date": day, **vals, "gross_pnl_bps": vals["gross_pnl"] * 10_000}
                          for day, vals in sorted(day_book.items())])
    return pd.DataFrame(trades), daily, pd.DataFrame(list(pair_book.values()))


def kalman_signal_models(close: pd.DataFrame, lookback: int, coverage: float,
                         process_noise_alpha: float = 1e-7,
                         process_noise_beta: float = 1e-7,
                         measurement_variance_scale: float = 1.0
                         ) -> dict[tuple[str, str], list[tuple | None]]:
    """Causal random-walk Kalman models for log-price alpha and beta.

    The z-score at t is a one-step-ahead innovation based on the state through
    t-1. The current close updates the state only after that signal is formed.
    """
    logs = np.log(close.where(close > 0))
    result: dict[tuple[str, str], list[tuple | None]] = {}
    minimum = int(np.ceil(lookback * coverage))
    process_noise = np.diag([process_noise_alpha, process_noise_beta])
    for left, right, _ in PAIRS:
        y, x = logs[left].to_numpy(float), logs[right].to_numpy(float)
        models: list[tuple | None] = [None] * len(logs)
        valid_indices = np.flatnonzero(np.isfinite(y) & np.isfinite(x))
        if len(valid_indices) <= lookback:
            result[(left, right)] = models
            continue
        initial_indices = valid_indices[:lookback]
        if len(initial_indices) < minimum:
            result[(left, right)] = models
            continue
        design = np.column_stack((np.ones(len(initial_indices)), x[initial_indices]))
        theta, *_ = np.linalg.lstsq(design, y[initial_indices], rcond=None)
        residual = y[initial_indices] - design @ theta
        measurement_variance = max(float(np.var(residual, ddof=2))
                                   * measurement_variance_scale, 1e-12)
        covariance = measurement_variance * np.linalg.pinv(design.T @ design)
        observations_seen = len(initial_indices)
        for i in range(int(initial_indices[-1]) + 1, len(logs)):
            covariance = covariance + process_noise
            if not (np.isfinite(y[i]) and np.isfinite(x[i])):
                continue
            observation = np.array([1.0, x[i]])
            innovation = float(y[i] - observation @ theta)
            predictive_variance = float(observation @ covariance @ observation + measurement_variance)
            if not np.isfinite(predictive_variance) or predictive_variance <= 1e-14:
                continue
            innovation_sd = float(np.sqrt(predictive_variance))
            z = innovation / innovation_sd
            gain = covariance @ observation / predictive_variance
            theta = theta + gain * innovation
            correction = np.eye(2) - np.outer(gain, observation)
            covariance = (correction @ covariance @ correction.T
                          + measurement_variance * np.outer(gain, gain))
            observations_seen += 1
            if np.isfinite(theta).all() and innovation_sd > 1e-12:
                models[i] = (float(theta[0]), float(theta[1]), innovation_sd,
                             float(z), observations_seen)
        result[(left, right)] = models
    return result

def summarize(trades: pd.DataFrame, costs: list[float]) -> list[dict]:
    if trades.empty:
        return []
    rows = []
    for pair in sorted(trades.pair.unique()):
        subset = trades.loc[trades.pair.eq(pair)]
        for cost in costs:
            net_trade = subset.gross_pnl - cost / 10_000 * subset.turnover
            rows.append({"pair": pair, "trades": len(subset), "gross_pnl_bps": subset.gross_pnl.sum() * 10_000,
                         "turnover_units": subset.turnover.sum(), "cost_bps_per_turnover": cost,
                         "cost_paid_bps": cost * subset.turnover.sum(),
                         "net_pnl_bps": net_trade.sum() * 10_000,
                         "profitable_trades": int((net_trade > 0).sum()),
                         "win_rate": float((net_trade > 0).mean())})
    for cost in costs:
        net_trade = trades.gross_pnl - cost / 10_000 * trades.turnover
        rows.append({"pair": "ALL", "trades": len(trades), "gross_pnl_bps": trades.gross_pnl.sum() * 10_000,
                     "turnover_units": trades.turnover.sum(), "cost_bps_per_turnover": cost,
                     "cost_paid_bps": cost * trades.turnover.sum(),
                     "net_pnl_bps": net_trade.sum() * 10_000,
                     "profitable_trades": int((net_trade > 0).sum()),
                     "win_rate": float((net_trade > 0).mean())})
    return rows

def daily_arrays(daily: pd.DataFrame, dates: pd.DatetimeIndex,
                 cost_bps: float) -> tuple[pd.Series, pd.Series]:
    if daily.empty:
        zeros = pd.Series(0.0, index=dates)
        return zeros, zeros.copy()
    frame = daily.set_index("date").reindex(dates)
    gross = frame.gross_pnl.fillna(0.0)
    turnover = frame.turnover.fillna(0.0)
    return gross, turnover


def portfolio_account(daily: pd.DataFrame, dates: pd.DatetimeIndex, model_name: str,
                      cost_bps: float, trade_count: int,
                      starting_capital_units: float = 7.0):
    """Convert unit-notional P&L into a fixed-capital, non-compounded account."""
    frame = daily.set_index("date").reindex(dates) if not daily.empty else pd.DataFrame(index=dates)
    gross = frame.get("gross_pnl", pd.Series(0.0, index=dates)).fillna(0.0).astype(float)
    turnover = frame.get("turnover", pd.Series(0.0, index=dates)).fillna(0.0).astype(float)
    gross_exposure = frame.get("gross_exposure_units", pd.Series(0.0, index=dates)).fillna(0.0).astype(float)
    net_exposure = frame.get("net_exposure_units", pd.Series(0.0, index=dates)).fillna(0.0).astype(float)
    cost_units = turnover * cost_bps / 10_000
    net = gross - cost_units
    daily_return = net / starting_capital_units
    equity = starting_capital_units + net.cumsum()
    nav = equity / starting_capital_units
    drawdown = nav / nav.cummax() - 1
    ledger = pd.DataFrame({
        "date": dates, "model": model_name, "cost_bps_per_turnover": cost_bps,
        "gross_pnl_units": gross.to_numpy(), "turnover_units": turnover.to_numpy(),
        "cost_units": cost_units.to_numpy(), "net_pnl_units": net.to_numpy(),
        "portfolio_daily_return": daily_return.to_numpy(),
        "account_equity_units": equity.to_numpy(), "nav": nav.to_numpy(),
        "drawdown_pct": drawdown.to_numpy() * 100,
        "gross_exposure_units": gross_exposure.to_numpy(),
        "gross_leverage": gross_exposure.to_numpy() / starting_capital_units,
        "net_exposure_units": net_exposure.to_numpy(),
        "net_exposure_leverage": net_exposure.to_numpy() / starting_capital_units,
    })
    tail = daily_return.to_numpy(float)
    tail = tail[np.isfinite(tail)]
    var95 = float(-np.quantile(tail, 0.05)) if len(tail) else np.nan
    cutoff = float(np.quantile(tail, 0.05)) if len(tail) else np.nan
    cvar95 = float(-tail[tail <= cutoff].mean()) if len(tail) else np.nan
    daily_sd = float(np.std(tail, ddof=1)) if len(tail) > 1 else np.nan
    sharpe = float(np.mean(tail) / daily_sd * np.sqrt(252)) if daily_sd and daily_sd > 0 else np.nan
    summary = {
        "model": model_name, "cost_bps_per_turnover": cost_bps,
        "completed_round_trips": trade_count,
        "starting_capital_units": starting_capital_units,
        "ending_account_equity_units": float(equity.iloc[-1]),
        "cumulative_return_pct": float((nav.iloc[-1] - 1) * 100),
        "annualized_sharpe": sharpe,
        "annualized_volatility_pct": daily_sd * np.sqrt(252) * 100 if np.isfinite(daily_sd) else np.nan,
        "max_drawdown_pct": float(drawdown.min() * 100),
        "historical_daily_var95_pct": var95 * 100,
        "historical_daily_cvar95_pct": cvar95 * 100,
        "total_gross_pnl_units": float(gross.sum()),
        "total_cost_units": float(cost_units.sum()),
        "total_net_pnl_units": float(net.sum()),
        "total_turnover_units": float(turnover.sum()),
        "mean_gross_leverage": float(ledger.gross_leverage.mean()),
        "max_gross_leverage": float(ledger.gross_leverage.max()),
        "mean_net_exposure_leverage": float(ledger.net_exposure_leverage.mean()),
        "max_abs_net_exposure_leverage": float(ledger.net_exposure_leverage.abs().max()),
    }
    return ledger, summary

def walkforward_zstar(strategy_runs: dict[float, tuple[pd.DataFrame, pd.DataFrame]],
                      eligible_at: dict[pd.Timestamp, set[tuple[str, str]]],
                      dates: pd.DatetimeIndex, train_years: int, min_train_trades: int,
                      selection_cost_bps: float, capital_units: float = 7.0,
                      fallback_z: float = 2.0):
    """Choose each year's entry z using only completed history before that year."""
    grid = sorted(strategy_runs)
    if fallback_z not in strategy_runs:
        raise ValueError("Fallback z threshold must be included in the tested grid.")
    first_eligible = min(date for date, pairs in eligible_at.items() if pairs)
    first_test_year = first_eligible.year + train_years
    last_test_year = dates[-1].year
    constant_net: dict[float, pd.Series] = {}
    candidate_trades: dict[float, pd.DataFrame] = {}
    for threshold, (trades, daily) in strategy_runs.items():
        gross, turnover = daily_arrays(daily, dates, selection_cost_bps)
        constant_net[threshold] = gross - turnover * selection_cost_bps / 10_000
        candidate_trades[threshold] = trades

    schedules, train_rows, cv_rows = [], [], []
    selected_by_year: dict[int, float] = {}
    for year in range(first_test_year, last_test_year + 1):
        train_start = pd.Timestamp(year=year - train_years, month=1, day=1)
        test_start = pd.Timestamp(year=year, month=1, day=1)
        test_end = pd.Timestamp(year=year + 1, month=1, day=1)
        train_mask = (dates >= train_start) & (dates < test_start)
        test_mask = (dates >= test_start) & (dates < test_end)
        metrics = []
        for threshold in grid:
            returns = constant_net[threshold].loc[train_mask] / capital_units
            sd = float(returns.std(ddof=1))
            sharpe = float(returns.mean() / sd * np.sqrt(252)) if sd > 0 else np.nan
            trades = candidate_trades[threshold]
            if trades.empty:
                completed = 0
            else:
                entries = pd.to_datetime(trades.entry_date)
                exits = pd.to_datetime(trades.exit_date)
                completed = int(((entries >= train_start) & (exits < test_start)).sum())
            metrics.append((threshold, sharpe, completed))
            train_rows.append({"test_year": year, "train_start": train_start.date(),
                               "train_end_exclusive": test_start.date(),
                               "entry_z": threshold, "training_sharpe_net": sharpe,
                               "completed_training_trades": completed,
                               "selection_cost_bps": selection_cost_bps})
        eligible_metrics = [m for m in metrics if m[2] >= min_train_trades and np.isfinite(m[1])]
        selected = max(eligible_metrics, key=lambda item: (item[1], item[0])) if eligible_metrics else None
        chosen_z = selected[0] if selected else fallback_z
        selected_sharpe = selected[1] if selected else np.nan
        selected_by_year[year] = chosen_z
        schedules.append({"test_year": year, "train_start": train_start.date(),
                          "train_end_exclusive": test_start.date(),
                          "test_end_exclusive": test_end.date(), "selected_entry_z": chosen_z,
                          "selected_training_sharpe_net": selected_sharpe,
                          "selection_cost_bps": selection_cost_bps,
                          "fallback_used": selected is None})

        validation_costs = sorted({selection_cost_bps, 5.0})
        for threshold in grid:
            gross, turnover = daily_arrays(strategy_runs[threshold][1], dates, selection_cost_bps)
            trades = candidate_trades[threshold]
            if trades.empty:
                fold_entries = 0
            else:
                entry_dates = pd.to_datetime(trades.entry_date)
                fold_entries = int(((entry_dates >= test_start) & (entry_dates < test_end)).sum())
            for test_cost in validation_costs:
                fold_net = (gross - turnover * test_cost / 10_000).loc[test_mask]
                cv_rows.append({"test_year": year, "entry_z": threshold,
                                "test_trade_entries": fold_entries,
                                "test_net_pnl_units": float(fold_net.sum()),
                                "test_return_pct_on_fixed_capital": float(fold_net.sum() / capital_units * 100),
                                "test_cost_bps": test_cost,
                                "selection_cost_bps": selection_cost_bps})

    threshold_by_date = {date: selected_by_year[date.year]
                         for date in dates if date.year in selected_by_year}
    return (pd.DataFrame(schedules), pd.DataFrame(train_rows), pd.DataFrame(cv_rows),
            threshold_by_date)

def run_fixed_backtest(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=FIXED_DATA_FILE)
    parser.add_argument("--output", type=Path, default=DESIGN_OUT_DIR)
    parser.add_argument("--beta-lookback", type=int, default=60)
    parser.add_argument("--min-coverage", type=float, default=0.95)
    parser.add_argument("--entry-z", type=float, default=2.0)
    parser.add_argument("--stop-z", type=float, default=4.0)
    parser.add_argument("--costs-bps", default="0,1,5,10")
    parser.add_argument("--z-grid", default="0.5,1,1.5,2,2.5")
    parser.add_argument("--train-years", type=int, default=3)
    parser.add_argument("--min-training-trades", type=int, default=5)
    parser.add_argument("--selection-cost-bps", type=float, default=1.0,
                        help="Standard per-turnover cost used to choose annual Z* (default: 1 bp).")
    parser.add_argument("--hedge-mode", choices=["rehedge", "frozen"], default="rehedge",
                        help="Daily rehedging or freeze the entry hedge throughout each trade.")
    parser.add_argument("--skip-zstar", action="store_true",
                        help="Skip annual walk-forward threshold selection (fixed entry z only).")
    parser.add_argument("--grid", action="store_true",
                        help="Run the hedge-mode x lookback x entry-z diagnostic grid.")
    parser.add_argument("--grid-lookbacks", default="60,120,252")
    args = parser.parse_args(argv)
    dynamic = args.hedge_mode == "rehedge"
    grid_lookbacks = [int(x.strip()) for x in args.grid_lookbacks.split(",") if x.strip()]
    if any(lb < 20 for lb in grid_lookbacks):
        parser.error("--grid-lookbacks values must be at least 20.")
    if not args.data.exists():
        parser.error("Daily data not found. Run download_yfinance_daily.py first.")

    prices = pd.read_parquet(args.data)
    prices["date"] = pd.to_datetime(prices.date).dt.tz_localize(None).dt.normalize()
    close = prices.pivot(index="date", columns="ticker", values="close").sort_index()
    open_px = prices.pivot(index="date", columns="ticker", values="open").reindex(close.index)
    open_px = open_px.reindex(columns=close.columns)
    # Activity control: retain the seven economically specified pairs as a fixed
    # watchlist after one year of pair-level price coverage, without daily
    # cointegration/stability requalification.
    fixed_watchlist_at: dict[pd.Timestamp, set[tuple[str, str]]] = {}
    for idx, date in enumerate(close.index):
        if idx + 1 < 252:
            continue
        window = close.iloc[idx - 251:idx + 1]
        fixed_watchlist_at[date] = {
            (left, right) for left, right, _ in base.PAIRS
            if window[[left, right]].dropna().shape[0] / len(window) >= args.min_coverage
        }

    costs = [float(x.strip()) for x in args.costs_bps.split(",") if x.strip()]
    z_grid = sorted(set(float(x.strip()) for x in args.z_grid.split(",") if x.strip()))
    if args.entry_z not in z_grid or any(value <= 0 for value in z_grid):
        parser.error("--z-grid must contain the default --entry-z and only positive thresholds.")
    if args.train_years < 2 or args.min_training_trades < 1 or args.selection_cost_bps < 0:
        parser.error("Invalid time-series CV training settings.")
    args.output.mkdir(parents=True, exist_ok=True)
    # Diagnostic benchmark: keep the seven economic pair hypotheses fixed and
    # require only 252-session pair price coverage, not a daily cointegration gate.
    label = f"fixed_watchlist_252bar_warmup_{args.hedge_mode}"
    trades, daily, diagnostics = coherent_backtest(
        open_px, close, fixed_watchlist_at, args.beta_lookback,
        args.min_coverage, args.entry_z, args.stop_z, 0, dynamic)
    trades.to_csv(args.output / f"{label}_trades.csv", index=False)
    daily.to_csv(args.output / f"{label}_daily_pnl.csv", index=False)
    diagnostics.to_csv(args.output / f"{label}_pair_daily.csv", index=False)
    performance = summarize(trades, costs)
    pd.DataFrame(performance).to_csv(args.output / f"{label}_performance.csv", index=False)
    comparisons = [{"design": label, **row} for row in performance if row["pair"] == "ALL"]

    # Test fixed z* thresholds across the development history, then create a
    # prequential policy whose threshold for each test year uses only earlier data.
    if not args.skip_zstar:
        strategy_runs = {args.entry_z: (trades, daily)}
        for threshold in z_grid:
            if threshold == args.entry_z:
                continue
            print(f"Testing fixed entry z={threshold:g}", flush=True)
            candidate_trades, candidate_daily, _ = coherent_backtest(
                open_px, close, fixed_watchlist_at, args.beta_lookback,
                args.min_coverage, threshold, args.stop_z, 0, dynamic)
            strategy_runs[threshold] = (candidate_trades, candidate_daily)

        schedules, train_grid, cv_folds, threshold_by_date = walkforward_zstar(
            strategy_runs, fixed_watchlist_at, pd.DatetimeIndex(close.index),
            args.train_years, args.min_training_trades, args.selection_cost_bps,
            fallback_z=args.entry_z)
        schedules.to_csv(args.output / "zstar_threshold_schedule.csv", index=False)
        train_grid.to_csv(args.output / "zstar_training_grid.csv", index=False)
        cv_folds.to_csv(args.output / "zstar_cv_folds.csv", index=False)
        cv_summary = cv_folds.groupby(["test_cost_bps", "entry_z"], as_index=False).agg(
            mean_test_trade_entries=("test_trade_entries", "mean"),
            mean_annual_net_pnl_units=("test_net_pnl_units", "mean"),
            total_test_net_pnl_units=("test_net_pnl_units", "sum"),
            total_test_fixed_capital_return_pct=("test_return_pct_on_fixed_capital", "sum"))
        cv_summary.to_csv(args.output / "zstar_cv_summary.csv", index=False)

        oos_trades, oos_daily, oos_diagnostics = coherent_backtest(
            open_px, close, fixed_watchlist_at, args.beta_lookback,
            args.min_coverage, args.entry_z, args.stop_z, 0, dynamic,
            entry_z_by_date=threshold_by_date)
        oos_label = "zstar_walkforward"
        oos_trades.to_csv(args.output / f"{oos_label}_trades.csv", index=False)
        oos_daily.to_csv(args.output / f"{oos_label}_daily_pnl.csv", index=False)
        oos_diagnostics.to_csv(args.output / f"{oos_label}_pair_daily.csv", index=False)
        oos_performance = summarize(oos_trades, costs)
        pd.DataFrame(oos_performance).to_csv(args.output / f"{oos_label}_performance.csv", index=False)
        comparisons.extend({"design": oos_label, **row} for row in oos_performance if row["pair"] == "ALL")
    else:
        oos_trades, oos_daily = trades, daily
        oos_label = label

    # Causal Kalman alternative, kept beside the OLS implementation for a
    # directly comparable walk-forward run. Fixed Q is explicit in metadata.
    kalman_models = kalman_signal_models(close, args.beta_lookback, args.min_coverage)
    kalman_fixed, kalman_fixed_daily, kalman_fixed_diag = coherent_backtest(
        open_px, close, fixed_watchlist_at, args.beta_lookback,
        args.min_coverage, args.entry_z, args.stop_z, 0, dynamic,
        signal_models=kalman_models)
    kalman_fixed_label = "kalman_fixed_z2"
    kalman_fixed.to_csv(args.output / f"{kalman_fixed_label}_trades.csv", index=False)
    kalman_fixed_daily.to_csv(args.output / f"{kalman_fixed_label}_daily_pnl.csv", index=False)
    kalman_fixed_diag.to_csv(args.output / f"{kalman_fixed_label}_pair_daily.csv", index=False)
    kalman_fixed_performance = summarize(kalman_fixed, costs)
    pd.DataFrame(kalman_fixed_performance).to_csv(
        args.output / f"{kalman_fixed_label}_performance.csv", index=False)
    comparisons.extend({"design": kalman_fixed_label, **row}
                       for row in kalman_fixed_performance if row["pair"] == "ALL")

    if not args.skip_zstar:
        kalman_strategy_runs = {args.entry_z: (kalman_fixed, kalman_fixed_daily)}
        for threshold in z_grid:
            if threshold == args.entry_z:
                continue
            print(f"Testing Kalman fixed entry z={threshold:g}", flush=True)
            candidate_trades, candidate_daily, _ = coherent_backtest(
                open_px, close, fixed_watchlist_at, args.beta_lookback,
                args.min_coverage, threshold, args.stop_z, 0, dynamic,
                signal_models=kalman_models)
            kalman_strategy_runs[threshold] = (candidate_trades, candidate_daily)

        kalman_schedule, kalman_train_grid, kalman_cv_folds, kalman_threshold_by_date = walkforward_zstar(
            kalman_strategy_runs, fixed_watchlist_at, pd.DatetimeIndex(close.index),
            args.train_years, args.min_training_trades, args.selection_cost_bps,
            fallback_z=args.entry_z)
        kalman_schedule.to_csv(args.output / "kalman_zstar_threshold_schedule.csv", index=False)
        kalman_train_grid.to_csv(args.output / "kalman_zstar_training_grid.csv", index=False)
        kalman_cv_folds.to_csv(args.output / "kalman_zstar_cv_folds.csv", index=False)
        kalman_cv_summary = kalman_cv_folds.groupby(["test_cost_bps", "entry_z"], as_index=False).agg(
            mean_test_trade_entries=("test_trade_entries", "mean"),
            mean_annual_net_pnl_units=("test_net_pnl_units", "mean"),
            total_test_net_pnl_units=("test_net_pnl_units", "sum"),
            total_test_fixed_capital_return_pct=("test_return_pct_on_fixed_capital", "sum"))
        kalman_cv_summary.to_csv(args.output / "kalman_zstar_cv_summary.csv", index=False)

        kalman_oos_trades, kalman_oos_daily, kalman_oos_diagnostics = coherent_backtest(
            open_px, close, fixed_watchlist_at, args.beta_lookback,
            args.min_coverage, args.entry_z, args.stop_z, 0, dynamic,
            entry_z_by_date=kalman_threshold_by_date, signal_models=kalman_models)
        kalman_oos_label = "kalman_zstar_walkforward"
        kalman_oos_trades.to_csv(args.output / f"{kalman_oos_label}_trades.csv", index=False)
        kalman_oos_daily.to_csv(args.output / f"{kalman_oos_label}_daily_pnl.csv", index=False)
        kalman_oos_diagnostics.to_csv(args.output / f"{kalman_oos_label}_pair_daily.csv", index=False)
        kalman_oos_performance = summarize(kalman_oos_trades, costs)
        pd.DataFrame(kalman_oos_performance).to_csv(
            args.output / f"{kalman_oos_label}_performance.csv", index=False)
        comparisons.extend({"design": kalman_oos_label, **row}
                           for row in kalman_oos_performance if row["pair"] == "ALL")
    else:
        kalman_oos_trades, kalman_oos_daily = kalman_fixed, kalman_fixed_daily
        kalman_oos_label = kalman_fixed_label

    if args.grid:
        grid_rows = []
        for lookback in grid_lookbacks:
            for mode_dynamic in (True, False):
                for threshold in z_grid:
                    mode = "rehedge" if mode_dynamic else "frozen"
                    print(f"Grid: {mode}, lookback={lookback}, entry z={threshold:g}", flush=True)
                    grid_trades, _, _ = coherent_backtest(
                        open_px, close, fixed_watchlist_at, lookback,
                        args.min_coverage, threshold, args.stop_z, 0, mode_dynamic)
                    gross = grid_trades.gross_pnl.to_numpy(float) * 10_000 if not grid_trades.empty else np.array([])
                    se = gross.std(ddof=1) / np.sqrt(len(gross)) if len(gross) > 1 else np.nan
                    grid_rows.append({
                        "hedge_mode": mode, "lookback": lookback, "entry_z": threshold,
                        "trades": len(gross), "gross_bps_per_trade": float(gross.mean()) if len(gross) else np.nan,
                        "gross_total_bps": float(gross.sum()) if len(gross) else 0.0,
                        "win_rate": float((gross > 0).mean()) if len(gross) else np.nan,
                        "mean_holding_sessions": float(grid_trades.holding_sessions.mean()) if not grid_trades.empty else np.nan,
                        "gross_mean_t_stat": float(gross.mean() / se) if np.isfinite(se) and se > 0 else np.nan,
                    })
        pd.DataFrame(grid_rows).to_csv(args.output / "variant_grid.csv", index=False)

    # Translate normalized one-unit pair P&L into a fixed-capital portfolio
    # account. One gross unit per pair and seven capital units caps target gross
    # exposure at 1x; idle capital remains cash and returns are not compounded.
    date_index = pd.DatetimeIndex(close.index)
    portfolio_runs = [
        ("Rolling OLS", oos_daily, len(oos_trades)),
        ("Kalman", kalman_oos_daily, len(kalman_oos_trades)),
    ]
    portfolio_ledgers, portfolio_summaries = [], []
    for model_name, daily_path, trade_count in portfolio_runs:
        for portfolio_cost in (1.0, 5.0):
            account, account_summary = portfolio_account(
                daily_path, date_index, model_name, portfolio_cost, trade_count, 7.0)
            portfolio_ledgers.append(account)
            portfolio_summaries.append(account_summary)
    pd.concat(portfolio_ledgers, ignore_index=True).to_csv(
        args.output / "portfolio_daily.csv", index=False)
    pd.DataFrame(portfolio_summaries).to_csv(
        args.output / "portfolio_summary.csv", index=False)

    baseline = pd.read_csv(BASELINE_TRADES_FILE) if BASELINE_TRADES_FILE.exists() else pd.DataFrame()
    baseline_summary = summarize(baseline, costs) if not baseline.empty else []
    for row in baseline_summary:
        if row["pair"] == "ALL":
            comparisons.append({"design": "baseline_dynamic_z_immediate_exit", **row})

    comparison = pd.DataFrame(comparisons)
    comparison.to_csv(args.output / "design_comparison.csv", index=False)
    metadata = {
        "selection_policy": "Fixed seven-pair economic watchlist; each pair becomes eligible after 252-session pair-price coverage.",
        "research_role": "Activity diagnostic only; it deliberately removes the daily cointegration/stability gate and is not a validated strategy.",
        "hedge_mode": args.hedge_mode,
        "signal_model": f"{args.beta_lookback} prior daily closes; signal close evaluated out of sample; next-session open execution.",
        "position_model": ("The preceding lookback alpha, beta, and residual scale update daily; at each next open legs are rebalanced to the updated hedge, and turnover is charged."
                           if dynamic else "Alpha, beta, and residual scale are frozen at entry; no daily rebalancing; exit z is measured against the entry model."),
        "exit_rules": ("Exit on z-score crossing zero, +/-4 z hard stop, or non-positive updated hedge beta."
                       if dynamic else "Exit on z-score crossing zero or +/-4 z hard stop against the frozen entry model."),
        "entry_z": args.entry_z, "exit_z": 0.0, "stop_z": args.stop_z,
        "zstar_grid": z_grid,
        "zstar_walkforward": None if args.skip_zstar else {
            "method": "annual prequential time-series validation; at each January, select from the trailing training window and apply to the next calendar year",
            "training_years": args.train_years,
            "minimum_completed_training_trades": args.min_training_trades,
            "selection_objective": "highest annualized Sharpe of daily fixed-capital returns net of the configured selection cost",
            "selection_cost_bps_per_turnover": args.selection_cost_bps,
            "fallback_entry_z": args.entry_z,
        },
        "variant_grid": {"enabled": args.grid, "lookbacks": grid_lookbacks,
                         "hedge_modes": ["rehedge", "frozen"], "entry_z_values": z_grid},
        "costs_bps_per_turnover": costs,
        "kalman_filter": {
            "model": "Two-state random walk on log-price intercept and hedge beta.",
            "signal_timing": "One-step-ahead innovation at close t uses the state through t-1; current close updates state; posterior hedge is used at next-session open.",
            "process_noise_alpha": 1e-7,
            "process_noise_beta": 1e-7,
            "measurement_variance_scale": 1.0,
            "initialization": f"OLS on first {args.beta_lookback} aligned closes; measurement variance initialized from residual variance.",
            "limitations": ["Fixed process-noise settings are exploratory and not tuned.",
                            "Kalman uses predictive-innovation scaling, so both hedge adaptation and z-score scale differ from OLS."],
        },
        "portfolio_account": {
            "starting_capital_units": 7.0,
            "pair_gross_notional_units": 1.0,
            "gross_leverage_target": 1.0,
            "marking_note": "Pair targets are one gross unit each and at most seven pairs are open; closing gross leverage can drift modestly above 1x between rebalances.",
            "return_method": "Daily net dollar P&L divided by seven starting capital units; fixed notional, no compounding, unused capital earns zero.",
            "exposure_metrics": "Daily close gross and signed net exposure are recorded, including leverage versus starting capital.",
        },
        "pnl_normalization": "Each trade starts at one unit of gross pair notional. Summed trade P&L is un-compounded, not a portfolio return.",
        "limitations": [
            "The fixed z=2 activity result and threshold sensitivity use the already-inspected 2006-2026 sample; they are not confirmatory holdout results.",
            "The annual z-threshold walk-forward is causal by fold, but repeated use of this inspected history remains exploratory rather than a pristine final holdout.",
            "Daily adjusted OHLC execution is approximate; spread, borrow, financing, and market impact are omitted.",
        ],
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(comparison.loc[comparison.pair.eq("ALL")].to_string(index=False), flush=True)

model = SimpleNamespace(PAIRS=PAIRS)


# ===== Consolidated from make_research_visuals.py =====

matplotlib.use("Agg")

DAILY_DATA_FILE = ROOT / "data" / "yfinance_daily.parquet"

DESIGN_OUT_DIR = ROOT / "outputs" / "design_test"

VISUAL_OUT_DIR = ROOT / "outputs" / "visuals"

def rolling_signal_models(close: pd.DataFrame, lookback: int = 60,
                          coverage: float = 0.95) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Estimate log-price OLS on prior closes and score the current residual."""
    logs = np.log(close.where(close > 0))
    zscores = pd.DataFrame(index=close.index)
    betas = pd.DataFrame(index=close.index)
    spreads = pd.DataFrame(index=close.index)
    minimum = int(np.ceil(lookback * coverage))
    for left, right, _ in PAIRS:
        z = np.full(len(logs), np.nan)
        beta_path = np.full(len(logs), np.nan)
        spread_path = np.full(len(logs), np.nan)
        y_all, x_all = logs[left].to_numpy(float), logs[right].to_numpy(float)
        for i in range(lookback, len(logs)):
            start = i - lookback
            train_mask = np.isfinite(y_all[start:i]) & np.isfinite(x_all[start:i])
            if train_mask.sum() < minimum or not np.isfinite(y_all[i]) or not np.isfinite(x_all[i]):
                continue
            y, x = y_all[start:i][train_mask], x_all[start:i][train_mask]
            design = np.column_stack((np.ones(len(x)), x))
            alpha, beta = np.linalg.lstsq(design, y, rcond=None)[0]
            residual = y - alpha - beta * x
            scale = float(np.std(residual, ddof=2))
            if not np.isfinite(scale) or scale <= 1e-12:
                continue
            spread_path[i] = y_all[i] - alpha - beta * x_all[i]
            z[i] = spread_path[i] / scale
            beta_path[i] = beta
        zscores[f"{left}/{right}"] = z
        betas[f"{left}/{right}"] = beta_path
        spreads[f"{left}/{right}"] = spread_path
    return zscores, betas, spreads

def load_inputs(data_path: Path, design_path: Path):
    prices = pd.read_parquet(data_path)
    prices["date"] = pd.to_datetime(prices.date).dt.tz_localize(None).dt.normalize()
    close = prices.pivot(index="date", columns="ticker", values="close").sort_index()

    # The backtest's hedge-mode label is "rehedge". Accept the older
    # "rehedged" spelling too so existing output folders remain readable.
    name_stems = [
        "fixed_watchlist_252bar_warmup_rehedge",
        "fixed_watchlist_252bar_warmup_rehedged",
    ]
    trade_path = next(
        (design_path / f"{stem}_trades.csv" for stem in name_stems
         if (design_path / f"{stem}_trades.csv").is_file()),
        None,
    )
    ledger_path = next(
        (design_path / f"{stem}_daily_pnl.csv" for stem in name_stems
         if (design_path / f"{stem}_daily_pnl.csv").is_file()),
        None,
    )
    if trade_path is None or ledger_path is None:
        raise FileNotFoundError(
            "Could not find rehedged OLS visual inputs under "
            f"{design_path}. Run the Part 3 backtest first; expected files named "
            "fixed_watchlist_252bar_warmup_rehedge[_d]_trades.csv and "
            "fixed_watchlist_252bar_warmup_rehedge[_d]_daily_pnl.csv."
        )
    trades = pd.read_csv(trade_path, parse_dates=["entry_date", "exit_date"])
    ledger = pd.read_csv(ledger_path, parse_dates=["date"]).sort_values("date")
    return close, trades, ledger

def plot_prices(close: pd.DataFrame, out: Path) -> None:
    columns = 3
    rows = (len(close.columns) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(16, 3.3 * rows), sharex=True)
    for ax, ticker in zip(axes.flat, close.columns):
        ax.plot(close.index, close[ticker], color="#1769aa", linewidth=0.8)
        ax.set_title(ticker)
        ax.set_ylabel("Adjusted close")
        ax.grid(alpha=0.25)
    for ax in axes.flat[len(close.columns):]:
        ax.set_visible(False)
    for ax in axes[-1, :]:
        ax.set_xlabel("Date")
    fig.suptitle("Daily split/dividend-adjusted close prices", fontsize=17)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out / "01_adjusted_close_time_series.png", dpi=160)
    plt.close(fig)

def plot_individual_prices(close: pd.DataFrame, out: Path) -> None:
    destination = out / "by_ticker"
    destination.mkdir(parents=True, exist_ok=True)
    for ticker in close.columns:
        series = close[ticker].dropna()
        fig, ax = plt.subplots(figsize=(12, 5.5))
        ax.plot(series.index, series, linewidth=1.0, color="#1769aa")
        ax.set(title=f"{ticker} daily adjusted close", xlabel="Date", ylabel="Adjusted close")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(destination / f"{ticker}_adjusted_close.png", dpi=160)
        plt.close(fig)

def plot_pair_relative_prices(close: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots((len(PAIRS) + 1) // 2, 2, figsize=(15, 15), sharex=True)
    for ax, (left, right, rationale) in zip(axes.flat, PAIRS):
        for ticker, color in ((left, "#1769aa"), (right, "#e68613")):
            series = close[ticker].dropna()
            rebased = series / series.iloc[0] * 100
            ax.plot(rebased.index, rebased, label=ticker, color=color, linewidth=1.0)
        ax.set_title(f"{left} / {right}: {rationale}", fontsize=10)
        ax.set_ylabel("Adjusted price, first valid close = 100")
        ax.grid(alpha=0.25)
        ax.legend(loc="upper left", ncol=2)
    for ax in axes.flat[len(PAIRS):]:
        ax.set_visible(False)
    for ax in axes[-1, :]:
        ax.set_xlabel("Date")
    fig.suptitle("Pair price histories rebased to 100 (visual comparison only)", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out / "02_pair_rebased_price_histories.png", dpi=160)
    plt.close(fig)

def plot_spread_zscores(close: pd.DataFrame, trades: pd.DataFrame, out: Path,
                        lookback: int, coverage: float) -> None:
    zscores, _, _ = rolling_signal_models(close, lookback, coverage)
    fig, axes = plt.subplots((len(PAIRS) + 1) // 2, 2, figsize=(16, 15), sharex=True, sharey=True)
    for ax, (left, right, _) in zip(axes.flat, PAIRS):
        pair = f"{left}/{right}"
        ax.plot(zscores.index, zscores[pair], color="#1769aa", linewidth=0.75,
                label="Rolling spread z-score")
        ax.axhline(0, color="#333333", linewidth=0.8)
        ax.axhline(2, color="#e68613", linestyle="--", linewidth=0.9)
        ax.axhline(-2, color="#e68613", linestyle="--", linewidth=0.9)
        ax.axhline(4, color="#c53030", linestyle=":", linewidth=0.9)
        ax.axhline(-4, color="#c53030", linestyle=":", linewidth=0.9)
        pair_trades = trades.loc[trades.pair.eq(pair)]
        entries = pair_trades.dropna(subset=["entry_date"])
        exits = pair_trades.dropna(subset=["exit_date"])
        for frame, date_col, marker, color, label in (
            (entries, "entry_date", "^", "#2f855a", "trade entry"),
            (exits, "exit_date", "v", "#805ad5", "trade exit"),
        ):
            locs = zscores.index.get_indexer(pd.DatetimeIndex(frame[date_col]), method="nearest")
            locs = [i for i in locs if i >= 0 and np.isfinite(zscores[pair].iloc[i])]
            if locs:
                ax.scatter(zscores.index[locs], zscores[pair].iloc[locs], marker=marker,
                           s=18, color=color, label=label, zorder=4)
        ax.set_title(pair)
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right", fontsize=7, ncol=2)
    for ax in axes.flat[len(PAIRS):]:
        ax.set_visible(False)
    for ax in axes[-1, :]:
        ax.set_xlabel("Date")
    for ax in axes[:, 0]:
        ax.set_ylabel("Rolling spread z-score")
    fig.suptitle(f"Out-of-sample close z-score: rolling log-price OLS, {lookback} prior sessions", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out / "03_rolling_spread_zscores_and_trades.png", dpi=160)
    plt.close(fig)

def plot_betas(close: pd.DataFrame, out: Path, lookback: int, coverage: float) -> None:
    _, betas, _ = rolling_signal_models(close, lookback, coverage)
    fig, axes = plt.subplots((len(PAIRS) + 1) // 2, 2, figsize=(15, 15), sharex=True)
    for ax, pair in zip(axes.flat, betas.columns):
        ax.plot(betas.index, betas[pair], linewidth=0.85, color="#2f855a")
        ax.axhline(0, color="#555555", linewidth=0.8)
        ax.set_title(pair)
        ax.set_ylabel("OLS log-price beta")
        ax.grid(alpha=0.25)
    for ax in axes.flat[len(PAIRS):]:
        ax.set_visible(False)
    for ax in axes[-1, :]:
        ax.set_xlabel("Date")
    fig.suptitle(f"Rolling hedge-ratio estimates using {lookback} prior sessions", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out / "04_rolling_hedge_ratios.png", dpi=160)
    plt.close(fig)

def plot_individual_pairs(close: pd.DataFrame, trades: pd.DataFrame,
                          out: Path, lookback: int, coverage: float) -> None:
    destination = out / "by_pair"
    destination.mkdir(parents=True, exist_ok=True)
    zscores, _, spreads = rolling_signal_models(close, lookback, coverage)
    for left, right, _ in PAIRS:
        pair = f"{left}/{right}"
        fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True,
                                 gridspec_kw={"height_ratios": [1.1, 1.15, 1.0]})
        for ticker, color in ((left, "#1769aa"), (right, "#e68613")):
            series = close[ticker].dropna()
            axes[0].plot(series.index, series / series.iloc[0] * 100,
                         label=ticker, color=color, linewidth=1.0)
        axes[0].set_ylabel("Price index\n(first close = 100)")
        axes[0].set_title(f"{pair} | rebased adjusted-price history")
        axes[0].legend(ncol=2, loc="upper left")
        axes[0].grid(alpha=0.25)

        spread = spreads[pair]
        rolling_mean = spread.rolling(lookback, min_periods=max(20, lookback // 3)).mean()
        rolling_sd = spread.rolling(lookback, min_periods=max(20, lookback // 3)).std(ddof=1)
        axes[1].plot(spread.index, spread, color="#1769aa", linewidth=0.7, alpha=0.65, label="Spread residual")
        axes[1].plot(rolling_mean.index, rolling_mean, color="#e68613", linewidth=1.2, label=f"{lookback}-session rolling mean")
        axes[1].fill_between(spread.index, (rolling_mean - 2 * rolling_sd),
                             (rolling_mean + 2 * rolling_sd), color="#e68613", alpha=0.12,
                             label="Rolling mean Â± 2 SD")
        axes[1].set_ylabel("Log-price spread")
        axes[1].set_title("Rolling spread level and mean")
        axes[1].legend(loc="upper left", ncol=3, fontsize=8)
        axes[1].grid(alpha=0.25)

        z = zscores[pair]
        axes[2].plot(z.index, z, color="#1769aa", linewidth=0.75, label="Rolling spread z-score")
        for level, color, style in ((0, "#333333", "-"), (2, "#e68613", "--"),
                                    (-2, "#e68613", "--"), (4, "#c53030", ":"),
                                    (-4, "#c53030", ":")):
            axes[2].axhline(level, color=color, linestyle=style, linewidth=0.9)
        pair_trades = trades.loc[trades.pair.eq(pair)]
        for frame, date_col, value_col, marker, color, label in (
            (pair_trades, "entry_date", "entry_z", "^", "#2f855a", "Trade entry"),
            (pair_trades, "exit_date", "exit_z", "v", "#c53030", "Trade exit"),
        ):
            frame = frame.dropna(subset=[date_col, value_col])
            if not frame.empty:
                axes[2].scatter(pd.to_datetime(frame[date_col]), frame[value_col],
                                marker=marker, color=color, s=28, label=label, zorder=4)
        axes[2].set_ylabel("Z-score")
        axes[2].set_xlabel("Date")
        axes[2].set_title("Rolling entry/exit signal with executed trade markers")
        axes[2].legend(loc="upper left", ncol=3, fontsize=8)
        axes[2].grid(alpha=0.25)
        fig.suptitle(f"{pair} rolling pair diagnostics", fontsize=16)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        filename = f"{left}_{right}_diagnostics.png"
        fig.savefig(destination / filename, dpi=160)
        plt.close(fig)

def plot_nav(ledger: pd.DataFrame, out: Path, capital: float = 7.0,
             costs: tuple[float, ...] = (0.0, 1.0, 5.0, 10.0)) -> pd.DataFrame:
    dates = pd.DatetimeIndex([ledger.date.min() - pd.Timedelta(days=1), *ledger.date.tolist()])
    fig, axes = plt.subplots(2, 1, figsize=(15, 11), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})
    rows = []
    series = {"date": dates}
    for cost in costs:
        cost_label = (f"{cost:g} bp (standard)" if cost == 1.0 else
                      f"{cost:g} bp (sensitivity)" if cost == 5.0 else f"{cost:g} bp")
        net = ledger.gross_pnl - ledger.turnover * cost / 10_000
        nav = pd.Series(np.r_[1.0, 1.0 + net.cumsum().to_numpy() / capital])
        drawdown = nav / nav.cummax() - 1
        series[f"nav_{cost:g}bp"] = nav
        series[f"drawdown_{cost:g}bp"] = drawdown
        axes[0].plot(dates, nav, label=cost_label, linewidth=1.2)
        axes[1].plot(dates, drawdown * 100, label=cost_label, linewidth=1.0)
        rows.append({"cost_bps_per_turnover": cost, "starting_capital_units": capital,
                     "ending_nav_index": nav.iloc[-1],
                     "total_return_pct_fixed_notional": (nav.iloc[-1] - 1) * 100,
                     "max_drawdown_pct": drawdown.min() * 100})
    axes[0].axhline(1, color="#555555", linewidth=0.8)
    axes[0].set_title("Fixed-notional equity index (no compounding)")
    axes[0].set_ylabel("Index (1.00 = starting capital)")
    axes[1].set_title("Peak-to-trough drawdown")
    axes[1].set_ylabel("Drawdown (%)")
    axes[1].set_xlabel("Date")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(title="Cost per turnover")
    fig.suptitle("Fixed-watchlist activity-control results", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out / "05_nav_and_drawdown_cost_sensitivity.png", dpi=160)
    plt.close(fig)
    pd.DataFrame(series).to_csv(out / "nav_timeseries.csv", index=False)
    return pd.DataFrame(rows)

def daily_returns(ledger: pd.DataFrame, sessions: pd.DatetimeIndex,
                  cost_bps: float, capital: float) -> pd.Series:
    daily = ledger.set_index("date").reindex(sessions)
    pnl = daily.gross_pnl.fillna(0.0) - daily.turnover.fillna(0.0) * cost_bps / 10_000
    return pnl / capital

def plot_pnl_risk(sessions: pd.DatetimeIndex, fixed_daily: pd.DataFrame,
                  walkforward_daily: pd.DataFrame, out: Path,
                  cost_levels: tuple[float, ...] = (1.0, 5.0), capital: float = 7.0) -> None:
    paths = [
        ("Fixed z=2 diagnostic", cost, daily_returns(fixed_daily, sessions, cost, capital))
        for cost in cost_levels
    ] + [
        ("Walk-forward z*", cost, daily_returns(walkforward_daily, sessions, cost, capital))
        for cost in cost_levels
    ]
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), sharex=True)
    rows = []
    output = pd.DataFrame({"date": sessions})
    for label, cost, returns in paths:
        color = "#1769aa" if label.startswith("Fixed") else "#e68613"
        linestyle = "-" if cost == 1.0 else "--"
        series_label = f"{label} ({cost:g} bp{' standard' if cost == 1.0 else ' sensitivity'})"
        cumulative = returns.cumsum() * 100
        sharpe = (returns.rolling(252, min_periods=126).mean()
                  / returns.rolling(252, min_periods=126).std(ddof=1) * np.sqrt(252))
        rolling = returns.rolling(252, min_periods=126)
        var95 = -rolling.quantile(0.05) * 100

        def expected_shortfall(window: np.ndarray) -> float:
            cutoff = np.quantile(window, 0.05)
            tail = window[window <= cutoff]
            return -float(tail.mean()) if len(tail) else np.nan

        cvar95 = rolling.apply(expected_shortfall, raw=True) * 100
        volatility = returns.rolling(252, min_periods=126).std(ddof=1) * np.sqrt(252) * 100
        drawdown = (1 + returns.cumsum()) / (1 + returns.cumsum()).cummax() - 1
        axes[0, 0].plot(sessions, cumulative, color=color, linewidth=1.1, linestyle=linestyle, label=series_label)
        axes[0, 1].plot(sessions, sharpe, color=color, linewidth=1.0, linestyle=linestyle, label=series_label)
        axes[1, 0].plot(sessions, var95, color=color, linewidth=1.0, linestyle=linestyle, label=f"{series_label} VaR")
        axes[1, 0].plot(sessions, cvar95, color=color, linewidth=1.0, linestyle=":" if cost == 1.0 else "-.", label=f"{series_label} CVaR")
        axes[1, 1].plot(sessions, drawdown * 100, color=color, linewidth=1.0, linestyle=linestyle, label=series_label)
        suffix = f"{cost:g}bp"
        output[f"{label}_{suffix}_daily_return"] = returns.to_numpy()
        output[f"{label}_{suffix}_cumulative_pnl_pct"] = cumulative.to_numpy()
        output[f"{label}_{suffix}_rolling_sharpe_252"] = sharpe.to_numpy()
        output[f"{label}_{suffix}_rolling_var95_pct"] = var95.to_numpy()
        output[f"{label}_{suffix}_rolling_cvar95_pct"] = cvar95.to_numpy()
        output[f"{label}_{suffix}_drawdown_pct"] = drawdown.to_numpy() * 100
        rows.append({"strategy": label, "cost_bps_per_turnover": cost,
                     "capital_units": capital,
                     "cumulative_fixed_capital_pnl_pct": float(cumulative.iloc[-1]),
                     "max_drawdown_pct": float(drawdown.min() * 100),
                     "latest_rolling_sharpe_252": float(sharpe.iloc[-1]),
                     "latest_rolling_var95_pct": float(var95.iloc[-1]),
                     "latest_rolling_cvar95_pct": float(cvar95.iloc[-1]),
                     "latest_rolling_volatility_pct": float(volatility.iloc[-1])})
    titles = (("Cumulative net P&L (fixed capital)", "252-session annualized Sharpe"),
              ("1-day historical VaR / CVaR (95%)", "Drawdown on fixed capital"))
    ylabels = (("Cumulative P&L (% of starting capital)", "Sharpe ratio"),
               ("Loss (% of starting capital)", "Drawdown (%)"))
    for row_idx in range(2):
        for col_idx in range(2):
            ax = axes[row_idx, col_idx]
            ax.set_title(titles[row_idx][col_idx])
            ax.set_ylabel(ylabels[row_idx][col_idx])
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)
    axes[1, 0].set_xlabel("Date")
    axes[1, 1].set_xlabel("Date")
    fig.suptitle("Portfolio P&L and rolling risk | 1 bp standard, 5 bp sensitivity | 7 fixed capital units", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out / "06_cumulative_pnl_and_rolling_risk.png", dpi=160)
    plt.close(fig)
    output.to_csv(out / "rolling_risk_timeseries.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "rolling_risk_summary.csv", index=False)

def plot_zstar_validation(design_dir: Path, sessions: pd.DatetimeIndex, out: Path,
                          standard_cost_bps: float = 1.0, sensitivity_cost_bps: float = 5.0,
                          capital: float = 7.0) -> None:
    cv = pd.read_csv(design_dir / "zstar_cv_summary.csv")
    schedule = pd.read_csv(design_dir / "zstar_threshold_schedule.csv")
    walkforward = pd.read_csv(design_dir / "zstar_walkforward_daily_pnl.csv", parse_dates=["date"])
    fig, axes = plt.subplots(figsize=(12, 6.5))
    right = axes.twinx()
    standard_cv = cv.loc[cv.test_cost_bps.eq(standard_cost_bps)]
    line1 = axes.plot(standard_cv.entry_z, standard_cv.mean_test_trade_entries, "o-", color="#1769aa",
                      label="Mean trade entries per test year")
    standard_pnl = cv.loc[cv.test_cost_bps.eq(standard_cost_bps)]
    sensitivity_pnl = cv.loc[cv.test_cost_bps.eq(sensitivity_cost_bps)]
    line2 = right.plot(standard_pnl.entry_z, standard_pnl.total_test_net_pnl_units, "o-", color="#e68613",
                       label=f"Summed net P&L ({standard_cost_bps:g} bp standard)")
    line3 = right.plot(sensitivity_pnl.entry_z, sensitivity_pnl.total_test_net_pnl_units, "s--", color="#bb3e03",
                       label=f"Summed net P&L ({sensitivity_cost_bps:g} bp sensitivity)")
    axes.set(title="Entry z* sensitivity across chronological test folds | 1 bp standard, 5 bp sensitivity",
             xlabel="Entry z* threshold", ylabel="Mean trades / test year")
    right.set_ylabel("Summed net P&L (one-unit-per-trade units)")
    axes.grid(alpha=0.3)
    axes.legend(line1 + line2 + line3, [line.get_label() for line in line1 + line2 + line3], loc="best")
    fig.tight_layout()
    fig.savefig(out / "07_zstar_chronological_cv_sensitivity.png", dpi=160)
    plt.close(fig)

    daily = walkforward.set_index("date").reindex(sessions)
    fig, left = plt.subplots(figsize=(13, 6.5))
    right = left.twinx()
    test_start = pd.to_datetime(schedule.test_year.astype(int).astype(str) + "-01-01")
    l1 = left.step(test_start, schedule.selected_entry_z, where="post",
                   color="#1769aa", label="Selected z* from prior data")
    lines = list(l1)
    for cost, color, linestyle in ((standard_cost_bps, "#e68613", "-"),
                                   (sensitivity_cost_bps, "#bb3e03", "--")):
        net = daily.gross_pnl.fillna(0.0) - daily.turnover.fillna(0.0) * cost / 10_000
        cumulative_pct = net.cumsum() / capital * 100
        l2 = right.plot(sessions, cumulative_pct, color=color, linestyle=linestyle,
                        label=f"Walk-forward cumulative net P&L ({cost:g} bp{' standard' if cost == standard_cost_bps else ' sensitivity'})")
        lines.extend(l2)
    left.set(title="Annual time-series validation: past-selected entry z* | 1 bp standard, 5 bp sensitivity",
             xlabel="Date", ylabel="Selected entry z*")
    right.set_ylabel("Cumulative P&L (% of seven-unit starting capital)")
    left.grid(alpha=0.3)
    left.legend(lines, [line.get_label() for line in lines], loc="best")
    fig.tight_layout()
    fig.savefig(out / "08_zstar_walkforward_test_path.png", dpi=160)
    plt.close(fig)

def plot_kalman_comparison(design_dir: Path, out: Path, sessions: pd.DatetimeIndex,
                           capital: float = 7.0) -> None:
    """Plot standalone Kalman and rolling-OLS walk-forward return/risk paths."""
    ols_path = design_dir / "zstar_walkforward_daily_pnl.csv"
    kalman_path = design_dir / "kalman_zstar_walkforward_daily_pnl.csv"
    if not (ols_path.exists() and kalman_path.exists()):
        return
    daily_by_model = {
        "Rolling OLS": (pd.read_csv(ols_path, parse_dates=["date"]).set_index("date"),
                        "zstar_walkforward_trades.csv"),
        "Kalman": (pd.read_csv(kalman_path, parse_dates=["date"]).set_index("date"),
                   "kalman_zstar_walkforward_trades.csv"),
    }
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    colors = {"Rolling OLS": "#1769aa", "Kalman": "#e68613"}
    rows = []
    for model_name, (daily, trade_file) in daily_by_model.items():
        frame = daily.reindex(sessions)
        gross = frame.gross_pnl.fillna(0.0)
        turnover = frame.turnover.fillna(0.0)
        trades_path = design_dir / trade_file
        trade_count = len(pd.read_csv(trades_path)) if trades_path.exists() else 0
        for cost, linestyle in ((1.0, "-"), (5.0, "--")):
            returns = (gross - turnover * cost / 10_000) / capital
            cumulative = returns.cumsum() * 100
            nav = 1 + returns.cumsum()
            drawdown = nav / nav.cummax() - 1
            label = f"{model_name}, {cost:g} bp{' standard' if cost == 1 else ' sensitivity'}"
            axes[0].plot(sessions, cumulative, color=colors[model_name], linestyle=linestyle,
                         linewidth=1.5, label=label)
            axes[1].plot(sessions, drawdown * 100, color=colors[model_name], linestyle=linestyle,
                         linewidth=1.4, label=label)
            rolling_sd = returns.rolling(252, min_periods=126).std(ddof=1)
            rolling_sharpe = (returns.rolling(252, min_periods=126).mean()
                              / rolling_sd * np.sqrt(252))
            rows.append({
                "model": model_name, "cost_bps_per_turnover": cost,
                "completed_round_trips": trade_count,
                "cumulative_fixed_capital_return_pct": float(cumulative.iloc[-1]),
                "max_drawdown_pct": float(drawdown.min() * 100),
                "latest_252_session_sharpe": (float(rolling_sharpe.dropna().iloc[-1])
                                              if rolling_sharpe.notna().any() else np.nan),
                "gross_pnl_units": float(gross.sum()),
                "turnover_units": float(turnover.sum()),
            })
    axes[0].set_title("Walk-forward hedge estimator comparison | 1 bp standard, 5 bp sensitivity")
    axes[0].set_ylabel("Cumulative P&L (% of seven-unit capital)")
    axes[1].set_ylabel("Drawdown (%)")
    axes[1].set_xlabel("Date")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "09_kalman_vs_rolling_ols.png", dpi=160)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(design_dir / "kalman_model_comparison.csv", index=False)


def plot_portfolio_accounting(design_dir: Path, out: Path) -> None:
    """Plot fixed-capital account value, drawdown, gross and net leverage."""
    source = design_dir / "portfolio_daily.csv"
    if not source.exists():
        return
    frame = pd.read_csv(source, parse_dates=["date"])
    if frame.empty:
        return
    fig, axes = plt.subplots(4, 1, figsize=(14, 13), sharex=True)
    colors = {"Rolling OLS": "#1769aa", "Kalman": "#e68613"}
    for model_name, group in frame.groupby("model", sort=False):
        for cost, linestyle in ((1.0, "-"), (5.0, "--")):
            path = group.loc[group.cost_bps_per_turnover.eq(cost)].sort_values("date")
            if path.empty:
                continue
            suffix = f"{cost:g} bp{' standard' if cost == 1 else ' sensitivity'}"
            label = f"{model_name}, {suffix}"
            axes[0].plot(path.date, (path.nav - 1) * 100, color=colors[model_name],
                         linestyle=linestyle, linewidth=1.2, label=label)
            axes[1].plot(path.date, path.drawdown_pct, color=colors[model_name],
                         linestyle=linestyle, linewidth=1.1, label=label)
            # Exposure is cost-independent; draw it once for each model.
            if cost == 1.0:
                axes[2].plot(path.date, path.gross_leverage, color=colors[model_name],
                             linewidth=1.1, label=f"{model_name} gross")
                axes[3].plot(path.date, path.net_exposure_leverage, color=colors[model_name],
                             linewidth=1.1, label=f"{model_name} net")
    axes[0].set_ylabel("Cumulative return (%)")
    axes[1].set_ylabel("Drawdown (%)")
    axes[2].set_ylabel("Gross leverage")
    axes[3].set_ylabel("Net exposure / capital")
    axes[3].set_xlabel("Date")
    axes[0].set_title("Portfolio account: seven fixed capital units, fixed notional, no compounding")
    axes[2].axhline(1.0, color="#555555", linestyle=":", linewidth=1.0, label="1x gross target")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "10_portfolio_account_nav_risk_and_exposure.png", dpi=160)
    plt.close(fig)

def run_visuals() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DAILY_DATA_FILE)
    parser.add_argument("--design-output", type=Path, default=DESIGN_OUT_DIR)
    parser.add_argument("--output", type=Path, default=VISUAL_OUT_DIR)
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--coverage", type=float, default=0.95)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    close, trades, ledger = load_inputs(args.data, args.design_output)
    walkforward_path = args.design_output / "zstar_walkforward_daily_pnl.csv"
    if not walkforward_path.exists():
        parser.error("Missing walk-forward z* outputs. Run fixed_watchlist_backtest.py first.")
    walkforward_ledger = pd.read_csv(walkforward_path, parse_dates=["date"])
    plot_prices(close, args.output)
    plot_individual_prices(close, args.output)
    plot_pair_relative_prices(close, args.output)
    plot_spread_zscores(close, trades, args.output, args.lookback, args.coverage)
    plot_betas(close, args.output, args.lookback, args.coverage)
    plot_individual_pairs(close, trades, args.output, args.lookback, args.coverage)
    summary = plot_nav(ledger, args.output)
    summary.to_csv(args.output / "nav_summary.csv", index=False)
    plot_pnl_risk(close.index, ledger, walkforward_ledger, args.output)
    plot_zstar_validation(args.design_output, close.index, args.output)
    plot_kalman_comparison(args.design_output, args.output, close.index)
    plot_portfolio_accounting(args.design_output, args.output)
    print(f"Created overview, individual ticker/pair plots, z* validation, and rolling-risk figures in {args.output}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Run the complete equity-pairs research workflow.")
    parser.add_argument("--stage", choices=["all", "data", "screen", "backtest", "visuals"], default="all",
                        help="Run one of the five presentation stages or the complete workflow.")
    parser.add_argument("--download", action="store_true",
                        help="Refresh adjusted daily market data from Yahoo Finance before analysis.")
    parser.add_argument("--start", default="2006-01-01",
                        help="Inclusive first download date; used with --download.")
    parser.add_argument("--end", default=None,
                        help="Exclusive download end date; omitted means latest available.")
    parser.add_argument("--hedge-mode", choices=["rehedge", "frozen"], default="rehedge")
    parser.add_argument("--grid", action="store_true",
                        help="Run the diagnostic hedge-mode/lookback/entry-z grid.")
    parser.add_argument("--grid-lookbacks", default="60,120,252")
    parser.add_argument("--skip-zstar", action="store_true")
    parser.add_argument("--output", type=Path, default=DESIGN_OUT_DIR,
                        help="Backtest output directory.")
    args = parser.parse_args(argv)

    if args.download or not DOWNLOAD_OUTPUT.exists():
        print("\n=== Download adjusted daily market data ===", flush=True)
        frame = download(args.start, args.end)
        DOWNLOAD_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(DOWNLOAD_OUTPUT, index=False)
        metadata = {
            "provider": "yfinance history() wrapper",
            "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
            "interval": "1d", "auto_adjust": True,
            "start_inclusive": args.start, "end_exclusive": args.end,
            "tickers": TICKERS, "rows": len(frame),
            "data_file": str(DOWNLOAD_OUTPUT.resolve()),
            "price_note": "OHLC adjusted by yfinance for splits/dividends; action columns are retained when returned.",
        }
        DOWNLOAD_OUTPUT.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"Saved {len(frame):,} rows to {DOWNLOAD_OUTPUT}", flush=True)
    else:
        print(f"Using existing data: {DOWNLOAD_OUTPUT}", flush=True)

    if not DAILY_DATA_FILE.exists():
        parser.error(f"Missing {DAILY_DATA_FILE}; rerun with --download.")
    if args.stage == "data":
        print(f"Data ready: {DOWNLOAD_OUTPUT}", flush=True)
        return

    original_argv = sys.argv[:]
    try:
        script = str(Path(__file__).resolve())
        if args.stage in {"all", "screen"}:
            sys.argv = [script]
            print("\n=== Rolling stationarity/cointegration screen ===", flush=True)
            run_rolling_screen()
        if args.stage in {"all", "backtest"}:
            backtest_args = ["--hedge-mode", args.hedge_mode,
                             "--grid-lookbacks", args.grid_lookbacks,
                             "--output", str(args.output)]
            if args.grid:
                backtest_args.append("--grid")
            if args.skip_zstar:
                backtest_args.append("--skip-zstar")
            print("\n=== OLS/Kalman and walk-forward backtests ===", flush=True)
            run_fixed_backtest(backtest_args)
        if args.stage in {"all", "visuals"}:
            sys.argv = [script]
            print("\n=== Figures and risk reports ===", flush=True)
            run_visuals()
    finally:
        sys.argv = original_argv

    if args.stage not in {"all", "backtest"}:
        return
    comparison = pd.read_csv(args.output / "design_comparison.csv")
    headline = comparison.loc[
        comparison.pair.eq("ALL") & comparison.cost_bps_per_turnover.isin([1.0, 5.0]),
        ["design", "trades", "cost_bps_per_turnover", "net_pnl_bps"]
    ]
    print("\nHeadline results (summed one-unit-per-trade bps; not account returns):", flush=True)
    print(headline.to_string(index=False), flush=True)
    print(f"\nCompleted stage '{args.stage}'. Results are under {args.output.parent}.", flush=True)


if __name__ == "__main__":
    main()
