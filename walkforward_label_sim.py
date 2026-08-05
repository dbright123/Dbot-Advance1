"""
walkforward_label_sim.py
────────────────────────────────────────────────────────────────────────
Leak-free WALK-FORWARD signal generator that mirrors the live behaviour of
market_prediction_extra.py (predict the current bar, then retrain, bar by
bar). It reproduces the server's exact configuration:

    symbol   : XAUUSDc
    model    : ExtraTrees (et)          class_weight="balanced"
    features : raw (default) | engineered
    labeler  : RobustPriceLabelerV3(atr=14, zz=0.5, hold=1, streak=0.01,
                                     target_hold_pct=0.01) + fix_pivot_labels

Why walk-forward?  The live server retrains every ~60 min (= 1 H1 bar), so
the model that predicts bar t was only ever trained on data < t. Training a
single model on the whole set (as a naive backtest does) leaks the future
into the labels the model "already knows". This script instead:

    for each of the last N bars (chronological):
        1. label + train ONLY on data strictly before bar t   (retrain)
        2. predict bar t with that model                       (predict)
        3. record the prediction

The result is a CSV of realistic, out-of-sample predictions that the MT5
Strategy-Tester replay EA can trade with the full market_predictor.mq5 logic.

USAGE
    python walkforward_label_sim.py --bars 1000
    python walkforward_label_sim.py --bars 1000 --retrain-every 4 --train-window 6000
    python walkforward_label_sim.py --features engineered --model et

Output CSV columns:  time, pred, pred_name, open, high, low, close, atr, label_hindsight
────────────────────────────────────────────────────────────────────────
"""

import argparse
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import MetaTrader5 as mt5

from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              GradientBoostingClassifier, HistGradientBoostingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Reuse the server's exact labeller so labels match production 1:1
from extra_function import RobustPriceLabelerV3, fix_pivot_labels

# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Walk-forward label simulator (predict-then-retrain)")
parser.add_argument("--terminal", default="C:\\Program Files\\MetaTrader 5\\terminal64.exe",
                    help="Full path to terminal64.exe")
parser.add_argument("--symbol", default="XAUUSD", help="Trading symbol")
parser.add_argument("--model", default="rf", choices=["et", "rf", "hgb", "gb", "logit"],
                    help="classifier (default et = ExtraTrees, same as server)")
parser.add_argument("--features", default="raw", choices=["engineered", "raw"],
                    help="feature set (default raw, same as server default)")
parser.add_argument("--bars", type=int, default=1000, help="how many recent bars to walk forward")
parser.add_argument("--retrain-every", type=int, default=1,
                    help="retrain cadence in bars (1 = every bar like the live server; raise to speed up)")
parser.add_argument("--train-window", type=int, default=0,
                    help="cap training rows to the last W bars (0 = use all history before t)")
parser.add_argument("--n-estimators", type=int, default=350,
                    help="trees for et/rf (lower = faster, default 400 like server)")
parser.add_argument("--out", default="",
                    help="output CSV path (default: MT5 Common\\Files\\market_predictor_signals.csv)")
args = parser.parse_args()


# ─────────────────────────────────────────────────────────────────────
# Server-identical feature/model helpers (copied from market_prediction_extra.py)
# ─────────────────────────────────────────────────────────────────────
FEATURE_NAMES = [
    'open', 'high', 'low', 'close', 'volume',
    'open_1h', 'high_1h', 'low_1h', 'close_1h', 'volume_1h',
    'open_4h', 'high_4h', 'low_4h', 'close_4h', 'volume_4h',
    'open_1d', 'high_1d', 'low_1d', 'close_1d', 'volume_1d',
    'hour','minute', 'day', 'month', 'day_of_week',
]

ENG_FEATURES = [
    "ret1", "ret3", "ret6", "ret12", "ret24", "rng", "body", "upwick", "lowick",
    "c_sma20", "c_sma50", "sma20_50", "vol20", "atr_pct", "rsi", "vchg",
    "c_vs_4h", "c_vs_1d", "h4_ret", "d1_ret",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
]


def fetch_rates(symbol, timeframe, count):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
    if rates is None:
        raise ValueError(f"MT5 returned None for {symbol}. Error: {mt5.last_error()}")
    df = pd.DataFrame(rates)
    vol_col = next((c for c in df.columns if "volume" in c.lower()), None)
    if vol_col is None:
        raise KeyError(f"No volume column found among: {list(df.columns)}")
    df = df[["time", "open", "high", "low", "close", vol_col]].copy()
    df.rename(columns={vol_col: "volume"}, inplace=True)
    return df.set_index("time")


def build_multi_tf(symbol):
    if not mt5.initialize(args.terminal):
        raise RuntimeError(f"MT5 initialize() failed: {mt5.last_error()}")
    df_15m = fetch_rates(symbol, mt5.TIMEFRAME_M15, 9_000_000).reset_index()
    df_1h = fetch_rates(symbol, mt5.TIMEFRAME_H1, 9_000_000).reset_index()
    df_4h = fetch_rates(symbol, mt5.TIMEFRAME_H4, 9_000_000).reset_index()
    df_1d = fetch_rates(symbol, mt5.TIMEFRAME_D1, 9_000_000).reset_index()

    def tag_and_shift(df, suffix):
        df = df.sort_values("time").rename(
            columns={c: f"{c}_{suffix}" for c in df.columns if c != "time"})
        val_cols = [c for c in df.columns if c != "time"]
        df[val_cols] = df[val_cols].shift(1)      # only use CLOSED higher-TF bars
        return df

    df_1h = tag_and_shift(df_1h, "1h")
    df_4h = tag_and_shift(df_4h, "4h")
    df_1d = tag_and_shift(df_1d, "1d")
    
    #df_1h = df_1h.sort_values("time")
    df_15m = df_15m.sort_values("time")
    merged = pd.merge_asof(df_15m, df_1h, on="time", direction="backward")
    merged = pd.merge_asof(merged, df_4h, on="time", direction="backward")
    #merged = pd.merge_asof(df_1h, df_4h, on="time", direction="backward")
    merged = pd.merge_asof(merged, df_1d, on="time", direction="backward")
    return merged.reset_index(drop=True)


def add_calendar(df):
    t = pd.to_datetime(df["time"], unit="s", errors="coerce")
    df["month"] = t.dt.month
    df["day"] = t.dt.day
    df["hour"] = t.dt.hour
    df["minute"] = t.dt.minute
    df["day_of_week"] = t.dt.dayofweek
    print(df)
    return df


def add_engineered_features(df):
    eps = 1e-9
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    v = df["volume"].astype(float)
    df["ret1"] = c.pct_change()
    df["ret3"] = c.pct_change(3)
    df["ret6"] = c.pct_change(6)
    df["ret12"] = c.pct_change(12)
    df["ret24"] = c.pct_change(24)
    df["rng"] = (h - l) / (c + eps)
    df["body"] = (c - o) / (c + eps)
    df["upwick"] = (h - np.maximum(o, c)) / (c + eps)
    df["lowick"] = (np.minimum(o, c) - l) / (c + eps)
    sma20 = c.rolling(20).mean()
    sma50 = c.rolling(50).mean()
    df["c_sma20"] = c / (sma20 + eps) - 1
    df["c_sma50"] = c / (sma50 + eps) - 1
    df["sma20_50"] = sma20 / (sma50 + eps) - 1
    df["vol20"] = df["ret1"].rolling(20).std()
    tr = pd.concat([(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    df["atr_pct"] = tr.rolling(14).mean() / (c + eps)
    delta = c.diff()
    up = delta.clip(lower=0).rolling(14).mean()
    dn = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi"] = (100 - 100 / (1 + up / (dn + eps))) / 100.0
    df["vchg"] = v.pct_change().clip(-5, 5)
    df["c_vs_4h"] = c / (df["close_4h"] + eps) - 1
    df["c_vs_1d"] = c / (df["close_1d"] + eps) - 1
    df["h4_ret"] = df["close_4h"].pct_change()
    df["d1_ret"] = df["close_1d"].pct_change()
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    return df, list(ENG_FEATURES)


def make_model(name):
    name = (name or "et").lower()
    if name == "rf":
        return RandomForestClassifier(n_estimators=args.n_estimators, class_weight="balanced",
                                      random_state=42, n_jobs=-1)
    if name == "hgb":
        return HistGradientBoostingClassifier(random_state=42, max_iter=300, learning_rate=0.08)
    if name == "gb":
        return GradientBoostingClassifier(random_state=42)
    if name == "logit":
        return make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=1000, class_weight="balanced"))
    return ExtraTreesClassifier(n_estimators=args.n_estimators, class_weight="balanced",
                                random_state=42, n_jobs=-1)


def label_window(df_slice):
    """Server-identical labelling, applied only to the given (causal) window."""
    labeler = RobustPriceLabelerV3(
        atr_period      = 14,
        zigzag_atr_mult = 5,
        hold_bars       = 5,
        min_streak      = 1.5,
        target_hold_pct = 1.5,
    )
    d = labeler.label(df_slice.copy())
    #d = fix_pivot_labels(d)
    return np.asarray(d["label"].values)


def default_out_path():
    appdata = os.environ.get("APPDATA", "")
    common = os.path.join(appdata, "MetaQuotes", "Terminal", "Common", "Files")
    os.makedirs(common, exist_ok=True)
    return os.path.join(common, "market_predictor_signals.csv")


# ─────────────────────────────────────────────────────────────────────
# Walk-forward
# ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 64)
    print("  Walk-forward label simulator (predict-then-retrain)")
    print(f"  symbol={args.symbol}  model={args.model}  features={args.features}")
    print(f"  bars={args.bars}  retrain_every={args.retrain_every}  "
          f"train_window={args.train_window or 'all'}  n_estimators={args.n_estimators}")
    print("=" * 64)

    df = build_multi_tf(args.symbol)
    df = add_calendar(df)
    if args.features == "engineered":
        df, feats = add_engineered_features(df)
    else:
        feats = list(FEATURE_NAMES)

    # ATR(14) on the base timeframe — passed through for the EA / analysis only
    tr = pd.concat([(df["high"] - df["low"]),
                    (df["high"] - df["close"].shift()).abs(),
                    (df["low"] - df["close"].shift()).abs()], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()

    N = len(df)
    if N < args.bars + 200:
        raise SystemExit(f"Not enough history: {N} rows, need >= {args.bars + 200}")
    start = N - args.bars

    # Full-history labels for hindsight accuracy analysis (NOT used for training)
    print("Computing hindsight labels for analysis ...")
    hindsight = label_window(df)

    feat_matrix = df[feats].replace([np.inf, -np.inf], np.nan)

    model = None
    rows = []
    t0 = time.time()
    n_retrains = 0

    for step, i in enumerate(range(start, N)):
        # ── retrain on data STRICTLY before bar i (leak-free) ──
        if model is None or step % max(1, args.retrain_every) == 0:
            win = df.iloc[:i]
            if args.train_window > 0:
                win = win.iloc[-args.train_window:]
            y = label_window(win)
            X = feat_matrix.iloc[win.index]
            mask = X.notna().all(axis=1)
            Xv = X[mask].values
            yv = y[mask.values]
            if len(Xv) >= 100 and len(np.unique(yv)) >= 2:
                model = make_model(args.model)
                model.fit(Xv, yv)
                n_retrains += 1

        # ── predict bar i with the current (past-only) model ──
        xrow = feat_matrix.iloc[i]
        if model is None or xrow.isna().any():
            pred = 0
        else:
            pred = int(model.predict(xrow.values.reshape(1, -1))[0])

        ts = pd.to_datetime(df.iloc[i]["time"], unit="s")
        rows.append({
            "time": ts.strftime("%Y.%m.%d %H:%M:%S"),
            "pred": pred,
            "pred_name": {0: "HOLD", 1: "BUY", 2: "SELL"}.get(pred, "?"),
            "open": round(float(df.iloc[i]["open"]), 5),
            "high": round(float(df.iloc[i]["high"]), 5),
            "low": round(float(df.iloc[i]["low"]), 5),
            "close": round(float(df.iloc[i]["close"]), 5),
            "atr": round(float(df.iloc[i]["atr"]) if pd.notna(df.iloc[i]["atr"]) else 0.0, 5),
            "label_hindsight": int(hindsight[i]),
        })

        if (step + 1) % 25 == 0 or step == args.bars - 1:
            el = time.time() - t0
            rate = (step + 1) / el if el > 0 else 0
            eta = (args.bars - step - 1) / rate if rate > 0 else 0
            print(f"  bar {step + 1}/{args.bars}  retrains={n_retrains}  "
                  f"{rate:.1f} bar/s  ETA {eta/60:.1f} min")
            
    out = pd.DataFrame(rows)
    out_path = args.out or default_out_path()
    out.to_csv(out_path, index=False)

    # quick agreement stat vs hindsight labels
    agree = float((out["pred"].values == out["label_hindsight"].values).mean() * 100)
    n_buy = int((out["pred"] == 1).sum())
    n_sell = int((out["pred"] == 2).sum())
    n_hold = int((out["pred"] == 0).sum())

    print("-" * 64)
    print(f"Wrote {len(out)} rows -> {out_path}")
    print(f"Predictions  BUY={n_buy}  SELL={n_sell}  HOLD={n_hold}")
    print(f"Agreement with hindsight labels: {agree:.1f}%")
    print(f"Total time: {(time.time() - t0)/60:.1f} min  ({n_retrains} retrains)")
    
    mt5.shutdown()


if __name__ == "__main__":
    main()
