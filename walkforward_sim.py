"""
walkforward_sim.py
==================================================================
Walk-forward simulation "server" for the ML Prediction system.

WHY THIS EXISTS
---------------
The MQL5 simulator in SERVER mode queries the *live* Flask server, whose
model is trained on ALL available history — including bars that come AFTER
the ones being back-tested. That is look-ahead bias: the model already
"knows the future" of every bar it scores, so the back-test looks far better
than live trading ever can. This is the single most likely reason the
simulation looks profitable while the real account loses.

This script removes that bias. It performs a proper WALK-FORWARD:
for each historical bar i it trains ONLY on bars [0 .. i-1] and predicts
bar i (retrain-per-prediction, honouring the "always current" idea). It
also uses a DIFFERENT model and a STATIONARY (return/ratio based) feature
set, because the production model is trained on raw price levels — a model
that learned gold at 2,000 cannot generalise to 4,000, another likely loss
source.

OUTPUTS
-------
1. sim_signals.csv  (epoch_time,signal)  → written into the MT5 Files folder
   so the MQL5 simulator can replay the SAME trade-management logic against
   these realistic, look-ahead-free signals (SignalSource = SIG_FILE).
2. walkforward_report.html → model diagnostics: walk-forward vs in-sample
   accuracy gap, confusion matrix, per-class precision/recall, feature
   importance, rolling accuracy, directional-error analysis. This is where
   you diagnose *where the losses come from*.

USAGE
-----
    python walkforward_sim.py --symbol XAUUSDc --model hgb --features engineered \
        --min-train 3000 --max-test 1000 --retrain-every 1

Increase --retrain-every (e.g. 10) to speed the run up massively at a small
fidelity cost. Requires the same environment as the live server
(MetaTrader5, scikit-learn, pandas, numpy) and MT5 running.
"""

import argparse
import os
import shutil
import time
from datetime import datetime

import numpy as np
import pandas as pd

import MetaTrader5 as mt5
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              HistGradientBoostingClassifier, ExtraTreesClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (confusion_matrix, accuracy_score,
                             precision_recall_fscore_support, f1_score)

from extra_function import engineer_features, RobustPriceLabelerV3, fix_pivot_labels


# ──────────────────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────────────────
DEFAULT_FILES_DIR = (r"C:\Users\omage\AppData\Roaming\MetaQuotes\Terminal"
                     r"\E3E3B02889D32F38295D39BF94B6AD4A\MQL5\Files")

parser = argparse.ArgumentParser(description="Walk-forward simulation server / backtester")
parser.add_argument("--terminal", default=r"C:\Program Files\HFM MetaTrader 5\terminal64.exe")
parser.add_argument("--symbol", default="XAUUSDc")
parser.add_argument("--model", default="hgb", choices=["hgb", "rf", "et", "gb", "logit"],
                    help="different model to test (default hgb = HistGradientBoosting)")
parser.add_argument("--all", action="store_true",
                    help="walk-forward test EVERY model and rank them by simulated profit")
parser.add_argument("--features", default="engineered", choices=["engineered", "raw"],
                    help="engineered = stationary returns/ratios (recommended); raw = production's raw prices")
parser.add_argument("--min-train", type=int, default=3000, help="bars required before the first prediction")
parser.add_argument("--max-test", type=int, default=1000, help="number of most-recent bars to walk-forward test (0 = all)")
parser.add_argument("--retrain-every", type=int, default=1, help="retrain cadence in bars (1 = every prediction)")
# ── quick-backtest economics (consistent across models for fair ranking) ──
parser.add_argument("--be-pips", type=float, default=40.0)
parser.add_argument("--lock-pips", type=float, default=10.0)
parser.add_argument("--sl-atr", type=float, default=1.5)
parser.add_argument("--spread-pips", type=float, default=2.0)
parser.add_argument("--vol", type=float, default=0.01)
parser.add_argument("--no-sl", action="store_true", help="disable the hard stop in the quick backtest")
parser.add_argument("--files-dir", default=DEFAULT_FILES_DIR, help="MT5 Files folder for the signal CSV")
parser.add_argument("--signal-csv", default="sim_signals.csv")
parser.add_argument("--report", default="walkforward_report.html")
args = parser.parse_args()

RAW_FEATURES = [
    'open', 'high', 'low', 'close', 'volume',
    'open_4h', 'high_4h', 'low_4h', 'close_4h', 'volume_4h',
    'open_1d', 'high_1d', 'low_1d', 'close_1d', 'volume_1d',
    'hour', 'day', 'month', 'day_of_week',
]

LABEL_NAMES = {0: "Hold", 1: "Buy", 2: "Sell"}


# ──────────────────────────────────────────────────────────────────
#  Data loading (mirrors the production server's multi-TF merge)
# ──────────────────────────────────────────────────────────────────
def fetch_rates(symbol, timeframe, count=9_000_000):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
    if rates is None:
        raise ValueError(f"MT5 returned None for {symbol}. Error: {mt5.last_error()}")
    df = pd.DataFrame(rates)
    vol_col = next((c for c in df.columns if "volume" in c.lower()), None)
    df = df[["time", "open", "high", "low", "close", vol_col]].copy()
    df.rename(columns={vol_col: "volume"}, inplace=True)
    return df


def build_multi_tf(symbol):
    if not mt5.initialize(args.terminal):
        raise RuntimeError(f"MT5 initialize() failed: {mt5.last_error()}")

    df_1h = fetch_rates(symbol, mt5.TIMEFRAME_H1)
    df_4h = fetch_rates(symbol, mt5.TIMEFRAME_H4)
    df_1d = fetch_rates(symbol, mt5.TIMEFRAME_D1)

    def tag_and_shift(df, suffix):
        df = df.sort_values("time").rename(
            columns={c: f"{c}_{suffix}" for c in df.columns if c != "time"})
        val_cols = [c for c in df.columns if c != "time"]
        df[val_cols] = df[val_cols].shift(1)      # only CLOSED higher-TF bars
        return df

    df_4h = tag_and_shift(df_4h, "4h")
    df_1d = tag_and_shift(df_1d, "1d")
    df_1h = df_1h.sort_values("time")

    merged = pd.merge_asof(df_1h, df_4h, on="time", direction="backward")
    merged = pd.merge_asof(merged, df_1d, on="time", direction="backward")
    return merged


# ──────────────────────────────────────────────────────────────────
#  Feature engineering
# ──────────────────────────────────────────────────────────────────
def add_calendar(df):
    t = pd.to_datetime(df["time"], unit="s", errors="coerce")
    df["month"] = t.dt.month
    df["day"] = t.dt.day
    df["hour"] = t.dt.hour
    df["day_of_week"] = t.dt.dayofweek
    return df


def add_engineered_features(df):
    """Stationary features that generalise across price levels."""
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

    feats = ["ret1", "ret3", "ret6", "ret12", "ret24", "rng", "body", "upwick", "lowick",
             "c_sma20", "c_sma50", "sma20_50", "vol20", "atr_pct", "rsi", "vchg",
             "c_vs_4h", "c_vs_1d", "h4_ret", "d1_ret",
             "hour_sin", "hour_cos", "dow_sin", "dow_cos"]
    return df, feats


# ──────────────────────────────────────────────────────────────────
#  Model factory (a DIFFERENT model from production's RandomForest)
# ──────────────────────────────────────────────────────────────────
def make_model(name):
    name = name.lower()
    if name == "rf":
        return RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                      random_state=42, n_jobs=-1)
    if name == "et":
        return ExtraTreesClassifier(n_estimators=250, class_weight="balanced",
                                    random_state=42, n_jobs=-1)
    if name == "gb":
        return GradientBoostingClassifier(random_state=42)
    if name == "logit":
        return make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=1000, class_weight="balanced"))
    # default: fast histogram gradient boosting
    return HistGradientBoostingClassifier(random_state=42, max_iter=300, learning_rate=0.08)


# ──────────────────────────────────────────────────────────────────
#  Walk-forward loop (retrain on [0:i], predict i — NO look-ahead)
# ──────────────────────────────────────────────────────────────────
def walk_forward(X, y, min_train, max_test, retrain_every, model_name):
    n = len(X)
    start = min_train
    if max_test > 0:
        start = max(min_train, n - max_test)
    preds = np.full(n, -1, dtype=int)

    model = None
    t0 = time.time()
    for i in range(start, n):
        if model is None or (i - start) % max(1, retrain_every) == 0:
            model = make_model(model_name)
            model.fit(X[:i], y[:i])
        preds[i] = int(model.predict(X[i:i + 1])[0])
        if (i - start) % 100 == 0:
            done = i - start + 1
            total = n - start
            print(f"  walk-forward {done}/{total}  ({100*done/total:5.1f}%)  "
                  f"elapsed {time.time()-t0:5.0f}s", end="\r")
    print()
    # also fit a final model on everything for feature importance
    final = make_model(model_name)
    final.fit(X[:n], y[:n])
    return preds, start, final


# ──────────────────────────────────────────────────────────────────
#  HTML report
# ──────────────────────────────────────────────────────────────────
def cell(v, cls=""):
    return f"<td class='{cls}'>{v}</td>"


def build_report(df, y, preds, start, final_model, feat_names, insample_acc, path):
    mask = preds >= 0
    yt = y[mask]
    yp = preds[mask]
    n_test = len(yt)

    wf_acc = accuracy_score(yt, yp) if n_test else 0.0
    wf_f1 = f1_score(yt, yp, average="macro", labels=[0, 1, 2], zero_division=0) if n_test else 0.0
    cm = confusion_matrix(yt, yp, labels=[0, 1, 2])
    prec, rec, f1c, sup = precision_recall_fscore_support(
        yt, yp, labels=[0, 1, 2], zero_division=0)

    # directional (costly) errors: predicted Buy while true Sell, and vice-versa
    buy_when_sell = int(np.sum((yp == 1) & (yt == 2)))
    sell_when_buy = int(np.sum((yp == 2) & (yt == 1)))
    directional_err = buy_when_sell + sell_when_buy
    dir_signals = int(np.sum((yp == 1) | (yp == 2)))

    # rolling accuracy (blocks of 50)
    blk = 50
    roll = []
    for k in range(0, n_test, blk):
        seg_t = yt[k:k + blk]
        seg_p = yp[k:k + blk]
        if len(seg_t):
            roll.append(accuracy_score(seg_t, seg_p))

    # label distribution (full set)
    full_counts = pd.Series(y).value_counts().sort_index()

    # ── build SVG rolling-accuracy sparkline ──
    def roll_svg(vals, w=860, h=140):
        if len(vals) < 2:
            return "<p>Not enough test blocks.</p>"
        pl, pt, pb = 40, 12, 22
        pw, ph = w - pl - 12, h - pt - pb
        pts = ""
        for i, val in enumerate(vals):
            x = pl + i / (len(vals) - 1) * pw
            yv = pt + (1 - val) * ph
            pts += f"{x:.1f},{yv:.1f} "
        svg = f"<svg width='{w}' height='{h}' viewBox='0 0 {w} {h}'>"
        svg += f"<rect width='{w}' height='{h}' fill='#0d1018'/>"
        for g in range(0, 5):
            val = g / 4.0
            yy = pt + (1 - val) * ph
            svg += f"<line x1='{pl}' y1='{yy:.1f}' x2='{w-12}' y2='{yy:.1f}' stroke='#232838'/>"
            svg += f"<text x='4' y='{yy+3:.1f}' fill='#8892a6' font-size='10'>{val*100:.0f}%</text>"
        # 33% random-chance baseline (3 classes)
        yb = pt + (1 - 1/3) * ph
        svg += f"<line x1='{pl}' y1='{yb:.1f}' x2='{w-12}' y2='{yb:.1f}' stroke='#5a6' stroke-dasharray='4 3'/>"
        svg += f"<polyline fill='none' stroke='#3d7bff' stroke-width='1.4' points='{pts}'/>"
        svg += "<text x='46' y='11' fill='#5a6' font-size='10'>random 33%</text></svg>"
        return svg

    # ── feature importance ──
    fi_html = "<p>Model does not expose feature importances.</p>"
    importances = getattr(final_model, "feature_importances_", None)
    if importances is not None and len(importances) == len(feat_names):
        order = np.argsort(importances)[::-1][:20]
        mx = float(np.max(importances)) or 1.0
        fi_html = "<table><tr><th class='l'>Feature</th><th>Importance</th><th></th></tr>"
        for idx in order:
            barw = int(200 * importances[idx] / mx)
            fi_html += (f"<tr><td class='l'>{feat_names[idx]}</td>"
                        f"<td>{importances[idx]:.4f}</td>"
                        f"<td class='l'><div style='background:#3d7bff;height:10px;width:{barw}px'></div></td></tr>")
        fi_html += "</table>"

    # ── confusion matrix table ──
    cm_html = "<table><tr><th></th><th>pred Hold</th><th>pred Buy</th><th>pred Sell</th><th>recall</th></tr>"
    row_lbl = ["true Hold", "true Buy", "true Sell"]
    for r in range(3):
        cm_html += f"<tr><th class='l'>{row_lbl[r]}</th>"
        rowsum = cm[r].sum()
        for cc in range(3):
            good = (r == cc)
            cm_html += f"<td class='{'pos' if good else 'neg'}'>{cm[r][cc]}</td>"
        cm_html += f"<td>{(cm[r][r]/rowsum*100 if rowsum else 0):.1f}%</td></tr>"
    cm_html += "<tr><th class='l'>precision</th>"
    for cc in range(3):
        colsum = cm[:, cc].sum()
        cm_html += f"<td>{(cm[cc][cc]/colsum*100 if colsum else 0):.1f}%</td>"
    cm_html += "<td></td></tr></table>"

    gap = insample_acc - wf_acc

    h = []
    h.append("<!DOCTYPE html><html><head><meta charset='utf-8'><title>Walk-Forward Model Report</title><style>")
    h.append("body{background:#0b0e16;color:#c9d1e0;font-family:Segoe UI,Arial,sans-serif;margin:0;padding:24px}")
    h.append("h1{font-size:20px;color:#fff;margin:0 0 4px}h2{font-size:15px;color:#9fb0ff;border-bottom:1px solid #232838;padding-bottom:6px;margin-top:28px}")
    h.append(".sub{color:#8892a6;font-size:12px;margin-bottom:14px}")
    h.append(".cards{display:flex;flex-wrap:wrap;gap:12px}.card{background:#141826;border:1px solid #232838;border-radius:8px;padding:12px 16px;min-width:150px}")
    h.append(".card .l{font-size:11px;color:#8892a6;text-transform:uppercase;letter-spacing:.5px}.card .v{font-size:20px;font-weight:600;margin-top:4px}")
    h.append(".pos{color:#25c281}.neg{color:#ff5c6c}.neu{color:#e8c85a}")
    h.append("table{border-collapse:collapse;font-size:12px;margin-top:8px}th,td{padding:6px 10px;border-bottom:1px solid #1c2130;text-align:right}th{color:#8892a6}.l{text-align:left}")
    h.append(".warn{background:#2a1416;border:1px solid #ff5c6c;color:#ffb3b8;border-radius:8px;padding:10px 14px;margin-top:12px;font-size:13px}")
    h.append("</style></head><body>")

    h.append("<h1>Walk-Forward Model Report</h1>")
    h.append(f"<div class='sub'>Symbol <b>{args.symbol}</b> &nbsp;|&nbsp; Model <b>{args.model}</b> &nbsp;|&nbsp; Features <b>{args.features}</b> "
             f"&nbsp;|&nbsp; Train ≥ <b>{args.min_train}</b> bars &nbsp;|&nbsp; Retrain every <b>{args.retrain_every}</b> bar(s) "
             f"&nbsp;|&nbsp; Test bars <b>{n_test}</b> &nbsp;|&nbsp; Generated {datetime.now():%Y-%m-%d %H:%M}</div>")

    h.append("<div class='cards'>")
    h.append(f"<div class='card'><div class='l'>Walk-Forward Accuracy</div><div class='v {'pos' if wf_acc>0.5 else 'neu' if wf_acc>1/3 else 'neg'}'>{wf_acc*100:.1f}%</div></div>")
    h.append(f"<div class='card'><div class='l'>In-Sample Accuracy</div><div class='v'>{insample_acc*100:.1f}%</div></div>")
    h.append(f"<div class='card'><div class='l'>Overfit / Look-ahead Gap</div><div class='v {'neg' if gap>0.15 else 'neu' if gap>0.05 else 'pos'}'>{gap*100:.1f}%</div></div>")
    h.append(f"<div class='card'><div class='l'>Macro F1 (walk-fwd)</div><div class='v'>{wf_f1:.3f}</div></div>")
    h.append(f"<div class='card'><div class='l'>Directional Errors</div><div class='v neg'>{directional_err}</div></div>")
    h.append(f"<div class='card'><div class='l'>of Directional Signals</div><div class='v'>{dir_signals}</div></div>")
    h.append("</div>")

    if gap > 0.15:
        h.append(f"<div class='warn'>⚠ Large gap ({gap*100:.1f}%) between in-sample and walk-forward accuracy — the model is heavily OVERFIT / relies on look-ahead. "
                 f"A live-realistic (walk-forward) accuracy of {wf_acc*100:.1f}% vs random 33% is the honest number your trade results will follow.</div>")
    if directional_err > 0 and dir_signals > 0:
        h.append(f"<div class='warn'>⚠ {directional_err} of {dir_signals} directional signals were exactly WRONG (predicted Buy on a true Sell leg or vice-versa) "
                 f"= {100*directional_err/dir_signals:.1f}%. These are the trades that produce the biggest losses (entering against the real move).</div>")

    h.append("<h2>Walk-Forward Accuracy Over Time (blocks of 50 bars)</h2>")
    h.append(roll_svg(roll))

    h.append("<h2>Confusion Matrix (walk-forward, no look-ahead)</h2>")
    h.append(cm_html)

    h.append("<h2>Per-Class Quality</h2>")
    h.append("<table><tr><th class='l'>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr>")
    for k in range(3):
        h.append(f"<tr><td class='l'>{LABEL_NAMES[k]}</td><td>{prec[k]*100:.1f}%</td>"
                 f"<td>{rec[k]*100:.1f}%</td><td>{f1c[k]:.3f}</td><td>{int(sup[k])}</td></tr>")
    h.append("</table>")

    h.append("<h2>Label Distribution (full dataset)</h2>")
    h.append("<table><tr><th class='l'>Class</th><th>Count</th><th>Share</th></tr>")
    tot = int(full_counts.sum())
    for k in [0, 1, 2]:
        cnt = int(full_counts.get(k, 0))
        h.append(f"<tr><td class='l'>{LABEL_NAMES[k]}</td><td>{cnt}</td><td>{100*cnt/tot:.1f}%</td></tr>")
    h.append("</table>")

    h.append("<h2>Top Feature Importances</h2>")
    h.append(fi_html)

    h.append("<p class='sub' style='margin-top:24px'>Interpretation: the walk-forward accuracy is the honest estimate of how the model performs on unseen bars. "
             "If it is near 33% (random for 3 classes) or the overfit gap is large, no amount of trade-management tuning will make it profitable live — the signal itself is the problem. "
             "Directional errors are the specific predictions that create losing trades. Feed the exported <b>sim_signals.csv</b> into the MQL5 simulator (SignalSource = SIG_FILE) to see the realistic trade result.</p>")
    h.append("</body></html>")

    with open(path, "w", encoding="utf-8") as f:
        f.write("".join(h))


# ──────────────────────────────────────────────────────────────────
#  Quick consistent backtest (for ranking models by profit)
# ──────────────────────────────────────────────────────────────────
def get_symbol_cfg(symbol):
    si = mt5.symbol_info(symbol)
    point = si.point if si else 0.01
    contract = si.trade_contract_size if si else 100.0
    return dict(point=point, pip=point * 10.0, contract=contract)


def quick_backtest(df, preds, start, cfg):
    """A single-position, flip-on-opposite backtest with trailing-breakeven +
    ATR stop + spread cost. Identical logic for every model, so the ranking is
    fair. It is a simplification of the full MQL5 engine (no lot accumulation /
    re-entry), used only to compare signal quality by profit."""
    o = df["open"].values; h = df["high"].values
    l = df["low"].values; c = df["close"].values
    atr = df["atr_abs"].values
    n = len(df)

    pip = cfg["pip"]; contract = cfg["contract"]; vol = cfg["vol"]
    be_thr = cfg["be_pips"] * pip
    trail = max(pip, (cfg["be_pips"] - cfg["lock_pips"]) * pip)
    cost = cfg["spread_pips"] * pip * contract * vol
    use_sl = cfg["use_sl"]; sl_atr = cfg["sl_atr"]

    pos = 0; entry = 0.0; sl = 0.0; be = False
    realized = 0.0; wins = 0; losses = 0; gp = 0.0; gl = 0.0; trades = 0
    peak = 0.0; maxdd = 0.0

    def book(px):
        nonlocal realized, wins, losses, gp, gl, trades
        pnl = (px - entry) * contract * vol * (1 if pos == 1 else -1) - cost
        realized += pnl
        if pnl >= 0: wins += 1; gp += pnl
        else:        losses += 1; gl += pnl
        trades += 1
        return pnl

    for i in range(start, n):
        a = atr[i]
        # 1) manage open position over bar i
        if pos != 0 and a > 0:
            if pos == 1:
                if not be and (h[i] - entry) >= be_thr:
                    be = True; sl = h[i] - trail
                elif be:
                    sl = max(sl, h[i] - trail)
                if sl > 0 and l[i] <= sl:
                    book(sl); pos = 0; be = False; sl = 0
            else:
                if not be and (entry - l[i]) >= be_thr:
                    be = True; sl = l[i] + trail
                elif be:
                    sl = sl and min(sl, l[i] + trail) or (l[i] + trail)
                if sl > 0 and h[i] >= sl:
                    book(sl); pos = 0; be = False; sl = 0
        # 2) apply signal at close[i]
        sig = preds[i]
        if sig == 1:
            if pos == -1: book(c[i]); pos = 0; be = False; sl = 0
            if pos == 0:
                pos = 1; entry = c[i]; be = False
                sl = (entry - sl_atr * a) if (use_sl and a > 0) else 0
        elif sig == 2:
            if pos == 1: book(c[i]); pos = 0; be = False; sl = 0
            if pos == 0:
                pos = -1; entry = c[i]; be = False
                sl = (entry + sl_atr * a) if (use_sl and a > 0) else 0
        elif sig == 0 and pos != 0:
            fl = (c[i] - entry) * contract * vol * (1 if pos == 1 else -1)
            if fl > 0:               # hold-basket style: bank a green book
                book(c[i]); pos = 0; be = False; sl = 0
        # 3) equity / drawdown
        fl = (c[i] - entry) * contract * vol * (1 if pos == 1 else -1) if pos != 0 else 0
        eq = realized + fl
        peak = max(peak, eq); maxdd = max(maxdd, peak - eq)

    if pos != 0:
        book(c[n - 1])

    pf = gp / abs(gl) if gl < 0 else (999.0 if gp > 0 else 0.0)
    win = 100.0 * wins / trades if trades else 0.0
    return dict(net=realized, pf=pf, win=win, maxdd=maxdd, trades=trades,
                gp=gp, gl=gl, wins=wins, losses=losses)


def _sparkline(vals, w=860, h=120):
    if len(vals) < 2:
        return ""
    pl, pt, pb = 40, 10, 18
    pw, ph = w - pl - 12, h - pt - pb
    pts = ""
    for i, v in enumerate(vals):
        x = pl + i / (len(vals) - 1) * pw
        y = pt + (1 - v) * ph
        pts += f"{x:.1f},{y:.1f} "
    s = f"<svg width='{w}' height='{h}'><rect width='{w}' height='{h}' fill='#0d1018'/>"
    yb = pt + (1 - 1 / 3) * ph
    s += f"<line x1='{pl}' y1='{yb:.1f}' x2='{w-12}' y2='{yb:.1f}' stroke='#5a6' stroke-dasharray='4 3'/>"
    s += f"<polyline fill='none' stroke='#3d7bff' stroke-width='1.3' points='{pts}'/></svg>"
    return s


# ──────────────────────────────────────────────────────────────────
#  Multi-model comparison report
# ──────────────────────────────────────────────────────────────────
def build_comparison_report(results, best_idx, cfg, feat_names, path):
    ranked = sorted(range(len(results)), key=lambda k: results[k]["bt"]["net"], reverse=True)

    h = ["<!DOCTYPE html><html><head><meta charset='utf-8'><title>Model Comparison</title><style>"]
    h.append("body{background:#0b0e16;color:#c9d1e0;font-family:Segoe UI,Arial,sans-serif;margin:0;padding:24px}")
    h.append("h1{font-size:20px;color:#fff;margin:0 0 4px}h2{font-size:15px;color:#9fb0ff;border-bottom:1px solid #232838;padding-bottom:6px;margin-top:26px}")
    h.append(".sub{color:#8892a6;font-size:12px;margin-bottom:14px}")
    h.append("table{border-collapse:collapse;font-size:12px;margin-top:8px}th,td{padding:7px 11px;border-bottom:1px solid #1c2130;text-align:right}th{color:#8892a6;background:#10131f}.l{text-align:left}")
    h.append(".pos{color:#25c281}.neg{color:#ff5c6c}.neu{color:#e8c85a}.best{background:#122019}")
    h.append("</style></head><body>")
    h.append("<h1>Walk-Forward Model Comparison</h1>")
    h.append(f"<div class='sub'>Symbol <b>{args.symbol}</b> &nbsp;|&nbsp; Features <b>{args.features}</b> "
             f"&nbsp;|&nbsp; Test bars <b>{results[0]['test_n']}</b> &nbsp;|&nbsp; Retrain every <b>{args.retrain_every}</b> bar(s) "
             f"&nbsp;|&nbsp; Backtest: BE {args.be_pips:.0f}/lock {args.lock_pips:.0f}, SL {'off' if args.no_sl else str(args.sl_atr)+'xATR'}, "
             f"spread {args.spread_pips:.1f}p, vol {args.vol} &nbsp;|&nbsp; {datetime.now():%Y-%m-%d %H:%M}</div>")

    h.append("<h2>Ranking (by simulated net profit)</h2>")
    h.append("<table><tr><th class='l'>#</th><th class='l'>Model</th><th>Net P/L</th><th>Profit Factor</th>"
             "<th>Win%</th><th>Trades</th><th>Max DD</th><th>WF Acc</th><th>Dir Err%</th><th>Overfit Gap</th></tr>")
    for rank, k in enumerate(ranked, 1):
        r = results[k]
        bt = r["bt"]
        derr = 100.0 * r["dir_err"] / r["dir_sig"] if r["dir_sig"] else 0.0
        cls = "best" if k == best_idx else ""
        h.append(f"<tr class='{cls}'><td class='l'>{rank}</td><td class='l'><b>{r['model']}</b></td>"
                 f"<td class='{'pos' if bt['net']>=0 else 'neg'}'>{bt['net']:.2f}</td>"
                 f"<td class='{'pos' if bt['pf']>=1 else 'neg'}'>{bt['pf']:.2f}</td>"
                 f"<td>{bt['win']:.1f}%</td><td>{bt['trades']}</td>"
                 f"<td class='neg'>{bt['maxdd']:.2f}</td>"
                 f"<td class='{'pos' if r['wf']>0.5 else 'neu' if r['wf']>1/3 else 'neg'}'>{r['wf']*100:.1f}%</td>"
                 f"<td class='neg'>{derr:.1f}%</td>"
                 f"<td class='{'neg' if r['gap']>0.15 else 'neu' if r['gap']>0.05 else 'pos'}'>{r['gap']*100:.1f}%</td></tr>")
    h.append("</table>")
    h.append(f"<div class='sub' style='margin-top:10px'>Best model: <b style='color:#25c281'>{results[best_idx]['model']}</b> "
             f"— its signals were copied to <b>sim_signals.csv</b>. Each model also has its own "
             f"<b>sim_signals_&lt;model&gt;.csv</b> in the MT5 Files folder for loading in the MQL5 simulator.</div>")

    # per-model classification snapshot
    h.append("<h2>Classification Quality (walk-forward)</h2>")
    h.append("<table><tr><th class='l'>Model</th><th>Buy Prec</th><th>Buy Rec</th><th>Sell Prec</th><th>Sell Rec</th>"
             "<th>Hold Rec</th><th>Macro F1</th></tr>")
    for k in ranked:
        r = results[k]
        p, rc, f1c = r["prec"], r["rec"], r["f1c"]
        macro = (f1c[0] + f1c[1] + f1c[2]) / 3.0
        h.append(f"<tr><td class='l'><b>{r['model']}</b></td>"
                 f"<td>{p[1]*100:.1f}%</td><td>{rc[1]*100:.1f}%</td>"
                 f"<td>{p[2]*100:.1f}%</td><td>{rc[2]*100:.1f}%</td>"
                 f"<td>{rc[0]*100:.1f}%</td><td>{macro:.3f}</td></tr>")
    h.append("</table>")

    # best model rolling accuracy + confusion matrix
    rb = results[best_idx]
    h.append(f"<h2>Best model ({rb['model']}) — walk-forward accuracy over time</h2>")
    h.append(_sparkline(rb["roll"]))
    cm = rb["cm"]
    h.append(f"<h2>Best model ({rb['model']}) — confusion matrix</h2>")
    h.append("<table><tr><th></th><th>pred Hold</th><th>pred Buy</th><th>pred Sell</th><th>recall</th></tr>")
    rl = ["true Hold", "true Buy", "true Sell"]
    for r in range(3):
        rs = cm[r].sum()
        h.append(f"<tr><th class='l'>{rl[r]}</th>")
        for cc in range(3):
            h.append(f"<td class='{'pos' if r==cc else 'neg'}'>{cm[r][cc]}</td>")
        h.append(f"<td>{(cm[r][r]/rs*100 if rs else 0):.1f}%</td></tr>")
    h.append("</table>")

    h.append("<p class='sub' style='margin-top:22px'>Net P/L here comes from a consistent simplified backtest (single position, flip on opposite, "
             "trailing-breakeven, ATR stop, spread cost) so models are ranked fairly by signal quality. For the exact strategy result, load the chosen "
             "model's CSV into the MQL5 simulator (SignalSource = SIG_FILE). A model can rank high on accuracy yet low on profit if it is right on small "
             "moves and wrong on big ones — always trust the profit ranking over raw accuracy.</p>")
    h.append("</body></html>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("".join(h))


# ──────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────
def main():
    print(f"MetaTrader5 {mt5.__version__} — building multi-timeframe dataset for {args.symbol} …")
    df = build_multi_tf(args.symbol)
    print(f"  merged rows: {len(df):,}")

    df = add_calendar(df)

    labeler = RobustPriceLabelerV3(atr_period=14, zigzag_atr_mult=0.5,
                                   hold_bars=1, min_streak=0.01, target_hold_pct=0.01)
    df = labeler.label(df)
    df = fix_pivot_labels(df)

    # absolute ATR for the backtest stop
    hh, ll, cc = df["high"], df["low"], df["close"]
    tr = pd.concat([(hh - ll), (hh - cc.shift()).abs(), (ll - cc.shift()).abs()], axis=1).max(axis=1)
    df["atr_abs"] = tr.rolling(14).mean()

    if args.features == "engineered":
        df, feat_names = add_engineered_features(df)
    else:
        feat_names = RAW_FEATURES

    keep = ["time", "label", "open", "high", "low", "close", "atr_abs"] + \
           [f for f in feat_names if f in df.columns]
    df = df[keep].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    print(f"  usable rows: {len(df):,}  features: {len(feat_names)}")

    X = df[feat_names].values.astype(float)
    y = df["label"].values.astype(int)
    if len(df) <= args.min_train + 50:
        raise SystemExit(f"Not enough data ({len(df)}) for min-train {args.min_train}.")

    cfg = get_symbol_cfg(args.symbol)
    cfg.update(dict(be_pips=args.be_pips, lock_pips=args.lock_pips, sl_atr=args.sl_atr,
                    spread_pips=args.spread_pips, vol=args.vol, use_sl=not args.no_sl))

    models = ["logit", "rf", "et", "hgb", "gb"] if args.all else [args.model]
    if args.all and args.retrain_every == 1:
        print("  NOTE: --all with retrain-every=1 is slow. Consider --retrain-every 10 for a quick first pass.")

    os.makedirs(args.files_dir, exist_ok=True)
    results = []

    for m in models:
        print(f"\n=== Walk-forward model: {m} ===")
        preds, start, final_model = walk_forward(
            X, y, args.min_train, args.max_test, args.retrain_every, m)

        ins = make_model(m); ins.fit(X, y)
        insample = accuracy_score(y, ins.predict(X))

        mask = preds >= 0
        yt, yp = y[mask], preds[mask]
        wf = accuracy_score(yt, yp) if len(yt) else 0.0
        cm = confusion_matrix(yt, yp, labels=[0, 1, 2])
        prec, rec, f1c, sup = precision_recall_fscore_support(
            yt, yp, labels=[0, 1, 2], zero_division=0)
        dir_err = int(np.sum((yp == 1) & (yt == 2)) + np.sum((yp == 2) & (yt == 1)))
        dir_sig = int(np.sum((yp == 1) | (yp == 2)))
        blk = 50
        roll = [accuracy_score(yt[k:k + blk], yp[k:k + blk])
                for k in range(0, len(yt), blk) if len(yt[k:k + blk])]

        bt = quick_backtest(df, preds, start, cfg)

        # export this model's signals
        csv_m = os.path.join(args.files_dir, f"sim_signals_{m}.csv")
        with open(csv_m, "w") as f:
            for i in range(start, len(df)):
                if preds[i] >= 0:
                    f.write(f"{int(df['time'].iat[i])},{int(preds[i])}\n")

        print(f"  {m}: net {bt['net']:.2f}  PF {bt['pf']:.2f}  win {bt['win']:.1f}%  "
              f"trades {bt['trades']}  WFacc {wf*100:.1f}%  gap {(insample-wf)*100:.1f}%")

        results.append(dict(model=m, wf=wf, insample=insample, gap=insample - wf,
                            cm=cm, prec=prec, rec=rec, f1c=f1c, sup=sup,
                            dir_err=dir_err, dir_sig=dir_sig, roll=roll, bt=bt,
                            test_n=int(mask.sum())))

    best_idx = max(range(len(results)), key=lambda k: results[k]["bt"]["net"])
    best_model = results[best_idx]["model"]
    shutil.copyfile(os.path.join(args.files_dir, f"sim_signals_{best_model}.csv"),
                    os.path.join(args.files_dir, args.signal_csv))

    report_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "model_comparison_report.html" if args.all else args.report)
    build_comparison_report(results, best_idx, cfg, feat_names, report_path)

    print("\n================ RANKING (by net profit) ================")
    for rank, k in enumerate(sorted(range(len(results)),
                                    key=lambda k: results[k]["bt"]["net"], reverse=True), 1):
        r = results[k]; b = r["bt"]
        star = "  <-- best" if k == best_idx else ""
        print(f" {rank}. {r['model']:5s}  net {b['net']:9.2f}  PF {b['pf']:5.2f}  "
              f"win {b['win']:5.1f}%  WFacc {r['wf']*100:4.1f}%  gap {r['gap']*100:4.1f}%{star}")
    print("=========================================================")
    print(f" Best signals → sim_signals.csv  |  report → {report_path}")
    mt5.shutdown()



if __name__ == "__main__":
    main()
