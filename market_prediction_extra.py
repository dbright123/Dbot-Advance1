import argparse
import threading
import time
from datetime import datetime, timedelta

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              GradientBoostingClassifier, HistGradientBoostingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from flask_cors import CORS
import MetaTrader5 as mt5
from extra_function import RobustPriceLabelerV3, engineer_features, plot_signals, predict_and_plot_signals
from extra_function import RobustPriceLabelerV4, RobustPriceLabelerV5
from imblearn.under_sampling import RandomUnderSampler

from extra_function import fix_pivot_labels

# ─────────────────────────────────────────────
# Parse CLI arguments — no more input() prompts
# ─────────────────────────────────────────────

parser = argparse.ArgumentParser(description="ML Prediction Server for MetaTrader 5")
parser.add_argument("--terminal", default = "C:\\Program Files\\HFM MetaTrader 5\\terminal64.exe", required=False,  help="Full path to terminal64.exe  e.g. C:\\Program Files\\MetaTrader 5\\terminal64.exe")
parser.add_argument("--symbol", default = "XAUUSDc", required=False,  help="Trading symbol to fetch data for  e.g. XAUUSD")
parser.add_argument("--port",      type=int, default=5000, help="Port for the Flask server  (default: 5000)")
parser.add_argument("--retrain-interval", type=int, default=(60), help="Minutes between automatic retrains  (default: 1 day)")
parser.add_argument("--model", default="et", choices=["et", "rf", "hgb", "gb", "logit"],
                    help="classifier to train (default et = ExtraTrees, best profit in walk-forward comparison)")
parser.add_argument("--features", default="raw", choices=["engineered", "raw"],
                    help="engineered = stationary returns/ratios computed server-side (recommended); raw = legacy raw-price features from the request")
args = parser.parse_args()   

# ─────────────────────────────────────────────
# App & globals
# ─────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

print("MetaTrader5 package author: ", mt5.__author__)
print("MetaTrader5 package version: ", mt5.__version__)

FEATURE_NAMES = [
    # ── H1 (base timeframe) ──────────────────────────────────────
    'open', 'high', 'low', 'close', 'volume',
    # ── H4 timeframe ─────────────────────────────────────────────
    'open_4h', 'high_4h', 'low_4h', 'close_4h', 'volume_4h',
    # ── D1 timeframe ─────────────────────────────────────────────
    'open_1d', 'high_1d', 'low_1d', 'close_1d', 'volume_1d',
    # ── Calendar features ─────────────────────────────────────────
    'hour', 'day', 'month', 'day_of_week',
]

# ── Shared training state (all access protected by a lock) ──
_model_lock          = threading.Lock()
model                = None          # active model served to /predict
active_features      = list(FEATURE_NAMES)  # feature columns the active model expects
training_in_progress = False
last_trained_at      = None          # datetime of last successful train
next_train_at        = None          # datetime of next scheduled train
training_error       = None          # last error message, if any


import MetaTrader5 as mt5
import pandas as pd

def fetch_rates(symbol, timeframe, count=9_000_000):
    """Pull rates from MT5 and return a tidy DataFrame."""
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
    
    if rates is None:
        raise ValueError(f"MT5 returned None for {symbol}. Error: {mt5.last_error()}")
    
    df = pd.DataFrame(rates)
    
    # ── Debug: show what columns MT5 actually gave us ──────────────────────
    print(f"Available columns: {list(df.columns)}")
    
    # ── Grab volume column whatever it is called ────────────────────────────
    vol_col = next((c for c in df.columns if "volume" in c.lower()), None)
    if vol_col is None:
        raise KeyError(f"No volume column found among: {list(df.columns)}")
    
    df = df[["time", "open", "high", "low", "close", vol_col]].copy()
    df.rename(columns={vol_col: "volume"}, inplace=True)
    #df["time"] = pd.to_datetime(df["time"], unit="s")
    return df.set_index("time")


def build_multi_tf(symbol, bars=None):
    if not mt5.initialize(args.terminal):
        raise RuntimeError(f"MT5 initialize() failed: {mt5.last_error()}")

    if bars:
        n1, n4, n1d = bars, max(80, bars // 4 + 20), max(40, bars // 24 + 20)
    else:
        n1 = n4 = n1d = 9_000_000

    df_1h = fetch_rates(symbol, mt5.TIMEFRAME_H1, n1).reset_index()
    df_4h = fetch_rates(symbol, mt5.TIMEFRAME_H4, n4).reset_index()
    df_1d = fetch_rates(symbol, mt5.TIMEFRAME_D1, n1d).reset_index()

    def tag_and_shift(df, suffix):
        df = df.sort_values("time").rename(
            columns={c: f"{c}_{suffix}" for c in df.columns if c != "time"})
        val_cols = [c for c in df.columns if c != "time"]
        df[val_cols] = df[val_cols].shift(1)      # <-- only use CLOSED bars
        return df

    df_4h = tag_and_shift(df_4h, "4h")
    df_1d = tag_and_shift(df_1d, "1d")
    df_1h = df_1h.sort_values("time")

    merged = pd.merge_asof(df_1h, df_4h, on="time", direction="backward")
    merged = pd.merge_asof(merged, df_1d, on="time", direction="backward")
    return merged


# ─────────────────────────────────────────────
# Engineered (stationary) feature pipeline — mirrors walkforward_sim.py
# ─────────────────────────────────────────────
ENG_FEATURES = [
    "ret1", "ret3", "ret6", "ret12", "ret24", "rng", "body", "upwick", "lowick",
    "c_sma20", "c_sma50", "sma20_50", "vol20", "atr_pct", "rsi", "vchg",
    "c_vs_4h", "c_vs_1d", "h4_ret", "d1_ret",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
]


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
    return df, list(ENG_FEATURES)


def make_model(name):
    """Classifier factory. Default 'et' (ExtraTrees) — best profit/PF in the
    walk-forward model comparison."""
    name = (name or "et").lower()
    if name == "rf":
        return RandomForestClassifier(n_estimators=350, class_weight="balanced",
                                      random_state=42, n_jobs=-1)
    if name == "hgb":
        return HistGradientBoostingClassifier(random_state=42, max_iter=300, learning_rate=0.08)
    if name == "gb":
        return GradientBoostingClassifier(random_state=42)
    if name == "logit":
        return make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=1000, class_weight="balanced"))
    return ExtraTreesClassifier(n_estimators=400, class_weight="balanced",
                                random_state=42, n_jobs=-1)


def build_predict_row(symbol, feats):
    """Compute the engineered feature vector for the latest bar, server-side."""
    try:
        df = build_multi_tf(symbol, bars=400)
    except Exception as e:
        print(f"build_predict_row: fetch failed: {e}")
        return None
    df = add_calendar(df)
    df, _ = add_engineered_features(df)
    df = df[feats].replace([np.inf, -np.inf], np.nan).dropna()
    if len(df) == 0:
        return None
    return df.iloc[-1].values.astype(float)


# ─────────────────────────────────────────────
# Core training routine (runs in bg thread)
# ─────────────────────────────────────────────
def _run_training():
    """Fetch latest market data, label, train a fresh RF model, swap it in."""
    global model, training_in_progress, last_trained_at, training_error

    print(f"\n[{datetime.now():%H:%M:%S}] ── Starting training run ──")

    try:
        # ── Fetch & merge multi-timeframe data ──────────────────────────────
        df = build_multi_tf(args.symbol)
        print(f"  Multi-TF merge complete: {len(df):,} rows, {len(df.columns)} columns")

        # ── Calendar features ───────────────────────
        df = add_calendar(df)

        # ── Labelling ───────────────────────────────
        labeler = RobustPriceLabelerV3(
            atr_period      = 14,
            zigzag_atr_mult = 0.5,
            hold_bars       = 1,
            min_streak      = 0.01,
            target_hold_pct = 0.01 ,
        )
        df = labeler.label(df)
        df = fix_pivot_labels(df)

        # ── Feature set (engineered = stationary returns/ratios) ─────
        global active_features
        if args.features == 'engineered':
            df, feats = add_engineered_features(df)
        else:
            feats = list(FEATURE_NAMES)
        df = df[feats + ['label']].replace([np.inf, -np.inf], np.nan).dropna()
        print(f"  Features: {len(feats)} ({args.features})  usable rows: {len(df):,}  model: {args.model}")

        # ── Train / test split ──────────────────────
        X = df[feats].values
        y = df['label'].values
        del df

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.1, random_state=42, stratify=y
        )
        del X, y

        # ── Fit new model (default ExtraTrees — best in walk-forward test) ──
        new_model = make_model(args.model)
        new_model.fit(X_train, y_train)
        del X_train, y_train

        # ── Evaluate ────────────────────────────────
        y_pred   = new_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
        print(f"  Classification Report:\n{classification_report(y_test, y_pred)}")
        del X_test, y_test, y_pred

        # ── Swap in the new model atomically ────────
        with _model_lock:
            model           = new_model
            active_features = feats
            last_trained_at = datetime.now()
            training_error  = None

        print(f"[{datetime.now():%H:%M:%S}] ✓ Model swapped in successfully")

    except Exception as exc:
        import traceback
        err_msg = str(exc)
        print(f"[{datetime.now():%H:%M:%S}] ✗ Training failed: {err_msg}")
        traceback.print_exc()
        with _model_lock:
            training_error = err_msg

    finally:
        with _model_lock:
            training_in_progress = False


# ─────────────────────────────────────────────
# Background scheduler — fires every N minutes
# ─────────────────────────────────────────────
def _training_scheduler(interval_minutes: int):
    """Daemon thread: kick off a training run immediately, then every interval."""
    global training_in_progress, next_train_at

    while True:
        # Mark training as in-progress and compute next window
        with _model_lock:
            training_in_progress = True
            next_train_at = datetime.now() + timedelta(minutes=interval_minutes)

        # Run training synchronously inside this scheduler thread so we don't
        # pile up overlapping train jobs; the scheduler sleeps until done, then
        # waits for the remaining slot time before firing again.
        train_start = time.monotonic()
        _run_training()
        elapsed = time.monotonic() - train_start

        # Sleep for whatever is left of the interval
        sleep_secs = max(0, interval_minutes * 60 - elapsed)
        next_wakeup = datetime.now() + timedelta(seconds=sleep_secs)
        with _model_lock:
            next_train_at = next_wakeup

        print(f"  Next retrain scheduled at {next_wakeup:%Y-%m-%d %H:%M:%S} "
              f"(sleeping {sleep_secs/60:.1f} min)")
        time.sleep(sleep_secs)


# ─────────────────────────────────────────────
# MT5 initialisation (called once at startup)
# ─────────────────────────────────────────────
def init_mt5():
    ok = mt5.initialize(args.terminal)
    if not ok:
        print("MT5 initialisation failed — check path and internet connection.")
        mt5.shutdown()
        raise SystemExit(1)

    acct = mt5.account_info()
    term = mt5.terminal_info()
    print(f"  Account equity : {acct.equity}")
    print(f"  Terminal path  : {term.path}")
    print(f"  Symbols total  : {mt5.symbols_total()}")


# ─────────────────────────────────────────────
# Flask endpoints
# ─────────────────────────────────────────────
@app.route('/predict', methods=['GET'])
def predict():
    """Return 0 (Hold), 1 (Buy), or 2 (Sell).

    In 'engineered' mode the server ignores the request payload and computes the
    stationary feature vector for the latest bar itself (from live MT5 data),
    because those features need a window of history. In 'raw' mode it reads the
    legacy raw-price features from the request query string.
    """
    try:
        with _model_lock:
            current_model = model   # grab a local ref while holding lock
            feats         = active_features

        if current_model is None:
            # First train still running — stay neutral
            print("Prediction requested but model not ready yet — returning Hold (0)")
            return "0", 200

        if args.features == 'engineered':
            row = build_predict_row(args.symbol, feats)
            if row is None:
                print("Predict: could not build engineered feature row — Hold (0)")
                return "0", 200
            arr = row.reshape(1, -1)
        else:
            features = []
            for name in feats:
                raw = request.args.get(name)
                if raw is None:
                    features.append(0.0)
                    continue
                try:
                    features.append(float(raw))
                except ValueError:
                    print(f"Invalid value for feature '{name}' — returning Hold (0)")
                    return "0", 200
            arr = np.array(features).reshape(1, -1)

        prediction = current_model.predict(arr)[0]
        label_map  = {0: "Hold", 1: "Buy", 2: "Sell"}
        print(f"Prediction: {int(prediction)} ({label_map.get(int(prediction), '?')})  [{args.features}/{args.model}]")
        return str(int(prediction)), 200

    except Exception as e:
        print(f"Error in /predict — returning Hold (0): {e}")
        return "0", 200


@app.route('/training/status', methods=['GET'])
def training_status():
    """
    Returns current training state and schedule.

    Example curl:
        curl http://localhost:5000/training/status
    """
    with _model_lock:
        in_progress  = training_in_progress
        last_at      = last_trained_at
        next_at      = next_train_at
        model_ready  = model is not None
        last_err     = training_error

    now = datetime.now()

    # Time remaining until next retrain (only meaningful when not training)
    if next_at and not in_progress:
        remaining_secs = max(0, (next_at - now).total_seconds())
        remaining_str  = str(timedelta(seconds=int(remaining_secs)))
    else:
        remaining_secs = 0
        remaining_str  = "N/A — training in progress"

    return jsonify({
        "training_in_progress" : in_progress,
        "model_ready"          : model_ready,
        "last_trained_at"      : last_at.strftime("%Y-%m-%d %H:%M:%S") if last_at else None,
        "next_train_at"        : next_at.strftime("%Y-%m-%d %H:%M:%S") if next_at else None,
        "time_until_retrain"   : remaining_str,
        "retrain_interval_mins": args.retrain_interval,
        "last_error"           : last_err,
    }), 200


@app.route('/health', methods=['GET'])
def health():
    with _model_lock:
        ready = model is not None
    return jsonify({"status": "healthy", "model_loaded": ready}), 200


@app.route('/model', methods=['GET'])
def model_info():
    """Active model / feature configuration — lets the EA HUD show what is
    actually serving predictions."""
    with _model_lock:
        ready = model is not None
        feats = list(active_features)
        last_at = last_trained_at
    return jsonify({
        "model"        : args.model,
        "features"     : args.features,
        "feature_count": len(feats),
        "model_ready"  : ready,
        "symbol"       : args.symbol,
        "last_trained_at": last_at.strftime("%Y-%m-%d %H:%M:%S") if last_at else None,
    }), 200


@app.route('/test', methods=['GET'])
def test():
    return "Server is running!", 200


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("  ML Prediction Server — starting up")
    print(f"  Symbol   : {args.symbol}")
    print(f"  Model    : {args.model}")
    print(f"  Features : {args.features}")
    print(f"  Terminal : {args.terminal}")
    print(f"  Port     : {args.port}")
    print(f"  Retrain  : every {args.retrain_interval} minutes")
    print("=" * 60)

    # 1. Connect to MT5 once
    init_mt5()

    # 2. Start the background training scheduler as a daemon thread
    scheduler = threading.Thread(
        target=_training_scheduler,
        args=(args.retrain_interval,),
        daemon=True,      # dies automatically when the main process exits
        name="TrainingScheduler"
    )
    scheduler.start()
    print(f"Background training scheduler started (interval: {args.retrain_interval} min)")

    # 3. Run Flask  (first model will be ready in ~5-6 min; /predict returns 503 until then)
    app.run(host='0.0.0.0', port=args.port, debug=False)


"""
Further information to needed
# ── 1. Basic server check ─────────────────────────────────────────────────
curl http://localhost:5000/test

# Expected: Server is running!


# ── 2. Health check ───────────────────────────────────────────────────────
curl http://localhost:5000/health

# Expected (model loaded):   {"model_loaded":true,"status":"healthy"}
# Expected (still training): {"model_loaded":false,"status":"healthy"}


# ── 3. Training status ────────────────────────────────────────────────────
curl http://localhost:5000/training/status

# While idle — expected:
# {
#   "training_in_progress": false,
#   "model_ready": true,
#   "last_trained_at": "2026-03-22 14:03:11",
#   "next_train_at":   "2026-03-22 14:33:11",
#   "time_until_retrain": "0:24:47",
#   "retrain_interval_mins": 30,
#   "last_error": null
# }

# While retraining — expected:
# {
#   "training_in_progress": true,
#   "model_ready": true,
#   "last_trained_at": "2026-03-22 14:03:11",
#   "next_train_at":   "2026-03-22 14:33:11",
#   "time_until_retrain": "N/A — training in progress",
#   "retrain_interval_mins": 30,
#   "last_error": null
# }


# ── 4. Predict — full 19-feature set (H1 + H4 + D1 multi-timeframe) ──────
# curl "http://localhost:5000/predict?open=3024.50&high=3027.80&low=3022.10&close=3026.40&volume=1823\
# &open_4h=3020.00&high_4h=3030.00&low_4h=3015.00&close_4h=3025.00&volume_4h=7200\
# &open_1d=3000.00&high_1d=3050.00&low_1d=2990.00&close_1d=3026.00&volume_1d=50000\
# &hour=14&day=22&month=3&day_of_week=6"
#
# Expected: 0  (Hold)
#        or 1  (Buy)
#        or 2  (Sell)


# ── 5. Predict — missing features (server defaults them to 0, still returns) 
curl "http://localhost:5000/predict?open=3024.50&close=3026.40"

# Expected: 0  (neutral — missing features logged on server side)


# ── 6. Predict — invalid feature value (bad data) ────────────────────────
curl "http://localhost:5000/predict?open=INVALID&close=3026.40"

# Expected: 0  (neutral — error caught, logged server side)


# ── 7. Predict — no features at all (total fallback) ─────────────────────
curl "http://localhost:5000/predict"

# Expected: 0  (neutral)






"""