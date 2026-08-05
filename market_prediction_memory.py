import argparse
import threading
import time
import gc
import os
import glob
import joblib
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
from extra_function import (RobustPriceLabelerV3, engineer_features, plot_signals, 
                            predict_and_plot_signals, RobustPriceLabelerV4, 
                            RobustPriceLabelerV5, fix_pivot_labels)
from imblearn.under_sampling import RandomUnderSampler

# ─────────────────────────────────────────────
# Parse CLI arguments
# ─────────────────────────────────────────────
parser = argparse.ArgumentParser(description="ML Prediction Server for MetaTrader 5")
parser.add_argument("--terminal", default="C:\\Program Files\\MetaTrader 5\\terminal64.exe", required=False, help="Full path to terminal64.exe")
parser.add_argument("--symbol", default="XAUUSD", required=False, help="Trading symbol to fetch data for")
parser.add_argument("--port", type=int, default=5000, help="Port for the Flask server (default: 5000)")
parser.add_argument("--retrain-interval", type=int, default=30, help="Minutes between automatic retrains")
parser.add_argument("--model", default="rf_et_combo", choices=["et", "rf", "hgb", "gb", "logit", "rf_et_combo"],
                    help="classifier to train (rf_et_combo requires both RF and ET to agree)")
parser.add_argument("--features", default="raw", choices=["engineered", "raw"],
                    help="engineered = stationary returns/ratios; raw = legacy raw-price features")
args = parser.parse_args()   

# ─────────────────────────────────────────────
# App & globals
# ─────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

print("MetaTrader5 package author: ", mt5.__author__)
print("MetaTrader5 package version: ", mt5.__version__)

FEATURE_NAMES = [
    'open', 'high', 'low', 'close', 'volume',
    'open_1h', 'high_1h', 'low_1h', 'close_1h', 'volume_1h',
    'open_4h', 'high_4h', 'low_4h', 'close_4h', 'volume_4h',
    'open_1d', 'high_1d', 'low_1d', 'close_1d', 'volume_1d',
    'hour','minute', 'day', 'month', 'day_of_week',
]

# ── Shared training state ──
_model_lock          = threading.Lock()
model                = None          
active_features      = list(FEATURE_NAMES)  
training_in_progress = False
last_trained_at      = None          
next_train_at        = None          
training_error       = None          

def fetch_rates(symbol, timeframe, count=9_000_000):
    """Pull rates from MT5 and return a tidy DataFrame."""
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
    
    if rates is None:
        raise ValueError(f"MT5 returned None for {symbol}. Error: {mt5.last_error()}")
    
    df = pd.DataFrame(rates)
    print(f"Available columns: {list(df.columns)}")
    
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
        df[val_cols] = df[val_cols].shift(1)      
        return df

    df_1h = tag_and_shift(df_1h, "1h")
    df_4h = tag_and_shift(df_4h, "4h")
    df_1d = tag_and_shift(df_1d, "1d")
    
    df_15m = df_15m.sort_values("time")
    merged = pd.merge_asof(df_15m, df_1h, on="time", direction="backward")
    merged = pd.merge_asof(merged, df_4h, on="time", direction="backward")
    merged = pd.merge_asof(merged, df_1d, on="time", direction="backward")
    return merged.reset_index(drop=True)

# ─────────────────────────────────────────────
# Engineered features & Model wrapper
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
    df["minute"] = t.dt.minute
    df["day_of_week"] = t.dt.dayofweek
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


class RFETComboModel:
    """Wrapper class combining Random Forest and Extra Trees. 
    Uses versioned joblib files for zero-downtime prediction thread isolation."""
    def __init__(self):
        self.version_id = None
        self.rf_path = None
        self.et_path = None

    def fit(self, X, y):
        # Create a unique version tag for this training run
        self.version_id = int(time.time())
        self.rf_path = f"rf_combo_{self.version_id}.joblib"
        self.et_path = f"et_combo_{self.version_id}.joblib"

        print(f"  [Combo v{self.version_id}] Initializing & Training Random Forest...")
        rf = RandomForestClassifier(n_estimators=350, class_weight="balanced",
                                    random_state=42, n_jobs=-1, verbose=1)
        rf.fit(X, y)
        print(f"  [Combo v{self.version_id}] Saving RF to disk...")
        joblib.dump(rf, self.rf_path)
        
        # Purge RF from RAM before starting ET
        del rf
        gc.collect()
        
        print(f"  [Combo v{self.version_id}] Initializing & Training Extra Trees...")
        et = ExtraTreesClassifier(n_estimators=400, class_weight="balanced",
                                  random_state=42, n_jobs=-1, verbose=1)
        et.fit(X, y)
        print(f"  [Combo v{self.version_id}] Saving ET to disk...")
        joblib.dump(et, self.et_path)
        
        # Purge ET from RAM
        del et
        gc.collect()

        return self

    def predict(self, X):
        if not self.rf_path or not os.path.exists(self.rf_path) or \
           not self.et_path or not os.path.exists(self.et_path):
            raise FileNotFoundError("Model files not found on disk. Ensure model is fitted.")

        # Load RF, predict, free memory immediately
        rf = joblib.load(self.rf_path)
        rf_preds = rf.predict(X)
        del rf
        gc.collect()
        
        # Load ET, predict, free memory immediately
        et = joblib.load(self.et_path)
        et_preds = et.predict(X)
        del et
        gc.collect()
        
        final_preds = []
        for r, e in zip(rf_preds, et_preds):
            if r == 1 and e == 1:
                final_preds.append(1)
            elif r == 2 and e == 2:
                final_preds.append(2)
            else:
                final_preds.append(0)
                
        return np.array(final_preds)

    def cleanup_old_files(self, keep_version_id):
        """Safely delete older model files while keeping the active model version alive."""
        for pattern in ["rf_combo_*.joblib", "et_combo_*.joblib"]:
            for filepath in glob.glob(pattern):
                try:
                    if str(keep_version_id) not in filepath:
                        os.remove(filepath)
                        print(f"  [Cleanup] Removed stale model file: {filepath}")
                except Exception as e:
                    print(f"  [Cleanup] Skipped removing {filepath}: {e}")


def make_model(name):
    name = (name or "et").lower()
    if name == "rf_et_combo":
        return RFETComboModel()
    if name == "rf":
        return RandomForestClassifier(n_estimators=350, class_weight="balanced",
                                      random_state=42, n_jobs=-1, verbose=1)
    if name == "hgb":
        return HistGradientBoostingClassifier(random_state=42, max_iter=300, learning_rate=0.08)
    if name == "gb":
        return GradientBoostingClassifier(random_state=42)
    if name == "logit":
        return make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=1000, class_weight="balanced"))
    return ExtraTreesClassifier(n_estimators=400, class_weight="balanced",
                                random_state=42, n_jobs=-1, verbose=1)

def build_predict_row(symbol, feats):
    try:
        df = build_multi_tf(symbol)
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
# Core training routine (Runs on background thread)
# ─────────────────────────────────────────────
def _run_training():
    global model, training_in_progress, last_trained_at, training_error

    print(f"\n[{datetime.now():%H:%M:%S}] ── Starting background training run ──")

    try:
        df = build_multi_tf(args.symbol)
        print(f"  Multi-TF merge complete: {len(df):,} rows, {len(df.columns)} columns")

        df = add_calendar(df)

        labeler = RobustPriceLabelerV3(
            atr_period      = 14,
            zigzag_atr_mult = 5,
            hold_bars       = 5,
            min_streak      = 1.5,
            target_hold_pct = 1.5,
        )
        df = labeler.label(df)

        global active_features
        if args.features == 'engineered':
            df, feats = add_engineered_features(df)
        else:
            feats = list(FEATURE_NAMES)
            
        df = df[feats + ['label']].replace([np.inf, -np.inf], np.nan).dropna()
        print(f"  Features: {len(feats)} ({args.features})  usable rows: {len(df):,}  model: {args.model}")

        X = df[feats].values
        y = df['label'].values
        del df

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.1, random_state=42, stratify=y
        )
        del X, y

        # Instantiate & Train the new model in local thread memory
        new_model = make_model(args.model)
        new_model.fit(X_train, y_train)
        del X_train, y_train

        # ── Evaluate (Accuracy & Confusion Matrix) ─────────────────
        y_pred   = new_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
        print(f"  Classification Report:\n{classification_report(y_test, y_pred)}")
        del X_test, y_test, y_pred

        # ── Complete Model Swap ───────────────────────────────────
        with _model_lock:
            old_model       = model
            model           = new_model
            active_features = feats
            last_trained_at = datetime.now()
            training_error  = None

        print(f"[{datetime.now():%H:%M:%S}] ✓ New model swapped in successfully")

        # ── Post-Swap Disk Cleanup ────────────────────────────────
        if isinstance(new_model, RFETComboModel):
            # Brief pause to allow inflight predictions on old files to complete
            time.sleep(2)
            new_model.cleanup_old_files(keep_version_id=new_model.version_id)

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


def _training_scheduler(interval_minutes: int):
    global training_in_progress, next_train_at

    while True:
        with _model_lock:
            training_in_progress = True
            next_train_at = datetime.now() + timedelta(minutes=interval_minutes)

        train_start = time.monotonic()
        _run_training()
        elapsed = time.monotonic() - train_start

        sleep_secs = max(0, interval_minutes * 60 - elapsed)
        next_wakeup = datetime.now() + timedelta(seconds=sleep_secs)
        with _model_lock:
            next_train_at = next_wakeup

        print(f"  Next retrain scheduled at {next_wakeup:%Y-%m-%d %H:%M:%S} "
              f"(sleeping {sleep_secs/60:.1f} min)")
        time.sleep(sleep_secs)


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
    try:
        # Non-blocking lock: grab current model reference and release immediately
        with _model_lock:
            current_model = model   
            feats         = active_features

        if current_model is None:
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

        # Run prediction on worker thread using current_model reference
        prediction = current_model.predict(arr)[0]
        label_map  = {0: "Hold", 1: "Buy", 2: "Sell"}
        print(f"Prediction: {int(prediction)} ({label_map.get(int(prediction), '?')})  [{args.features}/{args.model}]")
        return str(int(prediction)), 200

    except Exception as e:
        print(f"Error in /predict — returning Hold (0): {e}")
        return "0", 200

@app.route('/training/status', methods=['GET'])
def training_status():
    with _model_lock:
        in_progress  = training_in_progress
        last_at      = last_trained_at
        next_at      = next_train_at
        model_ready  = model is not None
        last_err     = training_error

    now = datetime.now()

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

    init_mt5()

    scheduler = threading.Thread(
        target=_training_scheduler,
        args=(args.retrain_interval,),
        daemon=True,
        name="TrainingScheduler"
    )
    scheduler.start()
    print(f"Background training scheduler started (interval: {args.retrain_interval} min)")

    # Enable explicit multi-threading in Flask server
    app.run(host='0.0.0.0', port=args.port, debug=False, threaded=True)