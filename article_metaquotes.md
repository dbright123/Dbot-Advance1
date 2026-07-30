# Building a Self-Retraining Machine Learning Trading System: A Python Prediction Server Connected to an MQL5 Expert Advisor

## Introduction

Most machine learning trading examples stop at the notebook: you train a model, print an accuracy score, and plot a confusion matrix. The hard part — turning that model into a live decision engine that an Expert Advisor can query, tick after tick, while the market keeps moving and the model keeps ageing — is rarely shown end to end.

This article describes a complete, working architecture that closes that gap. It is built from two cooperating components:

1. A **Python prediction server** that connects to the MetaTrader 5 terminal, pulls multi-timeframe price data, engineers features, labels the data, trains a `RandomForestClassifier`, and — importantly — **retrains itself on a schedule** so the model never goes stale.
2. An **MQL5 Expert Advisor** that requests a prediction on every new bar via `WebRequest`, then applies a disciplined trade-management layer (ATR-based exits, a trailing break-even ratchet, re-entry logic, split-leg entries, an ADX/DI trend filter and a rejection-wick filter) on top of the raw model signal.

The design goal throughout is **separation of concerns**: Python owns the intelligence (data, features, model, retraining), MQL5 owns the execution (order placement, risk, position bookkeeping and the on-chart dashboard). The two talk over a tiny HTTP contract that returns a single integer: `0` = Hold, `1` = Buy, `2` = Sell.

> **Disclaimer.** The code below is presented for educational purposes. Machine learning does not predict the future; it estimates conditional probabilities from historical patterns that may not repeat. Nothing here is financial advice, and no result guarantees future profitability. Always test on a demo account first.

## System Architecture

The two processes run side by side on the same machine and communicate over localhost:

```
┌──────────────────────────────┐         HTTP GET /predict?...        ┌──────────────────────────────┐
│      MQL5 Expert Advisor      │  ─────────────────────────────────► │     Python Flask Server       │
│  (execution + risk + HUD)     │                                     │  (data + features + model)    │
│                               │ ◄─────────────────────────────────  │                               │
│  • WebRequest each new bar    │        "0" | "1" | "2"              │  • MT5 multi-TF fetch         │
│  • ATR TP/SL, trailing BE     │                                     │  • feature engineering        │
│  • re-entry, split legs       │         GET /health                 │  • labelling                  │
│  • ADX/DI + rejection filter  │  ◄─────────────────────────────────►│  • RandomForest training      │
│  • 3-panel dashboard          │         GET /training/status        │  • background auto-retrain    │
└──────────────────────────────┘                                     └──────────────────────────────┘
```

Why HTTP rather than a shared file, a socket, or a DLL? Because it is the simplest contract that MQL5 supports natively through `WebRequest`, it is language-agnostic, it is trivial to test with `curl`, and it lets the EA poll a health endpoint to know whether the brain on the other side is actually awake.

## Part 1 — The Python Prediction Server

### Fetching multi-timeframe data

A single timeframe rarely tells the whole story. A signal that looks strong on H1 can be fighting a clear H4 or D1 trend. The server therefore fetches three timeframes and merges them so that every H1 bar carries the most recently **closed** H4 and D1 context.

```python
def fetch_rates(symbol, timeframe, count=9_000_000):
    """Pull rates from MT5 and return a tidy DataFrame."""
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
```

The critical detail is in the merge. Higher-timeframe values are **shifted by one bar** before the join, so the model can only ever see H4 and D1 candles that have already closed. This single line prevents a very common and very destructive form of look-ahead bias:

```python
def tag_and_shift(df, suffix):
    df = df.sort_values("time").rename(
        columns={c: f"{c}_{suffix}" for c in df.columns if c != "time"})
    val_cols = [c for c in df.columns if c != "time"]
    df[val_cols] = df[val_cols].shift(1)      # <-- only use CLOSED bars
    return df

merged = pd.merge_asof(df_1h, df_4h, on="time", direction="backward")
merged = pd.merge_asof(merged, df_1d, on="time", direction="backward")
```

`pd.merge_asof` with `direction="backward"` performs an "as-of" join: for each H1 timestamp it attaches the last known H4 and D1 row at or before that time. Combined with the shift, the result is a leak-free multi-timeframe feature table.

### The feature contract

The model is trained on a fixed, ordered list of features. The **exact same list, in the same order**, is used on both sides of the wire. Keeping this single source of truth is what makes the whole system reproducible:

```python
FEATURE_NAMES = [
    'open', 'high', 'low', 'close', 'volume',                 # H1 base
    'open_4h', 'high_4h', 'low_4h', 'close_4h', 'volume_4h',  # H4 context
    'open_1d', 'high_1d', 'low_1d', 'close_1d', 'volume_1d',  # D1 context
    'hour', 'day', 'month', 'day_of_week',                    # calendar
]
```

### Labelling: turning prices into targets

Supervised learning needs labels. Naively labelling "next bar up = Buy" produces noise, because a single bar's direction is mostly randomness. Instead the project uses a swing/zig-zag style labeller driven by ATR, so that a Buy or Sell label is only assigned when price makes a move that is meaningful relative to recent volatility:

```python
labeler = RobustPriceLabelerV3(
    atr_period      = 14,
    zigzag_atr_mult = 0.5,
    hold_bars       = 1,
    min_streak      = 0.01,
    target_hold_pct = 0.01,
)
df = labeler.label(df)
```

Volatility-scaled labelling matters because a 30-pip move is a genuine swing in a quiet session but pure noise during a news spike. Anchoring the threshold to ATR keeps the definition of "a real move" consistent across market regimes.

### Training the model

With features and labels in place, training a Random Forest is straightforward. Two choices are worth highlighting: a stratified split (to preserve the class balance in the test set) and `class_weight='balanced'` (because Hold labels usually dominate Buy and Sell):

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=y
)

new_model = RandomForestClassifier(
    n_estimators=350, random_state=42,
    class_weight='balanced', verbose=1, n_jobs=-1
)
new_model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, new_model.predict(X_test))
print(f"  Accuracy: {accuracy:.4f}")
```

### The part most examples skip: automatic retraining

A model trained once will slowly drift out of sync with the market. This server runs a background daemon thread that retrains at a configurable interval and then **atomically swaps** the new model in under a lock, so the live `/predict` endpoint never serves a half-built model:

```python
def _run_training():
    global model, training_in_progress, last_trained_at, training_error
    try:
        df = build_multi_tf(args.symbol)
        df = engineer_features(df)
        df = labeler.label(df)
        # ... split, fit, evaluate ...
        with _model_lock:              # atomic hand-over
            model           = new_model
            last_trained_at = datetime.now()
            training_error  = None
    except Exception as exc:
        with _model_lock:
            training_error = str(exc)
    finally:
        with _model_lock:
            training_in_progress = False
```

The scheduler runs the training **synchronously inside its own thread** so that two training runs can never overlap, then sleeps for the remainder of the interval:

```python
def _training_scheduler(interval_minutes: int):
    while True:
        with _model_lock:
            training_in_progress = True
            next_train_at = datetime.now() + timedelta(minutes=interval_minutes)

        train_start = time.monotonic()
        _run_training()
        elapsed = time.monotonic() - train_start

        sleep_secs = max(0, interval_minutes * 60 - elapsed)
        time.sleep(sleep_secs)
```

Because every access to the shared `model` object is guarded by `_model_lock`, the prediction thread and the training thread can run concurrently without ever seeing an inconsistent state.

### The HTTP contract

Three endpoints make the whole system observable and testable. `/predict` is deliberately forgiving — if the model is not ready yet, or a feature is missing, or the input is malformed, it returns a neutral Hold rather than throwing. In trading, "do nothing" is the correct behaviour when you are uncertain:

```python
@app.route('/predict', methods=['GET'])
def predict():
    with _model_lock:
        current_model = model
    if current_model is None:
        return "0", 200                # not ready yet → stay flat

    features = []
    for name in FEATURE_NAMES:
        raw = request.args.get(name)
        if raw is None:
            features.append(0.0)       # missing → default, logged server-side
            continue
        try:
            features.append(float(raw))
        except ValueError:
            return "0", 200            # bad data → stay flat

    arr        = np.array(features).reshape(1, -1)
    prediction = current_model.predict(arr)[0]
    return str(int(prediction)), 200
```

The companion `/health` and `/training/status` endpoints let the EA display, in real time, whether the model is ready, whether a retrain is running, and how long until the next one:

```python
@app.route('/training/status', methods=['GET'])
def training_status():
    with _model_lock:
        in_progress = training_in_progress
        last_at     = last_trained_at
        next_at     = next_train_at
        model_ready = model is not None
        last_err    = training_error

    remaining = (str(timedelta(seconds=int(max(0, (next_at - datetime.now()).total_seconds()))))
                 if next_at and not in_progress else "N/A — training in progress")

    return jsonify({
        "training_in_progress": in_progress,
        "model_ready"         : model_ready,
        "last_trained_at"     : last_at.strftime("%Y-%m-%d %H:%M:%S") if last_at else None,
        "time_until_retrain"  : remaining,
        "last_error"          : last_err,
    }), 200
```

You can verify the server with nothing more than `curl`:

```bash
curl http://localhost:5000/health
# {"model_loaded":true,"status":"healthy"}

curl "http://localhost:5000/predict?open=3024.5&high=3027.8&low=3022.1&close=3026.4&volume=1823&hour=14&day=22&month=3&day_of_week=6"
# 1
```

## Part 2 — The MQL5 Expert Advisor

### Requesting a prediction

On every **new bar** (never on every tick — that would flood the server and trade on noise), the EA copies one closed candle from H1, H4 and D1, assembles a query string in the exact feature order the model expects, and calls `WebRequest`:

```mql5
if(CopyRates(_Symbol, PERIOD_H1, 1, 1, r1)  <= 0) return -1;
if(CopyRates(_Symbol, PERIOD_H4, 1, 1, r4)  <= 0) return -1;
if(CopyRates(_Symbol, PERIOD_D1, 1, 1, r1d) <= 0) return -1;

url  = ServerURL + "?";
url += "open="       + DoubleToString(r1[0].open,  _Digits);
url += "&high="      + DoubleToString(r1[0].high,  _Digits);
// ... H4 and D1 blocks ...
url += "&hour="      + IntegerToString(dt.hour);
url += "&day_of_week=" + IntegerToString(dt.day_of_week);

int res = WebRequest("GET", url, hdr, RequestTimeout, post, result, hdr);
```

Robust error handling is essential here. A network outage must **not** be interpreted as a Hold — the EA must leave existing positions untouched rather than reacting to silence. The prediction function therefore uses a distinct `-1` sentinel for "no reliable answer":

```mql5
if(res == -1)
{
   int err = GetLastError();
   if(err == 4060) Print("URL not allowed — add '", g_serverBase, "' to MT5 allowed list");
   g_serverStatus = "OFFLINE";
   return -1;        // an outage must NOT look like Hold
}

int pred = (int)StringToInteger(response);
if(pred < 0 || pred > 2) return -1;
return pred;
```

> **Setup note.** For `WebRequest` to work, the server base URL (e.g. `http://127.0.0.1:5000`) must be added under *Tools → Options → Expert Advisors → Allow WebRequest for listed URL*.

The main loop keeps signal generation strictly on the bar boundary while running position management on every tick:

```mql5
void OnTick()
{
   RefreshTracking();       // snapshot positions for re-entry recovery
   breakeven();             // trailing break-even ratchet
   ManageCloseAtProfit();   // close flagged opposite baskets at profit
   ManagePendingLegs();     // fill second legs at the prior candle extreme
   ManageReentry();         // re-enter after a break-even stop

   if(g_currentBar == lastBarTime) return;   // one decision per bar
   lastBarTime = g_currentBar;

   prediction = GetMLPrediction();
   if     (prediction == 1) HandleBuySignal (g_currentBar);
   else if(prediction == 2) HandleSellSignal(g_currentBar);
   else if(prediction == 0) HandleHoldSignal();
   // prediction == -1 → server/data error: do nothing, keep positions intact
}
```

### Turning a raw signal into a managed trade

The model produces a direction; it says nothing about *how* to enter, protect and exit. That is the EA's job, and it is where a naive "buy on 1, sell on 2" bot separates from a disciplined one. The design exposes each behaviour as an input so it can be tuned or switched off:

- **Volatility-scaled exits.** Take-profit and stop-loss are computed from ATR (`TP_ATR_Mult`, `SL_ATR_Mult`) rather than fixed pips, so protection adapts to the current regime.
- **Trailing break-even ratchet.** Once a position is `BreakevenPips` in profit, the stop is ratcheted forward to lock in at least `BreakevenLockPips`, and it only ever moves in the favourable direction.
- **Re-entry after a break-even stop.** If a winner is stopped out at break-even but the trend resumes, the EA can re-enter at the previous entry price.
- **Split-leg (staged) entries.** Instead of committing full size at once, a second leg is armed and filled by tick-watch at the previous candle's low (for buys) or high (for sells), improving the average entry.
- **Trend and rejection filters.** An ADX/DI check (`ADX_MinLevel`) blocks entries against a weak or contradictory trend, and a rejection-wick filter (`RejectionWickRatio`) blocks entries into candles that show strong rejection.
- **Account protection.** A projected margin-level guard (`MinMarginLevelPct`) and a cap on concurrent positions (`MaxOpenPositions`) refuse new risk when the account is stretched.

Every one of these is a *rule the model never learned* — deliberately kept in MQL5, where execution logic belongs, rather than baked into the classifier.

### A dashboard that shows the brain's state

Because the intelligence lives in another process, the trader needs to *see* it. The EA polls `/health` and `/training/status` on a timer and renders a three-panel on-chart HUD: server status (online / model ready / training / next retrain), EA monitor (current signal, ADX, break-even level, GMT), and positions (accumulated buy/sell volume, staged legs, flip flags, duplicates). At a glance you know whether the system is trading on a fresh model, a stale one, or none at all.

```mql5
void CheckServerStatus()
{
   int code = WebRequest("GET", g_serverBase + "/health", "", RequestTimeout, post, result, hdr);
   g_serverStatus = (code == -1 || code == 0) ? "OFFLINE" : "ONLINE";

   code = WebRequest("GET", g_serverBase + "/training/status", "", RequestTimeout, post, result, hdr);
   string json = CharArrayToString(result);
   g_modelReady       = ExtractBool  (json, "model_ready");
   g_trainingActive   = ExtractBool  (json, "training_in_progress");
   g_timeUntilRetrain = ExtractString(json, "time_until_retrain");
   UpdateHUD();
}
```

## Design Decisions Worth Copying

A few principles from this project generalise well beyond this specific EA:

1. **Fail neutral.** Every uncertain path — server down, missing feature, malformed input, model not ready — resolves to Hold or "do nothing". Uncertainty should never open risk.
2. **One decision per bar.** Signals are generated only on new-bar events; only management runs on ticks. This keeps the model out of the noise and the server out of overload.
3. **A single feature contract.** The ordered `FEATURE_NAMES` list is the shared truth between training and inference. Drift here is the most common silent bug in ML trading.
4. **No look-ahead.** Higher-timeframe data is shifted to closed bars before merging. It is astonishingly easy to leak the future into a backtest and impossible to trade it live.
5. **Separation of concerns.** Python thinks, MQL5 acts. Each side can be developed, tested and replaced independently.

## Conclusion

This system demonstrates that a machine-learning trading model becomes genuinely usable only when it is wrapped in the unglamorous infrastructure around it: a reproducible feature contract, leak-free multi-timeframe data, automatic retraining with atomic model swaps, a forgiving HTTP interface, and — on the terminal side — a robust execution layer that treats the model as *one input among several* rather than an oracle.

The Random Forest at the centre could be swapped for gradient boosting, an LSTM, or an ensemble without touching a single line of MQL5, precisely because the two halves communicate through a minimal contract. That modularity is the real lesson: keep the intelligence and the execution separate, make each side observable, and let every uncertain path fail safely.

As always, validate everything on a demo account and over a long forward test before risking capital. A model that scores well on historical data has proven only that it can describe the past — never that it can trade the future.
