from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
from flask_cors import CORS
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import threading
import time
import re
import os

app = Flask(__name__)
CORS(app)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MODEL_PATH = "GeneratedXAUUSDRandom Forest.joblib"

# Get a free key at https://newsapi.org (500 req/day free)
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "2e3ede876f094cee925a093376827858")

# How many minutes before/after a high-impact news event to freeze predictions
NEWS_BLOCK_MINUTES_BEFORE = 5
NEWS_BLOCK_MINUTES_AFTER  = 15

# Minimum absolute sentiment score to override model (0.0 – 1.0)
SENTIMENT_OVERRIDE_THRESHOLD = 0.4

# How often to refresh news cache (seconds)
NEWS_REFRESH_INTERVAL = 300   # 5 minutes
CALENDAR_REFRESH_INTERVAL = 3600  # 1 hour

model = None

FEATURE_NAMES = [
    'open', 'high', 'low', 'close', 'volume',
    'hour', 'minute', 'day', 'month', 'day_of_week'
]

# ─────────────────────────────────────────────
# GOLD SENTIMENT KEYWORDS
# ─────────────────────────────────────────────

# Words that push gold UP (bullish for XAU)
BULLISH_GOLD_KEYWORDS = [
    "inflation", "inflation rise", "cpi higher", "pce higher",
    "war", "conflict", "geopolitical tension", "recession",
    "fed pause", "rate cut", "dovish", "rate hold",
    "safe haven", "gold rally", "gold surge", "gold climbs",
    "dollar falls", "dollar weakens", "usd drops",
    "banking crisis", "debt crisis", "uncertainty",
    "gold demand", "central bank buys gold",
    "stagflation", "yields fall", "bond yields drop",
]

# Words that push gold DOWN (bearish for XAU)
BEARISH_GOLD_KEYWORDS = [
    "rate hike", "hawkish", "fed hike", "interest rate increase",
    "dollar strengthens", "dollar surges", "usd rises",
    "gold falls", "gold drops", "gold decline", "gold sell-off",
    "risk-on", "equities rally", "stock market surge",
    "inflation cools", "cpi lower", "inflation eases",
    "yields rise", "bond yields surge",
    "gold outflows", "etf outflows",
]

# Economic event names that heavily impact gold
HIGH_IMPACT_GOLD_EVENTS = [
    "non-farm payrolls", "nfp",
    "fed interest rate decision", "fomc",
    "cpi", "consumer price index",
    "pce", "personal consumption expenditure",
    "gdp", "gross domestic product",
    "unemployment rate", "jobless claims",
    "ism manufacturing", "ism services",
    "retail sales",
    "powell", "fed chair",
    "ecb rate decision",
    "inflation report",
    "pmi",
    "aud", "usd", "eur", "gbp",   # major currencies affect gold
]

# ─────────────────────────────────────────────
# SHARED STATE (thread-safe via simple dict + lock)
# ─────────────────────────────────────────────
news_cache = {
    "sentiment_score": 0.0,       # -1.0 (very bearish) to +1.0 (very bullish)
    "sentiment_label": "neutral",
    "upcoming_events": [],         # list of dicts with event info
    "last_news_refresh": None,
    "last_calendar_refresh": None,
    "headlines": [],
}
cache_lock = threading.Lock()

# ─────────────────────────────────────────────
# NEWS SENTIMENT ANALYSIS
# ─────────────────────────────────────────────

def score_headline(text: str) -> float:
    """
    Score a single headline.
    Returns a float: positive = bullish gold, negative = bearish gold.
    """
    text_lower = text.lower()
    score = 0.0

    for kw in BULLISH_GOLD_KEYWORDS:
        if kw in text_lower:
            score += 1.0

    for kw in BEARISH_GOLD_KEYWORDS:
        if kw in text_lower:
            score -= 1.0

    return score


def fetch_news_sentiment():
    """
    Pull latest gold/economic headlines from NewsAPI and compute
    an aggregate sentiment score.
    """
    try:
        queries = [
            "gold XAU price",
            "Federal Reserve interest rates",
            "inflation CPI gold",
            "US dollar economy",
        ]
        all_headlines = []

        for q in queries:
            url = (
                "https://newsapi.org/v2/everything"
                f"?q={requests.utils.quote(q)}"
                "&sortBy=publishedAt"
                "&pageSize=10"
                "&language=en"
                f"&apiKey={NEWS_API_KEY}"
            )
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                articles = resp.json().get("articles", [])
                # Only look at news published in the last 6 hours
                cutoff = datetime.utcnow() - timedelta(hours=6)
                for art in articles:
                    published = art.get("publishedAt", "")
                    try:
                        pub_dt = datetime.strptime(published, "%Y-%m-%dT%H:%M:%SZ")
                        if pub_dt >= cutoff:
                            title = art.get("title", "") or ""
                            desc  = art.get("description", "") or ""
                            all_headlines.append(f"{title}. {desc}")
                    except Exception:
                        pass

        if not all_headlines:
            print("[News] No recent headlines found.")
            return 0.0, "neutral", []

        scores = [score_headline(h) for h in all_headlines]
        total  = sum(scores)
        # Normalise to [-1, 1]
        normalised = max(-1.0, min(1.0, total / (len(scores) * 3 + 1e-9)))

        if normalised > 0.15:
            label = "bullish"
        elif normalised < -0.15:
            label = "bearish"
        else:
            label = "neutral"

        print(f"[News] Sentiment: {label} ({normalised:.3f}) from {len(all_headlines)} headlines")
        return normalised, label, all_headlines[:5]  # return top 5 for logging

    except Exception as e:
        print(f"[News] Error fetching news: {e}")
        return 0.0, "neutral", []


# ─────────────────────────────────────────────
# ECONOMIC CALENDAR (ForexFactory scrape)
# ─────────────────────────────────────────────

def fetch_economic_calendar():
    """
    Scrape ForexFactory for today's high-impact USD events.
    Returns a list of dicts: {name, time_utc, impact}
    """
    events = []
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }
        url = "https://www.forexfactory.com/calendar"
        resp = requests.get(url, headers=headers, timeout=15)
        if resp.status_code != 200:
            print(f"[Calendar] ForexFactory returned {resp.status_code}")
            return events

        soup = BeautifulSoup(resp.text, "html.parser")
        rows = soup.select("tr.calendar__row")

        current_time_str = None
        today = datetime.utcnow().date()

        for row in rows:
            # ForexFactory reuses the time cell — track last seen time
            time_cell = row.select_one("td.calendar__time")
            if time_cell:
                t = time_cell.get_text(strip=True)
                if t and t not in ("", "All Day", "Tentative"):
                    current_time_str = t

            currency_cell = row.select_one("td.calendar__currency")
            impact_cell   = row.select_one("td.calendar__impact span")
            event_cell    = row.select_one("td.calendar__event")

            if not (currency_cell and impact_cell and event_cell):
                continue

            currency = currency_cell.get_text(strip=True).upper()
            impact   = impact_cell.get("class", [])
            event_name = event_cell.get_text(strip=True).lower()

            # Filter: USD/EUR events + high impact + gold-relevant keywords
            is_relevant_currency = currency in ("USD", "EUR", "GBP", "AUD", "JPY")
            is_high_impact = any("high" in c.lower() for c in impact)
            is_gold_relevant = any(kw in event_name for kw in HIGH_IMPACT_GOLD_EVENTS)

            if is_relevant_currency and (is_high_impact or is_gold_relevant):
                # Parse time (ForexFactory uses ET — convert to UTC roughly +5h)
                event_dt = None
                if current_time_str:
                    try:
                        # Handle formats like "8:30am"
                        t_clean = current_time_str.replace("\u00a0", "").strip()
                        dt = datetime.strptime(t_clean, "%I:%M%p")
                        # Assume ET = UTC-5 (approximate; adjust for DST if needed)
                        event_dt = datetime.combine(
                            today,
                            dt.time()
                        ) + timedelta(hours=5)
                    except Exception:
                        pass

                events.append({
                    "name": event_cell.get_text(strip=True),
                    "currency": currency,
                    "time_utc": event_dt,
                    "impact": "high" if is_high_impact else "medium",
                })

        print(f"[Calendar] Found {len(events)} gold-relevant events today")

    except Exception as e:
        print(f"[Calendar] Error fetching calendar: {e}")

    return events


# ─────────────────────────────────────────────
# BACKGROUND REFRESH THREAD
# ─────────────────────────────────────────────

def background_refresh():
    """Continuously refresh news & calendar in the background."""
    while True:
        now = datetime.utcnow()

        with cache_lock:
            last_news     = news_cache["last_news_refresh"]
            last_calendar = news_cache["last_calendar_refresh"]

        # Refresh news sentiment
        if last_news is None or (now - last_news).total_seconds() >= NEWS_REFRESH_INTERVAL:
            score, label, headlines = fetch_news_sentiment()
            with cache_lock:
                news_cache["sentiment_score"]  = score
                news_cache["sentiment_label"]  = label
                news_cache["headlines"]        = headlines
                news_cache["last_news_refresh"] = now

        # Refresh calendar
        if last_calendar is None or (now - last_calendar).total_seconds() >= CALENDAR_REFRESH_INTERVAL:
            events = fetch_economic_calendar()
            with cache_lock:
                news_cache["upcoming_events"]       = events
                news_cache["last_calendar_refresh"] = now

        time.sleep(60)  # check every minute


# ─────────────────────────────────────────────
# CORE FILTER LOGIC
# ─────────────────────────────────────────────

def get_news_filter():
    """
    Returns a dict describing whether to block/override the ML prediction.

    Possible actions:
        "allow"    – news is neutral / no event; pass model prediction through
        "block"    – high-impact event window active; return HOLD (0)
        "buy_only" – news strongly bullish; only allow BUY or HOLD
        "sell_only"– news strongly bearish; only allow SELL or HOLD
    """
    with cache_lock:
        score        = news_cache["sentiment_score"]
        label        = news_cache["sentiment_label"]
        events       = news_cache["upcoming_events"]

    now = datetime.utcnow()

    # 1. Check if we're inside a high-impact news window
    for event in events:
        evt_time = event.get("time_utc")
        if evt_time is None:
            continue
        delta_minutes = (evt_time - now).total_seconds() / 60.0
        if -NEWS_BLOCK_MINUTES_AFTER <= delta_minutes <= NEWS_BLOCK_MINUTES_BEFORE:
            print(
                f"[Filter] BLOCKING — '{event['name']}' "
                f"event in {delta_minutes:.1f} min window"
            )
            return {
                "action": "block",
                "reason": f"High-impact event: {event['name']} "
                          f"(within ±{NEWS_BLOCK_MINUTES_BEFORE}/{NEWS_BLOCK_MINUTES_AFTER} min window)",
                "sentiment_score": score,
                "sentiment_label": label,
            }

    # 2. Apply sentiment bias if strong enough
    if score >= SENTIMENT_OVERRIDE_THRESHOLD:
        return {
            "action": "buy_only",
            "reason": f"Strong bullish news sentiment ({score:.2f})",
            "sentiment_score": score,
            "sentiment_label": label,
        }
    elif score <= -SENTIMENT_OVERRIDE_THRESHOLD:
        return {
            "action": "sell_only",
            "reason": f"Strong bearish news sentiment ({score:.2f})",
            "sentiment_score": score,
            "sentiment_label": label,
        }

    return {
        "action": "allow",
        "reason": "No significant news interference",
        "sentiment_score": score,
        "sentiment_label": label,
    }


def apply_news_filter(model_prediction: int, news_filter: dict) -> int:
    """
    Takes the raw model prediction (0=Hold, 1=Buy, 2=Sell)
    and the news filter dict, and returns the final prediction.
    """
    action = news_filter["action"]

    if action == "block":
        return 0  # Force HOLD during news window

    if action == "buy_only" and model_prediction == 2:
        print("[Filter] Overriding SELL → HOLD (bearish news doesn't support sell)")
        return 0  # Downgrade sell to hold; don't fight bullish macro

    if action == "sell_only" and model_prediction == 1:
        print("[Filter] Overriding BUY → HOLD (bullish signal blocked by bearish macro)")
        return 0  # Downgrade buy to hold; don't fight bearish macro

    return model_prediction  # News agrees or is neutral — trust the model


# ─────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────

def load_model():
    global model
    try:
        model = joblib.load(MODEL_PATH)
        print("Model loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        exit()


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.route('/predict', methods=['GET'])
def predict():
    try:
        if model is None:
            return "0", 500

        # Extract features
        features = []
        missing_features = []
        for feature_name in FEATURE_NAMES:
            value = request.args.get(feature_name)
            if value is None:
                missing_features.append(feature_name)
                features.append(0.0)
            else:
                try:
                    features.append(float(value))
                except ValueError:
                    return f"Invalid value for {feature_name}", 400

        if missing_features:
            print(f"Warning: Missing features: {missing_features}")

        features_array = np.array(features).reshape(1, -1)

        # ML prediction
        raw_prediction = int(model.predict(features_array)[0])
        print(f"[Model] Raw prediction: {raw_prediction} (0=Hold, 1=Buy, 2=Sell)")

        # News filter
        news_filter     = get_news_filter()
        final_prediction = apply_news_filter(raw_prediction, news_filter)

        print(
            f"[Final] Prediction: {final_prediction} | "
            f"Filter: {news_filter['action']} | "
            f"Reason: {news_filter['reason']}"
        )

        return str(final_prediction), 200

    except Exception as e:
        print(f"Error in prediction: {e}")
        import traceback
        traceback.print_exc()
        return "0", 500


@app.route('/health', methods=['GET'])
def health():
    with cache_lock:
        cache_info = {
            "sentiment_score":  news_cache["sentiment_score"],
            "sentiment_label":  news_cache["sentiment_label"],
            "upcoming_events":  [
                {
                    "name": e["name"],
                    "currency": e["currency"],
                    "time_utc": e["time_utc"].isoformat() if e["time_utc"] else None,
                    "impact": e["impact"],
                }
                for e in news_cache["upcoming_events"]
            ],
            "last_news_refresh":     news_cache["last_news_refresh"].isoformat()
                                     if news_cache["last_news_refresh"] else None,
            "last_calendar_refresh": news_cache["last_calendar_refresh"].isoformat()
                                     if news_cache["last_calendar_refresh"] else None,
            "recent_headlines": news_cache["headlines"],
        }

    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "news_filter": cache_info,
    }), 200


@app.route('/news_status', methods=['GET'])
def news_status():
    """Human-readable current news filter state."""
    news_filter = get_news_filter()
    return jsonify(news_filter), 200


@app.route('/test', methods=['GET'])
def test():
    return "Server is running!", 200


# ─────────────────────────────────────────────
# STARTUP
# ─────────────────────────────────────────────

if __name__ == '__main__':
    print("Starting ML Prediction Server...")
    load_model()

    # Start background news/calendar refresh thread
    refresh_thread = threading.Thread(target=background_refresh, daemon=True)
    refresh_thread.start()
    print("Background news refresh thread started.")

    app.run(host='0.0.0.0', port=5002, debug=False)