from __future__ import annotations
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import math

from typing import List, Tuple
import numpy as np
import pandas as pd
 
#warnings.filterwarnings('ignore')


def engineer_features(df):
    df['time'] = pd.to_datetime(df['time'], unit='s', errors='coerce')
    df['month'] = df['time'].dt.month

    # Extract day of the month (1-31)
    df['day'] = df['time'].dt.day


    # --- New Additions ---
    # Extract hour (0-23)
    df['hour'] = df['time'].dt.hour
    df["minute"] = df['time'].dt.minute
    df['day_of_week'] = df['time'].dt.day_of_week

    #df['t_price'] = (df['open'] + df['close'])/2
    #df['hl_price'] = (df['high'] + df['low'])/2


    # --- End of New Additions ---

    # 3. Remove the original 'time' column
    df = df.drop(columns=['time'])
    # --- Clean up and drop NaNs (first ~200 rows) ---
    df = df.dropna().reset_index(drop=True)

    return df


# Creating a labeling function for XAUUSD close prices and demonstrating it on example data.
# The function labels each row as: 0 = hold, 1 = buy, 2 = sell.
# Behavior:
#  - Smooths the close with a rolling mean (to reduce noise).
#  - Uses percent-change between consecutive smoothed closes.
#  - If pct change magnitude < change_threshold -> preserve current state (so buys/sells continue).
#    If no current state yet, remain 'hold'.
#  - If pct change >= threshold -> set state to buy (positive) or sell (negative).
# This creates continuous streaks of buy or sell until a meaningful reversal happens.
# Adjust `smoothing_window` and `change_threshold` to control sensitivity (smaller threshold -> more labels).


def label_buy_sell_hold(df,
                        close_col='close',
                        smoothing_window=3,
                        change_threshold=0.0005,
                        hold_label=0,
                        buy_label=1,
                        sell_label=2):
    """
    Label dataframe rows as hold/buy/sell based on smoothed percent-change of close prices.
    - df: pandas DataFrame with at least the close_col.
    - smoothing_window: rolling mean window to reduce noise (int >=1).
    - change_threshold: relative change (fraction) that qualifies as meaningful movement.
      e.g., 0.0005 = 0.05% change between consecutive smoothed closes.
    Returns: a copy of df with columns 'close_smoothed' and 'label' added.
    """
    df = df.reset_index(drop=True)
    if close_col not in df.columns:
        raise ValueError(f"'{close_col}' column not found in dataframe")
    # Ensure float
    s = df[close_col].astype(float)
    # Smoothed close to reduce noise
    smoothed = s.rolling(window=smoothing_window, min_periods=1).mean()
    pct = smoothed.pct_change(fill_method=None).fillna(0.0)

    labels = [hold_label] * len(df)
    state = None  # None => no active buy/sell yet; we'll keep hold until a meaningful movement appears
    for i in range(1, len(df)):
        p = pct.iat[i]
        if abs(p) < change_threshold:
            # Not a large enough move: continue the current state (if any), otherwise hold
            if state is None:
                labels[i] = hold_label
            else:
                labels[i] = state
        else:
            # Significant move: update state and label accordingly
            if p > 0:
                state = buy_label
            else:
                state = sell_label
            labels[i] = state
    # First row: set to the same as second row (if exists) to avoid an isolated hold at start,
    # otherwise leave as hold.
    if len(df) > 1:
        labels[0] = labels[1]
    else:
        labels[0] = hold_label

    #df['close_smoothed'] = smoothed
    df['label'] = labels
    #df['label_name'] = df['label'].map({hold_label: 'hold', buy_label: 'buy', sell_label: 'sell'})
    return df




from typing import Tuple, List, Optional
import matplotlib.pyplot as plt

class RobustPriceLabeler:
    """
    A robust price labeling system using forward and backward passes
    to ensure consistent labeling without contradictory buy/sell signals.
    """

    def __init__(self,
                 smoothing_window: int = 5,
                 min_trend_change: float = 0.002,  # 0.2%
                 min_reversal_change: float = 0.003,  # 0.3%
                 consolidation_threshold: float = 0.001):  # 0.1%
        """
        Initialize the labeler with configurable parameters.

        Args:
            smoothing_window: Window for price smoothing
            min_trend_change: Minimum change to consider a trend
            min_reversal_change: Minimum change to consider a reversal
            consolidation_threshold: Threshold for consolidation/range detection
        """
        self.smoothing_window = smoothing_window
        self.min_trend_change = min_trend_change
        self.min_reversal_change = min_reversal_change
        self.consolidation_threshold = consolidation_threshold

    def calculate_support_resistance(self, prices: np.ndarray, window: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate dynamic support and resistance levels."""
        support = np.zeros_like(prices)
        resistance = np.zeros_like(prices)

        for i in range(len(prices)):
            start = max(0, i - window)
            end = i + 1

            window_prices = prices[start:end]
            if len(window_prices) >= 5:
                # Support is recent low with some margin
                support[i] = np.min(window_prices[-5:]) * 0.998
                # Resistance is recent high with some margin
                resistance[i] = np.max(window_prices[-5:]) * 1.002
            else:
                support[i] = prices[i] * 0.995
                resistance[i] = prices[i] * 1.005

        return support, resistance

    def detect_trend_strength(self, prices: np.ndarray) -> np.ndarray:
        """Calculate trend strength using multiple timeframes."""
        n = len(prices)
        trend_strength = np.zeros(n)

        # Short-term trend (5 periods)
        short_ma = pd.Series(prices).rolling(window=5, min_periods=3).mean().values
        short_slope = np.zeros(n)
        for i in range(2, n):
            if i >= 5:
                x = np.arange(-4, 1)
                y = prices[i-4:i+1]
                if len(y) == 5:
                    coeffs = np.polyfit(x, y, 1)
                    short_slope[i] = coeffs[0]

        # Medium-term trend (15 periods)
        medium_ma = pd.Series(prices).rolling(window=15, min_periods=8).mean().values
        medium_slope = np.zeros(n)
        for i in range(7, n):
            if i >= 15:
                x = np.arange(-14, 1)
                y = prices[i-14:i+1]
                if len(y) == 15:
                    coeffs = np.polyfit(x, y, 1)
                    medium_slope[i] = coeffs[0]

        # Combine trend strengths
        for i in range(n):
            if i >= 15:
                # Weight short-term more for responsiveness
                trend_strength[i] = (short_slope[i] * 0.6 + medium_slope[i] * 0.4)
            elif i >= 5:
                trend_strength[i] = short_slope[i]

        return trend_strength

    def forward_pass_labeling(self, prices: np.ndarray) -> np.ndarray:
        """
        First pass: Label from left to right, detecting trends and reversals.
        """
        n = len(prices)
        labels = np.zeros(n, dtype=int)  # 0 = hold, 1 = buy, 2 = sell

        # Calculate smoothed prices
        smoothed = pd.Series(prices).rolling(window=self.smoothing_window,
                                             min_periods=1).mean().values

        # Calculate support/resistance
        support, resistance = self.calculate_support_resistance(prices)

        # Calculate trend strength
        trend_strength = self.detect_trend_strength(smoothed)

        # Track state
        current_trend = 0  # 0 = range, 1 = uptrend, 2 = downtrend
        trend_start_idx = 0
        last_extreme_price = prices[0]
        last_extreme_idx = 0

        for i in range(1, n):
            current_price = prices[i]
            smoothed_price = smoothed[i]

            # Check for consolidation/range
            if i > 10:
                recent_range = np.max(prices[i-10:i+1]) - np.min(prices[i-10:i+1])
                range_pct = recent_range / prices[i-5]

                if range_pct < self.consolidation_threshold:
                    # In consolidation - hold
                    labels[i] = 0
                    current_trend = 0
                    continue

            # Detect potential reversal points
            is_potential_reversal = False

            if current_trend == 1:  # In uptrend
                # Check for bearish reversal
                if (current_price < support[i] * 1.01 and
                    current_price < prices[i-1] and
                    trend_strength[i] < -abs(trend_strength[trend_start_idx]) * 0.5):
                    is_potential_reversal = True

            elif current_trend == 2:  # In downtrend
                # Check for bullish reversal
                if (current_price > resistance[i] * 0.99 and
                    current_price > prices[i-1] and
                    trend_strength[i] > abs(trend_strength[trend_start_idx]) * 0.5):
                    is_potential_reversal = True

            # Determine label based on trend and conditions
            if is_potential_reversal:
                # Wait for confirmation - hold
                labels[i] = 0

                # Check if reversal is confirmed
                if current_trend == 1:  # Was uptrend, now potential downtrend
                    if current_price < last_extreme_price * (1 - self.min_reversal_change):
                        current_trend = 2
                        trend_start_idx = i
                        last_extreme_price = current_price
                        last_extreme_idx = i
                        labels[i] = 2  # Sell

                elif current_trend == 2:  # Was downtrend, now potential uptrend
                    if current_price > last_extreme_price * (1 + self.min_reversal_change):
                        current_trend = 1
                        trend_start_idx = i
                        last_extreme_price = current_price
                        last_extreme_idx = i
                        labels[i] = 1  # Buy

            else:
                # Continue current trend or start new one
                if current_trend == 0:  # No trend established
                    # Check for new trend
                    price_change = (current_price - prices[trend_start_idx]) / prices[trend_start_idx]

                    if abs(price_change) >= self.min_trend_change:
                        if price_change > 0:
                            current_trend = 1
                            labels[i] = 1  # Buy
                        else:
                            current_trend = 2
                            labels[i] = 2  # Sell
                        last_extreme_price = current_price
                        last_extreme_idx = i
                    else:
                        labels[i] = 0  # Hold

                elif current_trend == 1:  # Uptrend
                    # Continue buy if price is making higher highs
                    if current_price >= last_extreme_price:
                        labels[i] = 1  # Buy
                        last_extreme_price = current_price
                        last_extreme_idx = i
                    elif current_price < last_extreme_price * (1 - self.min_reversal_change * 0.7):
                        # Significant pullback in uptrend - hold
                        labels[i] = 0
                    else:
                        # Minor pullback - still buy
                        labels[i] = 1

                elif current_trend == 2:  # Downtrend
                    # Continue sell if price is making lower lows
                    if current_price <= last_extreme_price:
                        labels[i] = 2  # Sell
                        last_extreme_price = current_price
                        last_extreme_idx = i
                    elif current_price > last_extreme_price * (1 + self.min_reversal_change * 0.7):
                        # Significant bounce in downtrend - hold
                        labels[i] = 0
                    else:
                        # Minor bounce - still sell
                        labels[i] = 2

        return labels

    def backward_pass_verification(self, prices: np.ndarray, forward_labels: np.ndarray) -> np.ndarray:
        """
        Second pass: Verify labels from right to left, ensuring consistency.
        """
        n = len(prices)
        backward_labels = forward_labels

        # Calculate smoothed prices for backward analysis
        smoothed = pd.Series(prices).rolling(window=self.smoothing_window,
                                             min_periods=1).mean().values

        # Track from the end
        current_trend = 0
        last_label = backward_labels[-1]
        trend_extreme = prices[-1]

        # Group consecutive labels
        from collections import defaultdict
        label_groups = []
        current_group = []

        for i in range(n):
            if not current_group or backward_labels[i] == current_group[-1][1]:
                current_group.append((i, backward_labels[i]))
            else:
                label_groups.append(current_group)
                current_group = [(i, backward_labels[i])]

        if current_group:
            label_groups.append(current_group)

        # Verify each group makes sense in context
        for group_idx in range(len(label_groups)-1, -1, -1):  # Backward iteration
            group = label_groups[group_idx]
            group_start_idx = group[0][0]
            group_end_idx = group[-1][0]
            group_label = group[0][1]

            if group_label == 0:  # Hold group
                # Check if hold makes sense
                group_prices = prices[group_start_idx:group_end_idx+1]
                price_range = np.max(group_prices) - np.min(group_prices)
                avg_price = np.mean(group_prices)

                if price_range / avg_price > self.min_trend_change:
                    # This hold group has significant movement - might need re-labeling
                    # Check direction
                    price_change = (group_prices[-1] - group_prices[0]) / group_prices[0]

                    if abs(price_change) >= self.min_trend_change:
                        # This should probably not be all holds
                        if price_change > 0:
                            # Should have some buys
                            backward_labels[group_start_idx:group_end_idx+1] = 1
                        else:
                            # Should have some sells
                            backward_labels[group_start_idx:group_end_idx+1] = 2

            elif group_label == 1:  # Buy group
                # Verify buys are in uptrend
                group_prices = prices[group_start_idx:group_end_idx+1]

                # Check if price is actually going up
                if len(group_prices) > 3:
                    # Calculate slope
                    x = np.arange(len(group_prices))
                    coeffs = np.polyfit(x, group_prices, 1)
                    slope = coeffs[0]

                    if slope < 0:  # Price going down during buy labels
                        # Convert to holds or sells based on context
                        # Check next group
                        if group_idx < len(label_groups) - 1:
                            next_group_label = label_groups[group_idx+1][0][1]
                            if next_group_label == 2:  # Next is sell - these might be late sells
                                backward_labels[group_start_idx:group_end_idx+1] = 2
                            else:
                                backward_labels[group_start_idx:group_end_idx+1] = 0

            elif group_label == 2:  # Sell group
                # Verify sells are in downtrend
                group_prices = prices[group_start_idx:group_end_idx+1]

                # Check if price is actually going down
                if len(group_prices) > 3:
                    # Calculate slope
                    x = np.arange(len(group_prices))
                    coeffs = np.polyfit(x, group_prices, 1)
                    slope = coeffs[0]

                    if slope > 0:  # Price going up during sell labels
                        # Convert to holds or buys based on context
                        # Check next group
                        if group_idx < len(label_groups) - 1:
                            next_group_label = label_groups[group_idx+1][0][1]
                            if next_group_label == 1:  # Next is buy - these might be early buys
                                backward_labels[group_start_idx:group_end_idx+1] = 1
                            else:
                                backward_labels[group_start_idx:group_end_idx+1] = 0

        # Final consistency check: ensure no buy immediately followed by sell without hold
        for i in range(1, n-1):
            if backward_labels[i-1] == 1 and backward_labels[i] == 2:
                # Insert hold between buy and sell
                backward_labels[i] = 0
            elif backward_labels[i-1] == 2 and backward_labels[i] == 1:
                # Insert hold between sell and buy
                backward_labels[i] = 0

        return backward_labels

    def conflict_resolution(self, prices: np.ndarray,
                          forward_labels: np.ndarray,
                          backward_labels: np.ndarray) -> np.ndarray:
        """
        Resolve conflicts between forward and backward passes.
        Uses weighted scoring based on price action confirmation.
        """
        n = len(prices)
        final_labels = np.zeros(n, dtype=int)

        # Calculate confidence scores
        confidence_forward = np.zeros(n)
        confidence_backward = np.zeros(n)

        # Calculate moving averages for trend confirmation
        short_ma = pd.Series(prices).rolling(window=5, min_periods=3).mean().values
        medium_ma = pd.Series(prices).rolling(window=15, min_periods=8).mean().values

        for i in range(1, n):
            # Forward confidence based on recent price action
            if forward_labels[i] == 1:  # Buy
                # Check if price is above moving averages
                above_short = prices[i] > short_ma[i] if not np.isnan(short_ma[i]) else True
                above_medium = prices[i] > medium_ma[i] if not np.isnan(medium_ma[i]) else True
                confidence_forward[i] = 0.5 + 0.3 * above_short + 0.2 * above_medium

            elif forward_labels[i] == 2:  # Sell
                # Check if price is below moving averages
                below_short = prices[i] < short_ma[i] if not np.isnan(short_ma[i]) else True
                below_medium = prices[i] < medium_ma[i] if not np.isnan(medium_ma[i]) else True
                confidence_forward[i] = 0.5 + 0.3 * below_short + 0.2 * below_medium
            else:
                confidence_forward[i] = 0.5

            # Backward confidence (similar logic)
            if backward_labels[i] == 1:  # Buy
                above_short = prices[i] > short_ma[i] if not np.isnan(short_ma[i]) else True
                above_medium = prices[i] > medium_ma[i] if not np.isnan(medium_ma[i]) else True
                confidence_backward[i] = 0.5 + 0.3 * above_short + 0.2 * above_medium
            elif backward_labels[i] == 2:  # Sell
                below_short = prices[i] < short_ma[i] if not np.isnan(short_ma[i]) else True
                below_medium = prices[i] < medium_ma[i] if not np.isnan(medium_ma[i]) else True
                confidence_backward[i] = 0.5 + 0.3 * below_short + 0.2 * below_medium
            else:
                confidence_backward[i] = 0.5

        # Resolve conflicts
        for i in range(n):
            if forward_labels[i] == backward_labels[i]:
                final_labels[i] = forward_labels[i]
            else:
                # Conflict - use higher confidence
                if confidence_forward[i] > confidence_backward[i]:
                    final_labels[i] = forward_labels[i]
                elif confidence_backward[i] > confidence_forward[i]:
                    final_labels[i] = backward_labels[i]
                else:
                    # Equal confidence - check neighbors
                    if i > 0:
                        final_labels[i] = final_labels[i-1]
                    else:
                        final_labels[i] = 0  # Default to hold

        return final_labels

    def post_process_labels(self, prices: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Final cleanup: remove noise, ensure minimum trend duration.
        """
        n = len(prices)
        processed = labels

        # Remove isolated labels (single point trends)
        for i in range(1, n-1):
            if labels[i] != labels[i-1] and labels[i] != labels[i+1]:
                # Single point different from neighbors
                processed[i] = labels[i-1]

        # Ensure minimum trend duration of 3 periods
        current_label = processed[0]
        streak_start = 0

        for i in range(1, n):
            if processed[i] != current_label:
                streak_length = i - streak_start
                if streak_length < 3 and current_label != 0:
                    # Short trend - convert to holds
                    processed[streak_start:i] = 0

                current_label = processed[i]
                streak_start = i

        # Check last streak
        if n - streak_start < 3 and current_label != 0:
            processed[streak_start:] = 0

        # Smooth transitions
        for i in range(1, n-1):
            if processed[i-1] == processed[i+1] and processed[i] != processed[i-1]:
                # Point different from both neighbors
                processed[i] = processed[i-1]

        return processed

    def label(self, df: pd.DataFrame, price_column: str = 'close') -> pd.DataFrame:
        """
        Main labeling function with forward and backward passes.

        Args:
            df: DataFrame with price data
            price_column: Column name for prices

        Returns:
            DataFrame with added 'label' column
        """
        df = df.reset_index(drop=True)

        if price_column not in df.columns:
            raise ValueError(f"Column '{price_column}' not found in DataFrame")

        prices = df[price_column].astype(float).values

        # Step 1: Forward pass
        print("Running forward pass...")
        forward_labels = self.forward_pass_labeling(prices)

        # Step 2: Backward pass for verification
        print("Running backward verification pass...")
        backward_labels = self.backward_pass_verification(prices, forward_labels)

        # Step 3: Conflict resolution
        print("Resolving conflicts...")
        final_labels = self.conflict_resolution(prices, forward_labels, backward_labels)

        # Step 4: Post-processing
        print("Post-processing labels...")
        final_labels = self.post_process_labels(prices, final_labels)

        # Add to dataframe
        df['label'] = final_labels

        # Add label names
        label_names = {0: 'hold', 1: 'buy', 2: 'sell'}
        df['label_name'] = df['label'].map(label_names)

        return df




def plot_signals(df):
    """Plot price chart with buy/sell signals"""
    fig, ax = plt.subplots(figsize=(15, 8))

    # Plot close price
    ax.plot(df.index, df['close'], label='Close Price', color='blue', alpha=0.6)



    # Plot buy signals
    buy_signals = df[df['label'] == 1]
    ax.scatter(buy_signals.index, buy_signals['close'],
               color='green', marker='^', s=100, label='Buy Signal', zorder=5)

    # Plot sell signals
    sell_signals = df[df['label'] == 2]
    ax.scatter(sell_signals.index, sell_signals['close'],
               color='red', marker='v', s=100, label='Sell Signal', zorder=5)

    ax.set_xlabel('Index')
    ax.set_ylabel('Price')
    ax.set_title('Trading Signals: Buy and Sell Points')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('trading_signals.png', dpi=300)
    plt.show()

    print(f"\nSignal Distribution:")
    print(f"Hold/No Action (0): {len(df[df['label'] == 0])}")
    print(f"Buy Signals (1): {len(df[df['label'] == 1])}")
    print(f"Sell Signals (2): {len(df[df['label'] == 2])}")




import matplotlib.pyplot as plt

def predict_and_plot_signals(model, df, feature_columns):
    """
    Runs model predictions on the dataframe and plots signals over the close price.
    """
    # 1. Prepare features and make predictions
    X = df[feature_columns].values
    df['predicted_label'] = model.predict(X)
    
    # 2. Setup the plot
    plt.figure(figsize=(16, 8))
    
    # Plot the actual Close price line
    plt.plot(df.index, df['close'], label='Close Price', color='blue', alpha=0.5, linewidth=1)
    
    # 3. Identify and plot Buy signals (Label 1)
    buys = df[df['predicted_label'] == 1]
    plt.scatter(buys.index, buys['close'], 
                color='green', marker='^', s=100, label='Predicted BUY', zorder=5)
    
    # 4. Identify and plot Sell signals (Label 2)
    sells = df[df['predicted_label'] == 2]
    plt.scatter(sells.index, sells['close'], 
                color='red', marker='v', s=100, label='Predicted SELL', zorder=5)
    
    # Formatting the graph
    plt.title('Market Predictions: Buy/Sell Signals on Close Price', fontsize=14)
    plt.xlabel('Index / Time', fontsize=12)
    plt.ylabel('Price', fontsize=12)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

    print(f"\nSignal Distribution:")
    print(f"Hold/No Action (0): {len(df[df['predicted_label'] == 0])}")
    print(f"Buy Signals (1): {len(df[df['predicted_label'] == 1])}")
    print(f"Sell Signals (2): {len(df[df['predicted_label'] == 2])}")

# --- Example Usage ---
# Ensure your 'df' has gone through engineer_features() first
# features = ['open', 'high', 'low', 'volume', 'month', 'day', 'hour', 'minute', 'day_of_week']


"""
RobustPriceLabelerV3  —  Balanced Multi-Feature Labeler
========================================================
Uses a zigzag-based zone approach so every bar is actively assigned
to a rising leg (BUY), falling leg (SELL), or pivot transition (HOLD).
 
Target distribution: ~35% buy | ~35% sell | ~30% hold
Volume / session / candle features become CONFIDENCE scores,
not hard gates that push everything into HOLD.
 
Features used:
    open, high, low, close, volume,
    hour, minute, day, month, day_of_week
"""
 

# ══════════════════════════════════════════════════════════
#  ATR
# ══════════════════════════════════════════════════════════
 
def _atr(high: np.ndarray, low: np.ndarray,
         close: np.ndarray, period: int = 14) -> np.ndarray:
    n  = len(close)
    tr = np.empty(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i - 1]),
                    abs(low[i]  - close[i - 1]))
    return pd.Series(tr).rolling(period, min_periods=1).mean().values
 
 
# ══════════════════════════════════════════════════════════
#  Zigzag pivot finder  (ATR-adaptive threshold)
# ══════════════════════════════════════════════════════════
 
def _zigzag_pivots(high: np.ndarray, low: np.ndarray,
                   atr: np.ndarray,
                   atr_mult: float = 0.5
                   ) -> List[Tuple[int, float, str]]:
    """
    Returns a list of (index, price, 'H'|'L') pivot points.
    A new pivot is confirmed when price reverses by ≥ atr_mult × ATR
    from the last extreme.
    """
    n       = len(high)
    pivots  = []
    trend   = 0          # +1 looking for high, -1 looking for low
    ext_idx = 0
    ext_val = (high[0] + low[0]) / 2
 
    for i in range(1, n):
        threshold = atr[i] * atr_mult
 
        if trend >= 0:                       # looking for higher high
            if high[i] >= ext_val:
                ext_val = high[i]
                ext_idx = i
            elif ext_val - low[i] >= threshold:
                # confirmed swing HIGH at ext_idx
                pivots.append((ext_idx, ext_val, 'H'))
                trend   = -1
                ext_val = low[i]
                ext_idx = i
        else:                               # looking for lower low
            if low[i] <= ext_val:
                ext_val = low[i]
                ext_idx = i
            elif high[i] - ext_val >= threshold:
                # confirmed swing LOW at ext_idx
                pivots.append((ext_idx, ext_val, 'L'))
                trend   = 1
                ext_val = high[i]
                ext_idx = i
 
    # Close out last pivot
    if trend >= 0:
        pivots.append((ext_idx, ext_val, 'H'))
    else:
        pivots.append((ext_idx, ext_val, 'L'))
 
    # Remove duplicate consecutive types
    cleaned = [pivots[0]]
    for p in pivots[1:]:
        if p[2] != cleaned[-1][2]:
            cleaned.append(p)
        else:
            # Keep the more extreme one
            if p[2] == 'H' and p[1] > cleaned[-1][1]:
                cleaned[-1] = p
            elif p[2] == 'L' and p[1] < cleaned[-1][1]:
                cleaned[-1] = p
 
    return cleaned
 
 
# ══════════════════════════════════════════════════════════
#  Zone labeling from pivots
# ══════════════════════════════════════════════════════════
 
def _zone_labels(n: int,
                 pivots: List[Tuple[int, float, str]],
                 hold_bars: int = 2) -> np.ndarray:
    """
    Assign labels based on which zigzag leg each bar belongs to.
 
    L→H leg  = BUY  (1)
    H→L leg  = SELL (2)
    ±hold_bars around each pivot = HOLD (0)
    """
    labels = np.zeros(n, dtype=int)
 
    for k in range(len(pivots) - 1):
        p1_idx, _, p1_type = pivots[k]
        p2_idx, _, _       = pivots[k + 1]
 
        direction = 1 if p1_type == 'L' else 2   # L→H = buy, H→L = sell
 
        seg_start = p1_idx + hold_bars + 1
        seg_end   = p2_idx - hold_bars
 
        if seg_start <= seg_end:
            labels[seg_start: seg_end + 1] = direction
 
    # The very last pivot forward → replicate the last segment direction
    if len(pivots) >= 2:
        last_p_idx  = pivots[-1][0]
        last_p_type = pivots[-1][2]
        last_dir    = 2 if last_p_type == 'H' else 1   # after H expect sell, after L expect buy
        tail_start  = last_p_idx + hold_bars + 1
        if tail_start < n:
            labels[tail_start:] = last_dir
 
    return labels
 
 
# ══════════════════════════════════════════════════════════
#  Confidence scoring  (does NOT affect label direction)
# ══════════════════════════════════════════════════════════
 
def _confidence(df: pd.DataFrame,
                atr: np.ndarray,
                labels: np.ndarray) -> np.ndarray:
    """
    Returns a 0-1 confidence value per bar based on:
      - Volume z-score
      - Session quality (hour / day_of_week)
      - Candle body ratio in the direction of the label
    """
    n  = len(labels)
    o  = df["open"].astype(float).values
    h  = df["high"].astype(float).values
    l  = df["low"].astype(float).values
    c  = df["close"].astype(float).values
    v  = df["volume"].astype(float).values
 
    hour = df["hour"].astype(int).values if "hour" in df.columns else np.full(n, 12)
    dow  = df["day_of_week"].astype(int).values if "day_of_week" in df.columns else np.ones(n, dtype=int)
 
    # ── volume z-score ───────────────────────────────────────
    vol_s  = pd.Series(v)
    vol_mu = vol_s.rolling(20, min_periods=5).mean()
    vol_sd = vol_s.rolling(20, min_periods=5).std().replace(0, np.nan)
    vol_z  = ((vol_s - vol_mu) / vol_sd).fillna(0).values
    vol_c  = np.clip((vol_z + 2) / 4, 0, 1)   # rescale to [0,1]
 
    # ── session score ─────────────────────────────────────────
    sess = np.full(n, 0.65)
    sess[(hour >= 13) & (hour <= 17)] = 1.00   # London+NY overlap
    sess[(hour >= 8)  & (hour < 13)]  = 0.85   # London open
    sess[(hour >= 18) & (hour <= 20)] = 0.75   # NY afternoon
    sess[(hour >= 0)  & (hour <= 7)]  = 0.45   # Asian low-liq
    sess[dow == 4]  *= 0.85                    # Friday discount
    sess[dow >= 5]  *= 0.30                    # Weekend
 
    # ── candle body in direction of label ─────────────────────
    hl_range  = np.where(h - l > 0, h - l, np.nan)
    body      = c - o
    body_bull = np.clip(body / hl_range, 0, 1)   # 1 = full bullish body
    body_bear = np.clip(-body / hl_range, 0, 1)  # 1 = full bearish body
    body_bull = np.nan_to_num(body_bull)
    body_bear = np.nan_to_num(body_bear)
 
    body_align = np.where(labels == 1, body_bull,
                 np.where(labels == 2, body_bear, 0.5))
 
    conf = 0.35 * vol_c + 0.35 * sess + 0.30 * body_align
    return np.round(np.clip(conf, 0, 1), 3)
 
 
# ══════════════════════════════════════════════════════════
#  Balance enforcer  (post-processing)
# ══════════════════════════════════════════════════════════
 
def _enforce_balance(labels: np.ndarray,
                     high: np.ndarray,
                     low: np.ndarray,
                     close: np.ndarray,
                     atr: np.ndarray,
                     target_hold_pct: float = 0.30,
                     min_streak: int = 2) -> np.ndarray:
    """
    Two-stage balance enforcer:
 
    Stage 1 — Remove short hold streaks that interrupt clear trends.
      If a hold run ≤ min_streak sits between two same-direction
      segments, absorb it into that segment.
 
    Stage 2 — If hold% still exceeds target, promote the longest
      hold runs that have a clear directional bias.
    """
    arr = labels.copy()
    n   = len(arr)
 
    # ── Stage 1: absorb tiny hold islands ─────────────────────
    changed = True
    while changed:
        changed = False
        i = 0
        while i < n:
            j = i
            while j < n and arr[j] == arr[i]:
                j += 1
            run_lbl = int(arr[i])
            run_len = j - i
 
            if run_lbl == 0 and run_len <= min_streak:
                prev_lbl = int(arr[i - 1]) if i > 0 else -1
                next_lbl = int(arr[j])     if j < n else -1
 
                if prev_lbl != 0 and prev_lbl == next_lbl:
                    arr[i:j] = prev_lbl
                    changed  = True
                elif prev_lbl != 0 and next_lbl == 0:
                    arr[i:j] = prev_lbl
                    changed  = True
                elif next_lbl != 0 and prev_lbl == 0:
                    arr[i:j] = next_lbl
                    changed  = True
            i = j
 
    # ── Stage 2: promote long hold runs with directional bias ─
    hold_pct = np.mean(arr == 0)
    if hold_pct > target_hold_pct:
        # Collect hold runs sorted by length (largest first)
        hold_runs = []
        i = 0
        while i < n:
            j = i
            while j < n and arr[j] == arr[i]:
                j += 1
            if arr[i] == 0:
                hold_runs.append((j - i, i, j))
            i = j
 
        hold_runs.sort(reverse=True)
 
        for run_len, s, e in hold_runs:
            if np.mean(arr == 0) <= target_hold_pct:
                break
            seg_close = close[s:e]
            seg_atr   = np.mean(atr[s:e])
            move      = seg_close[-1] - seg_close[0]
            # Only promote if the move is > 0.5 ATR (genuine direction)
            if abs(move) >= seg_atr * 0.5:
                new_lbl = 1 if move > 0 else 2
                arr[s:e] = new_lbl
 
    # ── Final: no immediate BUY→SELL or SELL→BUY ─────────────
    for i in range(1, n - 1):
        if arr[i - 1] == 1 and arr[i] == 2:
            arr[i] = 0
        elif arr[i - 1] == 2 and arr[i] == 1:
            arr[i] = 0
 
    return arr
 
 
# ══════════════════════════════════════════════════════════
#  Main class
# ══════════════════════════════════════════════════════════
 
class RobustPriceLabelerV3:
    """
    Balanced zigzag-zone labeler with full OHLCV + time feature support.
 
    Key design principle
    --------------------
    Hold is ONLY used at pivot transition zones (±hold_bars around each
    swing point) and in flat/ambiguous periods.  All trend legs are
    labeled buy or sell.  This naturally produces balanced classes.
 
    Parameters
    ----------
    atr_period          : ATR look-back (default 14)
    zigzag_atr_mult     : min reversal in ATR units to confirm a new pivot
                          Lower → more pivots → more balanced but noisy.
                          Higher → fewer pivots → smoother but may over-hold.
                          Good starting range: 0.3 – 1.0
    hold_bars           : bars around each pivot marked as HOLD (transition zone)
                          2–4 is typical
    min_streak          : minimum consecutive bars to keep a label
    target_hold_pct     : maximum desired hold fraction (balance enforcer)
    """
 
    def __init__(
        self,
        atr_period: int = 14,
        zigzag_atr_mult: float = 0.5,
        hold_bars: int = 2,
        min_streak: int = 2,
        target_hold_pct: float = 0.30,
    ):
        self.atr_period       = atr_period
        self.zigzag_atr_mult  = zigzag_atr_mult
        self.hold_bars        = hold_bars
        self.min_streak       = min_streak
        self.target_hold_pct  = target_hold_pct
 
    # ─────────────────────────────────────────────────────────
 
    def label(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Label each bar as buy (1), sell (2), or hold (0).
 
        Required columns : open, high, low, close, volume
        Optional columns : hour, minute, day, month, day_of_week
 
        Returns
        -------
        df (copy) with new columns:
            label       int   0=hold 1=buy 2=sell
            label_name  str
            label_conf  float 0-1 quality confidence
        """
        required = {"open", "high", "low", "close", "volume"}
        missing  = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
 
        df  = df.reset_index(drop=True).copy()
        n   = len(df)
        h   = df["high"].astype(float).values
        l   = df["low"].astype(float).values
        c   = df["close"].astype(float).values
 
        # ── 1. ATR ──────────────────────────────────────────────
        print("Computing ATR...")
        atr = _atr(h, l, c, self.atr_period)
 
        # ── 2. Zigzag pivots ─────────────────────────────────────
        print("Finding zigzag pivots...")
        pivots = _zigzag_pivots(h, l, atr, self.zigzag_atr_mult)
        print(f"   {len(pivots)} pivots found")
 
        # ── 3. Zone labels ───────────────────────────────────────
        print("Assigning zone labels...")
        raw_labels = _zone_labels(n, pivots, self.hold_bars)
 
        # ── 4. Balance enforcer ──────────────────────────────────
        print("Enforcing class balance...")
        balanced = _enforce_balance(
            raw_labels, h, l, c, atr,
            target_hold_pct=self.target_hold_pct,
            min_streak=self.min_streak,
        )
 
        # ── 5. Confidence scores ─────────────────────────────────
        print("Computing confidence scores...")
        conf = _confidence(df, atr, balanced)
 
        # ── 6. Attach to dataframe ───────────────────────────────
        df["label"]      = balanced
        df["label_name"] = pd.Series(balanced).map({0:"hold", 1:"buy", 2:"sell"})
        df["label_conf"] = conf
 
        # ── 7. Distribution report ───────────────────────────────
        vc    = df["label_name"].value_counts()
        total = len(df)
        print("\n── Label Distribution ──────────────────────────")
        for lbl in ["buy", "sell", "hold"]:
            cnt = vc.get(lbl, 0)
            bar = "█" * int(cnt / total * 40)
            print(f"  {lbl:4s}: {cnt:6,d}  ({cnt/total*100:5.1f}%)  {bar}")
        print("────────────────────────────────────────────────")
 
        return df
 
    # ─────────────────────────────────────────────────────────
    #  Convenience: return only the pivot series for plotting
    # ─────────────────────────────────────────────────────────
 
    def get_pivots(self, df: pd.DataFrame):
        """
        Returns a DataFrame of swing pivot points for charting.
 
        Columns: index, price, type ('H' or 'L')
        """
        h   = df["high"].astype(float).values
        l   = df["low"].astype(float).values
        c   = df["close"].astype(float).values
        atr = _atr(h, l, c, self.atr_period)
        pivots = _zigzag_pivots(h, l, atr, self.zigzag_atr_mult)
        return pd.DataFrame(pivots, columns=["bar_index", "price", "type"])
 
### Major upgrade has been done 

"""
RobustPriceLabelerV4  —  Zone-Aware, Forward-Confirmed Labeler
==============================================================
Designed for 5-minute Gold (XAUUSD) data.

Root problems fixed vs V3
--------------------------
Problem 1 — BUY labels near resistance zones:
    The zigzag assigns BUY to the entire L→H leg, including bars very close
    to the eventual swing HIGH (resistance). Those bars get reversed in live
    trading, causing losses.

Problem 2 — SELL labels near support zones:
    Same issue in reverse — the H→L leg includes bars near the eventual
    swing LOW (support) where price bounces back up.

Solutions
---------
1. S/R Zone Filter  (BACKWARD confirmation)
   ──────────────────────────────────────────
   After each confirmed pivot is recorded, its price level is tracked as
   active resistance (swing HIGH) or active support (swing LOW).  Before
   assigning a directional label, we check:

     • BUY label at bar i:
         If close[i] is within zone_mult × ATR[i] of any active resistance
         AND price has not already broken above that level (no breakout) →
         demote to HOLD.

     • SELL label at bar i:
         If close[i] is within zone_mult × ATR[i] of any active support
         AND price has not already broken below that level (no breakdown) →
         demote to HOLD.

   The last `sr_lookback` swing highs and lows are tracked, so the filter
   catches multi-level zone clusters common in Gold.

2. Triple-Barrier Forward Confirmation  (FORWARD labeling)
   ─────────────────────────────────────────────────────────
   Every surviving directional label must prove itself by checking what
   actually happens in the next `max_confirm_bars` bars:

     • BUY at bar i is CONFIRMED only if:
           high[j] >= close[i] + profit_mult x ATR[i]   (TP hit)
         before
           low[j]  <= close[i] - stop_mult  x ATR[i]   (SL hit)
         before
           bar i + max_confirm_bars                      (time barrier)

     • SELL at bar i is CONFIRMED only if:
           low[j]  <= close[i] - profit_mult x ATR[i]
         before high[j] >= close[i] + stop_mult x ATR[i]

   Unconfirmed labels are demoted to HOLD.  These bars are LOCKED so the
   balance enforcer cannot re-promote them.

3. Optional Backward EMA Momentum Gate
   ──────────────────────────────────────
   Prevents buying in a confirmed downtrend or selling in a confirmed
   uptrend by checking EMA(fast) vs EMA(slow) alignment.
   Disabled by default (use_momentum_gate=False).

Label pipeline
--------------
  ATR
  -> Zigzag pivots
  -> Zone-based direction labels  (L->H = BUY, H->L = SELL, +/-hold_bars = HOLD)
  -> S/R zone filter              (backward confirmation, produces locked mask A)
  -> Optional EMA momentum gate   (backward trend filter, adds to locked mask)
  -> Triple-barrier filter        (forward confirmation, produces locked mask B)
  -> Balance enforcer             (lock-aware - never re-promotes locked bars)
  -> Confidence scoring

Label encoding
--------------
  0 = HOLD  |  1 = BUY  |  2 = SELL
"""

import numpy as np
import pandas as pd
from collections import deque
from typing import List, Tuple, Optional


# ======================================================
#  ATR
# ======================================================

def _atr(high: np.ndarray, low: np.ndarray,
         close: np.ndarray, period: int = 14) -> np.ndarray:
    n  = len(close)
    tr = np.empty(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i - 1]),
                    abs(low[i]  - close[i - 1]))
    return pd.Series(tr).rolling(period, min_periods=1).mean().values


# ======================================================
#  Zigzag pivot finder  (ATR-adaptive threshold)
# ======================================================

def _zigzag_pivots(high: np.ndarray, low: np.ndarray,
                   atr: np.ndarray,
                   atr_mult: float = 0.5
                   ) -> List[Tuple[int, float, str]]:
    """
    Returns a list of (index, price, 'H'|'L') confirmed swing pivots.
    A new pivot is only confirmed when price reverses >= atr_mult x ATR
    from the last extreme, giving built-in backward context per pivot.
    """
    n       = len(high)
    pivots  = []
    trend   = 0
    ext_idx = 0
    ext_val = (high[0] + low[0]) / 2

    for i in range(1, n):
        threshold = atr[i] * atr_mult

        if trend >= 0:                          # seeking higher high
            if high[i] >= ext_val:
                ext_val, ext_idx = high[i], i
            elif ext_val - low[i] >= threshold:
                pivots.append((ext_idx, ext_val, 'H'))
                trend, ext_val, ext_idx = -1, low[i], i
        else:                                   # seeking lower low
            if low[i] <= ext_val:
                ext_val, ext_idx = low[i], i
            elif high[i] - ext_val >= threshold:
                pivots.append((ext_idx, ext_val, 'L'))
                trend, ext_val, ext_idx = 1, high[i], i

    # Close out the last unconfirmed pivot
    pivots.append((ext_idx, ext_val, 'H' if trend >= 0 else 'L'))

    # Remove duplicate consecutive types -- keep more extreme
    cleaned = [pivots[0]]
    for p in pivots[1:]:
        if p[2] != cleaned[-1][2]:
            cleaned.append(p)
        elif p[2] == 'H' and p[1] > cleaned[-1][1]:
            cleaned[-1] = p
        elif p[2] == 'L' and p[1] < cleaned[-1][1]:
            cleaned[-1] = p

    return cleaned


# ======================================================
#  Zone labels from pivots
# ======================================================

def _zone_labels(n: int,
                 pivots: List[Tuple[int, float, str]],
                 hold_bars: int = 2) -> np.ndarray:
    """
    Assign initial directional labels from zigzag legs:
      L -> H  =  BUY  (1)
      H -> L  =  SELL (2)
      +/- hold_bars around each pivot  =  HOLD (0)
    """
    labels = np.zeros(n, dtype=int)

    for k in range(len(pivots) - 1):
        p1_idx, _, p1_type = pivots[k]
        p2_idx, _, _       = pivots[k + 1]
        direction = 1 if p1_type == 'L' else 2
        seg_start = p1_idx + hold_bars + 1
        seg_end   = p2_idx - hold_bars
        if seg_start <= seg_end:
            labels[seg_start:seg_end + 1] = direction

    # Extend beyond the last confirmed pivot
    if len(pivots) >= 2:
        last_p_idx  = pivots[-1][0]
        last_p_type = pivots[-1][2]
        last_dir    = 2 if last_p_type == 'H' else 1
        tail_start  = last_p_idx + hold_bars + 1
        if tail_start < n:
            labels[tail_start:] = last_dir

    return labels


# ======================================================
#  NEW -- Build S/R level arrays  (purely backward-looking)
# ======================================================

def _build_sr_levels(pivots: List[Tuple[int, float, str]],
                     n: int,
                     lookback: int = 3
                     ) -> Tuple[List[List[float]], List[List[float]]]:
    """
    For each bar i, maintain the last `lookback` confirmed:
      - swing HIGH prices  ->  resistance levels
      - swing LOW  prices  ->  support levels

    Only pivots at index <= i are visible (zero lookahead bias).
    Resistance list is sorted descending (nearest-above first).
    Support    list is sorted ascending  (nearest-below first).

    Returns
    -------
    resistance : list[list[float]]   length = n
    support    : list[list[float]]   length = n
    """
    highs_buf = deque(maxlen=lookback)
    lows_buf  = deque(maxlen=lookback)

    pivot_map: dict = {}
    for idx, price, ptype in sorted(pivots, key=lambda x: x[0]):
        pivot_map.setdefault(idx, []).append((price, ptype))

    resistance: List[List[float]] = [[] for _ in range(n)]
    support:    List[List[float]] = [[] for _ in range(n)]

    for i in range(n):
        if i in pivot_map:
            for price, ptype in pivot_map[i]:
                if ptype == 'H':
                    highs_buf.append(price)
                else:
                    lows_buf.append(price)
        # Nearest-above resistance first; nearest-below support first
        resistance[i] = sorted(highs_buf, reverse=True)
        support[i]    = sorted(lows_buf)

    return resistance, support


# ======================================================
#  NEW -- S/R Zone Filter  (BACKWARD confirmation)
# ======================================================

def _filter_sr_zones(labels:     np.ndarray,
                     close:      np.ndarray,
                     resistance: List[List[float]],
                     support:    List[List[float]],
                     atr:        np.ndarray,
                     zone_mult:  float = 1.0
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Demote directional labels that are in conflict with S/R zones.

    BUY at bar i -> HOLD if:
      Any resistance level r satisfies:  0 < (r - close[i]) <= zone_mult x ATR[i]
      i.e. price is approaching resistance from below (reversal risk high).
      EXCEPTION: if close[i] > r (price has broken ABOVE) the label is KEPT --
      breakout entries are valid and desirable.

    SELL at bar i -> HOLD if:
      Any support level s satisfies:  0 < (close[i] - s) <= zone_mult x ATR[i]
      i.e. price is approaching support from above (bounce risk high).
      EXCEPTION: if close[i] < s (price has broken BELOW) the label is KEPT --
      breakdown entries are valid and desirable.

    Returns
    -------
    filtered : np.ndarray  -- updated labels
    locked   : np.ndarray  -- bool mask; True = permanently HOLD (zone conflict)
    """
    filtered = labels.copy()
    locked   = np.zeros(len(labels), dtype=bool)
    n        = len(labels)

    for i in range(n):
        zone = atr[i] * zone_mult
        c    = close[i]

        if filtered[i] == 1:           # BUY candidate
            for r in resistance[i]:
                gap = r - c            # >0 -> below resistance | <=0 -> breakout above
                if 0 < gap <= zone:
                    # Approaching resistance, has NOT broken out -> demote
                    filtered[i] = 0
                    locked[i]   = True
                    break

        elif filtered[i] == 2:         # SELL candidate
            for s in support[i]:
                gap = c - s            # >0 -> above support | <=0 -> breakdown below
                if 0 < gap <= zone:
                    # Approaching support, has NOT broken down -> demote
                    filtered[i] = 0
                    locked[i]   = True
                    break

    return filtered, locked


# ======================================================
#  NEW -- Triple-Barrier Forward Confirmation  (FORWARD labeling)
# ======================================================

def _forward_triple_barrier(labels:      np.ndarray,
                             close:       np.ndarray,
                             high:        np.ndarray,
                             low:         np.ndarray,
                             atr:         np.ndarray,
                             profit_mult: float = 1.5,
                             stop_mult:   float = 1.0,
                             max_bars:    int   = 20
                             ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Validate each directional label by observing the next max_bars of price
    action -- the triple-barrier method (Lopez de Prado, AFML ch.3).

    BUY at bar i is CONFIRMED if, within the next max_bars bars:
        high[j] >= close[i] + profit_mult x ATR[i]   (take-profit hit FIRST)
      BEFORE
        low[j]  <= close[i] - stop_mult  x ATR[i]   (stop-loss hit)
      BEFORE
        bar i + max_bars                              (time barrier)

    SELL at bar i is CONFIRMED if:
        low[j]  <= close[i] - profit_mult x ATR[i]   (take-profit hit FIRST)
      BEFORE
        high[j] >= close[i] + stop_mult  x ATR[i]   (stop-loss hit)
      BEFORE
        bar i + max_bars                              (time barrier)

    Unconfirmed -> HOLD (0).  These bars are added to the `locked` mask so
    the balance enforcer does not re-promote them.

    Performance note: For large datasets (>200k bars), consider wrapping the
    inner loop in Numba (@njit) to reduce runtime from O(n*max_bars) scans.

    Returns
    -------
    confirmed : np.ndarray  -- updated labels
    locked    : np.ndarray  -- bool mask; True = forward-unconfirmed bar
    """
    confirmed = labels.copy()
    locked    = np.zeros(len(labels), dtype=bool)
    n         = len(labels)

    for i in range(n - 1):
        if labels[i] == 0:
            continue

        end = min(i + max_bars + 1, n)

        if labels[i] == 1:             # BUY forward check
            tp  = close[i] + profit_mult * atr[i]
            sl  = close[i] - stop_mult   * atr[i]
            hit = 'none'
            for j in range(i + 1, end):
                if high[j] >= tp:
                    hit = 'tp'; break
                if low[j]  <= sl:
                    hit = 'sl'; break
            if hit != 'tp':
                confirmed[i] = 0
                locked[i]    = True

        else:                          # SELL forward check
            tp  = close[i] - profit_mult * atr[i]
            sl  = close[i] + stop_mult   * atr[i]
            hit = 'none'
            for j in range(i + 1, end):
                if low[j]  <= tp:
                    hit = 'tp'; break
                if high[j] >= sl:
                    hit = 'sl'; break
            if hit != 'tp':
                confirmed[i] = 0
                locked[i]    = True

    return confirmed, locked


# ======================================================
#  NEW -- Backward EMA Momentum Gate  (optional)
# ======================================================

def _backward_ema_gate(labels:    np.ndarray,
                       close:     np.ndarray,
                       atr:       np.ndarray,
                       ema_fast:  int   = 8,
                       ema_slow:  int   = 21,
                       slope_tol: float = 0.5
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Optional backward trend-alignment gate using dual EMAs.

    A BUY label in a confirmed downtrend (EMA_fast significantly below
    EMA_slow) is demoted to HOLD.  A SELL label in a confirmed uptrend
    (EMA_fast significantly above EMA_slow) is demoted to HOLD.

    slope_tol controls how large the EMA separation must be (in ATR units)
    before the gate fires.  slope_tol=0 is strictest; higher = more lenient.

    Default EMAs 8/21 suit intraday 5-min Gold well.  Adjust for different
    timeframes.

    Returns
    -------
    filtered : np.ndarray
    locked   : np.ndarray
    """
    close_s = pd.Series(close)
    ema_f   = close_s.ewm(span=ema_fast, adjust=False).mean().values
    ema_s   = close_s.ewm(span=ema_slow, adjust=False).mean().values

    filtered = labels.copy()
    locked   = np.zeros(len(labels), dtype=bool)

    for i in range(len(labels)):
        thresh = atr[i] * slope_tol

        if labels[i] == 1:             # BUY needs fast > slow (or near-equal)
            if ema_f[i] < ema_s[i] - thresh:
                filtered[i] = 0
                locked[i]   = True

        elif labels[i] == 2:           # SELL needs fast < slow (or near-equal)
            if ema_f[i] > ema_s[i] + thresh:
                filtered[i] = 0
                locked[i]   = True

    return filtered, locked


# ======================================================
#  Balance enforcer  (lock-aware)
# ======================================================

def _enforce_balance(labels:          np.ndarray,
                     high:            np.ndarray,
                     low:             np.ndarray,
                     close:           np.ndarray,
                     atr:             np.ndarray,
                     target_hold_pct: float = 0.30,
                     min_streak:      int   = 2,
                     locked:          Optional[np.ndarray] = None
                     ) -> np.ndarray:
    """
    Two-stage balance enforcer -- fully respects the locked mask.

    Bars in `locked` are permanently HOLD and are never re-promoted
    regardless of balance pressure.  This ensures the zone/barrier filters
    are never undermined by class balancing.

    Stage 1: Absorb tiny hold islands (length <= min_streak) that sit
             between two same-direction segments.  Island bars must all be
             unlocked.

    Stage 2: If hold% still exceeds target, promote the longest unlocked
             hold runs that show a clear directional move (>= 0.5 ATR).
    """
    arr    = labels.copy()
    n      = len(arr)
    if locked is None:
        locked = np.zeros(n, dtype=bool)

    # -- Stage 1: absorb short hold islands ------------------
    changed = True
    while changed:
        changed = False
        i = 0
        while i < n:
            j = i
            while j < n and arr[j] == arr[i]:
                j += 1
            run_lbl = int(arr[i])
            run_len = j - i

            if run_lbl == 0 and run_len <= min_streak:
                prev_lbl = int(arr[i - 1]) if i > 0 else -1
                next_lbl = int(arr[j])     if j < n else -1
                # Only absorb if no bar in the island is locked
                if not np.any(locked[i:j]):
                    if prev_lbl != 0 and prev_lbl == next_lbl:
                        arr[i:j] = prev_lbl; changed = True
                    elif prev_lbl != 0 and next_lbl == 0:
                        arr[i:j] = prev_lbl; changed = True
                    elif next_lbl != 0 and prev_lbl == 0:
                        arr[i:j] = next_lbl; changed = True
            i = j

    # -- Stage 2: promote directionally biased hold runs -----
    hold_pct = np.mean(arr == 0)
    if hold_pct > target_hold_pct:
        hold_runs = []
        i = 0
        while i < n:
            j = i
            while j < n and arr[j] == arr[i]:
                j += 1
            if arr[i] == 0:
                hold_runs.append((j - i, i, j))
            i = j
        hold_runs.sort(reverse=True)

        for _, s, e in hold_runs:
            if np.mean(arr == 0) <= target_hold_pct:
                break
            if np.any(locked[s:e]):           # never touch locked bars
                continue
            seg_close = close[s:e]
            seg_atr   = np.mean(atr[s:e])
            move      = seg_close[-1] - seg_close[0]
            if abs(move) >= seg_atr * 0.5:
                arr[s:e] = 1 if move > 0 else 2

    # -- Final: prevent abrupt BUY <-> SELL transitions -------
    for i in range(1, n - 1):
        if arr[i - 1] == 1 and arr[i] == 2:
            arr[i] = 0
        elif arr[i - 1] == 2 and arr[i] == 1:
            arr[i] = 0

    return arr


# ======================================================
#  Confidence scoring  (unchanged from V3)
# ======================================================

def _confidence(df: pd.DataFrame,
                atr: np.ndarray,
                labels: np.ndarray) -> np.ndarray:
    """
    Returns a 0-1 confidence value per bar based on:
      - Volume z-score       (0.35 weight)
      - Session quality      (0.35 weight)
      - Candle body alignment(0.30 weight)
    """
    n   = len(labels)
    o   = df["open"].astype(float).values
    h   = df["high"].astype(float).values
    l   = df["low"].astype(float).values
    c   = df["close"].astype(float).values
    v   = df["volume"].astype(float).values

    hour = df["hour"].astype(int).values       if "hour"        in df.columns else np.full(n, 12)
    dow  = df["day_of_week"].astype(int).values if "day_of_week" in df.columns else np.ones(n, dtype=int)

    # Volume z-score
    vol_s  = pd.Series(v)
    vol_mu = vol_s.rolling(20, min_periods=5).mean()
    vol_sd = vol_s.rolling(20, min_periods=5).std().replace(0, np.nan)
    vol_z  = ((vol_s - vol_mu) / vol_sd).fillna(0).values
    vol_c  = np.clip((vol_z + 2) / 4, 0, 1)

    # Session quality
    sess = np.full(n, 0.65)
    sess[(hour >= 13) & (hour <= 17)] = 1.00   # London/NY overlap
    sess[(hour >=  8) & (hour < 13)]  = 0.85   # London open
    sess[(hour >= 18) & (hour <= 20)] = 0.75   # NY afternoon
    sess[(hour >=  0) & (hour <=  7)] = 0.45   # Asian session
    sess[dow == 4] *= 0.85                      # Friday discount
    sess[dow >= 5] *= 0.30                      # Weekend

    # Candle body alignment with label direction
    hl_range  = np.where(h - l > 0, h - l, np.nan)
    body      = c - o
    body_bull = np.clip( body / hl_range, 0, 1)
    body_bear = np.clip(-body / hl_range, 0, 1)
    body_bull = np.nan_to_num(body_bull)
    body_bear = np.nan_to_num(body_bear)

    body_align = np.where(labels == 1, body_bull,
                 np.where(labels == 2, body_bear, 0.5))

    conf = 0.35 * vol_c + 0.35 * sess + 0.30 * body_align
    return np.round(np.clip(conf, 0, 1), 3)


# ======================================================
#  Main class
# ======================================================

class RobustPriceLabelerV4:
    """
    Zone-aware, forward-confirmed price labeler for 5-min Gold (XAUUSD).

    Every label passes TWO independent validation layers before it is
    kept as a directional signal:

    BACKWARD (S/R zone context)
      Active resistance and support zones are built from the last
      sr_lookback confirmed zigzag pivots.  BUY labels inside resistance
      zones and SELL labels inside support zones are demoted to HOLD.
      Exception: breakout/breakdown moves (price already past the zone)
      are preserved as those are the highest-probability entries.

    FORWARD (triple-barrier confirmation)
      Each surviving directional label must be validated by actual future
      price movement within max_confirm_bars.  A BUY must reach
      close + profit_mult*ATR before close - stop_mult*ATR.
      A SELL must reach close - profit_mult*ATR before close + stop_mult*ATR.
      Failed confirmation -> HOLD, permanently locked.

    Parameters
    ----------
    atr_period        : ATR lookback (default 14)
    zigzag_atr_mult   : Min reversal in ATR units to confirm a new pivot.
                        Range 0.3-1.0 for 5-min Gold; lower = more pivots.
    hold_bars         : Bars around each pivot marked as HOLD (2-4 typical)
    min_streak        : Minimum consecutive same-label bars to preserve
    target_hold_pct   : Desired maximum hold fraction after balancing
    sr_lookback       : Number of recent swing H/L to track as active zones
    zone_mult         : S/R zone half-width in ATR units (0.5-2.0).
                        1.0 = 1 ATR from the pivot level.
    profit_mult       : Triple-barrier TP distance in ATR units
    stop_mult         : Triple-barrier SL distance in ATR units
    max_confirm_bars  : Forward look-ahead window for barrier confirmation.
                        20 bars = 100 min on 5-min chart.
    use_momentum_gate : If True, applies EMA trend-alignment filter.
    ema_fast          : Fast EMA period for momentum gate (default 8)
    ema_slow          : Slow EMA period for momentum gate (default 21)
    ema_slope_tol     : EMA gap tolerance in ATR units (0 = strict, 1 = loose)
    """

    def __init__(
        self,
        # Core zigzag
        atr_period:        int   = 14,
        zigzag_atr_mult:   float = 0.5,
        hold_bars:         int   = 2,
        min_streak:        int   = 2,
        target_hold_pct:   float = 0.30,
        # S/R zone filter  (backward confirmation)
        sr_lookback:       int   = 3,
        zone_mult:         float = 1.0,
        # Triple-barrier   (forward confirmation)
        profit_mult:       float = 1.5,
        stop_mult:         float = 1.0,
        max_confirm_bars:  int   = 20,
        # Optional EMA momentum gate
        use_momentum_gate: bool  = False,
        ema_fast:          int   = 8,
        ema_slow:          int   = 21,
        ema_slope_tol:     float = 0.5,
    ):
        self.atr_period        = atr_period
        self.zigzag_atr_mult   = zigzag_atr_mult
        self.hold_bars         = hold_bars
        self.min_streak        = min_streak
        self.target_hold_pct   = target_hold_pct
        self.sr_lookback       = sr_lookback
        self.zone_mult         = zone_mult
        self.profit_mult       = profit_mult
        self.stop_mult         = stop_mult
        self.max_confirm_bars  = max_confirm_bars
        self.use_momentum_gate = use_momentum_gate
        self.ema_fast          = ema_fast
        self.ema_slow          = ema_slow
        self.ema_slope_tol     = ema_slope_tol

    # --------------------------------------------------

    def label(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Label each bar as buy (1), sell (2), or hold (0).

        Required columns : open, high, low, close, volume
        Optional columns : hour, minute, day, month, day_of_week

        Returns
        -------
        df (copy) with new columns:
            label              int    0=hold 1=buy 2=sell
            label_name         str
            label_conf         float  0-1 quality confidence
            zone_locked        bool   True = demoted by S/R zone filter
            barrier_locked     bool   True = demoted by forward confirmation
            momentum_locked    bool   True = demoted by EMA momentum gate
        """
        required = {"open", "high", "low", "close", "volume"}
        missing  = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        df = df.reset_index(drop=True).copy()
        n  = len(df)
        h  = df["high"].astype(float).values
        l  = df["low"].astype(float).values
        c  = df["close"].astype(float).values

        # Step 1 - ATR
        print("Step 1/8 | Computing ATR ...")
        atr = _atr(h, l, c, self.atr_period)

        # Step 2 - Zigzag pivots
        print("Step 2/8 | Finding zigzag pivots ...")
        pivots = _zigzag_pivots(h, l, atr, self.zigzag_atr_mult)
        print(f"         | {len(pivots)} pivots found")

        # Step 3 - Initial zone labels
        print("Step 3/8 | Assigning initial zone labels ...")
        raw_labels = _zone_labels(n, pivots, self.hold_bars)

        # Step 4 - Build S/R levels (backward-only)
        print("Step 4/8 | Building S/R level arrays ...")
        resistance, support = _build_sr_levels(pivots, n, self.sr_lookback)

        # Step 5 - S/R zone filter (backward confirmation)
        print("Step 5/8 | Applying S/R zone filter (backward confirmation) ...")
        zone_filtered, zone_lock = _filter_sr_zones(
            raw_labels, c, resistance, support, atr, self.zone_mult
        )
        merged_lock = zone_lock.copy()

        # Step 6 - Optional EMA momentum gate
        momentum_lock = np.zeros(n, dtype=bool)
        if self.use_momentum_gate:
            print("Step 6/8 | Applying EMA momentum gate ...")
            zone_filtered, momentum_lock = _backward_ema_gate(
                zone_filtered, c, atr,
                self.ema_fast, self.ema_slow, self.ema_slope_tol
            )
            merged_lock |= momentum_lock
        else:
            print("Step 6/8 | EMA momentum gate disabled (use_momentum_gate=False)")

        # Step 7 - Triple-barrier forward confirmation
        print(
            f"Step 7/8 | Applying triple-barrier confirmation "
            f"(TP={self.profit_mult}xATR  SL={self.stop_mult}xATR "
            f"window={self.max_confirm_bars} bars) ..."
        )
        barrier_filtered, barrier_lock = _forward_triple_barrier(
            zone_filtered, c, h, l, atr,
            self.profit_mult, self.stop_mult, self.max_confirm_bars
        )
        merged_lock |= barrier_lock

        # Step 8 - Balance enforcer (lock-aware)
        print("Step 8/8 | Enforcing class balance (lock-aware) ...")
        balanced = _enforce_balance(
            barrier_filtered, h, l, c, atr,
            target_hold_pct=self.target_hold_pct,
            min_streak=self.min_streak,
            locked=merged_lock,
        )

        # Confidence scores
        conf = _confidence(df, atr, balanced)

        # Attach to dataframe
        df["label"]           = balanced
        df["label_name"]      = pd.Series(balanced).map({0: "hold", 1: "buy", 2: "sell"})
        df["label_conf"]      = conf
        df["zone_locked"]     = zone_lock
        df["barrier_locked"]  = barrier_lock
        df["momentum_locked"] = momentum_lock

        # Distribution report
        vc    = df["label_name"].value_counts()
        total = len(df)
        zl    = int(zone_lock.sum())
        bl    = int(barrier_lock.sum())

        print("\n-- Label Distribution ---------------------------------------")
        for lbl in ["buy", "sell", "hold"]:
            cnt = vc.get(lbl, 0)
            bar = "X" * int(cnt / total * 40)
            print(f"  {lbl:4s}: {cnt:7,d}  ({cnt / total * 100:5.1f}%)  {bar}")
        print("-------------------------------------------------------------")
        print(f"  Zone-locked  (S/R conflict)      : {zl:7,d}  ({zl / total * 100:5.1f}%)")
        print(f"  Barrier-locked (forward unconf.) : {bl:7,d}  ({bl / total * 100:5.1f}%)")
        if self.use_momentum_gate:
            ml = int(momentum_lock.sum())
            print(f"  Momentum-locked (EMA filter)     : {ml:7,d}  ({ml / total * 100:5.1f}%)")
        print("-------------------------------------------------------------\n")

        return df

    # --------------------------------------------------

    def get_pivots(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns a DataFrame of confirmed swing pivots for charting.
        Columns: bar_index, price, type ('H' or 'L')
        """
        h      = df["high"].astype(float).values
        l      = df["low"].astype(float).values
        c      = df["close"].astype(float).values
        atr    = _atr(h, l, c, self.atr_period)
        pivots = _zigzag_pivots(h, l, atr, self.zigzag_atr_mult)
        return pd.DataFrame(pivots, columns=["bar_index", "price", "type"])

    # --------------------------------------------------

    def get_sr_zones(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns a DataFrame showing the active S/R levels at each bar.
        Useful for visualising which zones were active during labeling.
        Columns: bar_index, resistance_levels (list), support_levels (list)
        """
        h      = df["high"].astype(float).values
        l      = df["low"].astype(float).values
        c      = df["close"].astype(float).values
        n      = len(df)
        atr    = _atr(h, l, c, self.atr_period)
        pivots = _zigzag_pivots(h, l, atr, self.zigzag_atr_mult)
        resistance, support = _build_sr_levels(pivots, n, self.sr_lookback)
        return pd.DataFrame({
            "bar_index":         range(n),
            "resistance_levels": resistance,
            "support_levels":    support,
        })




"""
RobustPriceLabelerV5  —  Binary Zone-Aware, Forward-Confirmed Labeler
======================================================================
Designed for 5-minute Gold (XAUUSD) data.

Binary label encoding
---------------------
  1 = BUY   |   2 = SELL
  HOLD is eliminated entirely.

Why binary instead of three-class?
-----------------------------------
Three-class labeling (buy / sell / hold) creates severe class imbalance
because every ambiguous bar is pushed into HOLD.  In practice this means
a model must learn to abstain — which is a separate meta-labeling problem.
Binary labeling forces every bar to carry a directional opinion, producing
naturally balanced classes and a cleaner supervised learning target.

How the V4 three-class logic maps to binary
-------------------------------------------

  V4 action              →  V5 binary action
  ─────────────────────────────────────────────────────────────────
  Zigzag ± hold_bars     →  Removed. Full leg direction applies to
    assigned HOLD            ALL bars including pivot transitions.
                             Pivot bars absorb into the incoming leg.

  S/R zone conflict      →  FLIP the label (not HOLD).
    BUY near resistance  →  Relabel as SELL  (resistance rejects price)
    SELL near support    →  Relabel as BUY   (support attracts price)
    Breakout / breakdown →  Keep original    (momentum through zone)

  Triple-barrier fail    →  FLIP based on actual outcome (not HOLD).
    TP hit first         →  Confirm the current label
    SL hit first         →  Flip to opposite label
    Time barrier         →  Assign by sign of net return over window
                             (close[i+max_bars] - close[i])

  EMA momentum gate      →  FLIP counter-trend labels (not HOLD).
    BUY in downtrend     →  Relabel as SELL
    SELL in uptrend      →  Relabel as BUY

Full label pipeline
--------------------
  1. ATR (14-bar, True-Range based)
  2. Zigzag pivots (ATR-adaptive threshold)
  3. Binary zone labels  — entire L→H leg = BUY, entire H→L leg = SELL.
                            No transition gaps.  Pivot bars belong to the
                            INCOMING leg direction.
  4. S/R zone filter     — flip labels that conflict with active S/R zones
                            (backward-only, zero lookahead bias)
  5. EMA momentum gate   — flip counter-trend labels (optional, default OFF)
  6. Triple-barrier      — re-assign each bar by its actual forward outcome
                            (forward-looking, the only intentional lookahead)
  7. Confidence scoring  — 0–1 quality score per bar

Diagnostic columns
------------------
  zone_flipped     : bool — label was flipped by S/R zone filter
  barrier_flipped  : bool — label was flipped by triple-barrier
  momentum_flipped : bool — label was flipped by EMA gate
  any_flipped      : bool — label was changed by ANY filter vs initial zigzag

Label encoding
--------------
  1 = BUY   |   2 = SELL
"""

import numpy as np
import pandas as pd
from collections import deque
from typing import List, Tuple, Optional


# ======================================================
#  ATR
# ======================================================

def _atr(high: np.ndarray, low: np.ndarray,
         close: np.ndarray, period: int = 14) -> np.ndarray:
    """Average True Range — True-Range rolling mean."""
    n  = len(close)
    tr = np.empty(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i - 1]),
                    abs(low[i]  - close[i - 1]))
    return pd.Series(tr).rolling(period, min_periods=1).mean().values


# ======================================================
#  Zigzag pivot finder  (ATR-adaptive threshold)
# ======================================================

def _zigzag_pivots(high: np.ndarray, low: np.ndarray,
                   atr: np.ndarray,
                   atr_mult: float = 0.5
                   ) -> List[Tuple[int, float, str]]:
    """
    Returns a list of (index, price, 'H'|'L') confirmed swing pivots.

    A new pivot is confirmed only when price reverses >= atr_mult x ATR
    from the last extreme.  This built-in reversal requirement means every
    pivot already has backward confirmation before it is recorded.

    atr_mult guidance for 5-min Gold:
      0.3  ->  many fine-grained pivots (noisy but rich label variety)
      0.5  ->  balanced default
      1.0  ->  only major swings (smooth but fewer signals)
    """
    n       = len(high)
    pivots  = []
    trend   = 0
    ext_idx = 0
    ext_val = (high[0] + low[0]) / 2

    for i in range(1, n):
        threshold = atr[i] * atr_mult

        if trend >= 0:                         # seeking higher high
            if high[i] >= ext_val:
                ext_val, ext_idx = high[i], i
            elif ext_val - low[i] >= threshold:
                pivots.append((ext_idx, ext_val, 'H'))
                trend, ext_val, ext_idx = -1, low[i], i
        else:                                  # seeking lower low
            if low[i] <= ext_val:
                ext_val, ext_idx = low[i], i
            elif high[i] - ext_val >= threshold:
                pivots.append((ext_idx, ext_val, 'L'))
                trend, ext_val, ext_idx = 1, high[i], i

    # Close out the last unconfirmed pivot
    pivots.append((ext_idx, ext_val, 'H' if trend >= 0 else 'L'))

    # Remove consecutive duplicates — keep the more extreme one
    cleaned = [pivots[0]]
    for p in pivots[1:]:
        if p[2] != cleaned[-1][2]:
            cleaned.append(p)
        elif p[2] == 'H' and p[1] > cleaned[-1][1]:
            cleaned[-1] = p
        elif p[2] == 'L' and p[1] < cleaned[-1][1]:
            cleaned[-1] = p

    return cleaned


# ======================================================
#  Binary zone labels  (no HOLD, no transition gaps)
# ======================================================

def _zone_labels_binary(n: int,
                        pivots: List[Tuple[int, float, str]]
                        ) -> np.ndarray:
    """
    Assign every bar to BUY (1) or SELL (2) based solely on the
    zigzag leg it belongs to.

    L -> H leg  =  BUY  (1)   (price rising from swing low to swing high)
    H -> L leg  =  SELL (2)   (price falling from swing high to swing low)

    Pivot bars themselves are assigned to the INCOMING leg direction.
    There are no transition gaps or HOLD bars.

    Bars before the first confirmed pivot: assigned to the inverse of
    the first leg (i.e., if the first leg is L->H/BUY, the bars before
    the swing low are labeled SELL because price was falling into it).
    """
    labels = np.ones(n, dtype=int)   # default BUY; overwritten below

    if not pivots:
        return labels

    # Bars from pivot k to pivot k+1 (inclusive both ends)
    # belong to the leg starting AT pivot k
    for k in range(len(pivots) - 1):
        p1_idx, _, p1_type = pivots[k]
        p2_idx, _, _       = pivots[k + 1]
        direction = 1 if p1_type == 'L' else 2
        labels[p1_idx: p2_idx + 1] = direction

    # Bars before the first pivot: assign inverse of first leg direction
    first_type = pivots[0][2]
    pre_dir    = 2 if first_type == 'L' else 1   # before a swing low = selling
    labels[: pivots[0][0]] = pre_dir

    # Bars after the last confirmed pivot
    last_p_idx  = pivots[-1][0]
    last_p_type = pivots[-1][2]
    # After a swing HIGH: price is expected to fall -> SELL
    # After a swing LOW:  price is expected to rise -> BUY
    last_dir = 2 if last_p_type == 'H' else 1
    labels[last_p_idx:] = last_dir

    return labels


# ======================================================
#  Build S/R level arrays  (purely backward-looking)
# ======================================================

def _build_sr_levels(pivots: List[Tuple[int, float, str]],
                     n: int,
                     lookback: int = 3
                     ) -> Tuple[List[List[float]], List[List[float]]]:
    """
    For each bar i, expose the last `lookback` confirmed:
      - swing HIGH prices  ->  resistance levels
      - swing LOW  prices  ->  support levels

    IMPORTANT: Only pivots confirmed at or BEFORE bar i are visible.
    This is a purely backward-looking construct — zero lookahead bias.

    Resistance: sorted descending  (nearest-above-close first)
    Support:    sorted ascending   (nearest-below-close first)
    """
    highs_buf = deque(maxlen=lookback)
    lows_buf  = deque(maxlen=lookback)

    pivot_map: dict = {}
    for idx, price, ptype in sorted(pivots, key=lambda x: x[0]):
        pivot_map.setdefault(idx, []).append((price, ptype))

    resistance: List[List[float]] = [[] for _ in range(n)]
    support:    List[List[float]] = [[] for _ in range(n)]

    for i in range(n):
        if i in pivot_map:
            for price, ptype in pivot_map[i]:
                if ptype == 'H':
                    highs_buf.append(price)
                else:
                    lows_buf.append(price)
        resistance[i] = sorted(highs_buf, reverse=True)
        support[i]    = sorted(lows_buf)

    return resistance, support


# ======================================================
#  S/R Zone Filter  (BACKWARD — flips labels, not HOLD)
# ======================================================

def _filter_sr_zones_binary(labels:     np.ndarray,
                             close:      np.ndarray,
                             resistance: List[List[float]],
                             support:    List[List[float]],
                             atr:        np.ndarray,
                             zone_mult:  float = 1.0
                             ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Flip directional labels that conflict with active S/R zones.

    BINARY MODE — no HOLD:  instead of demoting a conflicted label to HOLD,
    the label is FLIPPED to the opposite direction, because the zone itself
    predicts the reversal.

    BUY at bar i  ->  SELL if:
      close[i] is within zone_mult x ATR BELOW any resistance level.
      Rationale: price is approaching resistance from below; the zone
      will likely reject the move and push price back down.
      EXCEPTION: if close[i] is ALREADY ABOVE the resistance level,
      a breakout is in progress — the original BUY label is kept and
      may even strengthen.

    SELL at bar i  ->  BUY if:
      close[i] is within zone_mult x ATR ABOVE any support level.
      Rationale: price is near support from above; the zone will likely
      bounce price back up.
      EXCEPTION: if close[i] is ALREADY BELOW the support level,
      a breakdown is in progress — the original SELL label is kept.

    Parameters
    ----------
    zone_mult : float
        Half-width of the S/R zone in ATR units.
        1.0  ->  1 ATR buffer on each side of the pivot level.
        Increase for wider Gold swings, decrease for tight structure.

    Returns
    -------
    filtered     : np.ndarray — updated binary labels (1 or 2 only)
    zone_flipped : np.ndarray — bool mask; True = label was flipped here
    """
    filtered     = labels.copy()
    zone_flipped = np.zeros(len(labels), dtype=bool)
    n            = len(labels)

    for i in range(n):
        zone = atr[i] * zone_mult
        c    = close[i]

        if filtered[i] == 1:          # BUY candidate: check resistance above
            for r in resistance[i]:
                gap = r - c
                if 0 < gap <= zone:
                    # Inside resistance zone, no breakout yet -> flip to SELL
                    filtered[i]     = 2
                    zone_flipped[i] = True
                    break
                # gap <= 0 means close >= r (breakout above) -> keep BUY

        elif filtered[i] == 2:        # SELL candidate: check support below
            for s in support[i]:
                gap = c - s
                if 0 < gap <= zone:
                    # Inside support zone, no breakdown yet -> flip to BUY
                    filtered[i]     = 1
                    zone_flipped[i] = True
                    break
                # gap <= 0 means close <= s (breakdown below) -> keep SELL

    return filtered, zone_flipped


# ======================================================
#  Triple-Barrier Forward Assignment  (FORWARD — outcome-based)
# ======================================================

def _forward_triple_barrier_binary(labels:      np.ndarray,
                                   close:        np.ndarray,
                                   high:         np.ndarray,
                                   low:          np.ndarray,
                                   atr:          np.ndarray,
                                   profit_mult:  float = 1.5,
                                   stop_mult:    float = 1.0,
                                   max_bars:     int   = 20
                                   ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Re-assign each bar's label by the ACTUAL forward price outcome.

    This is the only intentional lookahead in the pipeline; it is used
    exclusively for constructing training labels (never in live inference).

    For each bar i with an initial label L, three barriers are placed:

      BUY initial (L=1):
        TP_up   = close[i] + profit_mult x ATR[i]   (upper barrier)
        SL_down = close[i] - stop_mult   x ATR[i]   (lower barrier)
        Time    = bar i + max_bars                   (vertical barrier)

      SELL initial (L=2):
        TP_down = close[i] - profit_mult x ATR[i]   (lower barrier)
        SL_up   = close[i] + stop_mult   x ATR[i]   (upper barrier)
        Time    = bar i + max_bars                   (vertical barrier)

    Outcome assignment (BINARY):
    ─────────────────────────────
      Barrier hit first    │  BUY initial  │  SELL initial
      ─────────────────────│───────────────│────────────────
      TP hit               │  keep BUY (1) │  keep SELL (2)
      SL hit               │  flip to SELL │  flip to BUY
      Time barrier (net+)  │  keep BUY (1) │  flip to BUY
      Time barrier (net-)  │  flip to SELL │  keep SELL (2)
      Time barrier (net=0) │  keep original│  keep original

    Tail bars (i + max_bars >= n):
      A partial forward window is used.  If the partial window is too
      short to hit either barrier (< 5 bars), the zigzag direction is
      preserved without modification.  The `partial_forward` column
      flags these bars.

    Parameters
    ----------
    profit_mult  : TP distance in ATR units  (must be > stop_mult for +EV)
    stop_mult    : SL distance in ATR units
    max_bars     : Forward window length in bars.
                   Recommended: 20 bars (~100 min on 5-min Gold)

    Returns
    -------
    final           : np.ndarray — binary labels (1 or 2 only)
    barrier_flipped : np.ndarray — bool mask; True = triple-barrier changed label
    partial_forward : np.ndarray — bool mask; True = tail bar, partial window used
    """
    final           = labels.copy()
    barrier_flipped = np.zeros(len(labels), dtype=bool)
    partial_forward = np.zeros(len(labels), dtype=bool)
    n               = len(labels)

    for i in range(n):
        end = min(i + max_bars + 1, n)
        window_len = end - i - 1

        # Flag and skip bars with too-short forward windows
        if window_len < 5:
            partial_forward[i] = True
            continue

        # Flag bars using a partial (but usable) forward window
        if end < i + max_bars + 1:
            partial_forward[i] = True

        current_label = labels[i]

        if current_label == 1:        # BUY: look for TP_up or SL_down
            tp = close[i] + profit_mult * atr[i]
            sl = close[i] - stop_mult   * atr[i]
            hit = 'time'
            for j in range(i + 1, end):
                if high[j] >= tp:
                    hit = 'tp'; break
                if low[j]  <= sl:
                    hit = 'sl'; break

            if hit == 'tp':
                pass                               # keep BUY (1) — confirmed
            elif hit == 'sl':
                final[i]           = 2             # SL hit -> flip to SELL
                barrier_flipped[i] = True
            else:                                  # time barrier
                net = close[end - 1] - close[i]
                if net < 0:
                    final[i]           = 2         # net negative -> flip to SELL
                    barrier_flipped[i] = True
                # net >= 0 -> keep BUY

        else:                         # SELL: look for TP_down or SL_up
            tp = close[i] - profit_mult * atr[i]
            sl = close[i] + stop_mult   * atr[i]
            hit = 'time'
            for j in range(i + 1, end):
                if low[j]  <= tp:
                    hit = 'tp'; break
                if high[j] >= sl:
                    hit = 'sl'; break

            if hit == 'tp':
                pass                               # keep SELL (2) — confirmed
            elif hit == 'sl':
                final[i]           = 1             # SL hit -> flip to BUY
                barrier_flipped[i] = True
            else:                                  # time barrier
                net = close[end - 1] - close[i]
                if net > 0:
                    final[i]           = 1         # net positive -> flip to BUY
                    barrier_flipped[i] = True
                # net <= 0 -> keep SELL

    return final, barrier_flipped, partial_forward


# ======================================================
#  Backward EMA Momentum Gate  (optional — flips, not HOLD)
# ======================================================

def _backward_ema_gate_binary(labels:    np.ndarray,
                               close:     np.ndarray,
                               atr:       np.ndarray,
                               ema_fast:  int   = 8,
                               ema_slow:  int   = 21,
                               slope_tol: float = 0.5
                               ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Optional backward trend-alignment gate.

    BINARY MODE — no HOLD: counter-trend labels are FLIPPED to align with
    the current EMA trend, not dropped to HOLD.

    BUY in a confirmed downtrend (EMA_fast < EMA_slow - slope_tol x ATR)
    ->  Flipped to SELL.  Rationale: the underlying trend is bearish;
        buying against it has low probability.  Relabeling as SELL treats
        the bar as a continuation signal in the dominant direction.

    SELL in a confirmed uptrend (EMA_fast > EMA_slow + slope_tol x ATR)
    ->  Flipped to BUY.

    slope_tol = 0.0  ->  strict: any EMA separation triggers the gate
    slope_tol = 1.0  ->  loose:  only large EMA gaps trigger the gate

    Parameters
    ----------
    ema_fast   : Fast EMA period.  Default 8 suits 5-min intraday Gold.
    ema_slow   : Slow EMA period.  Default 21 is one Fibonacci level above.
    slope_tol  : Minimum EMA gap in ATR units needed to fire the gate.

    Returns
    -------
    filtered          : np.ndarray — updated binary labels
    momentum_flipped  : np.ndarray — bool mask; True = EMA gate changed label
    """
    close_s = pd.Series(close)
    ema_f   = close_s.ewm(span=ema_fast, adjust=False).mean().values
    ema_s   = close_s.ewm(span=ema_slow, adjust=False).mean().values

    filtered         = labels.copy()
    momentum_flipped = np.zeros(len(labels), dtype=bool)

    for i in range(len(labels)):
        thresh = atr[i] * slope_tol

        if labels[i] == 1:            # BUY — needs fast > slow (bullish alignment)
            if ema_f[i] < ema_s[i] - thresh:
                filtered[i]          = 2   # bearish trend -> flip to SELL
                momentum_flipped[i]  = True

        elif labels[i] == 2:          # SELL — needs fast < slow (bearish alignment)
            if ema_f[i] > ema_s[i] + thresh:
                filtered[i]          = 1   # bullish trend -> flip to BUY
                momentum_flipped[i]  = True

    return filtered, momentum_flipped


# ======================================================
#  Confidence scoring
# ======================================================

def _confidence(df: pd.DataFrame,
                atr: np.ndarray,
                labels: np.ndarray) -> np.ndarray:
    """
    Returns a 0–1 confidence score per bar.

    Components
    ----------
    Volume z-score    (35%) — above-average volume validates the move
    Session quality   (35%) — London/NY overlap has highest reliability
    Candle alignment  (30%) — full bullish body for BUY, bearish for SELL
    """
    n   = len(labels)
    o   = df["open"].astype(float).values
    h   = df["high"].astype(float).values
    l   = df["low"].astype(float).values
    c   = df["close"].astype(float).values
    v   = df["volume"].astype(float).values

    hour = df["hour"].astype(int).values        if "hour"        in df.columns else np.full(n, 12)
    dow  = df["day_of_week"].astype(int).values if "day_of_week" in df.columns else np.ones(n, dtype=int)

    # Volume z-score -----------------------------------
    vol_s  = pd.Series(v)
    vol_mu = vol_s.rolling(20, min_periods=5).mean()
    vol_sd = vol_s.rolling(20, min_periods=5).std().replace(0, np.nan)
    vol_z  = ((vol_s - vol_mu) / vol_sd).fillna(0).values
    vol_c  = np.clip((vol_z + 2) / 4, 0, 1)

    # Session quality ----------------------------------
    sess = np.full(n, 0.65)
    sess[(hour >= 13) & (hour <= 17)] = 1.00   # London/NY overlap (peak)
    sess[(hour >=  8) & (hour < 13)]  = 0.85   # London open
    sess[(hour >= 18) & (hour <= 20)] = 0.75   # NY afternoon
    sess[(hour >=  0) & (hour <=  7)] = 0.45   # Asian low-liquidity
    sess[dow == 4] *= 0.85                      # Friday fade
    sess[dow >= 5] *= 0.30                      # Weekend (near-zero)

    # Candle body alignment ----------------------------
    hl_range  = np.where(h - l > 0, h - l, np.nan)
    body      = c - o
    body_bull = np.clip( body / hl_range, 0, 1)   # 1.0 = full bullish body
    body_bear = np.clip(-body / hl_range, 0, 1)   # 1.0 = full bearish body
    body_bull = np.nan_to_num(body_bull)
    body_bear = np.nan_to_num(body_bear)

    body_align = np.where(labels == 1, body_bull, body_bear)

    conf = 0.35 * vol_c + 0.35 * sess + 0.30 * body_align
    return np.round(np.clip(conf, 0, 1), 3)


# ======================================================
#  Main class
# ======================================================

class RobustPriceLabelerV5:
    """
    Binary (BUY/SELL) zone-aware, forward-confirmed price labeler.
    Designed for 5-minute Gold (XAUUSD).

    All ambiguous bars are assigned a definitive directional label.
    No HOLD class.

    How each filter resolves ambiguity in binary mode
    --------------------------------------------------
    1. S/R Zone Filter    (backward):
       BUY inside resistance -> SELL  (zone rejects upward move)
       SELL inside support   -> BUY   (zone attracts price back up)
       Price past the zone   -> keep original (breakout / breakdown)

    2. EMA Momentum Gate  (backward, optional):
       BUY in downtrend  -> SELL  (align with trend direction)
       SELL in uptrend   -> BUY   (align with trend direction)

    3. Triple-Barrier     (forward — outcome based):
       TP hit first      -> confirm current label
       SL hit first      -> flip to opposite label
       Time barrier      -> sign of net return decides

    Parameters
    ----------
    atr_period         : ATR lookback period (default 14)
    zigzag_atr_mult    : Min reversal in ATR units for a confirmed pivot.
                         Range 0.3–1.0 for 5-min Gold.  Lower = more pivots.
    sr_lookback        : Active S/R levels to track per side (swing H / swing L)
    zone_mult          : S/R zone half-width in ATR units (0.5 – 2.0)
    profit_mult        : Triple-barrier TP distance in ATR units
    stop_mult          : Triple-barrier SL distance in ATR units
    max_confirm_bars   : Forward window length for barrier assignment.
                         20 bars = 100 min on a 5-min chart.
    use_momentum_gate  : Enable EMA trend-alignment filter (default False)
    ema_fast           : Fast EMA period for momentum gate (default 8)
    ema_slow           : Slow EMA period for momentum gate (default 21)
    ema_slope_tol      : EMA gap threshold in ATR units (0 = strict)
    """

    def __init__(
        self,
        atr_period:        int   = 14,
        zigzag_atr_mult:   float = 0.5,
        # S/R zone filter  (backward)
        sr_lookback:       int   = 3,
        zone_mult:         float = 1.0,
        # Triple-barrier   (forward)
        profit_mult:       float = 1.5,
        stop_mult:         float = 1.0,
        max_confirm_bars:  int   = 20,
        # Optional EMA momentum gate  (backward)
        use_momentum_gate: bool  = False,
        ema_fast:          int   = 8,
        ema_slow:          int   = 21,
        ema_slope_tol:     float = 0.5,
    ):
        self.atr_period        = atr_period
        self.zigzag_atr_mult   = zigzag_atr_mult
        self.sr_lookback       = sr_lookback
        self.zone_mult         = zone_mult
        self.profit_mult       = profit_mult
        self.stop_mult         = stop_mult
        self.max_confirm_bars  = max_confirm_bars
        self.use_momentum_gate = use_momentum_gate
        self.ema_fast          = ema_fast
        self.ema_slow          = ema_slow
        self.ema_slope_tol     = ema_slope_tol

    # --------------------------------------------------

    def label(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Label every bar as buy (1) or sell (2).  No HOLD class.

        Required columns : open, high, low, close, volume
        Optional columns : hour, minute, day, month, day_of_week

        Returns
        -------
        df (copy) with new columns:
            label              int    1=buy | 2=sell
            label_name         str    'buy' | 'sell'
            label_conf         float  0–1 quality confidence
            zone_flipped       bool   S/R zone filter changed this label
            barrier_flipped    bool   Triple-barrier changed this label
            momentum_flipped   bool   EMA gate changed this label
            any_flipped        bool   ANY filter changed from zigzag baseline
            partial_forward    bool   Tail bar — partial forward window used
        """
        required = {"open", "high", "low", "close", "volume"}
        missing  = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        df = df.reset_index(drop=True).copy()
        n  = len(df)
        h  = df["high"].astype(float).values
        l  = df["low"].astype(float).values
        c  = df["close"].astype(float).values

        # ── Step 1: ATR ───────────────────────────────────
        print("Step 1/6 | Computing ATR ...")
        atr = _atr(h, l, c, self.atr_period)

        # ── Step 2: Zigzag pivots ─────────────────────────
        print("Step 2/6 | Finding zigzag pivots ...")
        pivots = _zigzag_pivots(h, l, atr, self.zigzag_atr_mult)
        print(f"         | {len(pivots)} pivots found")

        # ── Step 3: Binary zone labels (no HOLD, no gaps) ─
        print("Step 3/6 | Assigning binary zone labels (no HOLD) ...")
        raw_labels = _zone_labels_binary(n, pivots)
        baseline   = raw_labels.copy()   # kept for any_flipped tracking

        # ── Step 4: Build S/R levels (backward-only) ─────
        print("Step 4/6 | Building backward S/R level arrays ...")
        resistance, support = _build_sr_levels(pivots, n, self.sr_lookback)

        # ── Step 5a: S/R zone filter ──────────────────────
        print("Step 5/6 | Applying S/R zone filter (backward) ...")
        current_labels, zone_flipped = _filter_sr_zones_binary(
            raw_labels, c, resistance, support, atr, self.zone_mult
        )
        zf = int(zone_flipped.sum())
        print(f"         | {zf:,} labels flipped by S/R zone filter")

        # ── Step 5b: Optional EMA momentum gate ───────────
        momentum_flipped = np.zeros(n, dtype=bool)
        if self.use_momentum_gate:
            print("Step 5/6 | Applying EMA momentum gate (backward) ...")
            current_labels, momentum_flipped = _backward_ema_gate_binary(
                current_labels, c, atr,
                self.ema_fast, self.ema_slow, self.ema_slope_tol
            )
            mf = int(momentum_flipped.sum())
            print(f"         | {mf:,} labels flipped by EMA gate")
        else:
            print("         | EMA momentum gate disabled (use_momentum_gate=False)")

        # ── Step 6: Triple-barrier forward assignment ─────
        print(
            f"Step 6/6 | Applying triple-barrier outcome assignment "
            f"(TP={self.profit_mult}xATR  SL={self.stop_mult}xATR "
            f"window={self.max_confirm_bars} bars) ..."
        )
        final_labels, barrier_flipped, partial_forward = \
            _forward_triple_barrier_binary(
                current_labels, c, h, l, atr,
                self.profit_mult, self.stop_mult, self.max_confirm_bars
            )
        bf  = int(barrier_flipped.sum())
        pfw = int(partial_forward.sum())
        print(f"         | {bf:,} labels flipped by triple-barrier")
        print(f"         | {pfw:,} tail bars used partial forward window")

        # ── Confidence scores ─────────────────────────────
        conf = _confidence(df, atr, final_labels)

        # ── Attach to dataframe ───────────────────────────
        any_flipped = (final_labels != baseline)

        df["label"]           = final_labels
        df["label_name"]      = pd.Series(final_labels).map({1: "buy", 2: "sell"})
        df["label_conf"]      = conf
        df["zone_flipped"]    = zone_flipped
        df["barrier_flipped"] = barrier_flipped
        df["momentum_flipped"]= momentum_flipped
        df["any_flipped"]     = any_flipped
        df["partial_forward"] = partial_forward

        # ── Distribution report ───────────────────────────
        vc    = df["label_name"].value_counts()
        total = len(df)
        buy_n = vc.get("buy",  0)
        sel_n = vc.get("sell", 0)

        ratio = buy_n / sel_n if sel_n > 0 else float('inf')

        print("\n-- Binary Label Distribution --------------------------------")
        for lbl in ["buy", "sell"]:
            cnt = vc.get(lbl, 0)
            bar = "X" * int(cnt / total * 50)
            print(f"  {lbl:4s}: {cnt:7,d}  ({cnt / total * 100:5.1f}%)  {bar}")
        print(f"  buy:sell ratio = {ratio:.3f}  (1.000 = perfect balance)")
        print("-------------------------------------------------------------")
        print(f"  Zone-flipped    (S/R conflict)   : {zf:7,d}  ({zf / total * 100:5.1f}%)")
        print(f"  Barrier-flipped (outcome-based)  : {bf:7,d}  ({bf / total * 100:5.1f}%)")
        if self.use_momentum_gate:
            mf_t = int(momentum_flipped.sum())
            print(f"  Momentum-flipped (EMA gate)      : {mf_t:7,d}  ({mf_t / total * 100:5.1f}%)")
        print(f"  Any filter flipped               : {int(any_flipped.sum()):7,d}  "
              f"({any_flipped.mean() * 100:5.1f}%)")
        print(f"  Tail bars (partial window)       : {pfw:7,d}  ({pfw / total * 100:5.1f}%)")
        print("-------------------------------------------------------------\n")

        if ratio > 1.6 or ratio < 0.625:
            print(
                f"  [!] Class imbalance warning: buy:sell ratio = {ratio:.2f}\n"
                f"      Consider adjusting zigzag_atr_mult, profit_mult,\n"
                f"      stop_mult, or zone_mult to restore balance.\n"
                f"      You can also use sample_weights or class_weight='balanced'\n"
                f"      in your classifier as a downstream fix.\n"
            )

        return df

    # --------------------------------------------------

    def get_pivots(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns confirmed swing pivots for charting.
        Columns: bar_index, price, type ('H' or 'L')
        """
        h      = df["high"].astype(float).values
        l      = df["low"].astype(float).values
        c      = df["close"].astype(float).values
        atr    = _atr(h, l, c, self.atr_period)
        pivots = _zigzag_pivots(h, l, atr, self.zigzag_atr_mult)
        return pd.DataFrame(pivots, columns=["bar_index", "price", "type"])

    # --------------------------------------------------

    def get_sr_zones(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns active S/R levels at each bar.
        Columns: bar_index, resistance_levels (list), support_levels (list)
        """
        h      = df["high"].astype(float).values
        l      = df["low"].astype(float).values
        c      = df["close"].astype(float).values
        n      = len(df)
        atr    = _atr(h, l, c, self.atr_period)
        pivots = _zigzag_pivots(h, l, atr, self.zigzag_atr_mult)
        resistance, support = _build_sr_levels(pivots, n, self.sr_lookback)
        return pd.DataFrame({
            "bar_index":         range(n),
            "resistance_levels": resistance,
            "support_levels":    support,
        })



    
"""
XAUUSD Multi-Timeframe Label Generator
=======================================
Labels: 0 = Hold | 1 = Buy | 2 = Sell

Strategy
--------
1. ATR-normalised forward-return sets the raw label (buy/sell/hold).
2. Multi-timeframe trend alignment gates buy and sell signals
   — conflicts are demoted to Hold.
3. Consolidation detector (ATR compression + Bollinger squeeze)
   forces Hold during ranging / noisy markets.
4. A validation report confirms every Buy/Sell label is directionally
   correct before the dataset is used for training.
5. Two class-balancing options: undersample majority or SMOTE
   (requires imbalanced-learn).
"""


"""
XAUUSD Multi-Timeframe Labeling Engine
=======================================
Labels: 0=Hold | 1=Buy | 2=Sell

Strategy:
  - ATR-based dynamic thresholds (adapts to XAUUSD volatility regimes)
  - Forward-looking window to confirm actual price movement
  - Consolidation detection via ADX + price range filter → Hold
  - Multi-timeframe trend alignment (1h + 4h + 1d) to filter noise
  - Class balancing via hold-zone tightening / buy-sell subsample
  - Label validation pass: checks each label against realised future move
"""

import numpy as np
import pandas as pd
from collections import Counter


# ─────────────────────────────────────────────
#  1.  ATR HELPER
# ─────────────────────────────────────────────

def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range on 1h bars."""
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=1).mean()


# ─────────────────────────────────────────────
#  2.  ADX HELPER  (consolidation detector)
# ─────────────────────────────────────────────

def compute_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Simplified ADX. Values < 20 → ranging/consolidation."""
    up   = high.diff()
    down = -low.diff()
    plus_dm  = np.where((up > down) & (up > 0), up,  0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)

    atr = compute_atr(high, low, close, period)
    atr = atr.replace(0, np.nan)

    plus_di  = 100 * pd.Series(plus_dm,  index=close.index).rolling(period).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=close.index).rolling(period).mean() / atr

    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    adx = dx.rolling(period, min_periods=1).mean()
    return adx.fillna(0)


# ─────────────────────────────────────────────
#  3.  MULTI-TIMEFRAME TREND BIAS
# ─────────────────────────────────────────────

def mtf_trend(df: pd.DataFrame) -> pd.Series:
    """
    Returns a score per row:
      +1  if higher timeframe is bullish,
      -1  if bearish,
       0  if mixed / neutral.

    Uses 4h and 1D EMAs to determine bias.
    """
    ema20_4h = df['close_4h'].ewm(span=20, adjust=False).mean()
    ema50_4h = df['close_4h'].ewm(span=50, adjust=False).mean()
    ema20_1d = df['close_1d'].ewm(span=20, adjust=False).mean()
    ema50_1d = df['close_1d'].ewm(span=50, adjust=False).mean()

    bull_4h = (df['close_4h'] > ema20_4h) & (ema20_4h > ema50_4h)
    bear_4h = (df['close_4h'] < ema20_4h) & (ema20_4h < ema50_4h)
    bull_1d = (df['close_1d'] > ema20_1d) & (ema20_1d > ema50_1d)
    bear_1d = (df['close_1d'] < ema20_1d) & (ema20_1d < ema50_1d)

    score = (bull_4h.astype(int) - bear_4h.astype(int) +
             bull_1d.astype(int) - bear_1d.astype(int))

    # Normalise: +2 / +1 → bullish, -2 / -1 → bearish, 0 → neutral
    return score.clip(-1, 1)


# ─────────────────────────────────────────────
#  4.  CORE LABELER
# ─────────────────────────────────────────────

def label_xauusd(
    df: pd.DataFrame,
    atr_period: int  = 14,
    fwd_window: int  = 6,        # bars ahead to measure realised move
    atr_buy_mult: float  = 1.0,  # future gain must exceed  N × ATR
    atr_sell_mult: float = 1.0,  # future loss  must exceed  N × ATR
    adx_threshold: float = 20.0, # below this → consolidation → Hold
    require_mtf_align: bool = True,  # skip buys/sells against HTF trend
) -> pd.DataFrame:
    """
    Assigns labels to every row in *df*.

    Parameters
    ----------
    df               : DataFrame with 1h OHLCV + 4h + 1d columns
    atr_period       : ATR look-back (bars)
    fwd_window       : How many 1h bars ahead to assess the move
    atr_buy_mult     : Min forward gain (as multiple of ATR) to label Buy
    atr_sell_mult    : Min forward drop (as multiple of ATR) to label Sell
    adx_threshold    : ADX below this → force Hold regardless of price move
    require_mtf_align: If True, a Buy label requires non-bearish HTF trend
                       and a Sell label requires non-bullish HTF trend

    Returns
    -------
    df with 'label', 'future_return', 'atr', 'adx', 'mtf_score' columns added.
    """
    df = df.copy().reset_index(drop=True)

    # ── Indicators ────────────────────────────────────────────────────────
    df['atr'] = compute_atr(df['high'], df['low'], df['close'], atr_period)
    df['adx'] = compute_adx(df['high'], df['low'], df['close'], atr_period)
    df['mtf_score'] = mtf_trend(df)

    # ── Forward return: max high vs max low over next fwd_window bars ─────
    future_high = pd.Series([
        df['high'].iloc[i+1 : i+1+fwd_window].max() if i+1+fwd_window <= len(df)
        else np.nan
        for i in range(len(df))
    ], index=df.index)

    future_low = pd.Series([
        df['low'].iloc[i+1 : i+1+fwd_window].min() if i+1+fwd_window <= len(df)
        else np.nan
        for i in range(len(df))
    ], index=df.index)

    future_close = pd.Series([
        df['close'].iloc[i + fwd_window] if i + fwd_window < len(df)
        else np.nan
        for i in range(len(df))
    ], index=df.index)

    df['future_high']   = future_high
    df['future_low']    = future_low
    df['future_close']  = future_close

    # Potential gain / loss from current close
    df['max_gain'] = df['future_high'] - df['close']
    df['max_loss']  = df['close'] - df['future_low']
    df['net_move']  = df['future_close'] - df['close']   # direction confirmation

    # ── Thresholds ────────────────────────────────────────────────────────
    buy_threshold  = df['atr'] * atr_buy_mult
    sell_threshold = df['atr'] * atr_sell_mult

    # ── Raw signal ────────────────────────────────────────────────────────
    #  A bar can qualify as Buy if price rallies enough AND net move is up
    raw_buy  = (df['max_gain'] >= buy_threshold)  & (df['net_move'] > 0)
    #  A bar qualifies as Sell if price drops enough AND net move is down
    raw_sell = (df['max_loss'] >= sell_threshold) & (df['net_move'] < 0)

    # When both qualify (volatile candle), pick direction of net_move
    conflict = raw_buy & raw_sell
    raw_buy  = raw_buy  & ~conflict | (conflict & (df['net_move'] > 0))
    raw_sell = raw_sell & ~conflict | (conflict & (df['net_move'] < 0))

    # ── Consolidation filter → override to Hold ───────────────────────────
    is_consolidation = df['adx'] < adx_threshold

    # ── MTF alignment filter ──────────────────────────────────────────────
    if require_mtf_align:
        # Suppress buy when HTF is bearish; suppress sell when HTF is bullish
        raw_buy  = raw_buy  & (df['mtf_score'] >= 0)
        raw_sell = raw_sell & (df['mtf_score'] <= 0)

    # ── Assign labels ─────────────────────────────────────────────────────
    label = pd.Series(0, index=df.index)          # default = Hold
    label[raw_buy]  = 1                            # Buy
    label[raw_sell] = 2                            # Sell
    label[is_consolidation] = 0                    # consolidation → Hold

    # Last fwd_window bars have no future data → Hold
    label.iloc[-fwd_window:] = 0

    df['label'] = label
    return df


# ─────────────────────────────────────────────
#  5.  CLASS BALANCER
# ─────────────────────────────────────────────

def balance_labels(
    df: pd.DataFrame,
    strategy: str = "soft",          # "soft" | "hard" | "none"
    hold_ratio: float = 1.5,         # max hold count = hold_ratio × min(buy, sell)
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Balance class distribution.

    strategy="soft"  → trim Hold to hold_ratio × min(buy_count, sell_count).
                        Buy and Sell are kept fully; only excess Hold is dropped.
    strategy="hard"  → undersample all three classes to the size of the smallest.
    strategy="none"  → return df unchanged (rely on class_weight in model).

    Rows are dropped, not reordered; the time index remains monotonic.
    """
    if strategy == "none":
        return df

    counts = Counter(df['label'])
    print(f"[balance] Before → Hold:{counts[0]}  Buy:{counts[1]}  Sell:{counts[2]}")

    rng = np.random.default_rng(random_state)

    if strategy == "soft":
        target_bs  = min(counts[1], counts[2])           # keep all buy & sell
        target_hold = int(target_bs * hold_ratio)
        target_hold = min(target_hold, counts[0])        # can't add more than exist

        hold_idx = df.index[df['label'] == 0].tolist()
        keep_hold = rng.choice(hold_idx, size=target_hold, replace=False).tolist()
        buy_sell_idx = df.index[df['label'].isin([1, 2])].tolist()
        keep_idx = sorted(set(keep_hold) | set(buy_sell_idx))
        df = df.loc[keep_idx]

    elif strategy == "hard":
        target = min(counts.values())
        parts = []
        for lbl in [0, 1, 2]:
            idx = df.index[df['label'] == lbl].tolist()
            chosen = rng.choice(idx, size=min(target, len(idx)), replace=False)
            parts.append(df.loc[sorted(chosen)])
        df = pd.concat(parts).sort_index()

    counts_after = Counter(df['label'])
    print(f"[balance] After  → Hold:{counts_after[0]}  Buy:{counts_after[1]}  Sell:{counts_after[2]}")
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────
#  6.  LABEL VALIDATOR
# ─────────────────────────────────────────────

def validate_labels(df: pd.DataFrame, verbose: bool = True) -> dict:
    """
    Checks each label against the realised future move.

    Accuracy definition:
      Buy  label (1) → net_move > 0          correct direction
      Sell label (2) → net_move < 0          correct direction
      Hold label (0) → |net_move| < atr      move stayed small (stayed in range)

    Returns a dict with per-class accuracy and overall stats.
    """
    results = {}

    # --- Buy accuracy ---
    buy_rows = df[df['label'] == 1].dropna(subset=['net_move'])
    buy_correct = (buy_rows['net_move'] > 0).sum()
    buy_acc = buy_correct / len(buy_rows) if len(buy_rows) else 0
    results['buy_count']    = len(buy_rows)
    results['buy_accuracy'] = round(buy_acc, 4)

    # --- Sell accuracy ---
    sell_rows = df[df['label'] == 2].dropna(subset=['net_move'])
    sell_correct = (sell_rows['net_move'] < 0).sum()
    sell_acc = sell_correct / len(sell_rows) if len(sell_rows) else 0
    results['sell_count']    = len(sell_rows)
    results['sell_accuracy'] = round(sell_acc, 4)

    # --- Hold accuracy (move stayed within 1×ATR) ---
    hold_rows = df[df['label'] == 0].dropna(subset=['net_move', 'atr'])
    hold_correct = (hold_rows['net_move'].abs() <= hold_rows['atr']).sum()
    hold_acc = hold_correct / len(hold_rows) if len(hold_rows) else 0
    results['hold_count']    = len(hold_rows)
    results['hold_accuracy'] = round(hold_acc, 4)

    # --- Overall directional accuracy (buy + sell only) ---
    dir_total   = len(buy_rows) + len(sell_rows)
    dir_correct = buy_correct + sell_correct
    results['directional_accuracy'] = round(dir_correct / dir_total, 4) if dir_total else 0

    # --- Label distribution ---
    total = len(df)
    results['label_distribution'] = {
        'hold': f"{len(df[df['label']==0])/total*100:.1f}%",
        'buy':  f"{len(df[df['label']==1])/total*100:.1f}%",
        'sell': f"{len(df[df['label']==2])/total*100:.1f}%",
    }

    if verbose:
        print("\n" + "="*55)
        print("  LABEL VALIDATION REPORT")
        print("="*55)
        print(f"  Buy  → {results['buy_count']:>6} labels | accuracy: {results['buy_accuracy']*100:.1f}%")
        print(f"  Sell → {results['sell_count']:>6} labels | accuracy: {results['sell_accuracy']*100:.1f}%")
        print(f"  Hold → {results['hold_count']:>6} labels | accuracy: {results['hold_accuracy']*100:.1f}%")
        print(f"  Directional accuracy (Buy+Sell): {results['directional_accuracy']*100:.1f}%")
        print(f"  Distribution → {results['label_distribution']}")
        print("="*55 + "\n")

        # ── Flag problematic labels ────────────────────────────────────────
        bad_buy  = buy_rows[buy_rows['net_move'] <= 0]
        bad_sell = sell_rows[sell_rows['net_move'] >= 0]
        if len(bad_buy) > 0:
            print(f"  ⚠  {len(bad_buy)} Buy labels where price actually fell (check thresholds)")
        if len(bad_sell) > 0:
            print(f"  ⚠  {len(bad_sell)} Sell labels where price actually rose (check thresholds)")
        if results['directional_accuracy'] < 0.60:
            print("  ⚠  Directional accuracy below 60% — consider raising atr_buy_mult / atr_sell_mult")
        if results['buy_accuracy'] < 0.55 or results['sell_accuracy'] < 0.55:
            print("  ⚠  One class below 55% — labels may be too noisy for training")

    return results


# ─────────────────────────────────────────────
#  7.  MAIN PIPELINE
# ─────────────────────────────────────────────

def buildLabeledDataset(
    df: pd.DataFrame,
    # Labeling knobs
    atr_period: int   = 14,
    fwd_window: int   = 6,
    atr_buy_mult: float  = 1.0,
    atr_sell_mult: float = 1.0,
    adx_threshold: float = 20.0,
    require_mtf_align: bool = True,
    # Balancing knobs
    balance_strategy: str  = "soft",   # "soft" | "hard" | "none"
    hold_ratio: float      = 1.5,
    random_state: int      = 42,
) -> tuple[pd.DataFrame, dict]:
    """
    Full pipeline:
      1. Label
      2. Validate (pre-balance)
      3. Balance
      4. Validate again (post-balance)
      5. Return (labeled_df, validation_report)

    Usage
    -----
    feature_cols = ['open','high','low','close','volume',
                    'open_4h','high_4h','low_4h','close_4h','volume_4h',
                    'open_1d','high_1d','low_1d','close_1d','volume_1d',
                    'hour','day','month','day_of_week']

    labeled_df, report = build_labeled_dataset(df)

    X = labeled_df[feature_cols].values
    y = labeled_df['label'].values
    """
    print("── Step 1: Labeling ─────────────────────────────────")
    labeled = label_xauusd(
        df,
        atr_period=atr_period,
        fwd_window=fwd_window,
        atr_buy_mult=atr_buy_mult,
        atr_sell_mult=atr_sell_mult,
        adx_threshold=adx_threshold,
        require_mtf_align=require_mtf_align,
    )

    print("\n── Step 2: Pre-balance Validation ───────────────────")
    validate_labels(labeled, verbose=True)

    return labeled


# ─────────────────────────────────────────────
#  8.  QUICK-TUNE HELPER
# ─────────────────────────────────────────────

def tune_thresholds(
    df: pd.DataFrame,
    atr_mults: list = [0.5, 0.75, 1.0, 1.25, 1.5],
    fwd_windows: list = [3, 6, 9, 12],
    adx_thresholds: list = [15, 20, 25],
    target_directional_acc: float = 0.65,
) -> pd.DataFrame:
    """
    Grid-search over threshold combinations and return a summary DataFrame
    sorted by directional accuracy.  Use this to find the best params before
    calling build_labeled_dataset().
    """
    records = []
    for atr_m in atr_mults:
        for fwd in fwd_windows:
            for adx_t in adx_thresholds:
                labeled = label_xauusd(
                    df,
                    atr_buy_mult=atr_m,
                    atr_sell_mult=atr_m,
                    fwd_window=fwd,
                    adx_threshold=adx_t,
                    require_mtf_align=True,
                )
                r = validate_labels(labeled, verbose=False)
                records.append({
                    'atr_mult': atr_m,
                    'fwd_window': fwd,
                    'adx_threshold': adx_t,
                    'dir_acc': r['directional_accuracy'],
                    'buy_acc': r['buy_accuracy'],
                    'sell_acc': r['sell_accuracy'],
                    'hold_acc': r['hold_accuracy'],
                    'buy_n': r['buy_count'],
                    'sell_n': r['sell_count'],
                    'hold_n': r['hold_count'],
                })

    summary = pd.DataFrame(records).sort_values('dir_acc', ascending=False)
    good = summary[summary['dir_acc'] >= target_directional_acc]
    print(f"\nConfigurations with ≥{target_directional_acc*100:.0f}% directional accuracy:")
    print(good.to_string(index=False))
    return summary


"""
fix_pivot_labels.py
===================
Corrects mislabeled candles at pivot point edges in XAUUSD 1H ML dataset.

LABEL MAP:  0 = hold  |  1 = buy  |  2 = sell

WHAT THIS FIXES
---------------
Four classes of label errors found at pivot edges:

  Class A – Direct illegal transition  buy→sell or sell→buy
            The boundary candle must become hold.

  Class B – Stray opposite label sandwiched in a pivot hold zone
            Pattern:  buy…buy → hold → BUY → sell…sell
            The lone BUY candle inside the hold zone should be hold.
            Same logic for:  sell…sell → hold → SELL → buy…buy

  Class C – Isolated opposite spike (1–3 candles) inside a run
            e.g. a single sell candle surrounded by buy on both sides.
            These become hold.

  Class D – Small opposite-direction run between a hold zone and the
            correct direction (hold→buy→sell should be hold→hold→sell).

USAGE
-----
    import pandas as pd
    from fix_pivot_labels import fix_pivot_labels

    df = pd.read_csv("Generated_gold_dbot.csv")
    df_fixed = fix_pivot_labels(df)
    df_fixed.to_csv("Generated_gold_dbot_fixed.csv", index=False)

Or run directly:
    python fix_pivot_labels.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

LABEL_MAP   = {0: "hold", 1: "buy", 2: "sell"}
SPIKE_MAX   = 3   # runs of this length or shorter are treated as spikes


# ─── helpers ─────────────────────────────────────────────────────────────────

def _rebuild(labels: np.ndarray) -> list[dict]:
    """Return list of {label, start, end} for contiguous runs."""
    runs = []
    n = len(labels)
    i = 0
    while i < n:
        j = i
        while j < n and labels[j] == labels[i]:
            j += 1
        runs.append({"label": int(labels[i]), "start": i, "end": j - 1})
        i = j
    return runs


def _apply(labels: np.ndarray, corrections: dict) -> np.ndarray:
    out = labels.copy()
    for idx, lbl in corrections.items():
        out[idx] = lbl
    return out


def _report(df: pd.DataFrame, tag: str):
    counts = df["label_name"].value_counts()
    total  = len(df)
    print(f"\n  [{tag}]")
    for lbl in ["buy", "hold", "sell"]:
        cnt = counts.get(lbl, 0)
        print(f"    {lbl:<6} {cnt:>7,}  ({cnt/total*100:.1f}%)")


# ─── pass 1: direct illegal transitions ──────────────────────────────────────

def _fix_direct_transitions(labels: np.ndarray) -> tuple[np.ndarray, int]:
    """
    Any buy→sell or sell→buy boundary: flip the EARLIER candle to hold.
    Repeat until no violations remain (fixing one may expose another).
    """
    fixed = labels.copy()
    total_changed = 0
    while True:
        changed = 0
        for i in range(len(fixed) - 1):
            a, b = int(fixed[i]), int(fixed[i + 1])
            if (a == 1 and b == 2) or (a == 2 and b == 1):
                fixed[i] = 0          # make the earlier one hold
                changed += 1
        total_changed += changed
        if changed == 0:
            break
    return fixed, total_changed


# ─── pass 2: stray opposite label inside pivot hold zone ─────────────────────

def _fix_pivot_strays(labels: np.ndarray) -> tuple[np.ndarray, int]:
    """
    Pattern:  ... buy | hold* | buy | sell ...   → flip the interior buy to hold
    And:      ... sell | hold* | sell | buy ...  → flip the interior sell to hold

    'hold*' means zero or more hold candles.
    We scan every run and check whether it is a short opposite-direction spike
    sitting between a hold zone (or the same direction) and the opposite direction.
    """
    fixed  = labels.copy()
    changed = 0

    while True:
        runs = _rebuild(fixed)
        local_changed = 0

        for ri in range(1, len(runs) - 1):
            prev_run  = runs[ri - 1]
            curr_run  = runs[ri]
            next_run  = runs[ri + 1]

            pl = prev_run["label"]
            cl = curr_run["label"]
            nl = next_run["label"]
            run_len = curr_run["end"] - curr_run["start"] + 1

            # Only act on non-hold runs that are small enough to be strays
            if cl == 0:
                continue

            if run_len > SPIKE_MAX:
                continue

            # Case: buy run that is immediately followed by sell (or hold then sell nearby)
            # prev context must be hold or sell or buy-then-hold
            opposite = 2 if cl == 1 else 1

            # The current run is in the wrong place if:
            # - next run is opposite direction, OR
            # - prev run is opposite direction
            if nl == opposite or pl == opposite:
                for pos in range(curr_run["start"], curr_run["end"] + 1):
                    fixed[pos] = 0
                local_changed += run_len

        changed += local_changed
        if local_changed == 0:
            break

    return fixed, changed


# ─── pass 3: hold zone that fragments a continuous directional run ────────────
# buy…hold…buy → the hold becomes buy (continuation hold, not pivot hold)
# BUT only if the hold is short and surrounded by same-direction runs
# We do NOT merge across true pivots (where a direction change follows)

def _fix_continuation_holds(labels: np.ndarray) -> tuple[np.ndarray, int]:
    """
    If a hold zone sits between two runs of the SAME direction AND
    is short (≤ SPIKE_MAX candles), merge it into that direction.

    This handles the case where a 1-2 candle hold briefly interrupts
    a buy or sell run at the pivot edge, breaking the continuity.
    """
    fixed   = labels.copy()
    changed = 0

    while True:
        runs = _rebuild(fixed)
        local_changed = 0

        for ri in range(1, len(runs) - 1):
            prev_run = runs[ri - 1]
            curr_run = runs[ri]
            next_run = runs[ri + 1]

            if curr_run["label"] != 0:
                continue   # only holds

            run_len = curr_run["end"] - curr_run["start"] + 1
            if run_len > SPIKE_MAX:
                continue

            pl = prev_run["label"]
            nl = next_run["label"]

            if pl == nl and pl != 0:   # same direction on both sides
                for pos in range(curr_run["start"], curr_run["end"] + 1):
                    fixed[pos] = pl
                local_changed += run_len

        changed += local_changed
        if local_changed == 0:
            break

    return fixed, changed


# ─── main entry point ─────────────────────────────────────────────────────────

def fix_pivot_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies all three correction passes in sequence and returns a corrected copy.
    Required columns: open, high, low, close, label, label_name
    """
    #df     = df.copy()
    labels = df["label"].to_numpy(dtype=np.int8)

    print("Starting label corrections…")
    print(f"  Input rows: {len(df):,}")
    _report(df, "before")

    # ── pass 1
    labels, n1 = _fix_direct_transitions(labels)
    print(f"\n  Pass 1 (direct illegal transitions)  → {n1:,} candles changed")

    # ── pass 2
    labels, n2 = _fix_pivot_strays(labels)
    print(f"  Pass 2 (stray opposite labels)       → {n2:,} candles changed")

    # ── pass 3
    labels, n3 = _fix_continuation_holds(labels)
    print(f"  Pass 3 (continuation hold merges)    → {n3:,} candles changed")

    # Write back
    df["label"]      = labels.astype(int)
    df["label_name"] = df["label"].map(LABEL_MAP)

    total = n1 + n2 + n3
    print(f"\n  Total corrections: {total:,}  ({total/len(df)*100:.2f}% of rows)")
    _report(df, "after")

    # ── validation
    arr = df["label"].to_numpy()
    bad = np.where(
        ((arr[:-1] == 1) & (arr[1:] == 2)) |
        ((arr[:-1] == 2) & (arr[1:] == 1))
    )[0]
    print(f"\n  Remaining illegal transitions: {len(bad)}")
    if len(bad):
        print("  WARNING – some transitions could not be resolved automatically.")
        print("  First few at row indices:", bad[:10].tolist())

    return df


