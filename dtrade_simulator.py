import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from tensorflow.keras.models import load_model
import os
from scaler3d2d import preprocess_and_save_scalers,transform_data, inverse_transform_data, create_sequences
# --- Configuration ---

SEQ_LEN = 240
PRED_STEPS = 5
SYMBOL_TO_SIMULATE = 'EURUSD' # <-- CHANGE THIS SYMBOL TO TEST DIFFERENT MARKETS
N_CLUSTERS = 5
SL_BUFFER_PIPS = 20
PRICE_NEAR_CLUSTER_PIPS = 10
BREAKEVEN_PROFIT_PIPS = 10
TP_RISK_REWARD_RATIO = 5

average_gap = 0 # Corrected gap from real and predicted

# --- Load Model and Data ---
def load_data_and_model(symbol):
    """Loads the appropriate data and model based on the symbol."""
    
    model_path = f'Generated{symbol} lstm_best.keras'
    data_path_h1 = f'Generated{symbol} dbot.csv'
        

    required_files = [model_path, data_path_h1]
    for f in required_files:
        if not os.path.exists(f):
            print(f"Error: Required file not found: {f}")
            print("Please ensure all necessary files are in the same directory as this script.")
            exit()

    print(f"Loading H1 data from {data_path_h1}...")
    df_h1 = pd.read_csv(data_path_h1)
    
    X, y = create_sequences(df_h1.values, SEQ_LEN, PRED_STEPS)  # X: (n_samples, SEQ_LEN, n_features), y: (n_samples, n_features)
    X, y = preprocess_and_save_scalers(X, y,f'{symbol} scaler_x.joblib',f'{symbol} scaler_y.joblib')
    print(f"Loading model from {model_path}...")
    model = load_model(model_path)
    y_pred = model.predict(X[-1000:])
    #y_test = y[-3000:]

    _,y_test = inverse_transform_data(scaled_y=y[-1000:], scaler_y_filename=f'{symbol} scaler_y.joblib')
    _,y_pred = inverse_transform_data(scaled_y=y_pred, scaler_y_filename=f'{symbol} scaler_y.joblib')
    #print(y_test, " comparing to ", y_pred)
    # 1. Calculate the error (the gap between actual and predicted)
    error = y_test - y_pred

    # 2. Calculate the average gap (the bias)
    average_gap = np.mean(error)
    print(f"Average Gap (Bias): {average_gap:.7f}")

    # 3. Add the average gap to your predictions to create a corrected version
    y_pred_corrected = y_pred + average_gap

    # --- Verification ---
    # Let's check the first predicted value vs. the first actual value
    print("\n--- Example of Correction ---")
    print(f"Original Prediction: {y_pred[0, 0]:.7f}")
    print(f"Corrected Prediction: {y_pred_corrected[0, 0]:.7f}")
    print(f"Actual Value:        {y_test[0, 0]:.7f}")

    return df_h1, model

# --- Helper Functions ---

def get_pip_value(symbol):
    """
    Returns the pip value for a given symbol.
    This is the core of the dynamic pip update.
    """
    # Normalize symbol by removing '.Daily' suffix if it exists
    symbol_clean = symbol.split('.')[0]

    # JPY pairs
    if 'JPY' in symbol_clean:
        return 0.01
    
    # Gold (XAU) and many Oil/Index pairs
    if symbol_clean in ['XAUUSD', 'XBRUSD', 'XTIUSD']:
        return 0.01

    # Silver (XAG) and Natural Gas (XNG)
    if symbol_clean in ['XAGUSD', 'XNGUSD']:
        return 0.001

    # Cryptocurrencies (treating the second decimal as the standard "pip")
    crypto_list = [
        'BTCUSD', 'ETHUSD', 'LTCUSD', 'XRPUSD', 'BCHUSD', 'AAVEUSD', 'ADAUSD',
        'ALGOUSD', 'ATOMUSD', 'AVAXUSD', 'AXSUSD', 'BNBUSD', 'DASHUSD', 'DOGEUSD',
        'DOTUSD', 'FILUSD', 'GRTUSD', 'ICPUSD', 'IOTAUSD', 'LINKUSD', 'LRCUSD',
        'MANAUSD', 'NEARUSD', 'SOLUSD', 'UNIUSD', 'ZECUSD', 'ETCUSD', 'TRXUSD',
        'FETUSD', 'ARBUSD', 'APTUSD', 'SUIUSD'
    ]
    if symbol_clean in crypto_list:
        return 0.01

    # Default for most Forex pairs (EURUSD, GBPUSD, AUDUSD, etc.) and many exotics
    return 0.0001

def pips_to_price(pips, symbol):
    """Convert pips to price value based on the symbol."""
    return pips * get_pip_value(symbol)

def price_to_pips(price_diff, symbol):
    """Convert a price difference to pips based on the symbol."""
    return price_diff / get_pip_value(symbol)

def get_nearest_cluster(current_price, cluster_centers):
    """Find the cluster center nearest to current price."""
    distances = [abs(c[0] - current_price) for c in cluster_centers]
    nearest_idx = distances.index(min(distances))
    return cluster_centers[nearest_idx][0], nearest_idx

def is_price_near_cluster(current_price, cluster_value, tolerance_pips, symbol):
    """Check if price is at or very close to a cluster level."""
    tolerance = pips_to_price(tolerance_pips, symbol)
    return abs(current_price - cluster_value) <= tolerance

# --- Trading Simulation ---
def run_simulation(data_h1, model, symbol):
    """Runs the backtesting trading simulation."""
    print("\nStarting trading simulation...")
    print(f"SYMBOL: {symbol} | Pip Value: {get_pip_value(symbol)}")
    print(f"SL Buffer: {SL_BUFFER_PIPS} pips | Entry Tolerance: {PRICE_NEAR_CLUSTER_PIPS} pips")
    print(f"Breakeven: {BREAKEVEN_PROFIT_PIPS} pips | Risk:Reward Ratio: 1:{TP_RISK_REWARD_RATIO}\n")
    
    in_trade = False
    trade_entry_price = 0
    trade_entry_tick = 0
    trade_type = None
    current_breakeven = 0
    take_profit = 0
    stop_loss = 0
    trade_history = []
    balance = 100

    plt.ion()
    fig, ax = plt.subplots(figsize=(15, 8))

    start_index = SEQ_LEN
    if start_index >= len(data_h1):
        print("Error: Not enough H1 data to start the simulation based on SEQ_LEN.")
        return

    for i in range(start_index, len(data_h1)):
        current_price = data_h1['close'].iloc[i]

        cluster_data = data_h1['close'].iloc[i - SEQ_LEN:i].values.reshape(-1, 1)
        kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=0, n_init='auto').fit(cluster_data)
        cluster_centers = sorted(kmeans.cluster_centers_, key=lambda x: x[0])

        prediction_data_sequence = data_h1[['open', 'high', 'low', 'close']].iloc[i - SEQ_LEN:i].values
        model_input = np.reshape(prediction_data_sequence, (1, SEQ_LEN, 4))
        model_input, _ = transform_data(model_input,scaler_x_filename=f'{symbol} scaler_x.joblib')
        
        predictions = model.predict(model_input, verbose=0)
        _,predictions = inverse_transform_data(scaled_y= predictions, scaler_y_filename=f'{symbol} scaler_y.joblib')
        predictions = predictions + average_gap
        predictions = predictions[0] 
        first_pred = predictions[0]

        nearest_cluster, cluster_idx = get_nearest_cluster(current_price, cluster_centers)

        if in_trade:
            if trade_type == 'long':
                current_profit_pips = price_to_pips(current_price - trade_entry_price, symbol)
            else:
                current_profit_pips = price_to_pips(trade_entry_price - current_price, symbol)
            
            if current_profit_pips >= BREAKEVEN_PROFIT_PIPS:
                new_breakeven = (current_price + trade_entry_price) / 2
                if trade_type == 'long':
                    if new_breakeven > current_breakeven and new_breakeven < current_price:
                        current_breakeven = new_breakeven
                        stop_loss = current_breakeven
                        profit_pips = price_to_pips(current_price - trade_entry_price, symbol)
                        print(f"Tick {i}: 🛡️ BREAKEVEN (LONG) - In profit by {profit_pips:.1f} pips - SL moved to: {stop_loss:.5f}")
                elif trade_type == 'short':
                    if (current_breakeven == 0 or new_breakeven < current_breakeven) and new_breakeven > current_price:
                        current_breakeven = new_breakeven
                        stop_loss = current_breakeven
                        profit_pips = price_to_pips(trade_entry_price - current_price, symbol)
                        print(f"Tick {i}: 🛡️ BREAKEVEN (SHORT) - In profit by {profit_pips:.1f} pips - SL moved to: {stop_loss:.5f}")

            trade_closed = False
            profit_loss = 0
            exit_reason = ""
            
            # NOTE: The P/L calculation using '* 1000' might need adjustment
            # depending on your desired contract size/lot size.
            # This example assumes a fixed multiplier for simplicity.
            if trade_type == 'long':
                if current_price >= take_profit:
                    exit_reason = "TAKE PROFIT HIT (LONG)"
                    profit_loss = (take_profit - trade_entry_price) * 1000
                    trade_closed = True
                elif current_price <= stop_loss:
                    exit_reason = "STOP LOSS HIT (LONG)"
                    profit_loss = (stop_loss - trade_entry_price) * 1000
                    trade_closed = True
            elif trade_type == 'short':
                if current_price <= take_profit:
                    exit_reason = "TAKE PROFIT HIT (SHORT)"
                    profit_loss = (trade_entry_price - take_profit) * 1000
                    trade_closed = True
                elif current_price >= stop_loss:
                    exit_reason = "STOP LOSS HIT (SHORT)"
                    profit_loss = (trade_entry_price - stop_loss) * 1000
                    trade_closed = True

            if trade_closed:
                exit_price = take_profit if "TAKE PROFIT" in exit_reason else stop_loss
                print(f"Tick {i}: *** {exit_reason} at {exit_price:.5f} ***")
                balance += profit_loss
                trade_history.append({
                    'entry_tick': trade_entry_tick, 'exit_tick': i, 'entry_price': trade_entry_price,
                    'exit_price': exit_price, 'trade_type': trade_type, 'profit_loss': profit_loss,
                    'balance': balance, 'exit_reason': exit_reason
                })
                print(f"P/L: ${profit_loss:.2f} | New Balance: ${balance:.2f}\n")
                in_trade = False
                trade_type = None
                current_breakeven = 0
        
        if not in_trade:
            is_near_support = is_price_near_cluster(current_price, nearest_cluster, PRICE_NEAR_CLUSTER_PIPS, symbol)
            
            if (np.all(predictions > nearest_cluster) and is_near_support and
                current_price >= nearest_cluster - pips_to_price(PRICE_NEAR_CLUSTER_PIPS, symbol)):
                
                in_trade = True
                trade_type = 'long'
                trade_entry_price = current_price
                trade_entry_tick = i
                
                stop_loss = nearest_cluster - pips_to_price(SL_BUFFER_PIPS, symbol)
                sl_distance = trade_entry_price - stop_loss
                take_profit = trade_entry_price + (sl_distance * TP_RISK_REWARD_RATIO)
                #ake_profit = predictions[-1]
                current_breakeven = 0

                print(f"\n{'='*60}")
                print(f"Tick {i}: 🟢 LONG ENTRY (Following Bullish Prediction)")
                print(f"{'='*60}")
                print(f"Entry Price:           {trade_entry_price:.5f}")
                print(f"Support Cluster:       {nearest_cluster:.5f}")
                print(f"Stop Loss:             {stop_loss:.5f} ({SL_BUFFER_PIPS} pips below support)")
                print(f"SL Distance:           {price_to_pips(sl_distance, symbol):.1f} pips")
                print(f"Take Profit:           {take_profit:.5f}")
                print(f"{'='*60}\n")
                    
            elif (np.all(predictions < nearest_cluster) and is_near_support and
                  current_price <= nearest_cluster + pips_to_price(PRICE_NEAR_CLUSTER_PIPS, symbol)):
                
                in_trade = True
                trade_type = 'short'
                trade_entry_price = current_price
                trade_entry_tick = i
                
                stop_loss = nearest_cluster + pips_to_price(SL_BUFFER_PIPS, symbol)
                sl_distance = stop_loss - trade_entry_price
                take_profit = trade_entry_price - (sl_distance * TP_RISK_REWARD_RATIO)
                #take_profit = predictions[-1]
                current_breakeven = 0
                
                print(f"\n{'='*60}")
                print(f"Tick {i}: 🔴 SHORT ENTRY (Following Bearish Prediction)")
                print(f"{'='*60}")
                print(f"Entry Price:        {trade_entry_price:.5f}")
                print(f"Resistance Cluster: {nearest_cluster:.5f}")
                print(f"Stop Loss:          {stop_loss:.5f} ({SL_BUFFER_PIPS} pips above resistance)")
                print(f"SL Distance:        {price_to_pips(sl_distance, symbol):.1f} pips")
                print(f"Take Profit:        {take_profit:.5f}")
                print(f"{'='*60}\n")

        # --- Plotting ---
        ax.clear()
        plot_start = max(start_index, i - 200)
        plot_range = range(plot_start, i+1)
        
        actual_prices = data_h1['close'].iloc[plot_start:i+1].values
        ax.plot(plot_range, actual_prices, label='Actual Close Price', color='blue', linewidth=2, zorder=10)
        
        ax.plot(i, first_pred, 'ro', markersize=8, label='Model 1st Prediction', zorder=11, alpha=0.7)
        ax.plot([i, i], [current_price, first_pred], 'r--', linewidth=1, alpha=0.5)
        
        pred_diff_pips = price_to_pips(first_pred - current_price, symbol)
        ax.text(i, first_pred, f'  Pred: {first_pred:.5f}\n  ({pred_diff_pips:+.1f} pips)', fontsize=7, color='red', va='center')
        
        for j, center in enumerate(cluster_centers):
            ax.axhline(y=center[0], color='orange', linestyle='--', linewidth=1.5, alpha=0.7, zorder=3)
        
        ax.axhline(y=nearest_cluster, color='purple', linestyle='-', linewidth=2, label=f'Nearest Cluster: {nearest_cluster:.5f}', zorder=4)
        
        if in_trade:
            entry_x = trade_entry_tick if trade_entry_tick >= plot_start else plot_start
            ax.plot(entry_x, trade_entry_price, 'go' if trade_type == 'long' else 'ro', markersize=12, label=f'{trade_type.upper()} Entry', zorder=15)
            ax.axhline(y=take_profit, color='green', linestyle=':', linewidth=2.5, label=f'TP: {take_profit:.5f}', zorder=5)
            ax.axhline(y=stop_loss, color='red', linestyle='--', linewidth=2.5, label=f'SL: {stop_loss:.5f}', zorder=5)
            
            if current_breakeven > 0:
                ax.axhline(y=current_breakeven, color='cyan', linestyle='-', linewidth=2, label=f'Breakeven SL: {current_breakeven:.5f}', zorder=4)
            
            if trade_type == 'long':
                current_pl_pips = price_to_pips(current_price - trade_entry_price, symbol)
            else:
                current_pl_pips = price_to_pips(trade_entry_price - current_price, symbol)
            
            pl_color = 'green' if current_pl_pips > 0 else 'red'
            ax.text(0.02, 0.88, f'Current P/L: {current_pl_pips:.1f} pips', transform=ax.transAxes, fontsize=10, verticalalignment='top', color=pl_color, fontweight='bold', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.set_title(f'{symbol} - Tick: {i} | Balance: ${balance:.2f}', fontsize=12, fontweight='bold')
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.pause(0.01)

    plt.ioff()
    plt.show()

    print("\nSIMULATION FINISHED\n")
    if trade_history:
        trade_df = pd.DataFrame(trade_history)
        print("--- TRADE HISTORY ---")
        print(trade_df.to_string(index=False))
        # Add performance summary if needed...

# You just need to change the symbol string here to run the simulation on a different market
market_data_h1, lstm_model = load_data_and_model(SYMBOL_TO_SIMULATE)
run_simulation(market_data_h1, lstm_model, SYMBOL_TO_SIMULATE)  