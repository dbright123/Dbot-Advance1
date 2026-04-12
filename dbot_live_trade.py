import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from tensorflow.keras.models import load_model
import random
import time
import os
from datetime import datetime
from scaler3d2d import create_sequences, transform_data, inverse_transform_data
import threading


# --- Configuration ---
SYMBOL_TO_TRADE = 'Volatility 75 Index'
TIMEFRAME = mt5.TIMEFRAME_M5 
LOT_SIZE = 0.01
SEQ_LEN = 240
PRED_STEPS = 1
N_CLUSTERS = 5
SL_BUFFER_PIPS = 20
PRICE_NEAR_CLUSTER_PIPS = 10
BREAKEVEN_PROFIT_PIPS = 5
TP_RISK_REWARD_RATIO = 5
MAGIC_NUMBER = random.randrange(1, 1000000) # Unique identifier for trades placed by this EA
n_feature = 5
class MT5Trader:
    def __init__(self, symbol, timeframe, lot_size):
        self.symbol = symbol
        self.timeframe = timeframe
        self.lot_size = lot_size
        self.pip_value = None # Will be set after MT5 initialization
        self.model = None
        self.average_gap = 0
        self.plot_data = {
            'price_data': None,
            'clusters': None,
            'nearest_cluster': None,
            'prediction': None,
            'open_position': None
        }
        self.plot_lock = threading.Lock()
        self.plot_running = False
        self.plot_thread = None

    def _get_pip_value(self):
        """Determines the pip value for the symbol. Assumes MT5 is initialized."""
        symbol_info = mt5.symbol_info(self.symbol)
        if symbol_info is None:
            print(f"Could not find symbol info for {self.symbol}. Exiting.")
            print("Please check if the symbol name matches your broker's (e.g., 'AUDUSD.m') and is visible in Market Watch.")
            return None
        
        if "JPY" in self.symbol.upper():
            return 0.01
        elif symbol_info.digits == 5 or symbol_info.digits == 3:
            return 0.0001 if symbol_info.digits == 5 else 0.001
        else:
            return 0.01

    def pips_to_price(self, pips):
        return pips * self.pip_value

    def price_to_pips(self, price_diff):
        return price_diff / self.pip_value

    def initialize_mt5(self):
        """Initializes connection to MetaTrader 5 and selects the symbol."""
        if not mt5.initialize():
            print("initialize() failed, error code =", mt5.last_error())
            return False
        print("MetaTrader 5 connection initialized.")
        
        # Ensure the symbol is available and selected in Market Watch
        if not mt5.symbol_select(self.symbol, True):
            print(f"Failed to select {self.symbol}. The symbol may not exist on your broker's server or is not in the Market Watch.")
            mt5.shutdown()
            return False
            
        return True
    
    def load_model_and_scalers(self):
        """Loads the pre-trained model and calculates the prediction gap."""
        model_path = f'Generated{self.symbol} lstm_best.keras'
        scaler_y_path = f'{self.symbol} scaler_y.joblib'
        
        if not os.path.exists(model_path) or not os.path.exists(scaler_y_path):
            print(f"Error: Model or scaler not found for {self.symbol}.")
            return False
            
        print(f"Loading model from {model_path}...")
        self.model = load_model(model_path)
        
        # Calculate average prediction gap (bias correction)
        print("Calculating model prediction bias...")
        data = self.get_market_data(self.symbol, self.timeframe, SEQ_LEN + 1000) # Get more data for calculation
        if data is None or len(data) < SEQ_LEN + PRED_STEPS:
            print("Not enough data to calculate bias.")
            return False

        X, y = create_sequences(data[["open", "high","low", "close","tick_volume"]].values, SEQ_LEN, PRED_STEPS, 3)
        
        _, y_transformed = transform_data(data_y=y[-(SEQ_LEN):], scaler_y_filename=scaler_y_path)
        y_pred_transformed = self.model.predict(transform_data(X[-(SEQ_LEN):], scaler_x_filename=f'{self.symbol} scaler_x.joblib')[0])

        _, y_test = inverse_transform_data(scaled_y=y_transformed, scaler_y_filename=scaler_y_path)
        _, y_pred = inverse_transform_data(scaled_y=y_pred_transformed, scaler_y_filename=scaler_y_path)
        
        error = y_test - y_pred
        self.average_gap = np.mean(error)
        print(f"Average Prediction Gap (Bias Correction): {self.average_gap:.7f}")
        return True

    def get_market_data(self, symbol, timeframe, count):
        """Fetches historical price data from MT5."""
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
        if rates is None:
            print("Failed to get rates:", mt5.last_error())
            return None
        return pd.DataFrame(rates)

    def get_prediction(self, data):
        """Generates a price prediction using the loaded model."""
        sequence = data[["open", "high","low", "close","tick_volume"]].values
        model_input = np.reshape(sequence, (1, SEQ_LEN, n_feature))
        
        model_input_scaled, _ = transform_data(model_input, scaler_x_filename=f'{self.symbol} scaler_x.joblib')
        
        predictions_scaled = self.model.predict(model_input_scaled, verbose=0)
        
        _, predictions = inverse_transform_data(scaled_y=predictions_scaled, scaler_y_filename=f'{self.symbol} scaler_y.joblib')
        
        # Apply the bias correction
        #corrected_predictions = predictions + self.average_gap
        #return corrected_predictions[0]
        return predictions

    def execute_trade(self, trade_type, price, sl, tp):
        """Places a trade order in MT5."""
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": self.lot_size,
            "type": trade_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "magic": MAGIC_NUMBER,
            "comment": "Python LSTM Trade",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK,
        }
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"Order send failed, retcode={result.retcode}")
        return result

    def get_open_trades(self):
        """Retrieves currently open trades for this EA."""
        positions = mt5.positions_get(symbol=self.symbol)
        if positions is None:
            return []
        
        # Filter positions by magic number
        return [p for p in positions]

    def close_trade(self, position):
        """Closes an open position."""
        trade_type = mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
        price = mt5.symbol_info_tick(self.symbol).ask if position.type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(self.symbol).bid
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": position.volume,
            "type": trade_type,
            "position": position.ticket,
            "price": price,
            "magic": MAGIC_NUMBER,
            "comment": "Close Python Trade",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK,
        }
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"Close order failed, retcode={result.retcode}")
        return result

    def manage_breakeven(self, position):
        """Manages breakeven for a profitable trade."""
        if position.type == mt5.ORDER_TYPE_BUY: # Long position
            current_profit_pips = self.price_to_pips(position.price_current - position.price_open)
            if current_profit_pips > BREAKEVEN_PROFIT_PIPS:
                new_sl = position.price_open + self.pips_to_price(1) # Move SL to entry + 1 pip
                if position.sl < new_sl:
                    print(f"🛡️ Moving SL to breakeven for LONG position #{position.ticket}")
                    self.modify_position(position.ticket, new_sl, position.tp)

        elif position.type == mt5.ORDER_TYPE_SELL: # Short position
            current_profit_pips = self.price_to_pips(position.price_open - position.price_current)
            if current_profit_pips > BREAKEVEN_PROFIT_PIPS:
                new_sl = position.price_open - self.pips_to_price(1) # Move SL to entry - 1 pip
                if position.sl > new_sl:
                    print(f"🛡️ Moving SL to breakeven for SHORT position #{position.ticket}")
                    self.modify_position(position.ticket, new_sl, position.tp)

    def modify_position(self, ticket, sl, tp):
        """Modifies the SL/TP of an open position."""
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "sl": sl,
            "tp": tp,
        }
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"Modify failed, retcode={result.retcode}, comment={result.comment}")

    def plot_worker(self):
        """Background worker for plotting to prevent blocking."""
        plt.ion()
        fig, ax = plt.subplots(figsize=(15, 8))
        self.plot_running = True
        
        while self.plot_running:
            with self.plot_lock:
                if self.plot_data['price_data'] is not None:
                    ax.clear()
                    
                    price_data = self.plot_data['price_data']
                    clusters = self.plot_data['clusters']
                    nearest_cluster = self.plot_data['nearest_cluster']
                    prediction = self.plot_data['prediction']
                    open_position = self.plot_data['open_position']
                    
                    prices = price_data['close']
                    ax.plot(prices.index, prices, label='Close Price', color='blue', linewidth=2)
                    
                    # Plot clusters
                    if clusters is not None:
                        for center in clusters:
                            ax.axhline(y=center[0], color='orange', linestyle='--', linewidth=1.5, alpha=0.7)
                        ax.axhline(y=nearest_cluster, color='purple', linestyle='-', linewidth=2, label=f'Nearest Cluster: {nearest_cluster:.5f}')
                    
                    # Plot prediction
                    if prediction is not None:
                        ax.plot(prices.index[-1], prediction, 'ro', markersize=8, label='Prediction')

                    # Plot open trade
                    if open_position:
                        entry_price = open_position.price_open
                        trade_type = "LONG" if open_position.type == mt5.ORDER_TYPE_BUY else "SHORT"
                        ax.plot(prices.index[-1], entry_price, 'go' if trade_type == "LONG" else 'ro', markersize=12, label=f'{trade_type} Entry')
                        ax.axhline(y=open_position.sl, color='red', linestyle='--', linewidth=2, label=f'SL: {open_position.sl:.5f}')
                        ax.axhline(y=open_position.tp, color='green', linestyle=':', linewidth=2, label=f'TP: {open_position.tp:.5f}')
                    
                    ax.set_title(f'{self.symbol} Live Trading | Last Update: {datetime.now().strftime("%H:%M:%S")}')
                    ax.legend(loc='upper left')
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    
                    # Non-blocking plot update
                    fig.canvas.draw()
                    fig.canvas.flush_events()
            
            time.sleep(1)  # Update plot every 2 seconds

    def update_plot(self, price_data, clusters, nearest_cluster, prediction, open_position):
        """Updates plot data without blocking."""
        with self.plot_lock:
            self.plot_data.update({
                'price_data': price_data,
                'clusters': clusters,
                'nearest_cluster': nearest_cluster,
                'prediction': prediction,
                'open_position': open_position
            })

    def start_plotting(self):
        """Starts the plotting thread."""
        self.plot_thread = threading.Thread(target=self.plot_worker, daemon=True)
        self.plot_thread.start()
        print("Plotting thread started...")

    def stop_plotting(self):
        """Stops the plotting thread."""
        self.plot_running = False
        if self.plot_thread and self.plot_thread.is_alive():
            self.plot_thread.join(timeout=2)
        plt.close('all')

    def run(self):
        """Main trading loop."""
        try:
            if not self.initialize_mt5():
                return
            
            self.pip_value = self._get_pip_value()
            if self.pip_value is None:
                return # Error message is printed in _get_pip_value

            if not self.load_model_and_scalers():
                return

            print("\n--- Starting Live Trading Bot ---")
            print(f"Symbol: {self.symbol} | Lot Size: {self.lot_size} | Timeframe: {self.timeframe}")
            
            # Start plotting thread
            self.start_plotting()
            
            while True:
                # --- Data & Prediction ---
                market_data = self.get_market_data(self.symbol, self.timeframe, SEQ_LEN)
                if market_data is None or len(market_data) < SEQ_LEN:
                    time.sleep(5)
                    continue

                current_price = market_data['close'].iloc[-1]
                
                predictions = self.get_prediction(market_data)
                first_pred = predictions[0]


                # --- Clustering (Support/Resistance) ---
                cluster_data = market_data['close'].values.reshape(-1, 1)
                kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=0, n_init='auto').fit(cluster_data)
                cluster_centers = sorted(kmeans.cluster_centers_, key=lambda x: x[0])
                
                distances = [abs(c[0] - current_price) for c in cluster_centers]
                nearest_cluster = cluster_centers[np.argmin(distances)][0]

                # --- Trade Management ---
                open_trades = self.get_open_trades()
                
                if open_trades:
                    # Manage existing trade
                    self.manage_breakeven(open_trades[0])
                    # Update plot with current data
                    self.update_plot(market_data, cluster_centers, nearest_cluster, first_pred, open_trades[0])

                else:
                     # --- New Trade Logic ---
                    is_near_cluster = abs(current_price - nearest_cluster) <= self.pips_to_price(PRICE_NEAR_CLUSTER_PIPS)

                    # Buy Condition
                    if np.all(predictions > nearest_cluster) and is_near_cluster:
                        print("🟢 BUY SIGNAL DETECTED")
                        sl = nearest_cluster - self.pips_to_price(SL_BUFFER_PIPS)
                        tick = mt5.symbol_info_tick(self.symbol)
                        sl_distance = tick.ask - sl
                        tp = tick.ask + (sl_distance * TP_RISK_REWARD_RATIO)
                        self.execute_trade(mt5.ORDER_TYPE_BUY, tick.ask, sl, tp)

                    # Sell Condition
                    elif np.all(predictions < nearest_cluster) and is_near_cluster:
                        print("🔴 SELL SIGNAL DETECTED")
                        sl = nearest_cluster + self.pips_to_price(SL_BUFFER_PIPS)
                        tick = mt5.symbol_info_tick(self.symbol)
                        sl_distance = sl - tick.bid
                        tp = tick.bid - (sl_distance * TP_RISK_REWARD_RATIO)
                        self.execute_trade(mt5.ORDER_TYPE_SELL, tick.bid, sl, tp)

                    # Update plot with current data
                    self.update_plot(market_data, cluster_centers, nearest_cluster, first_pred, None)

                # Wait for the next bar
                print(f"Last price: {current_price:.5f} | Nearest Cluster: {nearest_cluster:.5f} | Prediction: {predictions} | Waiting for next bar...")
                time.sleep(30) # Check every minute for new bar

        except KeyboardInterrupt:
            print("Bot stopped by user.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
        finally:
            print("Shutting down MT5 connection and stopping plot...")
            self.stop_plotting()
            mt5.shutdown()


trader = MT5Trader(symbol=SYMBOL_TO_TRADE, 
                       timeframe=TIMEFRAME, 
                       lot_size=LOT_SIZE)
trader.run()