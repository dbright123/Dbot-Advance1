import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class TradingSimulator:
    def __init__(self, model_path, scaler_path=None, initial_balance=10000):
        """
        Initialize trading simulator
        
        Args:
            model_path: Path to saved RandomForest model
            scaler_path: Path to saved StandardScaler (optional)
            initial_balance: Starting account balance in USD
        """
        # Load trained model
        self.model = joblib.load(model_path)
        
        # Load scaler if provided
        self.scaler = None
        if scaler_path:
            self.scaler = joblib.load(scaler_path)
        
        # Trading parameters
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.position = None  # None, 'long', or 'short'
        self.position_size = 0
        self.entry_price = 0
        
        # Simulation tracking
        self.trades = []
        self.equity_curve = []
        self.predictions = []
        self.actual_labels = []
        
        # Performance metrics
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.max_drawdown = 0
        self.max_profit = 0
        
    def engineer_features(self, df):
        """Engineer time-based features"""
        df['time'] = pd.to_datetime(df['time'], unit='s', errors='coerce')
        df['month'] = df['time'].dt.month
        df['day'] = df['time'].dt.day
        df['hour'] = df['time'].dt.hour
        df['minute'] = df['time'].dt.minute
        df['day_of_week'] = df['time'].dt.day_of_week
        df = df.drop(columns=['time'])
        df = df.dropna().reset_index(drop=True)
        return df
    
    def add_technical_indicators(self, df):
        """Add technical indicators"""
        # RSI
        def calculate_rsi(data, period=14):
            delta = data.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        
        df['rsi'] = calculate_rsi(df['close'])
        
        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Moving Averages
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        
        # ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['atr'] = true_range.rolling(14).mean()
        
        # Momentum
        df['momentum'] = df['close'] - df['close'].shift(4)
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # ROC
        df['roc'] = ((df['close'] - df['close'].shift(10)) / df['close'].shift(10)) * 100
        
        # Stochastic
        n_k = 14
        lowest_low = df['low'].rolling(window=n_k).min()
        highest_high = df['high'].rolling(window=n_k).max()
        denom = (highest_high - lowest_low).replace(0, np.nan)
        df['stoch_k'] = 100 * ((df['close'] - lowest_low) / denom)
        df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
        
        # Williams %R
        df['williams_r'] = -100 * ((highest_high - df['close']) / denom)
        
        # CCI
        n_cci = 20
        tp = (df['high'] + df['low'] + df['close']) / 3.0
        sma_tp = tp.rolling(window=n_cci).mean()
        mean_dev = tp.rolling(window=n_cci).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
        df['cci'] = (tp - sma_tp) / (0.015 * mean_dev.replace(0, np.nan))
        
        # OBV
        direction = np.sign(df['close'].diff()).fillna(0)
        df['obv'] = (direction * df['volume']).fillna(0).cumsum()
        
        # Drop rows with NaN values
        df = df.dropna()
        
        return df
    
    def prepare_data_for_simulation(self, df):
        """
        Prepare data for simulation (includes feature engineering)
        
        Args:
            df: DataFrame with historical data
        """
        # Make a copy to avoid modifying original
        df_sim = df.copy()
        
        # Engineer features
        df_sim = self.engineer_features(df_sim)
        
        # Add technical indicators
        df_sim = self.add_technical_indicators(df_sim)
        
        return df_sim
    
    def get_features_for_prediction(self, df):
        """
        Extract features for prediction (excluding label and time columns)
        """
        # Drop non-feature columns
        feature_columns = [col for col in df.columns if col not in ['label', 'close_smoothed', 'label_name']]
        features = df[feature_columns]
        
        return features
    
    def simulate_trading(self, historical_data, label_col='label', 
                         trade_size_pct=0.1, stop_loss_pct=0.02, take_profit_pct=0.03,
                         commission=0.001, slippage=0.0005):
        """
        Simulate trading on historical data
        
        Args:
            historical_data: DataFrame with historical data including labels
            label_col: Column name for actual labels
            trade_size_pct: Percentage of balance to risk per trade
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
            commission: Trading commission per trade
            slippage: Price slippage percentage
        """
        # Prepare data
        print("Preparing data for simulation...")
        df = self.prepare_data_for_simulation(historical_data)
        
        # Get features
        features = self.get_features_for_prediction(df)
        
        # Scale features if scaler was provided
        if self.scaler is not None:
            features_scaled = self.scaler.transform(features)
        else:
            features_scaled = features.values
        
        # Make predictions
        print("Making predictions...")
        predictions = self.model.predict(features_scaled)
        
        # Map predictions to labels
        label_map = {0: 'hold', 1: 'buy', 2: 'sell'}
        prediction_labels = [label_map[p] for p in predictions]
        
        # Reset simulator state
        self._reset_simulator()
        
        # Track previous predictions for consecutive rule
        previous_prediction = None
        consecutive_count = 0
        required_consecutive = 2
        
        print(f"\nStarting simulation with ${self.balance:.2f}")
        print("-" * 80)
        
        # Simulation loop
        for i in range(len(df)):
            current_price = df['close'].iloc[i]
            current_time = df.index[i] if 'index' in df.columns else i
            
            # Get current prediction
            current_pred = predictions[i]
            current_pred_label = prediction_labels[i]
            
            # Update equity curve
            self.equity_curve.append(self._calculate_total_equity(current_price))
            
            # Skip first few rows to ensure we have previous prediction
            if i < required_consecutive:
                previous_prediction = current_pred
                continue
            
            # Check consecutive predictions rule
            if current_pred == previous_prediction:
                consecutive_count += 1
            else:
                consecutive_count = 1
            
            # Determine if we should trade
            should_trade = False
            trade_action = None
            
            # Rule 1: If already in position and opposite signal appears
            if self.position is not None:
                if (self.position == 'long' and current_pred_label == 'sell') or \
                   (self.position == 'short' and current_pred_label == 'buy'):
                    should_trade = True
                    trade_action = 'sell' if self.position == 'long' else 'buy'
            
            # Rule 2: New position with consecutive signals
            elif consecutive_count >= required_consecutive and current_pred_label != 'hold':
                should_trade = True
                trade_action = current_pred_label
            
            # Execute trade if needed
            if should_trade:
                # Close existing position if any
                if self.position is not None:
                    self._close_position(current_price, commission, slippage)
                
                # Open new position
                if trade_action in ['buy', 'sell']:
                    # Calculate position size based on percentage of balance
                    position_value = self.balance * trade_size_pct
                    self.position_size = position_value / current_price
                    
                    # Record entry
                    self.entry_price = current_price
                    self.position = 'long' if trade_action == 'buy' else 'short'
                    
                    # Set stop loss and take profit
                    if self.position == 'long':
                        self.stop_loss = current_price * (1 - stop_loss_pct)
                        self.take_profit = current_price * (1 + take_profit_pct)
                    else:
                        self.stop_loss = current_price * (1 + stop_loss_pct)
                        self.take_profit = current_price * (1 - take_profit_pct)
                    
                    # Record trade
                    trade_record = {
                        'entry_time': current_time,
                        'entry_price': current_price,
                        'position': self.position,
                        'size': self.position_size,
                        'stop_loss': self.stop_loss,
                        'take_profit': self.take_profit,
                        'prediction': current_pred_label
                    }
                    self.trades.append(trade_record)
                    self.total_trades += 1
            
            # Check for stop loss or take profit
            elif self.position is not None:
                if self.position == 'long':
                    if current_price <= self.stop_loss:
                        self._close_position(current_price, commission, slippage, reason='Stop Loss')
                    elif current_price >= self.take_profit:
                        self._close_position(current_price, commission, slippage, reason='Take Profit')
                else:  # short position
                    if current_price >= self.stop_loss:
                        self._close_position(current_price, commission, slippage, reason='Stop Loss')
                    elif current_price <= self.take_profit:
                        self._close_position(current_price, commission, slippage, reason='Take Profit')
            
            # Update previous prediction
            previous_prediction = current_pred
            
            # Store predictions and actuals for evaluation
            self.predictions.append(current_pred)
            if label_col in df.columns:
                self.actual_labels.append(df[label_col].iloc[i])
        
        # Close any open position at the end
        if self.position is not None:
            self._close_position(df['close'].iloc[-1], commission, slippage, reason='End of Simulation')
        
        # Final equity calculation
        final_equity = self._calculate_total_equity(df['close'].iloc[-1])
        
        return {
            'initial_balance': self.initial_balance,
            'final_balance': self.balance,
            'final_equity': final_equity,
            'total_return': ((final_equity / self.initial_balance) - 1) * 100,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0,
            'max_drawdown': self.max_drawdown,
            'trades': pd.DataFrame(self.trades) if self.trades else pd.DataFrame(),
            'equity_curve': self.equity_curve,
            'predictions': self.predictions,
            'actual_labels': self.actual_labels
        }
    
    def _close_position(self, exit_price, commission, slippage, reason='Signal'):
        """Close current position"""
        if self.position is None or self.position_size == 0:
            return
        
        # Apply slippage
        if self.position == 'long':
            exit_price *= (1 - slippage)
        else:
            exit_price *= (1 + slippage)
        
        # Calculate profit/loss
        if self.position == 'long':
            profit = (exit_price - self.entry_price) * self.position_size
        else:  # short
            profit = (self.entry_price - exit_price) * self.position_size
        
        # Apply commission
        trade_value = self.entry_price * self.position_size
        commission_cost = trade_value * commission
        profit -= commission_cost
        
        # Update balance
        self.balance += profit
        
        # Update trade statistics
        if profit > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
        
        # Update trade record
        if self.trades:
            last_trade = self.trades[-1]
            last_trade.update({
                'exit_time': len(self.equity_curve) - 1,
                'exit_price': exit_price,
                'profit': profit,
                'return_pct': (profit / (self.entry_price * self.position_size)) * 100,
                'exit_reason': reason
            })
        
        # Reset position
        self.position = None
        self.position_size = 0
        self.entry_price = 0
    
    def _calculate_total_equity(self, current_price):
        """Calculate total equity (balance + position value)"""
        if self.position is None or self.position_size == 0:
            return self.balance
        
        position_value = self.position_size * current_price
        if self.position == 'short':
            # For short positions, we need to calculate differently
            # Simplified calculation
            position_value = self.position_size * (self.entry_price - current_price)
        
        return self.balance + position_value
    
    def _reset_simulator(self):
        """Reset simulator state"""
        self.balance = self.initial_balance
        self.position = None
        self.position_size = 0
        self.entry_price = 0
        self.trades = []
        self.equity_curve = []
        self.predictions = []
        self.actual_labels = []
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.max_drawdown = 0
        self.max_profit = 0
    
    def analyze_performance(self, results, plot=True):
        """
        Analyze and visualize simulation results
        
        Args:
            results: Dictionary containing simulation results
            plot: Whether to create performance plots
        """
        print("\n" + "="*80)
        print("TRADING SIMULATION RESULTS")
        print("="*80)
        
        # Basic metrics
        print(f"\nInitial Balance: ${results['initial_balance']:.2f}")
        print(f"Final Balance: ${results['final_balance']:.2f}")
        print(f"Final Equity: ${results['final_equity']:.2f}")
        print(f"Total Return: {results['total_return']:.2f}%")
        print(f"\nTotal Trades: {results['total_trades']}")
        print(f"Winning Trades: {results['winning_trades']}")
        print(f"Losing Trades: {results['losing_trades']}")
        print(f"Win Rate: {results['win_rate']:.2f}%")
        
        # Calculate additional metrics
        if results['total_trades'] > 0:
            trades_df = results['trades']
            if not trades_df.empty and 'profit' in trades_df.columns:
                avg_win = trades_df[trades_df['profit'] > 0]['profit'].mean() if len(trades_df[trades_df['profit'] > 0]) > 0 else 0
                avg_loss = trades_df[trades_df['profit'] <= 0]['profit'].mean() if len(trades_df[trades_df['profit'] <= 0]) > 0 else 0
                profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
                max_profit = trades_df['profit'].max()
                max_loss = trades_df['profit'].min()
                
                print(f"Average Win: ${avg_win:.2f}")
                print(f"Average Loss: ${avg_loss:.2f}")
                print(f"Profit Factor: {profit_factor:.2f}")
                print(f"Max Profit: ${max_profit:.2f}")
                print(f"Max Loss: ${max_loss:.2f}")
                
                # Calculate Sharpe Ratio (simplified)
                returns = trades_df['profit'].values
                if len(returns) > 1:
                    sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)  # Annualized
                    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        
        # Model performance metrics
        if len(results['actual_labels']) > 0 and len(results['predictions']) > 0:
            print("\n" + "-"*80)
            print("MODEL PERFORMANCE METRICS")
            print("-"*80)
            
            # Trim predictions to match actual labels length
            min_len = min(len(results['actual_labels']), len(results['predictions']))
            actuals = results['actual_labels'][:min_len]
            preds = results['predictions'][:min_len]
            
            print(f"\nClassification Report:")
            print(classification_report(actuals, preds, target_names=['Hold', 'Buy', 'Sell']))
            
            # Confusion Matrix
            cm = confusion_matrix(actuals, preds)
            accuracy = np.trace(cm) / np.sum(cm)
            print(f"Accuracy: {accuracy:.4f}")
            
            if plot:
                plt.figure(figsize=(10, 4))
                plt.subplot(1, 2, 1)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                           xticklabels=['Hold', 'Buy', 'Sell'],
                           yticklabels=['Hold', 'Buy', 'Sell'])
                plt.title('Confusion Matrix')
                plt.ylabel('Actual')
                plt.xlabel('Predicted')
        
        # Plot equity curve
        if plot and len(results['equity_curve']) > 0:
            plt.figure(figsize=(15, 10))
            
            # Equity curve
            plt.subplot(2, 2, 1)
            plt.plot(results['equity_curve'])
            plt.title('Equity Curve')
            plt.xlabel('Time')
            plt.ylabel('Equity ($)')
            plt.grid(True, alpha=0.3)
            
            # Drawdown
            plt.subplot(2, 2, 2)
            equity_array = np.array(results['equity_curve'])
            running_max = np.maximum.accumulate(equity_array)
            drawdown = (equity_array - running_max) / running_max * 100
            plt.fill_between(range(len(drawdown)), drawdown, 0, color='red', alpha=0.3)
            plt.plot(drawdown, color='red')
            plt.title('Drawdown')
            plt.xlabel('Time')
            plt.ylabel('Drawdown (%)')
            plt.grid(True, alpha=0.3)
            
            # Trade distribution
            if not results['trades'].empty and 'profit' in results['trades'].columns:
                plt.subplot(2, 2, 3)
                profits = results['trades']['profit'].values
                plt.hist(profits, bins=30, edgecolor='black', alpha=0.7)
                plt.axvline(x=0, color='red', linestyle='--', linewidth=2)
                plt.title('Profit Distribution per Trade')
                plt.xlabel('Profit ($)')
                plt.ylabel('Frequency')
                plt.grid(True, alpha=0.3)
                
                # Cumulative profit
                plt.subplot(2, 2, 4)
                cumulative_profit = results['trades']['profit'].cumsum()
                plt.plot(cumulative_profit)
                plt.title('Cumulative Profit')
                plt.xlabel('Trade Number')
                plt.ylabel('Cumulative Profit ($)')
                plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
        
        # Print trade log
        if not results['trades'].empty:
            print("\n" + "-"*80)
            print("TRADE LOG (Last 10 trades)")
            print("-"*80)
            print(results['trades'].tail(10).to_string())
        
        return results


def run_simulation():
    """
    Main function to run trading simulation
    """
    # Load your trained model and scaler
    model_path = 'GeneratedXAUUSDRandom Forest.joblib'  # Update with your model path
    scaler_path = 'GeneratedXAUUSD scaler.joblib'  # Update if you have a scaler
    
    # Load your historical data
    # Assuming you have a DataFrame with your historical data
    # This should be the same format as your training data
    data_path = 'GeneratedXAUUSD dbot.csv'  # Update with your data path
    
    try:
        # Load data (adjust based on your data format)
        print(f"Loading data from {data_path}...")
        historical_data = pd.read_csv(data_path)
        
        # If your data has time in Unix format, convert it
        if 'time' in historical_data.columns:
            historical_data['time'] = pd.to_datetime(historical_data['time'], unit='s', errors='coerce')
        
        print(f"Data loaded: {len(historical_data)} rows")
        print(f"Columns: {historical_data.columns.tolist()}")
        
    except FileNotFoundError:
        print(f"Data file not found at {data_path}")
        print("Creating sample data for demonstration...")
        
        # Create sample data for demonstration
        np.random.seed(42)
        n_samples = 1000
        
        # Create sample features similar to your training data
        historical_data = pd.DataFrame({
            'time': pd.date_range(start='2023-01-01', periods=n_samples, freq='H').astype(int) // 10**9,
            'open': np.random.uniform(1800, 1900, n_samples),
            'high': np.random.uniform(1810, 1920, n_samples),
            'low': np.random.uniform(1790, 1890, n_samples),
            'close': np.random.uniform(1800, 1900, n_samples),
            'volume': np.random.randint(1000, 50000, n_samples),
            'spread': np.random.randint(0, 50, n_samples),
            'label': np.random.choice([0, 1, 2], n_samples, p=[0.7, 0.15, 0.15])  # Simulated labels
        })
        
        print(f"Sample data created: {len(historical_data)} rows")
    
    # Initialize simulator
    print("\nInitializing Trading Simulator...")
    simulator = TradingSimulator(
        model_path=model_path,
        scaler_path=scaler_path,
        initial_balance=1000
    )
    
    # Run simulation with different parameters
    print("\nRunning simulation...")
    
    # Test different trading parameters
    scenarios = [
        {'name': 'Conservative', 'trade_size_pct': 0.05, 'stop_loss_pct': 0.01, 'take_profit_pct': 0.02},
        {'name': 'Moderate', 'trade_size_pct': 0.1, 'stop_loss_pct': 0.02, 'take_profit_pct': 0.03},
        {'name': 'Aggressive', 'trade_size_pct': 0.2, 'stop_loss_pct': 0.03, 'take_profit_pct': 0.05},
    ]
    
    all_results = []
    
    for scenario in scenarios:
        print(f"\n{'='*60}")
        print(f"Testing {scenario['name']} Strategy")
        print(f"{'='*60}")
        
        # Run simulation
        results = simulator.simulate_trading(
            historical_data=historical_data,
            label_col='label',
            trade_size_pct=scenario['trade_size_pct'],
            stop_loss_pct=scenario['stop_loss_pct'],
            take_profit_pct=scenario['take_profit_pct'],
            commission=0.001,  # 0.1% commission
            slippage=0.0005    # 0.05% slippage
        )
        
        # Analyze results
        analyzed_results = simulator.analyze_performance(results, plot=False)
        
        # Store scenario name
        analyzed_results['scenario_name'] = scenario['name']
        all_results.append(analyzed_results)
    
    # Compare all scenarios
    print("\n" + "="*80)
    print("SCENARIO COMPARISON")
    print("="*80)
    
    comparison_data = []
    for result in all_results:
        comparison_data.append({
            'Scenario': result['scenario_name'],
            'Final Equity': f"${result['final_equity']:.2f}",
            'Total Return': f"{result['total_return']:.2f}%",
            'Total Trades': result['total_trades'],
            'Win Rate': f"{result['win_rate']:.2f}%",
            'Max Drawdown': f"{result.get('max_drawdown', 0):.2f}%",
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print("\n" + comparison_df.to_string(index=False))
    
    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Return comparison
    returns = [r['total_return'] for r in all_results]
    scenarios_names = [r['scenario_name'] for r in all_results]
    
    axes[0, 0].bar(scenarios_names, returns, color=['green', 'blue', 'orange'])
    axes[0, 0].set_title('Total Return by Scenario')
    axes[0, 0].set_ylabel('Return (%)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Win rate comparison
    win_rates = [r['win_rate'] for r in all_results]
    axes[0, 1].bar(scenarios_names, win_rates, color=['green', 'blue', 'orange'])
    axes[0, 1].set_title('Win Rate by Scenario')
    axes[0, 1].set_ylabel('Win Rate (%)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Equity curves comparison
    for i, result in enumerate(all_results):
        if len(result['equity_curve']) > 0:
            normalized_equity = np.array(result['equity_curve']) / result['initial_balance']
            axes[1, 0].plot(normalized_equity, label=result['scenario_name'])
    
    axes[1, 0].set_title('Normalized Equity Curves')
    axes[1, 0].set_xlabel('Time')
    axes[1, 0].set_ylabel('Equity (Normalized)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Trade count comparison
    trade_counts = [r['total_trades'] for r in all_results]
    axes[1, 1].bar(scenarios_names, trade_counts, color=['green', 'blue', 'orange'])
    axes[1, 1].set_title('Total Trades by Scenario')
    axes[1, 1].set_ylabel('Number of Trades')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return all_results


if __name__ == "__main__":
    print("TRADING MODEL PROFITABILITY SIMULATION")
    print("="*80)
    
    # Run simulation
    results = run_simulation()
    
    print("\n" + "="*80)
    print("SIMULATION COMPLETE")
    print("="*80)
    
    # Summary
    best_scenario = max(results, key=lambda x: x['total_return'])
    print(f"\nBest Performing Scenario: {best_scenario['scenario_name']}")
    print(f"Best Return: {best_scenario['total_return']:.2f}%")
    print(f"Final Equity: ${best_scenario['final_equity']:.2f}")