"""
Python HTTP REST API Prediction Server for MT5 Trading Bot
Uses HTTP for communication between MT5 and Python
Expects exactly 39 features (label removed from training data)
"""

import joblib
import numpy as np
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime
import threading

# Feature names (39 total - matching your training data)
FEATURE_NAMES = [
    'open', 'high', 'low', 'close', 'volume', 'spread',
    'hour', 'minute', 'day', 'month', 'day_of_week',
    'rsi', 'macd', 'macd_signal', 'macd_hist',
    'sma_20', 'sma_50', 'ema_12',
    'bb_upper', 'bb_middle', 'bb_lower', 'atr',
    'momentum', 'volume_sma', 'volume_ratio', 'roc', 'typical_price',
    'stoch_k', 'stoch_d', 'williams_r', 'cci', 'obv', 'cmf',
    'adx', 'trix', 'fisher', 'kc_upper', 'kc_lower', 'kc_middle'
]

class MT5PredictionServer:
    def __init__(self, model_path, scaler_path):
        """Initialize the prediction server with model and scaler"""
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = None
        self.scaler = None
        self.feature_count = 39
        self.prediction_count = 0
        self.load_model()
        
    def load_model(self):
        """Load the trained model and scaler"""
        try:
            print(f"Loading model from: {self.model_path}")
            self.model = joblib.load(self.model_path)
            print(f"✓ Model loaded successfully")
            print(f"  Model type: {type(self.model).__name__}")
            
            print(f"\nLoading scaler from: {self.scaler_path}")
            self.scaler = joblib.load(self.scaler_path)
            print(f"✓ Scaler loaded successfully")
            print(f"  Scaler type: {type(self.scaler).__name__}")
            
            # Verify feature count
            if hasattr(self.scaler, 'n_features_in_'):
                print(f"  Expected features: {self.scaler.n_features_in_}")
                if self.scaler.n_features_in_ != self.feature_count:
                    print(f"⚠ WARNING: Scaler expects {self.scaler.n_features_in_} features, but we have {self.feature_count}")
            
            return True
        except Exception as e:
            print(f"✗ Error loading model or scaler: {e}")
            return False
    
    def validate_features(self, features):
        """Validate feature array"""
        if len(features) != self.feature_count:
            raise ValueError(f"Expected {self.feature_count} features, got {len(features)}")
        
        # Check for NaN or Inf values
        if np.any(np.isnan(features)):
            raise ValueError("Features contain NaN values")
        
        if np.any(np.isinf(features)):
            raise ValueError("Features contain Inf values")
        
        return True
    
    def predict(self, features):
        """
        Make prediction using the loaded model
        
        Args:
            features: numpy array or list of 39 features
            
        Returns:
            dict: Prediction result with probabilities
        """
        try:
            # Convert to numpy array if needed
            if isinstance(features, list):
                features = np.array(features)
            
            # Validate
            self.validate_features(features)
            
            # Reshape if needed
            if len(features.shape) == 1:
                features = features.reshape(1, -1)
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Make prediction
            prediction = self.model.predict(features_scaled)[0]
            
            # Get probabilities if available
            probabilities = None
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(features_scaled)[0].tolist()
            
            self.prediction_count += 1
            
            return {
                'prediction': int(prediction),
                'probabilities': probabilities,
                'timestamp': datetime.now().isoformat(),
                'count': self.prediction_count
            }
            
        except Exception as e:
            print(f"✗ Error making prediction: {e}")
            return {
                'prediction': 0,
                'probabilities': None,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

# Global server instance
prediction_server = None

class PredictionHandler(BaseHTTPRequestHandler):
    """HTTP request handler for predictions"""
    
    def log_message(self, format, *args):
        """Override to customize logging"""
        pass  # Suppress default logging
    
    def do_GET(self):
        """Handle GET requests - server status"""
        if self.path == '/status' or self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            status = {
                'status': 'online',
                'model_loaded': prediction_server.model is not None,
                'scaler_loaded': prediction_server.scaler is not None,
                'predictions_count': prediction_server.prediction_count,
                'expected_features': 39
            }
            
            self.wfile.write(json.dumps(status).encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def do_POST(self):
        """Handle POST requests - predictions"""
        if self.path == '/predict':
            try:
                # Get content length
                content_length = int(self.headers['Content-Length'])
                
                # Read request body
                post_data = self.rfile.read(content_length)
                
                # Parse JSON
                data = json.loads(post_data.decode('utf-8'))
                
                if 'features' not in data:
                    self.send_error(400, 'Missing features in request')
                    return
                
                features = data['features']
                
                # Log request
                print(f"\n[Prediction #{prediction_server.prediction_count + 1}]")
                print(f"├─ Client: {self.client_address[0]}:{self.client_address[1]}")
                print(f"├─ Received: {len(features)} features")
                
                # Make prediction
                result = prediction_server.predict(features)
                
                # Log result
                signal_names = {0: "HOLD", 1: "BUY", 2: "SELL"}
                signal_name = signal_names.get(result['prediction'], "UNKNOWN")
                
                print(f"├─ Prediction: {result['prediction']} ({signal_name})")
                
                if result.get('probabilities'):
                    probs = result['probabilities']
                    print(f"├─ Probabilities:")
                    print(f"│  ├─ Hold: {probs[0]:.4f} ({probs[0]*100:.1f}%)")
                    print(f"│  ├─ Buy:  {probs[1]:.4f} ({probs[1]*100:.1f}%)")
                    print(f"│  └─ Sell: {probs[2]:.4f} ({probs[2]*100:.1f}%)")
                
                print(f"└─ ✓ Response sent")
                
                # Send response
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                
                self.wfile.write(json.dumps(result).encode())
                
            except json.JSONDecodeError as e:
                print(f"✗ JSON decode error: {e}")
                self.send_error(400, f'Invalid JSON: {str(e)}')
            except Exception as e:
                print(f"✗ Error processing request: {e}")
                self.send_error(500, f'Server error: {str(e)}')
        else:
            self.send_error(404, 'Endpoint not found')
    
    def do_OPTIONS(self):
        """Handle OPTIONS requests for CORS"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

def start_server(host='localhost', port=9090):
    """Start the HTTP server"""
    print("="*70)
    print("MT5 HTTP REST API PREDICTION SERVER")
    print("="*70)
    print(f"Host: {host}")
    print(f"Port: {port}")
    print(f"Expected features: 39")
    print("="*70)
    
    if not prediction_server.model or not prediction_server.scaler:
        print("✗ Failed to initialize server - Model or Scaler not loaded")
        return
    
    print("\n✓ Server initialized successfully")
    print(f"\nAPI Endpoints:")
    print(f"  - Status:     http://{host}:{port}/status")
    print(f"  - Predict:    http://{host}:{port}/predict (POST)")
    print("\n" + "="*70)
    print("WAITING FOR REQUESTS...")
    print("="*70)
    print("Press Ctrl+C to stop\n")
    
    try:
        server = HTTPServer((host, port), PredictionHandler)
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n\n" + "="*70)
        print("SERVER STOPPED BY USER")
        print("="*70)
        print(f"Total predictions: {prediction_server.prediction_count}")
        server.shutdown()

def test_prediction(model_path, scaler_path):
    """Test the prediction system with sample data"""
    print("="*70)
    print("TESTING PREDICTION SYSTEM")
    print("="*70)
    
    server = MT5PredictionServer(model_path, scaler_path)
    
    if not server.model or not server.scaler:
        print("✗ Failed to initialize server")
        return
    
    print("\n✓ Server initialized successfully")
    
    # Create sample features (39 features)
    print(f"\nGenerating sample features (39 features)...")
    
    # Realistic sample features for XAUUSD
    sample_features = [
        2650.0, 2655.0, 2648.0, 2652.0, 1500.0, 15.0,
        10.0, 30.0, 15.0, 12.0, 1.0,
        55.0, 2.5, 2.0, 0.5,
        2645.0, 2640.0, 2648.0,
        2660.0, 2650.0, 2640.0, 3.5,
        5.0, 1400.0, 1.07, 0.5, 2651.0,
        60.0, 58.0, -40.0, 50.0, 15000.0, 0.1,
        25.0, 0.02, 0.5, 2658.0, 2642.0, 2650.0
    ]
    
    print(f"Feature count: {len(sample_features)}")
    
    print("\nMaking prediction...")
    result = server.predict(sample_features)
    
    signal_names = {0: "HOLD", 1: "BUY", 2: "SELL"}
    
    print(f"\n{'='*70}")
    print(f"PREDICTION RESULT")
    print(f"{'='*70}")
    print(f"Prediction: {result['prediction']} ({signal_names[result['prediction']]})")
    
    if result.get('probabilities'):
        probs = result['probabilities']
        print(f"\nProbabilities:")
        print(f"  Hold: {probs[0]:.4f} ({probs[0]*100:.2f}%)")
        print(f"  Buy:  {probs[1]:.4f} ({probs[1]*100:.2f}%)")
        print(f"  Sell: {probs[2]:.4f} ({probs[2]*100:.2f}%)")
    
    print(f"\n✓ Test completed successfully!")

def verify_model_files(model_path, scaler_path):
    """Verify that model files exist and are valid"""
    import os
    
    print("="*70)
    print("VERIFYING MODEL FILES")
    print("="*70)
    
    if not os.path.exists(model_path):
        print(f"✗ Model file not found: {model_path}")
        return False
    else:
        model_size = os.path.getsize(model_path) / (1024 * 1024)
        print(f"✓ Model file found: {model_path}")
        print(f"  Size: {model_size:.2f} MB")
    
    if not os.path.exists(scaler_path):
        print(f"✗ Scaler file not found: {scaler_path}")
        return False
    else:
        scaler_size = os.path.getsize(scaler_path) / 1024
        print(f"✓ Scaler file found: {scaler_path}")
        print(f"  Size: {scaler_size:.2f} KB")
    
    print(f"\n✓ All files verified")
    return True

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='MT5 HTTP REST API Prediction Server')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to the trained model (.joblib)')
    parser.add_argument('--scaler', type=str, required=True,
                       help='Path to the scaler (.joblib)')
    parser.add_argument('--mode', type=str, choices=['server', 'test', 'verify'],
                       default='server',
                       help='Mode: server (run HTTP server), test (test prediction), verify (check files)')
    parser.add_argument('--host', type=str, default='127.0.0.1',
                       help='HTTP server host (default: localhost)')
    parser.add_argument('--port', type=int, default=9090,
                       help='HTTP server port (default: 9090)')
    
    args = parser.parse_args()
    
    global prediction_server
    
    if args.mode == 'verify':
        verify_model_files(args.model, args.scaler)
    elif args.mode == 'test':
        if verify_model_files(args.model, args.scaler):
            test_prediction(args.model, args.scaler)
    else:  # server mode
        if verify_model_files(args.model, args.scaler):
            prediction_server = MT5PredictionServer(args.model, args.scaler)
            start_server(args.host, args.port)

if __name__ == "__main__":
    # Example usage:
    # python prediction_server.py --model "GeneratedXAUUSD Random Forest.joblib" --scaler "GeneratedXAUUSD scaler.joblib" --mode server
    # python prediction_server.py --model "model.joblib" --scaler "scaler.joblib" --mode test
    
    main()