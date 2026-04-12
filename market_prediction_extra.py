from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained model and scaler
MODEL_PATH = "GeneratedXAUUSDRandom Forest-octa.joblib"
# SCALER_PATH = "GeneratedXAUUSD scaler.joblib"  # If you use scaling

model = None
# scaler = None

def load_model():
    global model
    try:
        model = joblib.load(MODEL_PATH, mmap_mode='r')
        # scaler = joblib.load(SCALER_PATH)  # Uncomment if using scaler
        print("Model loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

# Feature names in the exact order
FEATURE_NAMES = ['open', 'high', 'low', 'close', 'volume',
                'hour', 'minute', 'day', 'month','day_of_week'        
                ]

@app.route('/predict', methods=['GET'])
def predict():
    try:
        if model is None:
            return "0", 500  # Return hold on error
        
        # Extract features from query parameters
        features = []
        missing_features = []
        
        for feature_name in FEATURE_NAMES:
            value = request.args.get(feature_name)
            if value is None:
                missing_features.append(feature_name)
                features.append(0.0)  # Default value
            else:
                try:
                    features.append(float(value))
                except ValueError:
                    return f"Invalid value for {feature_name}", 400
        
        if missing_features:
            print(f"Warning: Missing features: {missing_features}")
        
        # Convert to numpy array and reshape
        features_array = np.array(features).reshape(1, -1)
        
        # Scale features if using scaler
        # if scaler is not None:
        #     features_array = scaler.transform(features_array)
        
        # Make prediction
        prediction = model.predict(features_array)[0]
        """
        y_prob = model.predict_proba(features_array)[0]
        print(f"Prediction probabilities: {y_prob}")

        for(i, prob) in enumerate(y_prob):
            print(f"Class {i} probability: {prob}")
            if(len(y_prob) < 3): n = i + 1 
            else: n = i
            if(prob > 0.6):
                print(f"High confidence in class {n} with probability {prob}")
                prediction = n
                break
            else:
                prediction = 0  # Default to hold if no high confidence
        """
        # Log the prediction
        print(f"Prediction: {prediction} (0=Hold, 1=Buy, 2=Sell)")
        
        # Return prediction as plain text
        return str(int(prediction)), 200
        
    except Exception as e:
        print(f"Error in prediction: {e}")
        import traceback
        traceback.print_exc()
        return "0", 500  # Return hold on error

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    status = {
        'status': 'healthy',
        'model_loaded': model is not None,
        # 'scaler_loaded': scaler is not None
    }
    return jsonify(status), 200

@app.route('/test', methods=['GET'])
def test():
    """Test endpoint to verify server is running"""
    return "Server is running!", 200

if __name__ == '__main__':
    print("Starting ML Prediction Server...")
    
    # Load model at startup
    if not load_model():
        print("Warning: Model could not be loaded. Server will start but predictions will fail.")
    
    # Run server
    app.run(host='0.0.0.0', port=5001, debug=False)