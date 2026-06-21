from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime
import yfinance as yf
import os

app = Flask(__name__, static_folder='../frontend')
CORS(app)

# Stock symbols
INDICES = {
    "NIFTY": "^NSEI",
    "SENSEX": "^BSESN"
}

POPULAR_STOCKS = {
    "TCS": "TCS.NS",
    "RELIANCE": "RELIANCE.NS",
    "INFY": "INFY.NS",
    "HDFC": "HDFCBANK.NS",
    "WIPRO": "WIPRO.NS"
}

# Global storage
models = {}
historical_data = {}

def load_and_train_model(symbol_name, symbol_code):
    try:
        print(f"Downloading data for {symbol_name} ({symbol_code})...")
        data = yf.download(symbol_code, start="2015-01-01", end=datetime.today().strftime('%Y-%m-%d'))
        
        if data.empty:
            print(f"No data found for {symbol_name}")
            return None, None
            
        # Basic feature preparation
        data['Day'] = data.index.day
        data['Month'] = data.index.month
        data['Year'] = data.index.year
        
        # Train model
        X = data[['Day', 'Month', 'Year']]
        y = data['Close']
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        print(f"Successfully loaded {symbol_name}")
        return model, data
    except Exception as e:
        print(f"Error loading {symbol_name}: {str(e)}")
        return None, None

# Initialize models
print("Initializing models...")
for name, symbol in INDICES.items():
    model, data = load_and_train_model(name, symbol)
    if model is not None:
        models[name] = model
        historical_data[name] = data

for name, symbol in POPULAR_STOCKS.items():
    model, data = load_and_train_model(name, symbol)
    if model is not None:
        models[name] = model
        historical_data[name] = data

@app.route('/')
def serve_frontend():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_files(path):
    return send_from_directory(app.static_folder, path)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        date_str = data.get('date', '')
        symbol = data.get('symbol', 'NIFTY')
        
        if symbol not in models:
            return jsonify({'error': f'Symbol {symbol} not available'})
            
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        X_pred = np.array([[date_obj.day, date_obj.month, date_obj.year]])
        
        prediction = models[symbol].predict(X_pred)[0]
        current_price = historical_data[symbol]['Close'].iloc[-1]
        
        return jsonify({
            'predicted_price': round(float(prediction), 2),
            'current_price': round(float(current_price), 2),
            'change_percent': round(((prediction - current_price) / current_price) * 100, 2)
        })
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/history', methods=['GET'])
def history():
    try:
        symbol = request.args.get('symbol', 'NIFTY')
        days = int(request.args.get('days', '60'))
        
        if symbol not in historical_data:
            return jsonify({'error': f'Symbol {symbol} not available'})
            
        data = historical_data[symbol].tail(days)
        
        return jsonify({
            'dates': data.index.strftime('%Y-%m-%d').tolist(),
            'prices': data['Close'].tolist(),
            'volumes': data['Volume'].tolist(),
            'indicators': {
                'sma20': data['Close'].rolling(window=20).mean().tolist(),
                'sma50': data['Close'].rolling(window=50).mean().tolist(),
                'ema9': data['Close'].ewm(span=9, adjust=False).mean().tolist(),
                'rsi': [50] * len(data),  # Placeholder RSI
                'macd': [0] * len(data),  # Placeholder MACD
                'macd_signal': [0] * len(data)  # Placeholder MACD Signal
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/available_symbols', methods=['GET'])
def available_symbols():
    return jsonify({
        'indices': [k for k in INDICES.keys() if k in models],
        'stocks': [k for k in POPULAR_STOCKS.keys() if k in models]
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)