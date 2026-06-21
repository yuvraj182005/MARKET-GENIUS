from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta
import yfinance as yf

app = Flask(__name__)
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

def load_and_train_model(symbol):
    print(f"Downloading data for {symbol}...")
    stock_data = yf.download(symbol, start="2010-01-01", end=datetime.today().strftime('%Y-%m-%d'))
    
    # Prepare data
    stock_data['Date'] = stock_data.index
    stock_data['Day'] = stock_data['Date'].dt.day
    stock_data['Month'] = stock_data['Date'].dt.month
    stock_data['Year'] = stock_data['Date'].dt.year
    X = stock_data[['Day', 'Month', 'Year']]
    y = stock_data['Close'].values.ravel()  # Convert to 1D array

    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    return model, stock_data

# Global variables to store models and data
models = {}
historical_data = {}

print("Initializing models...")
# Load and train models for all symbols
for name, symbol in {**INDICES, **POPULAR_STOCKS}.items():
    try:
        models[name], historical_data[name] = load_and_train_model(symbol)
        print(f"Successfully loaded {name}")
    except Exception as e:
        print(f"Error loading {name}: {e}")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data_in = request.get_json()
        date_str = data_in.get('date', '')
        symbol = data_in.get('symbol', 'NIFTY')
        
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        X_pred = np.array([[date_obj.day, date_obj.month, date_obj.year]])
        
        if symbol not in models:
            return jsonify({'error': 'Invalid symbol'})
            
        prediction = models[symbol].predict(X_pred)[0]
        return jsonify({'predicted_price': round(float(prediction), 2)})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/history', methods=['GET'])
def history():
    symbol = request.args.get('symbol', 'NIFTY')
    if symbol not in historical_data:
        return jsonify({'error': 'Invalid symbol'})
        
    last_data = historical_data[symbol].tail(60)
    return jsonify({
        'dates': last_data.index.strftime('%Y-%m-%d').tolist(),
        'prices': last_data['Close'].tolist()
    })

@app.route('/available_symbols', methods=['GET'])
def available_symbols():
    return jsonify({
        'indices': list(INDICES.keys()),
        'stocks': list(POPULAR_STOCKS.keys())
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)