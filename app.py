from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
from datetime import datetime
import os
import threading
import pickle
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__, static_folder='../frontend')
CORS(app)

# Dataset paths
DATASET_DIR = 'd:\\market dataset'
INDEX_FILES = {
    'NIFTY': os.path.join(DATASET_DIR, 'NIFTY_50.csv'),
    'SENSEX': os.path.join(DATASET_DIR, 'SENSEX.csv')
}
COMPANIES_FILES = [
    os.path.join(DATASET_DIR, 'NIFTY_50_COMPANIES.csv'),
    os.path.join(DATASET_DIR, 'SENSEX_COMPANIES.csv')
]

# Global storage for data
historical_data = {}
models = {}
available_symbols = {'indices': [], 'stocks': []}

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def calculate_adx(df, period=14):
    high = df['High'].fillna(df['Close'])
    low = df['Low'].fillna(df['Close'])
    close = df['Close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    
    up = high - high.shift()
    down = low.shift() - low
    
    pos_dm = up.where((up > 0) & (up > down), 0)
    neg_dm = down.where((down > 0) & (down > up), 0)
    
    pos_di = 100 * pos_dm.rolling(period).mean() / atr.replace(0, 1e-10)
    neg_di = 100 * neg_dm.rolling(period).mean() / atr.replace(0, 1e-10)
    
    di_diff = abs(pos_di - neg_di)
    di_sum = pos_di + neg_di
    dx = 100 * di_diff / di_sum.replace(0, 1e-10)
    adx = dx.rolling(period).mean()
    
    return adx.fillna(0)

def load_csv_data():
    """Load all CSV files from the market dataset directory."""
    global historical_data, models, available_symbols
    
    print("Loading data from CSV files...")
    
    # Load index data (NIFTY, SENSEX)
    for symbol, filepath in INDEX_FILES.items():
        try:
            if os.path.exists(filepath):
                print(f"Loading {symbol} from {filepath}...")
                df = pd.read_csv(filepath)
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.sort_values('Date')
                
                # Convert Close to numeric, handle any non-numeric values
                df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
                df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
                
                # Remove rows with NaN in Close or Volume
                df = df.dropna(subset=['Close', 'Volume'])
                
                historical_data[symbol] = df
                available_symbols['indices'].append(symbol)
                print(f"[OK] Loaded {symbol}: {len(df)} records")
            else:
                print(f"[FAIL] File not found: {filepath}")
        except Exception as e:
            print(f"[FAIL] Error loading {symbol}: {e}")
    
    # Load company data from NIFTY_50_COMPANIES.csv and SENSEX_COMPANIES.csv
    stocks_loaded = set()
    for filepath in COMPANIES_FILES:
        try:
            if os.path.exists(filepath):
                print(f"Loading companies from {filepath}...")
                df = pd.read_csv(filepath)
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.sort_values('Date')
                
                # Convert numeric columns
                df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
                df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
                df = df.dropna(subset=['Close', 'Volume'])
                
                # Group by Ticker and store separately
                for ticker in df['Ticker'].unique():
                    if pd.isna(ticker) or ticker == '':
                        continue
                    
                    ticker_data = df[df['Ticker'] == ticker].copy()
                    ticker_data = ticker_data.sort_values('Date')
                    
                    # Extract symbol name from ticker (e.g., RELIANCE.NS -> RELIANCE)
                    symbol_name = ticker.split('.')[0] if '.' in ticker else ticker
                    
                    if symbol_name not in stocks_loaded and len(ticker_data) > 0:
                        historical_data[symbol_name] = ticker_data
                        available_symbols['stocks'].append(symbol_name)
                        stocks_loaded.add(symbol_name)
                        print(f"  [OK] {symbol_name}: {len(ticker_data)} records")
        except Exception as e:
            print(f"[FAIL] Error loading companies from {filepath}: {e}")
    
    # Remove duplicates from stocks list
    available_symbols['stocks'] = list(set(available_symbols['stocks']))
    available_symbols['indices'] = list(set(available_symbols['indices']))
    
    print(f"\nData loading complete!")
    print(f"Indices: {available_symbols['indices']}")
    print(f"Stocks: {available_symbols['stocks']}")

def train_models():
    """Train ML models for each symbol using historical data."""
    global models
    
    print("\n" + "="*60)
    print("Training prediction models on CSV data...")
    print("="*60)
    
    trained_count = 0
    max_models = 1  # Limit to 1 model for quick demonstration
    
    # Skip SENSEX as it seems to have issues
    skip_symbols = ['SENSEX']
    
    for symbol in [s for s in list(historical_data.keys())[:max_models] if s not in skip_symbols]:
        try:
            print(f"\nTraining model for {symbol}...")
            df = historical_data[symbol].copy()
            
            # Need at least 20 data points to train
            if len(df) < 20:
                print(f"  ⚠ Skipping {symbol}: not enough data ({len(df)} rows)")
                continue
            
            # Prepare features from available columns
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date')
            
            df['DayOfYear'] = df['Date'].dt.dayofyear
            df['Month'] = df['Date'].dt.month
            df['Quarter'] = df['Date'].dt.quarter
            df['Year'] = df['Date'].dt.year
            df['WeekOfYear'] = df['Date'].dt.isocalendar().week
            df['DayOfWeek'] = df['Date'].dt.dayofweek
            
            for lag in [1, 2, 3, 5, 10]:
                df[f'Close_lag_{lag}'] = df['Close'].shift(lag)
            
            for lag in [1, 2, 3]:
                df[f'Volume_lag_{lag}'] = df['Volume'].shift(lag)
            
            df['Close_MA_5'] = df['Close'].rolling(window=5).mean()
            df['Close_MA_10'] = df['Close'].rolling(window=10).mean()
            df['Close_MA_20'] = df['Close'].rolling(window=20).mean()
            df['Close_MA_50'] = df['Close'].rolling(window=50).mean()
            
            df['Close_Volatility_5'] = df['Close'].rolling(window=5).std()
            df['Close_Volatility_10'] = df['Close'].rolling(window=10).std()
            df['Close_Volatility_20'] = df['Close'].rolling(window=20).std()
            
            df['Volume_MA_5'] = df['Volume'].rolling(window=5).mean()
            df['Volume_MA_10'] = df['Volume'].rolling(window=10).mean()
            
            df['Price_Range'] = df['High'].fillna(df['Close']) - df['Low'].fillna(df['Close'])
            df['Price_Range_MA'] = df['Price_Range'].rolling(window=5).mean()
            
            df['Returns'] = df['Close'].pct_change()
            df['Returns_MA_5'] = df['Returns'].rolling(window=5).mean()
            df['Returns_Volatility'] = df['Returns'].rolling(window=10).std()
            
            df['Close_Position'] = (df['Close'] - df['Close'].rolling(20).min()) / (df['Close'].rolling(20).max() - df['Close'].rolling(20).min())
            
            df['Momentum_5'] = df['Close'] - df['Close'].shift(5)
            df['Momentum_10'] = df['Close'] - df['Close'].shift(10)
            
            df['RSI_14'] = calculate_rsi(df['Close'], 14)
            df['ADX_14'] = calculate_adx(df, 14)
            
            df['Volume_Ratio'] = df['Volume'] / df['Volume'].rolling(window=20).mean()
            
            technical_cols = ['SMA_20', 'SMA_50', 'EMA_12', 'EMA_26', 'MACD', 'RSI_14', 'BB_Mid', 'ADX_14']
            for col in technical_cols:
                if col in df.columns:
                    df[col] = df[col].ffill().bfill()
            
            # Remove rows with any NaN values
            df = df.dropna()
            
            if len(df) < 20:
                print(f"  ⚠ Skipping {symbol}: not enough data after feature engineering ({len(df)} rows)")
                continue
            
            feature_cols = [
                'DayOfYear', 'Month', 'Quarter', 'Year', 'WeekOfYear', 'DayOfWeek',
                'Close_lag_1', 'Close_lag_2', 'Close_lag_3', 'Close_lag_5', 'Close_lag_10',
                'Volume_lag_1', 'Volume_lag_2', 'Volume_lag_3',
                'Close_MA_5', 'Close_MA_10', 'Close_MA_20', 'Close_MA_50',
                'Close_Volatility_5', 'Close_Volatility_10', 'Close_Volatility_20',
                'Volume_MA_5', 'Volume_MA_10', 'Volume', 'Volume_Ratio',
                'Price_Range', 'Price_Range_MA',
                'Returns', 'Returns_MA_5', 'Returns_Volatility',
                'Close_Position',
                'Momentum_5', 'Momentum_10',
                'RSI_14', 'ADX_14'
            ]
            
            available_technical = [col for col in technical_cols if col in df.columns]
            feature_cols.extend(available_technical)
            
            feature_cols = [col for col in feature_cols if col in df.columns]
            
            if len(feature_cols) < 10:
                print(f"  ⚠ Skipping {symbol}: insufficient features ({len(feature_cols)})")
                continue
            
            X = df[feature_cols]
            y = df['Close']
            
            split_idx = int(len(df) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            if len(X_train) < 10 or len(X_test) < 2:
                print(f"  ⚠ Skipping {symbol}: insufficient train/test samples")
                continue
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            rf_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                min_samples_split=3,
                min_samples_leaf=1,
                random_state=42,
                n_jobs=-1
            )
            
            gb_model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                min_samples_split=3,
                min_samples_leaf=1,
                random_state=42
            )
            
            rf_model.fit(X_train_scaled, y_train)
            gb_model.fit(X_train_scaled, y_train)
            
            rf_pred_train = rf_model.predict(X_train_scaled)
            rf_pred_test = rf_model.predict(X_test_scaled)
            gb_pred_train = gb_model.predict(X_train_scaled)
            gb_pred_test = gb_model.predict(X_test_scaled)

            y_pred_train = (rf_pred_train * 0.6 + gb_pred_train * 0.4)
            y_pred_test = (rf_pred_test * 0.6 + gb_pred_test * 0.4)

            # Calculate individual model predictions for direction accuracy
            gb_pred_train_direction = (gb_pred_train[1:] > gb_pred_train[:-1]).astype(int)
            gb_pred_test_direction = (gb_pred_test[1:] > gb_pred_test[:-1]).astype(int)
            
            model = {'rf': rf_model, 'gb': gb_model, 'scaler': scaler, 'ensemble_weights': [0.6, 0.4]}
            
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            train_mae = mean_absolute_error(y_train, y_pred_train)
            test_mae = mean_absolute_error(y_test, y_pred_test)
            
            # Convert regression to classification: 1 if price goes up, 0 if down
            y_train_direction = (y_train.diff() > 0).astype(int)[1:]  # Remove first NaN
            y_test_direction = (y_test.diff() > 0).astype(int)[1:]    # Remove first NaN
            
            y_pred_train_direction = (y_pred_train[1:] > y_pred_train[:-1]).astype(int)
            y_pred_test_direction = (y_pred_test[1:] > y_pred_test[:-1]).astype(int)
            
            # Calculate classification metrics if we have enough samples
            if len(y_test_direction) > 1:
                test_accuracy = accuracy_score(y_test_direction, y_pred_test_direction)
                test_precision = precision_score(y_test_direction, y_pred_test_direction, zero_division=0)
                test_recall = recall_score(y_test_direction, y_pred_test_direction, zero_division=0)
                test_f1 = f1_score(y_test_direction, y_pred_test_direction, zero_division=0)

                train_accuracy = accuracy_score(y_train_direction, y_pred_train_direction)
                train_precision = precision_score(y_train_direction, y_pred_train_direction, zero_division=0)
                train_recall = recall_score(y_train_direction, y_pred_train_direction, zero_division=0)
                train_f1 = f1_score(y_train_direction, y_pred_train_direction, zero_division=0)

                # Calculate Gradient Boosting model accuracy
                gb_test_accuracy = accuracy_score(y_test_direction, gb_pred_test_direction)
                gb_train_accuracy = accuracy_score(y_train_direction, gb_pred_train_direction)
            else:
                test_accuracy = train_accuracy = 0.0
                test_precision = train_precision = 0.0
                test_recall = train_recall = 0.0
                test_f1 = train_f1 = 0.0
                gb_test_accuracy = gb_train_accuracy = 0.0
            
            models[symbol] = {
                'model': model,
                'rf': model['rf'],
                'gb': model['gb'],
                'scaler': model['scaler'],
                'features': feature_cols,
                'last_close': y.iloc[-1],
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'train_precision': train_precision,
                'test_precision': test_precision,
                'train_recall': train_recall,
                'test_recall': test_recall,
                'train_f1': train_f1,
                'test_f1': test_f1,
                'gb_train_accuracy': gb_train_accuracy,
                'gb_test_accuracy': gb_test_accuracy,
                'last_date': df['Date'].iloc[-1]
            }
            
            # Print detailed metrics
            print(f"  [OK] Trained {symbol}:")
            print(f"       Regression Metrics:")
            print(f"          R² Score (Train): {train_r2:.4f} ({train_r2*100:.2f}%)")
            print(f"          R² Score (Test):  {test_r2:.4f} ({test_r2*100:.2f}%)")
            print(f"          RMSE (Test):      {test_rmse:.2f}")
            print(f"          MAE (Test):       {test_mae:.2f}")
            print(f"       Classification Metrics (Direction Prediction):")
            print(f"          Ensemble Accuracy (Test): {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
            print(f"          GB Model Accuracy (Test):  {gb_test_accuracy:.4f} ({gb_test_accuracy*100:.2f}%)")
            print(f"          F1 Score (Test): {test_f1:.4f} ({test_f1*100:.2f}%)")
            print(f"          Precision (Test): {test_precision:.4f} ({test_precision*100:.2f}%)")
            print(f"          Recall (Test):    {test_recall:.4f} ({test_recall*100:.2f}%)")
            print(f"       Training Set Metrics:")
            print(f"          Ensemble Accuracy (Train): {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
            print(f"          GB Model Accuracy (Train):  {gb_train_accuracy:.4f} ({gb_train_accuracy*100:.2f}%)")
            print(f"          F1 Score (Train): {train_f1:.4f} ({train_f1*100:.2f}%)")
            
            trained_count += 1
            
        except Exception as e:
            print(f"  [FAIL] Error training model for {symbol}: {e}")
            import traceback
            traceback.print_exc()
    
    # Print summary of all trained models
    print(f"\n{'='*80}")
    print(f"MODEL TRAINING SUMMARY")
    print(f"{'='*80}")
    print(f"Total models trained: {len(models)}")
    print(f"\n{'Symbol':<15} {'Ensemble Acc':<15} {'GB Model Acc':<15} {'Test F1 Score':<18} {'Test R²':<12} {'Status':<10}")
    print(f"{'-'*95}")

    if models:
        for symbol, info in models.items():
            test_accuracy = info.get('test_accuracy', 0) * 100
            gb_test_accuracy = info.get('gb_test_accuracy', 0) * 100
            test_f1 = info.get('test_f1', 0) * 100
            test_r2 = info.get('test_r2', 0) * 100

            # Color code based on performance
            if test_accuracy >= 60:
                status = "Good"
            elif test_accuracy >= 50:
                status = "Fair"
            else:
                status = "Poor"

            print(f"{symbol:<15} {test_accuracy:>6.2f}% ({test_accuracy/100:.4f})  {gb_test_accuracy:>6.2f}% ({gb_test_accuracy/100:.4f})  {test_f1:>6.2f}% ({test_f1/100:.4f})  {test_r2:>6.2f}%   {status:<10}")

        # Calculate average metrics
        avg_accuracy = sum(info.get('test_accuracy', 0) for info in models.values()) / len(models) * 100
        avg_gb_accuracy = sum(info.get('gb_test_accuracy', 0) for info in models.values()) / len(models) * 100
        avg_f1 = sum(info.get('test_f1', 0) for info in models.values()) / len(models) * 100
        avg_r2 = sum(info.get('test_r2', 0) for info in models.values()) / len(models) * 100

        print(f"{'-'*95}")
        print(f"{'AVERAGE':<15} {avg_accuracy:>6.2f}%         {avg_gb_accuracy:>6.2f}%         {avg_f1:>6.2f}%              {avg_r2:>6.2f}%")
    else:
        print("No models were successfully trained.")
    
    print(f"{'='*80}\n")

@app.route('/')
def serve_frontend():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_files(path):
    return send_from_directory(app.static_folder, path)

@app.route('/available_symbols', methods=['GET'])
def get_available_symbols():
    """Return the list of available symbols (indices and stocks)."""
    return jsonify({
        'indices': available_symbols['indices'],
        'stocks': available_symbols['stocks'],
        'loaded': list(historical_data.keys())
    })

@app.route('/history', methods=['GET'])
def history():
    """Return historical data for a symbol."""
    try:
        symbol = request.args.get('symbol', 'NIFTY')
        days = int(request.args.get('days', '60'))
        
        if symbol not in historical_data:
            return jsonify({'error': f'Symbol {symbol} not available'})
        
        data = historical_data[symbol].tail(days).copy()
        
        # Helper to convert to list and handle NaN
        def to_list(obj):
            import json
            try:
                if hasattr(obj, 'tolist'):
                    lst = obj.tolist()
                elif hasattr(obj, 'values'):
                    lst = obj.values.tolist()
                else:
                    lst = list(obj)
                # Replace NaN with None (converts to null in JSON)
                result = []
                for x in lst:
                    if isinstance(x, (int, float)):
                        if pd.isna(x):
                            result.append(None)
                        else:
                            result.append(x)
                    else:
                        result.append(str(x) if x is not None else None)
                return result
            except Exception as e:
                print(f"Error in to_list: {e}")
                return lst
        
        # Normalize Close and Volume columns
        prices_series = data['Close']
        if isinstance(prices_series, pd.DataFrame):
            prices_series = prices_series.iloc[:, 0]
        
        volume_series = data['Volume']
        if isinstance(volume_series, pd.DataFrame):
            volume_series = volume_series.iloc[:, 0]
        
        # Calculate technical indicators
        sma20 = prices_series.rolling(window=20).mean()
        sma50 = prices_series.rolling(window=50).mean()
        ema9 = prices_series.ewm(span=9, adjust=False).mean()
        
        # Calculate RSI if not available in data
        if 'RSI_14' in data.columns:
            rsi_data = data['RSI_14']
        else:
            # Calculate RSI manually
            delta = prices_series.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi_data = 100 - (100 / (1 + rs))
            rsi_data = rsi_data.fillna(50)  # Fill NaN with neutral RSI
        
        # Calculate MACD if not available
        if 'MACD' in data.columns and 'Signal_Line' in data.columns:
            macd_data = data['MACD']
            macd_signal = data['Signal_Line']
        else:
            # Calculate MACD manually
            ema12 = prices_series.ewm(span=12, adjust=False).mean()
            ema26 = prices_series.ewm(span=26, adjust=False).mean()
            macd_data = ema12 - ema26
            macd_signal = macd_data.ewm(span=9, adjust=False).mean()
            macd_data = macd_data.fillna(0)
            macd_signal = macd_signal.fillna(0)
        
        return jsonify({
            'dates': to_list(data['Date'].dt.strftime('%Y-%m-%d')),
            'prices': to_list(prices_series),
            'volumes': to_list(volume_series),
            'indicators': {
                'sma20': to_list(sma20),
                'sma50': to_list(sma50),
                'ema9': to_list(ema9),
                'rsi': to_list(rsi_data) if hasattr(rsi_data, 'tolist') or hasattr(rsi_data, 'values') else [50] * len(data),
                'macd': to_list(macd_data) if hasattr(macd_data, 'tolist') or hasattr(macd_data, 'values') else [0] * len(data),
                'macd_signal': to_list(macd_signal) if hasattr(macd_signal, 'tolist') or hasattr(macd_signal, 'values') else [0] * len(data)
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/predict', methods=['POST'])
def predict():
    """Predict stock price using ensemble ML models."""
    try:
        data = request.get_json()
        date_str = data.get('date', '')
        symbol = data.get('symbol', 'NIFTY')
        
        if symbol not in historical_data:
            return jsonify({'error': f'Symbol {symbol} not available'})
        
        pred_date = pd.to_datetime(date_str)
        
        day_of_week = pred_date.dayofweek
        if day_of_week >= 5:
            days_to_add = 7 - day_of_week if day_of_week == 5 else 1
            next_weekday = pred_date + pd.Timedelta(days=days_to_add)
            return jsonify({
                'error': f'Markets are closed on weekends. Please select a weekday. The next trading day would be {next_weekday.strftime("%Y-%m-%d")} ({next_weekday.strftime("%A")}).'
            })
        
        df = historical_data[symbol].copy()
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        
        current_price = df['Close'].iloc[-1]
        last_date = df['Date'].iloc[-1]
        
        days_ahead = 0
        current_date = last_date
        while current_date < pred_date:
            current_date += pd.Timedelta(days=1)
            if current_date.dayofweek < 5:
                days_ahead += 1
        
        if symbol in models and days_ahead <= 30:
            model_info = models[symbol]
            rf_model = model_info['rf']
            gb_model = model_info['gb']
            scaler = model_info['scaler']
            features = model_info['features']
            
            try:
                df_temp = df.copy()
                df_temp['Date'] = pd.to_datetime(df_temp['Date'])
                df_temp = df_temp.sort_values('Date')
                
                df_temp['DayOfYear'] = df_temp['Date'].dt.dayofyear
                df_temp['Month'] = df_temp['Date'].dt.month
                df_temp['Quarter'] = df_temp['Date'].dt.quarter
                df_temp['Year'] = df_temp['Date'].dt.year
                df_temp['WeekOfYear'] = df_temp['Date'].dt.isocalendar().week
                df_temp['DayOfWeek'] = df_temp['Date'].dt.dayofweek
                
                for lag in [1, 2, 3, 5, 10]:
                    df_temp[f'Close_lag_{lag}'] = df_temp['Close'].shift(lag)
                
                for lag in [1, 2, 3]:
                    df_temp[f'Volume_lag_{lag}'] = df_temp['Volume'].shift(lag)
                
                df_temp['Close_MA_5'] = df_temp['Close'].rolling(window=5).mean()
                df_temp['Close_MA_10'] = df_temp['Close'].rolling(window=10).mean()
                df_temp['Close_MA_20'] = df_temp['Close'].rolling(window=20).mean()
                df_temp['Close_MA_50'] = df_temp['Close'].rolling(window=50).mean()
                
                df_temp['Close_Volatility_5'] = df_temp['Close'].rolling(window=5).std()
                df_temp['Close_Volatility_10'] = df_temp['Close'].rolling(window=10).std()
                df_temp['Close_Volatility_20'] = df_temp['Close'].rolling(window=20).std()
                
                df_temp['Volume_MA_5'] = df_temp['Volume'].rolling(window=5).mean()
                df_temp['Volume_MA_10'] = df_temp['Volume'].rolling(window=10).mean()
                
                df_temp['Price_Range'] = df_temp['High'].fillna(df_temp['Close']) - df_temp['Low'].fillna(df_temp['Close'])
                df_temp['Price_Range_MA'] = df_temp['Price_Range'].rolling(window=5).mean()
                
                df_temp['Returns'] = df_temp['Close'].pct_change()
                df_temp['Returns_MA_5'] = df_temp['Returns'].rolling(window=5).mean()
                df_temp['Returns_Volatility'] = df_temp['Returns'].rolling(window=10).std()
                
                df_temp['Close_Position'] = (df_temp['Close'] - df_temp['Close'].rolling(20).min()) / (df_temp['Close'].rolling(20).max() - df_temp['Close'].rolling(20).min() + 1e-10)
                
                df_temp['Momentum_5'] = df_temp['Close'] - df_temp['Close'].shift(5)
                df_temp['Momentum_10'] = df_temp['Close'] - df_temp['Close'].shift(10)
                
                df_temp['RSI_14'] = calculate_rsi(df_temp['Close'], 14)
                df_temp['ADX_14'] = calculate_adx(df_temp, 14)
                
                df_temp['Volume_Ratio'] = df_temp['Volume'] / (df_temp['Volume'].rolling(window=20).mean() + 1e-10)
                
                df_temp = df_temp.dropna()
                
                if len(df_temp) > 0:
                    X_latest = df_temp[features].iloc[-1:].values
                    X_scaled = scaler.transform(X_latest)
                    
                    rf_pred = rf_model.predict(X_scaled)[0]
                    gb_pred = gb_model.predict(X_scaled)[0]
                    
                    predicted_price = rf_pred * 0.6 + gb_pred * 0.4
                    
                    if predicted_price < 0:
                        predicted_price = current_price
                else:
                    predicted_price = current_price
            except Exception as e:
                print(f"Error in model prediction: {e}")
                predicted_price = current_price
        else:
            if len(df) >= 30:
                recent_returns = (df['Close'].iloc[-1] - df['Close'].iloc[-30]) / df['Close'].iloc[-30]
                daily_return = recent_returns / 30
                predicted_price = current_price * (1 + daily_return * days_ahead)
            else:
                predicted_price = current_price
            
            if predicted_price < 0:
                predicted_price = current_price
        
        change_percent = ((predicted_price - current_price) / current_price) * 100
        
        model_accuracy = None
        model_f1 = None
        if symbol in models:
            model_info = models[symbol]
            model_accuracy = round(model_info.get('test_accuracy', 0) * 100, 2)
            model_f1 = round(model_info.get('test_f1', 0) * 100, 2)
        
        response_data = {
            'predicted_price': round(float(predicted_price), 2),
            'current_price': round(float(current_price), 2),
            'change_percent': round(float(change_percent), 2),
            'days_ahead': days_ahead,
        }
        
        if model_accuracy is not None:
            response_data['model_accuracy'] = model_accuracy
        if model_f1 is not None:
            response_data['model_f1'] = model_f1
        
        return jsonify(response_data)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)})

@app.route('/metrics', methods=['GET'])
def get_metrics():
    """Return model performance metrics for all trained symbols."""
    if not models:
        return jsonify({'error': 'No models trained yet'})
    
    metrics_data = {}
    for symbol, info in models.items():
        metrics_data[symbol] = {
            'regression': {
                'train_r2': round(info.get('train_r2', 0), 4),
                'test_r2': round(info.get('test_r2', 0), 4),
                'train_rmse': round(info.get('train_rmse', 0), 4),
                'test_rmse': round(info.get('test_rmse', 0), 4),
                'train_mae': round(info.get('train_mae', 0), 4),
                'test_mae': round(info.get('test_mae', 0), 4),
            },
            'classification': {
                'train_accuracy': round(info.get('train_accuracy', 0), 4),
                'test_accuracy': round(info.get('test_accuracy', 0), 4),
                'gb_train_accuracy': round(info.get('gb_train_accuracy', 0), 4),
                'gb_test_accuracy': round(info.get('gb_test_accuracy', 0), 4),
                'train_precision': round(info.get('train_precision', 0), 4),
                'test_precision': round(info.get('test_precision', 0), 4),
                'train_recall': round(info.get('train_recall', 0), 4),
                'test_recall': round(info.get('test_recall', 0), 4),
                'train_f1': round(info.get('train_f1', 0), 4),
                'test_f1': round(info.get('test_f1', 0), 4),
            }
        }
    
    return jsonify(metrics_data)

if __name__ == '__main__':
    # Load data from CSV files at startup
    load_csv_data()
    
    # Train models in background thread
    print("\nStarting model training in background...")
    trainer_thread = threading.Thread(target=train_models, daemon=True)
    trainer_thread.start()
    
    print("\nStarting Flask server...")
    # Run Flask without debug reloader to maintain stable connections
    app.run(debug=False, host='0.0.0.0', port=5000, use_reloader=False)