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
import warnings
warnings.filterwarnings('ignore')

# Dataset paths
DATASET_DIR = 'd:\\market dataset'
INDEX_FILES = {
    'NIFTY': os.path.join(DATASET_DIR, 'NIFTY_50.csv'),
}
COMPANIES_FILES = [
    os.path.join(DATASET_DIR, 'NIFTY_50_COMPANIES.csv'),
]

# Global storage for data
historical_data = {}
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
    global historical_data, available_symbols

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

def train_single_model():
    """Train ML models for NIFTY to show gradient boosting accuracy."""
    global models

    print("\n" + "="*60)
    print("Training prediction model for NIFTY...")
    print("="*60)

    symbol = 'NIFTY'
    try:
        print(f"\nTraining model for {symbol}...")
        df = historical_data[symbol].copy()

        # Need at least 20 data points to train
        if len(df) < 20:
            print(f"  Skipping {symbol}: not enough data ({len(df)} rows)")
            return

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
            print(f"  Skipping {symbol}: not enough data after feature engineering ({len(df)} rows)")
            return

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
            print(f"  Skipping {symbol}: insufficient features ({len(feature_cols)})")
            return

        X = df[feature_cols]
        y = df['Close']

        split_idx = int(len(df) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        if len(X_train) < 10 or len(X_test) < 2:
            print(f"  Skipping {symbol}: insufficient train/test samples")
            return

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

        # Calculate individual model predictions for direction accuracy
        rf_pred_train_direction = (rf_pred_train[1:] > rf_pred_train[:-1]).astype(int)
        rf_pred_test_direction = (rf_pred_test[1:] > rf_pred_test[:-1]).astype(int)
        gb_pred_train_direction = (gb_pred_train[1:] > gb_pred_train[:-1]).astype(int)
        gb_pred_test_direction = (gb_pred_test[1:] > gb_pred_test[:-1]).astype(int)

        train_r2 = r2_score(y_train, gb_pred_train)
        test_r2 = r2_score(y_test, gb_pred_test)
        train_rmse = np.sqrt(mean_squared_error(y_train, gb_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, gb_pred_test))
        train_mae = mean_absolute_error(y_train, gb_pred_train)
        test_mae = mean_absolute_error(y_test, gb_pred_test)

        # Convert regression to classification: 1 if price goes up, 0 if down
        y_train_direction = (y_train.diff() > 0).astype(int)[1:]  # Remove first NaN
        y_test_direction = (y_test.diff() > 0).astype(int)[1:]    # Remove first NaN

        # Calculate classification metrics if we have enough samples
        if len(y_test_direction) > 1:
            rf_test_accuracy = accuracy_score(y_test_direction, rf_pred_test_direction)
            rf_train_accuracy = accuracy_score(y_train_direction, rf_pred_train_direction)
            gb_test_accuracy = accuracy_score(y_test_direction, gb_pred_test_direction)
            gb_train_accuracy = accuracy_score(y_train_direction, gb_pred_train_direction)
        else:
            rf_test_accuracy = rf_train_accuracy = 0.0
            gb_test_accuracy = gb_train_accuracy = 0.0

        # Print detailed metrics
        print(f"  [OK] Trained {symbol} Models:")
        print(f"       Classification Metrics (Direction Prediction):")
        print(f"          Random Forest Accuracy (Test):  {rf_test_accuracy:.4f} ({rf_test_accuracy*100:.2f}%)")
        print(f"          Gradient Boosting Accuracy (Test):  {gb_test_accuracy:.4f} ({gb_test_accuracy*100:.2f}%)")
        print(f"       Training Set Metrics:")
        print(f"          Random Forest Accuracy (Train):  {rf_train_accuracy:.4f} ({rf_train_accuracy*100:.2f}%)")
        print(f"          Gradient Boosting Accuracy (Train):  {gb_train_accuracy:.4f} ({gb_train_accuracy*100:.2f}%)")

        print(f"\n{'='*60}")
        print(f"MODEL ACCURACY SUMMARY")
        print(f"{'='*60}")
        print(f"Symbol: {symbol}")
        print(f"Random Forest Test Accuracy: {rf_test_accuracy:.4f} ({rf_test_accuracy*100:.2f}%)")
        print(f"Random Forest Train Accuracy: {rf_train_accuracy:.4f} ({rf_train_accuracy*100:.2f}%)")
        print(f"Gradient Boosting Test Accuracy: {gb_test_accuracy:.4f} ({gb_test_accuracy*100:.2f}%)")
        print(f"Gradient Boosting Train Accuracy: {gb_train_accuracy:.4f} ({gb_train_accuracy*100:.2f}%)")
        print(f"{'='*60}")

    except Exception as e:
        print(f"  [FAIL] Error training model for {symbol}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    load_csv_data()
    train_single_model()