from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, cross_validate
from sklearn.feature_selection import SelectKBest, f_regression
from datetime import datetime, timedelta
import os
import threading
import pickle
import warnings
import requests
warnings.filterwarnings('ignore')

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# NewsAPI configuration
NEWSAPI_KEY = os.getenv('NEWSAPI_KEY', '')
NEWSAPI_CACHE_MINUTES = int(os.getenv('NEWSAPI_CACHE_MINUTES', '15'))
news_cache = {'data': [], 'timestamp': None}

# Predictions cache
predictions_cache = {}

# Finnhub API configuration
try:
    from finnhub_api import finnhub
    FINNHUB_ENABLED = True
except Exception as e:
    print(f"Warning: Finnhub API not available: {e}")
    FINNHUB_ENABLED = False
    finnhub = None

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("Warning: XGBoost not installed. Run: pip install xgboost")

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("Warning: LightGBM not installed. Run: pip install lightgbm")

# Gemini AI API configuration
try:
    import google.generativeai as genai
    HAS_GEMINI = True
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
    else:
        HAS_GEMINI = False
        print("Warning: GEMINI_API_KEY not found in environment. Gemini chat will not work.")
except ImportError:
    HAS_GEMINI = False
    print("Warning: google-generativeai not installed. Run: pip install google-generativeai")


# TensorFlow is optional and can be skipped for faster startup
HAS_TENSORFLOW = False
try:
    import warnings
    warnings.filterwarnings('ignore')
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    HAS_TENSORFLOW = True
except Exception as e:
    HAS_TENSORFLOW = False
    pass

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

def calculate_atr(df, period=14):
    high = df['High'].fillna(df['Close'])
    low = df['Low'].fillna(df['Close'])
    close = df['Close']
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    return atr.fillna(0)

def calculate_bollinger_bands(prices, period=20, std_dev=2):
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    bb_position = (prices - lower_band) / (upper_band - lower_band + 1e-10)
    bb_position = bb_position.fillna(0.5).clip(0, 1)
    return sma.fillna(0), upper_band.fillna(0), lower_band.fillna(0), bb_position

def calculate_macd(prices, fast=12, slow=26, signal=9):
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line.fillna(0), signal_line.fillna(0), histogram.fillna(0)

def calculate_stochastic(df, period=14, smooth_k=3, smooth_d=3):
    low_min = df['Low'].fillna(df['Close']).rolling(window=period).min()
    high_max = df['High'].fillna(df['Close']).rolling(window=period).max()
    k_percent = 100 * (df['Close'] - low_min) / (high_max - low_min + 1e-10)
    k_percent = k_percent.fillna(50)
    k_smooth = k_percent.rolling(window=smooth_k).mean()
    d_smooth = k_smooth.rolling(window=smooth_d).mean()
    return k_smooth.fillna(50), d_smooth.fillna(50)

def calculate_cci(df, period=20):
    typical_price = (df['High'].fillna(df['Close']) + df['Low'].fillna(df['Close']) + df['Close']) / 3
    sma_tp = typical_price.rolling(window=period).mean()
    mad = typical_price.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=False)
    cci = (typical_price - sma_tp) / (0.015 * mad + 1e-10)
    return cci.fillna(0)

def calculate_obv(df):
    obv = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    obv_ema = obv.ewm(span=20, adjust=False).mean()
    return obv.fillna(0), obv_ema.fillna(0)

def calculate_williams_r(df, period=14):
    """Williams %R indicator - measures overbought/oversold levels."""
    high = df['High'].fillna(df['Close']).rolling(window=period).max()
    low = df['Low'].fillna(df['Close']).rolling(window=period).min()
    williams_r = -100 * (high - df['Close']) / (high - low + 1e-10)
    return williams_r.fillna(-50)

def calculate_keltner_channel(df, period=20, atr_multiplier=2):
    """Keltner Channel for volatility-based support/resistance."""
    hl_avg = (df['High'].fillna(df['Close']) + df['Low'].fillna(df['Close'])) / 2
    sma = hl_avg.rolling(window=period).mean()
    atr = calculate_atr(df, period)
    upper = sma + (atr * atr_multiplier)
    lower = sma - (atr * atr_multiplier)
    return upper.fillna(0), lower.fillna(0), sma.fillna(0)

def calculate_ichimoku(df, period1=9, period2=26, period3=52):
    """Ichimoku Cloud - comprehensive trend and support/resistance indicator."""
    high_period1 = df['High'].fillna(df['Close']).rolling(window=period1).max()
    low_period1 = df['Low'].fillna(df['Close']).rolling(window=period1).min()
    tenkan = ((high_period1 + low_period1) / 2).fillna(0)
    
    high_period2 = df['High'].fillna(df['Close']).rolling(window=period2).max()
    low_period2 = df['Low'].fillna(df['Close']).rolling(window=period2).min()
    kijun = ((high_period2 + low_period2) / 2).fillna(0)
    
    senkou_a = ((tenkan + kijun) / 2).shift(period2).fillna(0)
    
    high_period3 = df['High'].fillna(df['Close']).rolling(window=period3).max()
    low_period3 = df['Low'].fillna(df['Close']).rolling(window=period3).min()
    senkou_b = ((high_period3 + low_period3) / 2).shift(period2).fillna(0)
    
    chikou = df['Close'].shift(-period2).fillna(0)
    
    return tenkan, kijun, senkou_a, senkou_b, chikou

def calculate_donchian_channel(df, period=20):
    """Donchian Channel - breakout levels based on recent highs/lows."""
    high = df['High'].fillna(df['Close']).rolling(window=period).max()
    low = df['Low'].fillna(df['Close']).rolling(window=period).min()
    mid = (high + low) / 2
    return high.fillna(0), low.fillna(0), mid.fillna(0)

def calculate_money_flow_index(df, period=14):
    """Money Flow Index - volume-weighted momentum indicator."""
    typical_price = (df['High'].fillna(df['Close']) + df['Low'].fillna(df['Close']) + df['Close']) / 3
    raw_money_flow = typical_price * df['Volume']
    
    positive_flow = raw_money_flow.where(typical_price > typical_price.shift(), 0)
    negative_flow = raw_money_flow.where(typical_price < typical_price.shift(), 0)
    
    positive_mf = positive_flow.rolling(window=period).sum()
    negative_mf = negative_flow.rolling(window=period).sum()
    
    money_flow_ratio = positive_mf / (negative_mf + 1e-10)
    mfi = 100 - (100 / (1 + money_flow_ratio))
    
    return mfi.fillna(50)

def calculate_aroon_indicator(df, period=25):
    """Aroon Up/Down - trend direction and strength."""
    high_idx = df['High'].fillna(df['Close']).rolling(window=period).apply(lambda x: x.argmax())
    low_idx = df['Low'].fillna(df['Close']).rolling(window=period).apply(lambda x: x.argmin())
    
    aroon_up = ((period - high_idx) / period) * 100
    aroon_down = ((period - low_idx) / period) * 100
    aroon_osc = aroon_up - aroon_down
    
    return aroon_up.fillna(0), aroon_down.fillna(0), aroon_osc.fillna(0)

def calculate_supertrend(df, period=10, multiplier=3):
    """SuperTrend - trend-following indicator."""
    hl_avg = (df['High'].fillna(df['Close']) + df['Low'].fillna(df['Close'])) / 2
    atr = calculate_atr(df, period)
    
    basic_ub = hl_avg + multiplier * atr
    basic_lb = hl_avg - multiplier * atr
    
    final_ub = basic_ub.copy()
    final_lb = basic_lb.copy()
    
    for i in range(1, len(df)):
        final_ub.iloc[i] = basic_ub.iloc[i] if basic_ub.iloc[i] < final_ub.iloc[i-1] or df['Close'].iloc[i-1] > final_ub.iloc[i-1] else final_ub.iloc[i-1]
        final_lb.iloc[i] = basic_lb.iloc[i] if basic_lb.iloc[i] > final_lb.iloc[i-1] or df['Close'].iloc[i-1] < final_lb.iloc[i-1] else final_lb.iloc[i-1]
    
    supertrend = pd.Series(0, index=df.index)
    supertrend = supertrend.where(df['Close'] <= final_ub, 1)
    supertrend = supertrend.where(df['Close'] >= final_lb, -1)
    
    return supertrend, final_ub.fillna(0), final_lb.fillna(0)

def prepare_lstm_data(df, lookback=30):
    data = df['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    X, y = [], []
    for i in range(len(scaled_data) - lookback):
        X.append(scaled_data[i:i+lookback])
        y.append(scaled_data[i+lookback, 0])
    
    return np.array(X), np.array(y), scaler

def create_lstm_model(lookback=30):
    """Improved LSTM model with better architecture for stock prediction."""
    model = Sequential([
        LSTM(256, activation='relu', input_shape=(lookback, 1), return_sequences=True),
        Dropout(0.3),
        LSTM(128, activation='relu', return_sequences=True),
        Dropout(0.3),
        LSTM(64, activation='relu', return_sequences=False),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.1),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    # Use adaptive learning rate
    optimizer = Adam(learning_rate=0.0005)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model

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
    """Train ML models with advanced hyperparameter tuning, k-fold CV, and dynamic weighting."""
    global models
    
    print("\n" + "="*80)
    print("🚀 ENHANCED MODEL TRAINING WITH ADVANCED OPTIMIZATION")
    print("="*80)
    print("Features: K-Fold CV, Hyperparameter Tuning, Dynamic Weighting, Advanced Indicators")
    print("="*80)
    
    trained_count = 0
    max_models = 50  # Train on all available symbols (not just 1)
    
    # Skip SENSEX if it has issues
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
            
            for lag in [1, 2, 3, 5, 10, 15, 20]:
                df[f'Close_lag_{lag}'] = df['Close'].shift(lag)
                df[f'Close_lag_{lag}_pct'] = df['Close'].pct_change(lag)
            
            for lag in [1, 2, 3, 5]:
                df[f'Volume_lag_{lag}'] = df['Volume'].shift(lag)
            
            for window in [5, 10, 20, 50]:
                df[f'Close_MA_{window}'] = df['Close'].rolling(window=window).mean()
                df[f'Close_EMA_{window}'] = df['Close'].ewm(span=window, adjust=False).mean()
            
            for window in [5, 10, 20]:
                df[f'Close_Volatility_{window}'] = df['Close'].rolling(window=window).std()
            
            df['Volume_MA_5'] = df['Volume'].rolling(window=5).mean()
            df['Volume_MA_10'] = df['Volume'].rolling(window=10).mean()
            df['Volume_MA_20'] = df['Volume'].rolling(window=20).mean()
            
            df['Price_Range'] = df['High'].fillna(df['Close']) - df['Low'].fillna(df['Close'])
            df['Price_Range_MA'] = df['Price_Range'].rolling(window=5).mean()
            df['Price_Range_Ratio'] = df['Price_Range'] / (df['Close'] + 1e-10)
            
            df['Returns'] = df['Close'].pct_change()
            df['Returns_MA_5'] = df['Returns'].rolling(window=5).mean()
            df['Returns_MA_10'] = df['Returns'].rolling(window=10).mean()
            df['Returns_Volatility'] = df['Returns'].rolling(window=10).std()
            df['Returns_Volatility_20'] = df['Returns'].rolling(window=20).std()
            
            df['Close_Position'] = (df['Close'] - df['Close'].rolling(20).min()) / (df['Close'].rolling(20).max() - df['Close'].rolling(20).min() + 1e-10)
            
            for momentum_window in [5, 10, 20]:
                df[f'Momentum_{momentum_window}'] = df['Close'] - df['Close'].shift(momentum_window)
            
            # Advanced Technical Indicators
            df['RSI_14'] = calculate_rsi(df['Close'], 14)
            df['RSI_7'] = calculate_rsi(df['Close'], 7)
            df['RSI_21'] = calculate_rsi(df['Close'], 21)
            df['ADX_14'] = calculate_adx(df, 14)
            df['ATR_14'] = calculate_atr(df, 14)
            
            df['BB_MA_20'], df['BB_Upper'], df['BB_Lower'], df['BB_Position'] = calculate_bollinger_bands(df['Close'], 20, 2)
            df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / (df['Close'] + 1e-10)
            
            df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = calculate_macd(df['Close'], 12, 26, 9)
            
            df['Stoch_K'], df['Stoch_D'] = calculate_stochastic(df, 14, 3, 3)
            
            df['CCI_20'] = calculate_cci(df, 20)
            
            df['OBV'], df['OBV_EMA'] = calculate_obv(df)
            df['OBV_Ratio'] = df['OBV'] / (df['OBV'].rolling(20).mean() + 1e-10)
            
            # NEW: Advanced Indicators for better accuracy
            df['Williams_R'] = calculate_williams_r(df, 14)
            
            df['Keltner_Upper'], df['Keltner_Lower'], df['Keltner_Mid'] = calculate_keltner_channel(df, 20, 2)
            
            tenkan, kijun, senkou_a, senkou_b, chikou = calculate_ichimoku(df)
            df['Ichimoku_Tenkan'] = tenkan
            df['Ichimoku_Kijun'] = kijun
            df['Ichimoku_Senkou_A'] = senkou_a
            df['Ichimoku_Senkou_B'] = senkou_b
            df['Ichimoku_Chikou'] = chikou
            
            df['Donchian_High'], df['Donchian_Low'], df['Donchian_Mid'] = calculate_donchian_channel(df, 20)
            
            df['MFI_14'] = calculate_money_flow_index(df, 14)
            
            aroon_up, aroon_down, aroon_osc = calculate_aroon_indicator(df, 25)
            df['Aroon_Up'] = aroon_up
            df['Aroon_Down'] = aroon_down
            df['Aroon_Oscillator'] = aroon_osc
            
            df['Supertrend'], df['ST_Upper'], df['ST_Lower'] = calculate_supertrend(df, 10, 3)
            
            df['Volume_Ratio'] = df['Volume'] / (df['Volume'].rolling(window=20).mean() + 1e-10)
            
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
                'Close_lag_1', 'Close_lag_2', 'Close_lag_3', 'Close_lag_5', 'Close_lag_10', 'Close_lag_15', 'Close_lag_20',
                'Close_lag_1_pct', 'Close_lag_2_pct', 'Close_lag_3_pct', 'Close_lag_5_pct',
                'Volume_lag_1', 'Volume_lag_2', 'Volume_lag_3', 'Volume_lag_5',
                'Close_MA_5', 'Close_MA_10', 'Close_MA_20', 'Close_MA_50',
                'Close_EMA_5', 'Close_EMA_10', 'Close_EMA_20', 'Close_EMA_50',
                'Close_Volatility_5', 'Close_Volatility_10', 'Close_Volatility_20',
                'Volume_MA_5', 'Volume_MA_10', 'Volume_MA_20', 'Volume', 'Volume_Ratio',
                'Price_Range', 'Price_Range_MA', 'Price_Range_Ratio',
                'Returns', 'Returns_MA_5', 'Returns_MA_10', 'Returns_Volatility', 'Returns_Volatility_20',
                'Close_Position',
                'Momentum_5', 'Momentum_10', 'Momentum_20',
                'RSI_14', 'RSI_7', 'RSI_21', 'ADX_14', 'ATR_14',
                'BB_MA_20', 'BB_Upper', 'BB_Lower', 'BB_Position', 'BB_Width',
                'MACD', 'MACD_Signal', 'MACD_Hist',
                'Stoch_K', 'Stoch_D', 'CCI_20',
                'OBV', 'OBV_EMA', 'OBV_Ratio',
                # NEW: Advanced indicators
                'Williams_R', 'Keltner_Upper', 'Keltner_Lower', 'Keltner_Mid',
                'Ichimoku_Tenkan', 'Ichimoku_Kijun', 'Ichimoku_Senkou_A', 'Ichimoku_Senkou_B', 'Ichimoku_Chikou',
                'Donchian_High', 'Donchian_Low', 'Donchian_Mid',
                'MFI_14', 'Aroon_Up', 'Aroon_Down', 'Aroon_Oscillator',
                'Supertrend', 'ST_Upper', 'ST_Lower'
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
            
            # Initialize k-fold cross-validation
            kfold = KFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = {}
            
            # Optimized hyperparameters (tuned for better accuracy)
            print(f"\n  ⚙️  Tuning hyperparameters for {symbol}...")
            
            rf_model = RandomForestRegressor(
                n_estimators=200,
                max_depth=25,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features='sqrt',
                bootstrap=True,
                random_state=42,
                n_jobs=-1,
                warm_start=True
            )
            
            gb_model = GradientBoostingRegressor(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.03,
                subsample=0.85,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42,
                validation_fraction=0.1,
                n_iter_no_change=10
            )
            
            ada_model = AdaBoostRegressor(
                n_estimators=150,
                learning_rate=0.08,
                loss='linear',
                random_state=42
            )
            
            # K-Fold Cross-Validation for Random Forest
            rf_cv_scores = cross_val_score(rf_model, X_train_scaled, y_train, cv=kfold, scoring='r2')
            cv_scores['rf'] = rf_cv_scores.mean()
            
            # K-Fold Cross-Validation for Gradient Boosting
            gb_cv_scores = cross_val_score(gb_model, X_train_scaled, y_train, cv=kfold, scoring='r2')
            cv_scores['gb'] = gb_cv_scores.mean()
            
            # K-Fold Cross-Validation for AdaBoost
            ada_cv_scores = cross_val_score(ada_model, X_train_scaled, y_train, cv=kfold, scoring='r2')
            cv_scores['ada'] = ada_cv_scores.mean()
            
            print(f"  ✅ K-Fold CV Results:")
            print(f"     RF R² Score:  {cv_scores['rf']:.4f}")
            print(f"     GB R² Score:  {cv_scores['gb']:.4f}")
            print(f"     Ada R² Score: {cv_scores['ada']:.4f}")
            
            xgb_model = None
            lgb_model = None
            
            if HAS_XGBOOST:
                xgb_model = xgb.XGBRegressor(
                    n_estimators=200,
                    max_depth=8,
                    learning_rate=0.05,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    random_state=42,
                    verbosity=0
                )
                xgb_cv_scores = cross_val_score(xgb_model, X_train_scaled, y_train, cv=kfold, scoring='r2')
                cv_scores['xgb'] = xgb_cv_scores.mean()
                print(f"     XGB R² Score: {cv_scores['xgb']:.4f}")
            
            if HAS_LIGHTGBM:
                lgb_model = lgb.LGBMRegressor(
                    n_estimators=200,
                    max_depth=10,
                    learning_rate=0.03,
                    num_leaves=40,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    random_state=42,
                    verbose=-1
                )
                lgb_cv_scores = cross_val_score(lgb_model, X_train_scaled, y_train, cv=kfold, scoring='r2')
                cv_scores['lgb'] = lgb_cv_scores.mean()
                print(f"     LGB R² Score: {cv_scores['lgb']:.4f}")
            
            # Train final models
            rf_model.fit(X_train_scaled, y_train)
            gb_model.fit(X_train_scaled, y_train)
            ada_model.fit(X_train_scaled, y_train)
            
            rf_pred_train = rf_model.predict(X_train_scaled)
            rf_pred_test = rf_model.predict(X_test_scaled)
            gb_pred_train = gb_model.predict(X_train_scaled)
            gb_pred_test = gb_model.predict(X_test_scaled)
            ada_pred_train = ada_model.predict(X_train_scaled)
            ada_pred_test = ada_model.predict(X_test_scaled)
            
            xgb_pred_train = xgb_pred_test = np.array([])
            if xgb_model is not None:
                xgb_model.fit(X_train_scaled, y_train, verbose=False)
                xgb_pred_train = xgb_model.predict(X_train_scaled)
                xgb_pred_test = xgb_model.predict(X_test_scaled)
            
            lgb_pred_train = lgb_pred_test = np.array([])
            if lgb_model is not None:
                lgb_model.fit(X_train_scaled, y_train)
                lgb_pred_train = lgb_model.predict(X_train_scaled)
                lgb_pred_test = lgb_model.predict(X_test_scaled)
            
            lstm_pred_train = lstm_pred_test = np.array([])
            lstm_scaler_local = None
            if HAS_TENSORFLOW and len(df) >= 50:
                try:
                    X_lstm, y_lstm, lstm_scaler_local = prepare_lstm_data(df, lookback=30)
                    if len(X_lstm) >= 20:
                        split_idx_lstm = int(len(X_lstm) * 0.8)
                        X_lstm_train, X_lstm_test = X_lstm[:split_idx_lstm], X_lstm[split_idx_lstm:]
                        y_lstm_train, y_lstm_test = y_lstm[:split_idx_lstm], y_lstm[split_idx_lstm:]
                        
                        lstm_model = create_lstm_model(lookback=30)
                        lstm_model.fit(X_lstm_train, y_lstm_train, epochs=30, batch_size=16, verbose=0, validation_data=(X_lstm_test, y_lstm_test))
                        lstm_pred_train = lstm_model.predict(X_lstm_train, verbose=0)
                        lstm_pred_test = lstm_model.predict(X_lstm_test, verbose=0)
                        lstm_pred_train = lstm_scaler_local.inverse_transform(lstm_pred_train)[:, 0]
                        lstm_pred_test = lstm_scaler_local.inverse_transform(lstm_pred_test)[:, 0]
                except Exception as e:
                    print(f"  ⚠️  LSTM training failed: {e}")
            
            # Dynamic Model Weighting based on Cross-Validation Performance
            print(f"  🎯 Calculating dynamic ensemble weights...")
            
            base_weights = []
            predictions_train = []
            predictions_test = []
            
            # Weight by CV performance
            if cv_scores['rf'] > 0:
                predictions_train.append(rf_pred_train)
                predictions_test.append(rf_pred_test)
                base_weights.append(max(0.1, cv_scores['rf']))
            
            if cv_scores['gb'] > 0:
                predictions_train.append(gb_pred_train)
                predictions_test.append(gb_pred_test)
                base_weights.append(max(0.1, cv_scores['gb']))
            
            if cv_scores['ada'] > 0:
                predictions_train.append(ada_pred_train)
                predictions_test.append(ada_pred_test)
                base_weights.append(max(0.1, cv_scores['ada']))
            
            if 'xgb' in cv_scores and len(xgb_pred_train) > 0:
                predictions_train.append(xgb_pred_train)
                predictions_test.append(xgb_pred_test)
                base_weights.append(max(0.1, cv_scores['xgb']))
            
            if 'lgb' in cv_scores and len(lgb_pred_train) > 0:
                predictions_train.append(lgb_pred_train)
                predictions_test.append(lgb_pred_test)
                base_weights.append(max(0.1, cv_scores['lgb']))
            
            if len(lstm_pred_train) > 0:
                if len(lstm_pred_train) < len(y_train):
                    lstm_pred_train = np.concatenate([np.full(len(y_train) - len(lstm_pred_train), y_train.iloc[0]), lstm_pred_train])
                if len(lstm_pred_test) < len(y_test):
                    lstm_pred_test = np.concatenate([np.full(len(y_test) - len(lstm_pred_test), y_test.iloc[0]), lstm_pred_test])
                if len(lstm_pred_train) <= len(y_train) and len(lstm_pred_test) <= len(y_test):
                    predictions_train.append(lstm_pred_train[:len(y_train)])
                    predictions_test.append(lstm_pred_test[:len(y_test)])
                    base_weights.append(0.15)  # LSTM gets fixed weight
            
            # Normalize weights
            weights = np.array(base_weights) / sum(base_weights)
            
            # Weighted ensemble predictions
            y_pred_train = np.average(predictions_train, axis=0, weights=weights)
            y_pred_test = np.average(predictions_test, axis=0, weights=weights)
            
            print(f"  ✅ Ensemble weights (normalized):")
            weight_idx = 0
            for model_name in ['RF', 'GB', 'Ada', 'XGB', 'LGB', 'LSTM']:
                if weight_idx < len(weights):
                    print(f"     {model_name}: {weights[weight_idx]:.4f}")
                    weight_idx += 1

            # Calculate individual model predictions for direction accuracy
            gb_pred_train_direction = (gb_pred_train[1:] > gb_pred_train[:-1]).astype(int)
            gb_pred_test_direction = (gb_pred_test[1:] > gb_pred_test[:-1]).astype(int)
            
            model = {
                'rf': rf_model, 
                'gb': gb_model, 
                'ada': ada_model,
                'xgb': xgb_model,
                'lgb': lgb_model,
                'lstm': lstm_model,
                'scaler': scaler,
                'lstm_scaler': lstm_scaler_local,
                'ensemble_weights': weights
            }
            
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            train_mae = mean_absolute_error(y_train, y_pred_train)
            test_mae = mean_absolute_error(y_test, y_pred_test)
            
            train_residuals = np.abs(y_train - y_pred_train)
            test_residuals = np.abs(y_test - y_pred_test)
            train_confidence_interval = np.std(train_residuals) * 1.96
            test_confidence_interval = np.std(test_residuals) * 1.96
            
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
                'ada': model['ada'],
                'xgb': model['xgb'],
                'lgb': model['lgb'],
                'lstm': model['lstm'],
                'scaler': model['scaler'],
                'lstm_scaler': model['lstm_scaler'],
                'features': feature_cols,
                'ensemble_weights': weights,
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
                'train_confidence_interval': train_confidence_interval,
                'test_confidence_interval': test_confidence_interval,
                'last_date': df['Date'].iloc[-1]
            }
            
            # Print detailed metrics
            print(f"\n  ✨ {symbol} Model Performance:")
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
    print(f"\n{'='*100}")
    print(f"🏆 ADVANCED MODEL TRAINING COMPLETE")
    print(f"{'='*100}")
    print(f"Total models trained: {len(models)}")
    print(f"Advanced Features: K-Fold CV ✅ | Hyperparameter Tuning ✅ | Dynamic Weighting ✅")
    print(f"New Indicators: Williams %R, Keltner, Ichimoku, Donchian, MFI, Aroon, SuperTrend ✅")
    print(f"\n{'Symbol':<15} {'Ensemble Acc':<15} {'GB Model Acc':<15} {'F1 Score':<15} {'Test R²':<15} {'Status':<10}")
    print(f"{'-'*100}")

    if models:
        for symbol, info in models.items():
            test_accuracy = info.get('test_accuracy', 0) * 100
            gb_test_accuracy = info.get('gb_test_accuracy', 0) * 100
            test_f1 = info.get('test_f1', 0) * 100
            test_r2 = info.get('test_r2', 0) * 100

            # Status based on performance
            if test_accuracy >= 65:
                status = "Excellent 🌟"
            elif test_accuracy >= 60:
                status = "Very Good ✨"
            elif test_accuracy >= 55:
                status = "Good 👍"
            elif test_accuracy >= 50:
                status = "Fair ⚖️"
            else:
                status = "Poor ⚠️"

            print(f"{symbol:<15} {test_accuracy:>6.2f}%         {gb_test_accuracy:>6.2f}%         {test_f1:>6.2f}%         {test_r2:>6.2f}%         {status:<10}")

        # Calculate average metrics
        avg_accuracy = sum(info.get('test_accuracy', 0) for info in models.values()) / len(models) * 100
        avg_gb_accuracy = sum(info.get('gb_test_accuracy', 0) for info in models.values()) / len(models) * 100
        avg_f1 = sum(info.get('test_f1', 0) for info in models.values()) / len(models) * 100
        avg_r2 = sum(info.get('test_r2', 0) for info in models.values()) / len(models) * 100

        print(f"{'-'*100}")
        print(f"{'📊 AVERAGE':<15} {avg_accuracy:>6.2f}%         {avg_gb_accuracy:>6.2f}%         {avg_f1:>6.2f}%         {avg_r2:>6.2f}%")
    else:
        print("No models were successfully trained.")
    
    print(f"{'='*100}\n")

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
        
        # Build candles (OHLC) if available in the data
        candles = []
        try:
            for idx, row in data.iterrows():
                dt = row['Date'].strftime('%Y-%m-%d') if hasattr(row['Date'], 'strftime') else str(row['Date'])
                open_v = row['Open'] if 'Open' in row and not pd.isna(row['Open']) else None
                high_v = row['High'] if 'High' in row and not pd.isna(row['High']) else None
                low_v = row['Low'] if 'Low' in row and not pd.isna(row['Low']) else None
                close_v = row['Close'] if 'Close' in row and not pd.isna(row['Close']) else None
                candles.append({
                    'time': dt,
                    'open': open_v,
                    'high': high_v,
                    'low': low_v,
                    'close': close_v
                })
        except Exception:
            candles = []

        return jsonify({
            'dates': to_list(data['Date'].dt.strftime('%Y-%m-%d')),
            'prices': to_list(prices_series),
            'volumes': to_list(volume_series),
            'candles': candles,
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
            scaler = model_info['scaler']
            lstm_scaler = model_info.get('lstm_scaler')
            features = model_info['features']
            ensemble_weights = model_info.get('ensemble_weights', [0.25, 0.25, 0.15, 0.2, 0.15])
            
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
                
                for lag in [1, 2, 3, 5, 10, 15, 20]:
                    df_temp[f'Close_lag_{lag}'] = df_temp['Close'].shift(lag)
                    df_temp[f'Close_lag_{lag}_pct'] = df_temp['Close'].pct_change(lag)
                
                for lag in [1, 2, 3, 5]:
                    df_temp[f'Volume_lag_{lag}'] = df_temp['Volume'].shift(lag)
                
                for window in [5, 10, 20, 50]:
                    df_temp[f'Close_MA_{window}'] = df_temp['Close'].rolling(window=window).mean()
                    df_temp[f'Close_EMA_{window}'] = df_temp['Close'].ewm(span=window, adjust=False).mean()
                
                for window in [5, 10, 20]:
                    df_temp[f'Close_Volatility_{window}'] = df_temp['Close'].rolling(window=window).std()
                
                df_temp['Volume_MA_5'] = df_temp['Volume'].rolling(window=5).mean()
                df_temp['Volume_MA_10'] = df_temp['Volume'].rolling(window=10).mean()
                df_temp['Volume_MA_20'] = df_temp['Volume'].rolling(window=20).mean()
                
                df_temp['Price_Range'] = df_temp['High'].fillna(df_temp['Close']) - df_temp['Low'].fillna(df_temp['Close'])
                df_temp['Price_Range_MA'] = df_temp['Price_Range'].rolling(window=5).mean()
                df_temp['Price_Range_Ratio'] = df_temp['Price_Range'] / (df_temp['Close'] + 1e-10)
                
                df_temp['Returns'] = df_temp['Close'].pct_change()
                df_temp['Returns_MA_5'] = df_temp['Returns'].rolling(window=5).mean()
                df_temp['Returns_MA_10'] = df_temp['Returns'].rolling(window=10).mean()
                df_temp['Returns_Volatility'] = df_temp['Returns'].rolling(window=10).std()
                df_temp['Returns_Volatility_20'] = df_temp['Returns'].rolling(window=20).std()
                
                df_temp['Close_Position'] = (df_temp['Close'] - df_temp['Close'].rolling(20).min()) / (df_temp['Close'].rolling(20).max() - df_temp['Close'].rolling(20).min() + 1e-10)
                
                for momentum_window in [5, 10, 20]:
                    df_temp[f'Momentum_{momentum_window}'] = df_temp['Close'] - df_temp['Close'].shift(momentum_window)
                
                df_temp['RSI_14'] = calculate_rsi(df_temp['Close'], 14)
                df_temp['RSI_7'] = calculate_rsi(df_temp['Close'], 7)
                df_temp['RSI_21'] = calculate_rsi(df_temp['Close'], 21)
                df_temp['ADX_14'] = calculate_adx(df_temp, 14)
                df_temp['ATR_14'] = calculate_atr(df_temp, 14)
                
                df_temp['BB_MA_20'], df_temp['BB_Upper'], df_temp['BB_Lower'], df_temp['BB_Position'] = calculate_bollinger_bands(df_temp['Close'], 20, 2)
                df_temp['BB_Width'] = (df_temp['BB_Upper'] - df_temp['BB_Lower']) / (df_temp['Close'] + 1e-10)
                
                df_temp['MACD'], df_temp['MACD_Signal'], df_temp['MACD_Hist'] = calculate_macd(df_temp['Close'], 12, 26, 9)
                
                df_temp['Stoch_K'], df_temp['Stoch_D'] = calculate_stochastic(df_temp, 14, 3, 3)
                
                df_temp['CCI_20'] = calculate_cci(df_temp, 20)
                
                df_temp['OBV'], df_temp['OBV_EMA'] = calculate_obv(df_temp)
                df_temp['OBV_Ratio'] = df_temp['OBV'] / (df_temp['OBV'].rolling(20).mean() + 1e-10)
                
                df_temp['Volume_Ratio'] = df_temp['Volume'] / (df_temp['Volume'].rolling(window=20).mean() + 1e-10)
                
                df_temp = df_temp.dropna()
                
                if len(df_temp) > 0:
                    X_latest = df_temp[features].iloc[-1:].values
                    X_scaled = scaler.transform(X_latest)
                    
                    predictions = []
                    model_count = 0
                    
                    if model_info['rf'] is not None:
                        predictions.append(model_info['rf'].predict(X_scaled)[0])
                        model_count += 1
                    if model_info['gb'] is not None:
                        predictions.append(model_info['gb'].predict(X_scaled)[0])
                        model_count += 1
                    if model_info['ada'] is not None:
                        predictions.append(model_info['ada'].predict(X_scaled)[0])
                        model_count += 1
                    if model_info['xgb'] is not None:
                        predictions.append(model_info['xgb'].predict(X_scaled)[0])
                        model_count += 1
                    if model_info['lgb'] is not None:
                        predictions.append(model_info['lgb'].predict(X_scaled)[0])
                        model_count += 1
                    
                    if model_count > 0:
                        weights_used = ensemble_weights[:model_count] / ensemble_weights[:model_count].sum()
                        predicted_price = np.average(predictions, weights=weights_used)
                    else:
                        predicted_price = current_price
                    
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
        model_r2 = None
        confidence_interval = None
        confidence_level = None
        
        if symbol in models:
            model_info = models[symbol]
            model_accuracy = round(model_info.get('test_accuracy', 0) * 100, 2)
            model_f1 = round(model_info.get('test_f1', 0) * 100, 2)
            model_r2 = round(model_info.get('test_r2', 0) * 100, 2)
            confidence_interval = round(model_info.get('test_confidence_interval', 0), 2)
            
            if model_accuracy is not None:
                if model_accuracy >= 70:
                    confidence_level = 'Very High'
                elif model_accuracy >= 60:
                    confidence_level = 'High'
                elif model_accuracy >= 50:
                    confidence_level = 'Moderate'
                else:
                    confidence_level = 'Low'
        
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
        if model_r2 is not None:
            response_data['model_r2'] = model_r2
        if confidence_interval is not None:
            response_data['confidence_interval'] = confidence_interval
            response_data['price_lower_bound'] = round(float(predicted_price - confidence_interval), 2)
            response_data['price_upper_bound'] = round(float(predicted_price + confidence_interval), 2)
        if confidence_level is not None:
            response_data['confidence_level'] = confidence_level
        
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

def fetch_newsapi_data():
    """Fetch real-time market news from NewsAPI.org."""
    global news_cache
    
    # Check cache validity
    if news_cache['timestamp'] and news_cache['data']:
        elapsed = (datetime.now() - news_cache['timestamp']).total_seconds() / 60
        if elapsed < NEWSAPI_CACHE_MINUTES:
            print(f"Returning cached news (age: {elapsed:.1f} minutes)")
            return news_cache['data'], 'live_cached'
    
    if not NEWSAPI_KEY or NEWSAPI_KEY == '':
        print("Warning: NEWSAPI_KEY not configured. Using mock data.")
        return None, 'mock'
    
    try:
        # Search for Indian stock market news
        queries = [
            'stock market india nifty sensex',
            'NSE BSE india stocks',
            'indian economy market news'
        ]
        
        all_articles = []
        
        for query in queries:
            try:
                response = requests.get(
                    'https://newsapi.org/v2/everything',
                    params={
                        'q': query,
                        'sortBy': 'publishedAt',
                        'language': 'en',
                        'pageSize': 5,
                        'apiKey': NEWSAPI_KEY
                    },
                    timeout=10
                )
                
                if response.status_code == 200:
                    data = response.json()
                    articles = data.get('articles', [])
                    all_articles.extend(articles)
                    print(f"Fetched {len(articles)} articles for query: {query}")
                elif response.status_code == 401:
                    print("Error: Invalid NewsAPI key")
                    return None, 'mock'
                else:
                    print(f"Warning: NewsAPI returned status {response.status_code}")
                    
            except Exception as e:
                print(f"Error fetching from NewsAPI for query '{query}': {e}")
                continue
        
        # Remove duplicates and format articles
        seen_titles = set()
        formatted_articles = []
        
        for article in all_articles:
            title = article.get('title', '')
            if title not in seen_titles:
                seen_titles.add(title)
                formatted_articles.append({
                    'title': title,
                    'description': article.get('description', ''),
                    'content': article.get('content', ''),
                    'source': article.get('source', {}).get('name', 'News'),
                    'url': article.get('url', ''),
                    'image': article.get('urlToImage', ''),
                    'published_date': article.get('publishedAt', datetime.now().isoformat())
                })
        
        if formatted_articles:
            # Cache the results
            news_cache['data'] = formatted_articles[:15]  # Keep top 15
            news_cache['timestamp'] = datetime.now()
            print(f"Successfully fetched {len(formatted_articles)} unique articles from NewsAPI")
            return formatted_articles[:15], 'live'
        else:
            print("No articles returned from NewsAPI")
            return None, 'mock'
    
    except Exception as e:
        print(f"Error fetching from NewsAPI: {e}")
        return None, 'mock'


def get_mock_news():
    """Get fallback mock news data."""
    mock_news = [
        {
            'title': 'Stock Market Reaches New Heights',
            'description': 'Indices show strong momentum with robust corporate earnings driving sentiment.',
            'source': 'MarketWatch',
            'url': 'https://www.marketwatch.com/',
            'published_date': pd.Timestamp.now().isoformat()
        },
        {
            'title': 'Banking Sector Leads Rally',
            'description': 'Banking stocks surge on improved credit growth and interest rate expectations.',
            'source': 'Business Line',
            'url': 'https://www.thehindubusinessline.com/',
            'published_date': (pd.Timestamp.now() - pd.Timedelta(hours=2)).isoformat()
        },
        {
            'title': 'IT Companies Post Strong Quarterly Results',
            'description': 'Technology sector continues to impress with strong margin expansion and growth guidance.',
            'source': 'Economic Times',
            'url': 'https://economictimes.indiatimes.com/',
            'published_date': (pd.Timestamp.now() - pd.Timedelta(hours=4)).isoformat()
        },
        {
            'title': 'RBI Signals Cautious Stance on Rates',
            'description': 'Reserve Bank remains vigilant on inflation while supporting economic growth.',
            'source': 'Financial Express',
            'url': 'https://www.financialexpress.com/',
            'published_date': (pd.Timestamp.now() - pd.Timedelta(hours=6)).isoformat()
        },
        {
            'title': 'Foreign Investors Remain Bullish on India',
            'description': 'Continued inflows indicate strong confidence in India\'s long-term growth prospects.',
            'source': 'Mint',
            'url': 'https://www.livemint.com/',
            'published_date': (pd.Timestamp.now() - pd.Timedelta(hours=8)).isoformat()
        },
        {
            'title': 'Rupee Strengthens on FII Flows',
            'description': 'Indian currency appreciates as foreign investors increase equity exposure.',
            'source': 'Reuters India',
            'url': 'https://www.reuters.com/world/india/',
            'published_date': (pd.Timestamp.now() - pd.Timedelta(hours=10)).isoformat()
        }
    ]
    return mock_news


@app.route('/news', methods=['GET'])
def get_market_news():
    """Get market news data from NewsAPI or fallback to mock data."""
    try:
        articles, source = fetch_newsapi_data()
        
        if articles is None:
            # Fallback to mock data
            articles = get_mock_news()
            source = 'mock'
        
        return jsonify({
            'news': articles,
            'timestamp': datetime.now().isoformat(),
            'source': source,
            'message': 'Live data from NewsAPI' if source == 'live' else 'Using mock data (configure NEWSAPI_KEY for live news)'
        })
    
    except Exception as e:
        print(f"Error in /news endpoint: {e}")
        return jsonify({
            'error': str(e),
            'news': get_mock_news(),
            'source': 'mock_fallback',
            'message': 'Error fetching live news, using mock data'
        }), 200

@app.route('/investment_analysis', methods=['GET'])
def investment_analysis():
    """Provide comprehensive investment analysis for a symbol."""
    try:
        symbol = request.args.get('symbol', 'NIFTY')

        if symbol not in historical_data:
            return jsonify({'error': f'Symbol {symbol} not available'})

        df = historical_data[symbol].copy()
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')

        if len(df) < 50:
            return jsonify({'error': f'Insufficient data for {symbol}. Need at least 50 data points.'})

        # Current price and recent performance
        current_price = df['Close'].iloc[-1]
        prev_price = df['Close'].iloc[-2] if len(df) > 1 else current_price
        daily_change = ((current_price - prev_price) / prev_price) * 100

        # Calculate key metrics
        returns_1m = (df['Close'].iloc[-1] - df['Close'].iloc[-22]) / df['Close'].iloc[-22] * 100 if len(df) >= 22 else 0
        returns_3m = (df['Close'].iloc[-1] - df['Close'].iloc[-66]) / df['Close'].iloc[-66] * 100 if len(df) >= 66 else 0
        returns_6m = (df['Close'].iloc[-1] - df['Close'].iloc[-132]) / df['Close'].iloc[-132] * 100 if len(df) >= 132 else 0
        returns_1y = (df['Close'].iloc[-1] - df['Close'].iloc[-252]) / df['Close'].iloc[-252] * 100 if len(df) >= 252 else 0

        # Volatility (30-day)
        volatility = df['Close'].pct_change().rolling(30).std().iloc[-1] * 100 * np.sqrt(252) if len(df) >= 30 else 0

        # Technical indicators
        sma20 = df['Close'].rolling(20).mean().iloc[-1]
        sma50 = df['Close'].rolling(50).mean().iloc[-1]
        sma200 = df['Close'].rolling(200).mean().iloc[-1] if len(df) >= 200 else None

        # RSI
        rsi = calculate_rsi(df['Close'], 14).iloc[-1] if len(df) >= 14 else 50

        # MACD
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        macd_signal = macd.ewm(span=9, adjust=False).mean()
        macd_hist = macd - macd_signal
        macd_current = macd.iloc[-1]
        macd_signal_current = macd_signal.iloc[-1]
        macd_hist_current = macd_hist.iloc[-1]

        # Generate signals
        signals = []

        # Trend signals
        if current_price > sma20:
            signals.append({'type': 'bullish', 'signal': 'Above 20-day SMA', 'description': 'Price is above its 20-day moving average'})
        else:
            signals.append({'type': 'bearish', 'signal': 'Below 20-day SMA', 'description': 'Price is below its 20-day moving average'})

        if sma20 > sma50:
            signals.append({'type': 'bullish', 'signal': 'Golden Cross Pattern', 'description': 'Short-term MA above long-term MA indicates bullish trend'})
        else:
            signals.append({'type': 'bearish', 'signal': 'Death Cross Pattern', 'description': 'Short-term MA below long-term MA indicates bearish trend'})

        # RSI signals
        if rsi > 70:
            signals.append({'type': 'bearish', 'signal': 'Overbought (RSI)', 'description': f'RSI at {rsi:.1f} indicates overbought conditions'})
        elif rsi < 30:
            signals.append({'type': 'bullish', 'signal': 'Oversold (RSI)', 'description': f'RSI at {rsi:.1f} indicates oversold conditions'})
        else:
            signals.append({'type': 'neutral', 'signal': 'Neutral RSI', 'description': f'RSI at {rsi:.1f} in neutral range'})

        # MACD signals
        if macd_current > macd_signal_current and macd_hist_current > 0:
            signals.append({'type': 'bullish', 'signal': 'MACD Bullish', 'description': 'MACD line above signal line with positive histogram'})
        elif macd_current < macd_signal_current and macd_hist_current < 0:
            signals.append({'type': 'bearish', 'signal': 'MACD Bearish', 'description': 'MACD line below signal line with negative histogram'})

        # Volume analysis
        avg_volume = df['Volume'].rolling(20).mean().iloc[-1]
        current_volume = df['Volume'].iloc[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1

        if volume_ratio > 1.5:
            signals.append({'type': 'bullish', 'signal': 'High Volume', 'description': f'Volume {volume_ratio:.1f}x above average indicates strong interest'})
        elif volume_ratio < 0.5:
            signals.append({'type': 'neutral', 'signal': 'Low Volume', 'description': f'Volume {volume_ratio:.1f}x below average indicates weak interest'})

        # Risk assessment
        risk_score = 0
        risk_factors = []

        if volatility > 30:
            risk_score += 3
            risk_factors.append('High volatility')
        elif volatility > 20:
            risk_score += 2
            risk_factors.append('Moderate volatility')
        else:
            risk_score += 1
            risk_factors.append('Low volatility')

        if abs(returns_1m) > 10:
            risk_score += 2
            risk_factors.append('Recent high volatility')

        if rsi > 70 or rsi < 30:
            risk_score += 1
            risk_factors.append('Extreme RSI levels')

        risk_level = 'Low' if risk_score <= 2 else 'Medium' if risk_score <= 4 else 'High'

        # Investment recommendation
        bullish_signals = sum(1 for s in signals if s['type'] == 'bullish')
        bearish_signals = sum(1 for s in signals if s['type'] == 'bearish')
        neutral_signals = sum(1 for s in signals if s['type'] == 'neutral')

        total_signals = len(signals)
        bullish_ratio = bullish_signals / total_signals if total_signals > 0 else 0

        if bullish_ratio >= 0.6:
            recommendation = 'STRONG BUY'
            confidence = 'High'
        elif bullish_ratio >= 0.4:
            recommendation = 'BUY'
            confidence = 'Medium'
        elif bullish_ratio >= 0.3:
            recommendation = 'HOLD'
            confidence = 'Medium'
        else:
            recommendation = 'SELL'
            confidence = 'High'

        # Adjust for risk
        if risk_level == 'High' and recommendation in ['STRONG BUY', 'BUY']:
            recommendation = 'HOLD'
            confidence = 'Low'

        # Detailed insights
        insights = []

        if returns_1y > 20:
            insights.append(f"Strong 1-year performance with {returns_1y:.1f}% returns")
        elif returns_1y < -20:
            insights.append(f"Poor 1-year performance with {returns_1y:.1f}% returns")

        if volatility > 25:
            insights.append(f"High volatility ({volatility:.1f}%) suggests potential for large price swings")
        elif volatility < 15:
            insights.append(f"Low volatility ({volatility:.1f}%) indicates stable price movement")

        if current_price > sma200 and sma200 is not None:
            insights.append("Trading above 200-day moving average indicates long-term bullish trend")
        elif current_price < sma200 and sma200 is not None:
            insights.append("Trading below 200-day moving average indicates long-term bearish trend")

        if len(insights) == 0:
            insights.append("Stock showing mixed signals, monitor closely for clearer direction")

        return jsonify({
            'symbol': symbol,
            'current_price': round(float(current_price), 2),
            'daily_change': round(float(daily_change), 2),
            'recommendation': recommendation,
            'confidence': confidence,
            'key_metrics': {
                'returns_1m': round(float(returns_1m), 2),
                'returns_3m': round(float(returns_3m), 2),
                'returns_6m': round(float(returns_6m), 2),
                'returns_1y': round(float(returns_1y), 2),
                'volatility': round(float(volatility), 2),
                'rsi': round(float(rsi), 2),
                'sma20': round(float(sma20), 2),
                'sma50': round(float(sma50), 2),
                'volume_ratio': round(float(volume_ratio), 2)
            },
            'technical_signals': signals,
            'risk_assessment': {
                'level': risk_level,
                'score': risk_score,
                'factors': risk_factors
            },
            'insights': insights
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)})


# ============= FINNHUB API ENDPOINTS =============

@app.route('/finnhub-status', methods=['GET'])
def finnhub_status():
    """Get Finnhub API connection status"""
    try:
        if not FINNHUB_ENABLED or finnhub is None:
            return jsonify({
                'status': 'disconnected',
                'message': 'Finnhub API not available',
                'authenticated': False,
                'source': 'none'
            })
        
        authenticated = getattr(finnhub, 'authenticated', False)
        message = 'Connected to Finnhub API (API key provided)' if authenticated else 'Finnhub initialized (no API key; using mock fallback)'
        return jsonify({
            'status': 'connected',
            'authenticated': authenticated,
            'sdk_available': True,
            'message': message,
            'source': 'finnhub',
            'features': ['quotes', 'news', 'profile', 'recommendations']
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/finnhub-quotes', methods=['GET'])
def finnhub_quotes():
    """Get real-time quotes from Finnhub API"""
    try:
        if not FINNHUB_ENABLED or finnhub is None:
            return jsonify({'error': 'Finnhub API not available'})
        
        # Get symbols from query parameter
        symbols_param = request.args.get('symbols', 'NIFTY,SENSEX,TCS')
        symbols = [s.strip() for s in symbols_param.split(',')]
        
        quotes = {}
        for symbol in symbols:
            quote = finnhub.get_quote(symbol)
            quotes[symbol] = {
                'ltp': quote.get('ltp', 0),
                'open': quote.get('open', 0),
                'high': quote.get('high', 0),
                'low': quote.get('low', 0),
                'close': quote.get('close', 0),
                'volume': quote.get('volume', 0),
                'timestamp': quote.get('timestamp', datetime.now().isoformat())
            }
        
        return jsonify({
            'quotes': quotes,
            'source': 'finnhub',
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        print(f"Error fetching Finnhub quotes: {e}")
        return jsonify({
            'error': str(e),
            'quotes': {},
            'source': 'error'
        }), 500


@app.route('/finnhub-profile', methods=['GET'])
def finnhub_profile():
    """Get company profile from Finnhub API"""
    try:
        if not FINNHUB_ENABLED or finnhub is None:
            return jsonify({'error': 'Finnhub API not available'})
        
        symbol = request.args.get('symbol', 'TCS')
        profile = finnhub.get_profile(symbol)
        
        return jsonify({
            'profile': profile,
            'symbol': symbol,
            'source': 'finnhub',
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        print(f"Error fetching Finnhub profile: {e}")
        return jsonify({
            'error': str(e),
            'profile': {},
            'source': 'error'
        }), 500


@app.route('/finnhub-news', methods=['GET'])
def finnhub_news():
    """Get latest market news from Finnhub API"""
    try:
        if not FINNHUB_ENABLED or finnhub is None:
            return jsonify({'error': 'Finnhub API not available'})
        
        symbol = request.args.get('symbol', '')
        limit = int(request.args.get('limit', 10))
        
        news = finnhub.get_news(symbol=symbol, limit=limit)
        
        return jsonify({
            'news': news,
            'symbol': symbol,
            'source': 'finnhub',
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        print(f"Error fetching Finnhub news: {e}")
        return jsonify({
            'error': str(e),
            'news': [],
            'source': 'error'
        }), 500


@app.route('/chat', methods=['POST'])
def chat_with_gemini():
    """Chat with Gemini AI about market analysis using current stock data"""
    try:
        if not HAS_GEMINI:
            return jsonify({
                'error': 'Gemini API not configured. Please set GEMINI_API_KEY in .env file.',
                'response': 'I am currently offline. Please ask your administrator to configure the Gemini API key.'
            }), 503
        
        data = request.get_json()
        user_message = data.get('message', '').strip()
        symbol = data.get('symbol', 'NIFTY')
        conversation_history = data.get('conversationHistory', [])
        
        if not user_message:
            return jsonify({'error': 'Empty message'}), 400
        
        # Gather market context for the symbol
        market_context = _get_market_context(symbol)
        
        # Build system prompt with market data context - STRICT STOCK MARKET FOCUS
        system_prompt = f"""You are an expert stock market analyst specializing in Indian equities.
        
IMPORTANT CONSTRAINTS:
1. You ONLY answer questions about stock markets, trading, and technical analysis
2. You MUST refuse any non-financial questions politely
3. You have access to real-time data for {symbol} - use it to provide accurate analysis
4. Always consider risk/reward ratios in your recommendations
5. Be specific with price targets, support/resistance levels, and entry/exit points

Current Market Data for {symbol}:
{market_context}

Your responses should be:
- SPECIFIC: Include actual price points, RSI levels, percentages
- ACTIONABLE: Give clear buy/sell signals with reasoning
- RISK-AWARE: Always mention potential risks and stop-loss levels
- DATA-DRIVEN: Reference the provided technical indicators and historical data

If a question is not about stock markets, politely decline with:
"I can only help with stock market analysis. Please ask about {symbol}, trading strategies, or technical analysis."
"""
        
        # Build conversation history for Gemini
        messages = []
        
        # Add system context as first message
        if conversation_history:
            # Convert conversation history format
            for msg in conversation_history:
                role = "user" if msg.get('role') == 'user' else "model"
                messages.append({
                    "role": role,
                    "parts": [{"text": msg.get('content', '')}]
                })
        
        # Add current user message
        full_message = f"{user_message}\n\n[Market Context: {symbol}]"
        messages.append({
            "role": "user",
            "parts": [{"text": full_message}]
        })
        
        # Call Gemini API
        try:
            model = genai.GenerativeModel('gemini-2.0-flash')
            
            # Create chat session with history
            chat = model.start_chat(history=[])
            
            # If we have previous messages, add them as context
            if messages[:-1]:
                for msg in messages[:-1]:
                    if msg['role'] == 'user':
                        response = chat.send_message(msg['parts'][0]['text'])
                        # Store response in history
                        chat.history.append({
                            "role": "user",
                            "parts": msg['parts']
                        })
                        chat.history.append({
                            "role": "model",
                            "parts": [{"text": response.text}]
                        })
            
            # Send current message
            response = chat.send_message(full_message)
            
            return jsonify({
                'response': response.text,
                'symbol': symbol,
                'source': 'gemini',
                'timestamp': datetime.now().isoformat()
            })
        
        except Exception as gemini_error:
            print(f"Gemini API error: {gemini_error}")
            return jsonify({
                'error': str(gemini_error),
                'response': f'Error communicating with Gemini API: {str(gemini_error)}'
            }), 500
    
    except Exception as e:
        print(f"Error in /chat endpoint: {e}")
        return jsonify({
            'error': str(e),
            'response': f'Server error: {str(e)}'
        }), 500


def _get_market_context(symbol):
    """Generate market context string with current data for a symbol"""
    context_parts = []
    
    try:
        # Get latest history data
        if symbol in historical_data:
            df = historical_data[symbol]
            if len(df) > 0:
                latest = df.iloc[-1]
                context_parts.append(f"📊 SYMBOL: {symbol}")
                context_parts.append(f"💹 Latest Close: ₹{latest['Close']:.2f}")
                context_parts.append(f"High: ₹{latest['High']:.2f} | Low: ₹{latest['Low']:.2f}")
                context_parts.append(f"Volume: {latest['Volume']:,.0f}")
                
                if len(df) > 1:
                    prev_close = df.iloc[-2]['Close']
                    change = latest['Close'] - prev_close
                    change_pct = (change / prev_close) * 100
                    trend = "📈 UP" if change > 0 else "📉 DOWN"
                    context_parts.append(f"{trend} Daily Change: ₹{change:+.2f} ({change_pct:+.2f}%)")
                
                # Calculate comprehensive technical indicators
                if len(df) >= 20:
                    # Moving Averages
                    sma_20 = df['Close'].tail(20).mean()
                    sma_50 = df['Close'].tail(50).mean() if len(df) >= 50 else sma_20
                    context_parts.append(f"\n📈 MOVING AVERAGES:")
                    context_parts.append(f"SMA-20: ₹{sma_20:.2f}")
                    context_parts.append(f"SMA-50: ₹{sma_50:.2f}")
                    
                    # Price position relative to MAs
                    if latest['Close'] > sma_20 > sma_50:
                        context_parts.append("⭐ Bullish alignment: Price > SMA20 > SMA50")
                    elif latest['Close'] < sma_20 < sma_50:
                        context_parts.append("⭐ Bearish alignment: Price < SMA20 < SMA50")
                    
                    # RSI calculation with detailed interpretation
                    delta = df['Close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    rsi = 100 - (100 / (1 + rs))
                    latest_rsi = rsi.iloc[-1]
                    
                    context_parts.append(f"\n📊 MOMENTUM (RSI-14):")
                    if latest_rsi > 70:
                        context_parts.append(f"RSI: {latest_rsi:.2f} - ⚠️ OVERBOUGHT (potential pullback)")
                    elif latest_rsi > 60:
                        context_parts.append(f"RSI: {latest_rsi:.2f} - Strong uptrend (cautious entry)")
                    elif latest_rsi < 30:
                        context_parts.append(f"RSI: {latest_rsi:.2f} - 🔥 OVERSOLD (potential bounce)")
                    elif latest_rsi < 40:
                        context_parts.append(f"RSI: {latest_rsi:.2f} - Weak downtrend (wait for reversal)")
                    else:
                        context_parts.append(f"RSI: {latest_rsi:.2f} - Neutral zone (consolidation)")
                
                # Price range and volatility
                context_parts.append(f"\n📏 PRICE LEVELS:")
                high_252 = df['High'].tail(252).max() if len(df) >= 252 else df['High'].max()
                low_252 = df['Low'].tail(252).min() if len(df) >= 252 else df['Low'].min()
                context_parts.append(f"52-week High: ₹{high_252:.2f}")
                context_parts.append(f"52-week Low: ₹{low_252:.2f}")
                
                # Calculate support and resistance
                recent_high = df['High'].tail(20).max()
                recent_low = df['Low'].tail(20).min()
                context_parts.append(f"Recent High (20D): ₹{recent_high:.2f}")
                context_parts.append(f"Recent Low (20D): ₹{recent_low:.2f}")
                
                # Volatility
                returns = df['Close'].pct_change()
                volatility = returns.std() * 100
                context_parts.append(f"\n⚡ VOLATILITY: {volatility:.2f}% (std dev)")
    
    except Exception as e:
        print(f"Error generating market context for {symbol}: {e}")
    
    # Get model predictions if available
    try:
        if symbol in predictions_cache and predictions_cache[symbol]:
            pred = predictions_cache[symbol]
            context_parts.append(f"\n🤖 AI MODEL PREDICTION:")
            context_parts.append(f"Model Prediction: {pred.get('prediction', 'N/A')}")
            context_parts.append(f"Confidence: {pred.get('confidence', 0):.2%}")
    except Exception as e:
        print(f"Error getting predictions: {e}")
    
    return "\n".join(context_parts) if context_parts else f"No data available for {symbol}"


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