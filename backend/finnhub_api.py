"""Finnhub API Integration Module

Provides real-time and historical market data from Finnhub
Uses free tier API (no key required for basic features, optional for rate limits)
"""

import requests
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

class FinnhubAPI:
    """Finnhub API wrapper for market data"""
    
    def __init__(self):
        self.api_key = os.getenv('FINNHUB_API_KEY', '')
        self.base_url = 'https://finnhub.io/api/v1'
        self.has_sdk = True  # Finnhub is always available via free tier
        # Consider authenticated only when an API key is provided. Without
        # a key some endpoints may return 401 and we'll fall back to mocks.
        self.authenticated = bool(self.api_key)
        # One-time flag to avoid repeating the same warning
        self._unauth_warned = False
        # Indian stock symbols mapping to Finnhub format (using NSE prefix)
        self.indian_symbols = {
            'TCS': 'TCS.NS',
            'RELIANCE': 'RELIANCE.NS',
            'INFY': 'INFY.NS',
            'NIFTY': 'NIFTY50.IN',
            'SENSEX': 'SENSEX.IN',
            'HDFCBANK': 'HDFCBANK.NS',
            'ICICIBANK': 'ICICIBANK.NS',
            'SBIN': 'SBIN.NS',
            'WIPRO': 'WIPRO.NS',
            'BAJAJFINSV': 'BAJAJFINSV.NS',
            'HDFC': 'HDFC.NS',
            'KOTAKBANK': 'KOTAKBANK.NS',
            'MARUTI': 'MARUTI.NS',
            'SUNPHARMA': 'SUNPHARMA.NS',
            'ASIANPAINT': 'ASIANPAINT.NS',
            'TITAN': 'TITAN.NS',
            'BRITANNIA': 'BRITANNIA.NS',
        }
        
        if self.api_key:
            print("Finnhub API initialized (API key detected)")
        else:
            print("Finnhub API initialized (no API key; using mock fallback)")
    
    def get_quote(self, symbol):
        """Get real-time quote for a symbol"""
        try:
            # Convert symbol to Finnhub format if in our mapping
            finnhub_symbol = self.indian_symbols.get(symbol, symbol)
            
            params = {'symbol': finnhub_symbol}
            if self.api_key:
                params['token'] = self.api_key
            
            response = requests.get(
                f'{self.base_url}/quote',
                params=params,
                timeout=5
            )
            response.raise_for_status()
            data = response.json()
            
            if data and data.get('c'):  # 'c' is current price
                return {
                    'symbol': symbol,
                    'ltp': data.get('c', 0),
                    'open': data.get('o', 0),
                    'high': data.get('h', 0),
                    'low': data.get('l', 0),
                    'close': data.get('pc', 0),
                    'volume': data.get('v', 0),
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return self._get_mock_quote(symbol)
        except Exception as e:
            msg = str(e)
            if (not getattr(self, '_unauth_warned', False)) and ('401' in msg or 'Unauthorized' in msg):
                print("Finnhub API unauthorized: set FINNHUB_API_KEY in .env to enable live data. Falling back to mock data.")
                self._unauth_warned = True
            return self._get_mock_quote(symbol)
    
    def get_profile(self, symbol):
        """Get company profile"""
        try:
            finnhub_symbol = self.indian_symbols.get(symbol, symbol)
            
            params = {'symbol': finnhub_symbol}
            if self.api_key:
                params['token'] = self.api_key
            
            response = requests.get(
                f'{self.base_url}/company/profile2',
                params=params,
                timeout=5
            )
            response.raise_for_status()
            data = response.json()
            
            return {
                'name': data.get('name', symbol),
                'country': data.get('country', 'IN'),
                'currency': data.get('currency', 'INR'),
                'exchange': data.get('exchange', 'NSE'),
                'ipo': data.get('ipo', ''),
                'marketCapitalization': data.get('marketCapitalization', 0),
                'logo': data.get('logo', ''),
                'phone': data.get('phone', ''),
                'weburl': data.get('weburl', ''),
            }
        except Exception as e:
            msg = str(e)
            if (not getattr(self, '_unauth_warned', False)) and ('401' in msg or 'Unauthorized' in msg):
                print("Finnhub API unauthorized: set FINNHUB_API_KEY in .env to enable live data. Falling back to mock profile.")
                self._unauth_warned = True
            # Return a sensible mock profile when Finnhub is not reachable or requires API key
            return {
                'name': symbol,
                'industry': '',
                'country': 'IN',
                'currency': 'INR',
                'exchange': 'NSE',
                'ipo': '',
                'marketCap': 0,
                'logo': '',
                'phone': '',
                'weburl': ''
            }
    
    def get_news(self, symbol='', limit=10):
        """Get latest news"""
        try:
            finnhub_symbol = self.indian_symbols.get(symbol, symbol) if symbol else ''
            
            params = {'minId': 0}
            if finnhub_symbol:
                params['symbol'] = finnhub_symbol
            if self.api_key:
                params['token'] = self.api_key
            
            # Use general news endpoint
            response = requests.get(
                f'{self.base_url}/news',
                params=params,
                timeout=5
            )
            response.raise_for_status()
            data = response.json()
            
            if isinstance(data, list):
                return data[:limit]
            return []
        except Exception as e:
            msg = str(e)
            if (not getattr(self, '_unauth_warned', False)) and ('401' in msg or 'Unauthorized' in msg):
                print("Finnhub API unauthorized: set FINNHUB_API_KEY in .env to enable live data. Falling back to mock news.")
                self._unauth_warned = True
            return self._get_mock_news()
    
    def get_recommendations(self, symbol):
        """Get analyst recommendations"""
        try:
            finnhub_symbol = self.indian_symbols.get(symbol, symbol)
            
            params = {'symbol': finnhub_symbol}
            if self.api_key:
                params['token'] = self.api_key
            
            response = requests.get(
                f'{self.base_url}/stock/recommendation',
                params=params,
                timeout=5
            )
            response.raise_for_status()
            data = response.json()
            
            if isinstance(data, list) and len(data) > 0:
                latest = data[0]
                return {
                    'symbol': symbol,
                    'buy': latest.get('buy', 0),
                    'hold': latest.get('hold', 0),
                    'sell': latest.get('sell', 0),
                    'strongBuy': latest.get('strongBuy', 0),
                    'strongSell': latest.get('strongSell', 0),
                    'date': latest.get('period', '')
                }
            return {}
        except Exception as e:
            print(f"Error fetching recommendations for {symbol}: {e}")
            return {}
    
    @staticmethod
    def _get_mock_quote(symbol):
        """Mock quote data for development/fallback"""
        mock_quotes = {
            'TCS': {'ltp': 3456.75, 'open': 3450, 'high': 3470, 'low': 3440, 'close': 3456, 'volume': 1500000},
            'RELIANCE': {'ltp': 2850.50, 'open': 2845, 'high': 2870, 'low': 2835, 'close': 2850, 'volume': 3000000},
            'INFY': {'ltp': 1895.25, 'open': 1890, 'high': 1910, 'low': 1880, 'close': 1895, 'volume': 2500000},
            'NIFTY': {'ltp': 23456.75, 'open': 23450, 'high': 23500, 'low': 23400, 'close': 23456, 'volume': 0},
            'SENSEX': {'ltp': 78950.25, 'open': 78900, 'high': 79050, 'low': 78850, 'close': 78950, 'volume': 0},
        }
        data = mock_quotes.get(symbol, {'ltp': 1000, 'open': 1000, 'high': 1050, 'low': 950, 'close': 1000, 'volume': 100000})
        data['symbol'] = symbol
        data['timestamp'] = datetime.now().isoformat()
        return data
    
    @staticmethod
    def _get_mock_news():
        """Mock news data"""
        return [
            {
                'id': 1,
                'category': 'company news',
                'datetime': int((datetime.now() - timedelta(hours=1)).timestamp()),
                'headline': 'Indian Markets Show Strong Recovery',
                'image': '',
                'related': 'NIFTY',
                'source': 'Economic Times',
                'summary': 'Stock indices bounce back with strong gains.',
                'url': '#'
            },
            {
                'id': 2,
                'category': 'general news',
                'datetime': int((datetime.now() - timedelta(hours=2)).timestamp()),
                'headline': 'Tech Stocks Lead Market Rally',
                'image': '',
                'related': 'INFY,TCS,WIPRO',
                'source': 'Business Today',
                'summary': 'IT companies announce strong quarterly results.',
                'url': '#'
            },
        ]


# Initialize global instance
finnhub = FinnhubAPI()
