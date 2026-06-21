"""Angel One SmartAPI Integration Module

Handles real-time market data from Angel One (Official Indian Trading API)
"""

import os
import json
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

class AngelOneAPI:
    """Angel One SmartAPI wrapper for market data"""
    
    def __init__(self):
        self.client_id = os.getenv('ANGEL_ONE_CLIENT_ID', '')
        self.api_key = os.getenv('ANGEL_ONE_API_KEY', '')
        self.password = os.getenv('ANGEL_ONE_PASSWORD', '')
        self.totp = os.getenv('ANGEL_ONE_TOTP', '000000')
        self.token = None
        self.authenticated = False
        
        # Try to import SmartAPI
        try:
            # Support both package name variants: 'smartapi' and 'SmartApi'
            try:
                from smartapi import SmartConnect
            except Exception:
                from SmartApi.smartConnect import SmartConnect
            import pyotp
            self.SmartConnect = SmartConnect
            self.pyotp = pyotp
            self.has_sdk = True
        except ImportError:
            self.has_sdk = False
            print("Warning: SmartAPI SDK not installed. Using mock data.")
            print("Install with: pip install smartapi-python pyotp")
    
    def authenticate(self):
        """Authenticate with Angel One API"""
        if not self.has_sdk:
            print("SmartAPI SDK not available")
            return False
        
        if not self.client_id or self.client_id == '':
            print("Angel One credentials not configured in .env")
            return False
        
        try:
            # Initialize connection
            obj = self.SmartConnect(api_key=self.api_key)
            
            # Generate TOTP if available
            totp = None
            if self.totp != '000000':
                totp_obj = self.pyotp.TOTP(self.totp)
                totp = totp_obj.now()
            else:
                totp = '000000'
            
            # Authenticate
            data = obj.generateSession(
                clientID=self.client_id,
                password=self.password,
                TOTP=totp
            )
            
            if data['status']:
                self.token = data['data']['jwtToken']
                self.obj = obj
                self.authenticated = True
                print("✓ Connected to Angel One SmartAPI")
                return True
            else:
                print(f"✗ Authentication failed: {data.get('message', 'Unknown error')}")
                return False
        
        except Exception as e:
            print(f"✗ Angel One connection error: {e}")
            return False
    
    def get_quote(self, symbol, exchange="NSE"):
        """Get real-time quote for a symbol"""
        if not self.authenticated:
            return self._get_mock_quote(symbol)
        
        try:
            # Format mode for quote
            mode = 'FULL'  # Full mode includes OHLC
            
            params = {
                'mode': mode,
                'exchangeTokens': {
                    exchange: ['token_value']  # Token mapping needed
                }
            }
            
            response = self.obj.getQuote(
                exchangeTokens=[exchange],
                tokens=['token_value'],
                mode=mode
            )
            
            if response['status']:
                return response['data']['fetched'][0]
            else:
                return self._get_mock_quote(symbol)
        
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            return self._get_mock_quote(symbol)
    
    def get_portfolio(self):
        """Get portfolio information"""
        if not self.authenticated:
            return self._get_mock_portfolio()
        
        try:
            response = self.obj.getProfile()
            if response['status']:
                return response['data']
            else:
                return self._get_mock_portfolio()
        
        except Exception as e:
            print(f"Error fetching portfolio: {e}")
            return self._get_mock_portfolio()
    
    def get_orderbook(self):
        """Get order history"""
        if not self.authenticated:
            return self._get_mock_orderbook()
        
        try:
            response = self.obj.orderBook()
            if response['status']:
                return response['data']
            else:
                return self._get_mock_orderbook()
        
        except Exception as e:
            print(f"Error fetching orders: {e}")
            return self._get_mock_orderbook()
    
    @staticmethod
    def _get_mock_quote(symbol):
        """Mock quote data for development"""
        mock_quotes = {
            'TCS': {'ltp': 3456.75, 'open': 3450, 'high': 3470, 'low': 3440, 'close': 3456},
            'RELIANCE': {'ltp': 2850.50, 'open': 2845, 'high': 2870, 'low': 2835, 'close': 2850},
            'INFY': {'ltp': 1895.25, 'open': 1890, 'high': 1910, 'low': 1880, 'close': 1895},
            'NIFTY': {'ltp': 23456.75, 'open': 23450, 'high': 23500, 'low': 23400, 'close': 23456},
            'SENSEX': {'ltp': 78950.25, 'open': 78900, 'high': 79050, 'low': 78850, 'close': 78950},
        }
        return mock_quotes.get(symbol, {'ltp': 1000, 'open': 1000, 'high': 1050, 'low': 950, 'close': 1000})
    
    @staticmethod
    def _get_mock_portfolio():
        """Mock portfolio data"""
        return {
            'name': 'Demo User',
            'email': 'user@example.com',
            'phone': '9999999999',
            'balance': 500000,
            'used_margin': 100000,
            'available_margin': 400000,
        }
    
    @staticmethod
    def _get_mock_orderbook():
        """Mock orderbook data"""
        return [
            {'order_id': '1', 'symbol': 'TCS', 'quantity': 10, 'price': 3450, 'status': 'Complete'},
            {'order_id': '2', 'symbol': 'RELIANCE', 'quantity': 5, 'price': 2840, 'status': 'Complete'},
        ]


# Global instance
angel_api = None

def init_angel_one():
    """Initialize Angel One API"""
    global angel_api
    angel_api = AngelOneAPI()
    if angel_api.has_sdk:
        angel_api.authenticate()
    return angel_api

def get_angel_api():
    """Get Angel One API instance"""
    global angel_api
    if angel_api is None:
        angel_api = init_angel_one()
    return angel_api
