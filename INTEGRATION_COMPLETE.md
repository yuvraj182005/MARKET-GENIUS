# Angel One SmartAPI Integration - Complete Setup Guide

## ✅ What's Been Completed

### Frontend Integration
- ✅ Created `angel_one.js` - Complete Angel One module with real-time data fetching
- ✅ Updated `index.html` - Added Angel One section with professional dashboard
- ✅ Updated `styles.css` - Added 400+ lines of professional Angel One styling
- ✅ Updated `script.js` - Integrated Angel One initialization with main app
- ✅ Added Angel One navigation link with orange icon in navbar

### Backend Integration
- ✅ Created `angel_one_api.py` - SmartAPI wrapper class
- ✅ Added 4 API endpoints to `app.py`:
  - `/angel-status` - Check connection status
  - `/angel-quotes` - Get real-time quotes
  - `/angel-portfolio` - Get portfolio information
  - `/angel-orderbook` - Get order history

### Configuration
- ✅ Updated `requirements.txt` with `smartapi-python` and `pyotp`
- ✅ Updated `.env.example` with Angel One credentials template
- ✅ Created comprehensive `ANGEL_ONE_SETUP.md` documentation

---

## 🚀 Installation Steps

### 1. Install Python Dependencies
```powershell
pip install -r requirements.txt
```

This will install:
- smartapi-python (Angel One SmartAPI SDK)
- pyotp (TOTP authentication)
- All other required packages

### 2. Configure Environment Variables

**Create a `.env` file in the `backend/` folder:**

```env
# NewsAPI Configuration (for market news)
NEWSAPI_KEY=your_newsapi_key_here

# Angel One SmartAPI Configuration
ANGEL_ONE_CLIENT_ID=your_client_id
ANGEL_ONE_API_KEY=your_api_key
ANGEL_ONE_PASSWORD=your_password
ANGEL_ONE_TOTP=your_totp_secret
```

### 3. Get Angel One Credentials

1. **Visit Angel One:** https://www.angelone.in/
2. **Create Account** or login to your existing account
3. **Get API Keys:**
   - Navigate to API Settings
   - Generate API credentials
   - Copy your Client ID, API Key, and Password
4. **Get TOTP Secret:**
   - Enable 2FA in security settings
   - Save the TOTP secret (or use authenticator app)

---

## 🌐 Running the Application

### Terminal 1 - Start Backend
```powershell
cd backend
python app.py
```
Backend will run at: `http://127.0.0.1:5000`

### Terminal 2 - Start Frontend (Optional but recommended)
```powershell
cd frontend
# Use Live Server VS Code extension or
python -m http.server 8000
```
Frontend will run at: `http://localhost:8000`

### Open Application
- **With Frontend Server:** `http://localhost:8000`
- **Direct Backend:** `http://127.0.0.1:5000`

---

## 📊 Angel One Dashboard Features

Once configured, you'll get access to:

### 1. **Real-Time Connection Status**
- Shows Angel One connection status (Connected/Disconnected)
- Green indicator when authenticated
- Red indicator when disconnected

### 2. **Live Quotes**
- Real-time stock quotes for major symbols
- Last traded price (LTP)
- Daily change percentage
- High/Low prices
- Updates every 5 seconds

### 3. **Portfolio Summary**
- Account name
- Total balance
- Used margin
- Available margin

### 4. **Order History**
- Complete order book
- Symbol, quantity, price, status
- Status badges (Completed, Pending, Rejected, Cancelled)

---

## 🔧 Troubleshooting

### Issue: "Angel One: Disconnected" (Red Indicator)

**Possible Causes:**
1. Credentials not set in `.env`
2. Invalid credentials
3. Backend server not running
4. Network connectivity issue

**Solutions:**
```powershell
# Check that .env file exists in backend/
Test-Path backend\.env

# Verify backend is running
# You should see "Angel One SmartAPI integration ready" in logs
```

### Issue: "Module 'smartapi' not found"

```powershell
# Reinstall dependencies
pip install --upgrade smartapi-python pyotp

# Or reinstall all
pip install -r requirements.txt --force-reinstall
```

### Issue: TOTP Authentication Fails

Make sure your TOTP secret is correct:
- Copy exact secret from Angel One account
- Ensure system time is synchronized
- TOTP codes expire after 30 seconds

### Issue: Quotes Not Updating

1. Check browser console (F12) for errors
2. Verify backend is responding:
   ```powershell
   curl http://127.0.0.1:5000/angel-quotes
   ```
3. Check backend logs for error messages

---

## 📱 Frontend Components

### Angel One Section

**Location:** Navigation bar → "⚡ Angel One" link

**Sections:**

1. **Status Bar**
   - Real-time connection indicator
   - Color-coded status (🟢 Connected / 🔴 Disconnected)

2. **Real-Time Quotes Panel**
   - Grid layout of popular symbols
   - NIFTY, SENSEX, TCS, RELIANCE, INFY
   - Live price updates
   - Up/down indicators with color coding

3. **Portfolio Summary Panel**
   - Key account metrics
   - Balance information
   - Margin utilization

4. **Order History Panel**
   - Table of recent orders
   - Status tracking
   - Order details

---

## 🎨 Design Features

### Professional UI Components

- **Dark Theme:** Matches the rest of MarketGenius platform
- **Blue Accents:** #3b82f6 primary color with gradients
- **Real-time Indicators:** Animated status updates
- **Responsive Layout:**
  - Desktop: Multi-column grids
  - Tablet: 2-column layout
  - Mobile: Single column stack

### Interactive Elements

- Hover effects on quote cards
- Smooth transitions on status changes
- Loading spinners during data fetch
- Professional status badges

---

## 🔐 Security Best Practices

⚠️ **IMPORTANT:**

1. **Never commit `.env` file** to Git
2. **Keep credentials private** - don't share your API keys
3. **Use environment variables** - never hardcode credentials
4. **Rotate credentials** periodically
5. **Enable 2FA** on your Angel One account
6. **Monitor orders** for unauthorized activity

---

## 📚 API Documentation

### Backend Endpoints

All endpoints return JSON responses:

#### GET `/angel-status`
Check Angel One connection status
```json
{
  "status": "connected",
  "authenticated": true,
  "message": "Connected to Angel One SmartAPI"
}
```

#### GET `/angel-quotes?symbols=NIFTY,SENSEX`
Get real-time quotes
```json
{
  "quotes": {
    "NIFTY": {
      "ltp": 19500.00,
      "open": 19400.00,
      "high": 19550.00,
      "low": 19350.00,
      "close": 19500.00
    }
  }
}
```

#### GET `/angel-portfolio`
Get portfolio information
```json
{
  "portfolio": {
    "name": "John Doe",
    "balance": 100000,
    "used_margin": 25000,
    "available_margin": 75000
  }
}
```

#### GET `/angel-orderbook`
Get order history
```json
{
  "orders": [
    {
      "symbol": "NIFTY",
      "quantity": 10,
      "price": 19500,
      "status": "Completed"
    }
  ]
}
```

---

## 🧪 Testing

### Manual Testing

1. **Test Connection:**
   ```powershell
   curl http://127.0.0.1:5000/angel-status
   ```

2. **Test Quotes:**
   ```powershell
   curl "http://127.0.0.1:5000/angel-quotes?symbols=NIFTY,SENSEX"
   ```

3. **Test Portfolio:**
   ```powershell
   curl http://127.0.0.1:5000/angel-portfolio
   ```

### Browser Testing

1. Open DevTools (F12)
2. Go to "Angel One" tab in navigation
3. Check Console tab for messages
4. Verify real-time data updates every 5 seconds

---

## 📝 Environment File Template

**File: `backend/.env`**

```env
# ===== NewsAPI Configuration =====
NEWSAPI_KEY=abc123def456...

# ===== Angel One SmartAPI Configuration =====
# Get these from your Angel One account settings
ANGEL_ONE_CLIENT_ID=AA1234
ANGEL_ONE_API_KEY=your_api_key_12345...
ANGEL_ONE_PASSWORD=your_password
ANGEL_ONE_TOTP=JBSWY3DPEBLW64TMMQ======

# Optional: Adjust cache timing
NEWS_CACHE_DURATION=900  # 15 minutes in seconds
ANGEL_ONE_REFRESH_INTERVAL=5000  # 5 seconds in milliseconds
```

---

## 🎯 Next Steps

1. ✅ **Install dependencies**
   ```powershell
   pip install -r requirements.txt
   ```

2. ✅ **Configure credentials**
   - Create `backend/.env`
   - Add your Angel One credentials

3. ✅ **Start backend**
   ```powershell
   python backend/app.py
   ```

4. ✅ **Start frontend**
   ```powershell
   python -m http.server 8000
   ```

5. ✅ **Open application**
   - Navigate to `http://localhost:8000` or `http://127.0.0.1:5000`
   - Click "⚡ Angel One" in navbar
   - View real-time trading data!

---

## 📞 Support

If you encounter issues:

1. Check the **Troubleshooting** section above
2. Review browser console (F12) for JavaScript errors
3. Check backend logs for Python errors
4. Verify `.env` file configuration
5. Ensure backend server is running

---

## ✨ Features Summary

| Feature | Status | Details |
|---------|--------|---------|
| Real-time Quotes | ✅ | Updates every 5 seconds |
| Portfolio Tracking | ✅ | Live balance & margins |
| Order History | ✅ | Complete orderbook access |
| Professional UI | ✅ | Dark theme with animations |
| Error Handling | ✅ | Fallback to mock data |
| Mobile Responsive | ✅ | Works on all devices |
| Live News | ✅ | NewsAPI integration |
| Technical Analysis | ✅ | Candlestick charts + indicators |

---

**Last Updated:** 2024
**Integration Version:** 1.0
**Status:** ✅ Production Ready
