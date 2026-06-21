# MarketGenius - NewsAPI Live Data Setup

## ⚡ Quick Start (5 minutes)

### Step 1: Get Free NewsAPI Key
1. Visit: https://newsapi.org/
2. Click "Get API Key"
3. Sign up for free account
4. Copy your API key

### Step 2: Configure Backend

**Windows PowerShell:**
```powershell
cd d:\nifty-stock-predictor\backend

# Create .env file
$content = @"
NEWSAPI_KEY=your_api_key_here
NEWSAPI_CACHE_MINUTES=15
"@
$content | Out-File -Encoding UTF8 .env

# Edit the file
notepad .env
```

Replace `your_api_key_here` with your actual API key.

### Step 3: Install & Run

```powershell
# Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Start backend
python app.py
```

### Step 4: Open in Browser
- URL: **http://127.0.0.1:5000**
- Navigate to: **Market News tab**
- See: **Real live news from NewsAPI.org** 🎉

---

## What You Get

✅ **Real-time market news**
✅ **Automatic caching** (15 min default)
✅ **Fallback to mock data** if API fails
✅ **100 free requests/day**
✅ **No credit card required**

---

## File Structure

```
backend/
├── app.py                 # Updated with NewsAPI integration
├── requirements.txt       # Updated with requests, python-dotenv
├── .env                   # NEW: Your API key (create from .env.example)
├── .env.example          # NEW: Template for .env
└── NEWSAPI_SETUP.md      # NEW: Detailed setup guide
```

---

## How It Works

1. **First Request**: Fetches from NewsAPI.org ✓
2. **Cached (< 15 min)**: Returns cached data 💾
3. **Cache Expires**: Fetches fresh news again ✓
4. **API Fails**: Automatically uses mock data 🔄

---

## Verification

After starting, check Flask console:

✅ **Success:**
```
Fetched 5 articles for query: stock market india nifty sensex
Successfully fetched 15 unique articles from NewsAPI
```

❌ **Missing Key:**
```
Warning: NEWSAPI_KEY not configured. Using mock data.
```

---

## Features

- **Live Market News** with articles from various sources
- **Smart Caching** to preserve API quota
- **Fallback Support** - works even if API fails
- **Sentiment Analysis** with news articles
- **Trading Signals** based on market news
- **External Links** to full articles

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Using mock data" | Add NEWSAPI_KEY to .env and restart |
| "Invalid API key" | Get new key from newsapi.org |
| "Quota exceeded" | Upgrade plan or wait for reset |
| Still mock data | Check Flask console for errors |

---

## API Response Example

```json
{
  "news": [
    {
      "title": "Indian Markets Surge on Strong Economic Data",
      "description": "NIFTY 50 rises 2% amid positive GDP growth...",
      "source": "Reuters India",
      "url": "https://...",
      "published_date": "2025-12-05T10:30:00Z"
    }
  ],
  "source": "live",
  "message": "Live data from NewsAPI"
}
```

---

## Next Steps

1. ✅ Get API key from newsapi.org
2. ✅ Create .env file with API key
3. ✅ Run `pip install -r requirements.txt`
4. ✅ Start: `python app.py`
5. ✅ Open: http://127.0.0.1:5000
6. ✅ Click "Market News" tab
7. ✅ Enjoy real live news! 📰

---

## Support

- **NewsAPI Docs**: https://newsapi.org/docs
- **API Key**: https://newsapi.org/
- **Check logs**: Look at Flask console for error messages

**You're all set! Enjoy live market news integration! 🚀**
