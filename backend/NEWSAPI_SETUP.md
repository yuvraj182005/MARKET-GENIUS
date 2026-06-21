# NewsAPI.org Integration Guide

## Quick Setup (2 minutes)

### Step 1: Get Your Free API Key

1. Go to **https://newsapi.org/**
2. Click **"Get API Key"** button
3. Sign up for a free account (free tier includes 100 requests/day)
4. Copy your **API Key**

### Step 2: Configure Your Backend

**Option A: Using .env file (Recommended)**

1. Navigate to backend folder:
```bash
cd d:\nifty-stock-predictor\backend
```

2. Create `.env` file from template:
```bash
copy .env.example .env
```

3. Edit `.env` file with your editor:
```
NEWSAPI_KEY=your_api_key_here
NEWSAPI_CACHE_MINUTES=15
```

4. Replace `your_api_key_here` with your actual API key

**Option B: Direct Environment Variable (Windows)**

Open PowerShell and run:
```powershell
$env:NEWSAPI_KEY="your_api_key_here"
```

### Step 3: Install Dependencies

```bash
cd d:\nifty-stock-predictor\backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Start the Backend

```bash
python app.py
```

You should see:
```
Starting Flask server...
```

### Step 5: Test Live News

1. Open browser: **http://127.0.0.1:5000**
2. Navigate to **"Market News"** tab
3. You should see **real live news articles** from NewsAPI.org

## How It Works

- **First request**: Fetches live news from NewsAPI.org
- **Cached requests** (within 15 min): Returns cached data to save API quota
- **Cache reset**: Every 15 minutes, fresh news is fetched
- **Fallback**: If API fails or key missing, uses mock data

## Free Tier Limits

| Plan | Requests/Day | Rate Limit |
|------|------------|-----------|
| **Free** | 100 | 1 req/sec |
| Paid | 10,000+ | Higher |

## Response Format

```json
{
  "news": [
    {
      "title": "Article Title",
      "description": "Brief description",
      "source": "News Source Name",
      "url": "https://full-article-link.com",
      "image": "https://image-url.jpg",
      "published_date": "2025-12-05T10:30:00Z"
    }
  ],
  "source": "live",
  "timestamp": "2025-12-05T10:35:00Z",
  "message": "Live data from NewsAPI"
}
```

## Troubleshooting

### Issue: "Using mock data (configure NEWSAPI_KEY)"
**Solution**: Your API key is not configured. Follow Setup Step 2 above.

### Issue: "Invalid NewsAPI key"
**Solution**: Your API key is wrong or expired. Get a new one from newsapi.org.

### Issue: "API quota exceeded"
**Solution**: Free tier = 100 requests/day. Upgrade or wait until next day.

### Issue: ".env file not working"
**Solution**: 
1. Make sure `python-dotenv` is installed: `pip install python-dotenv`
2. Restart Flask after creating .env file
3. Check .env is in `backend/` folder, not root

### Issue: Still getting mock data
**Solution**:
1. Check Flask console for error messages
2. Verify .env file has correct NEWSAPI_KEY
3. Test API key at: https://newsapi.org/account
4. Restart Flask: `python app.py`

## Real-Time News Topics Covered

The integration fetches news for:
- Indian stock market
- NIFTY and SENSEX indices
- NSE/BSE trading
- Indian economy and markets
- Corporate earnings
- Market trends and analysis

## Monitoring & Logs

Watch the Flask console for:

```
Fetched 5 articles for query: stock market india nifty sensex
Successfully fetched 15 unique articles from NewsAPI
Returning cached news (age: 5.2 minutes)
```

## Advanced Configuration

Edit `.env` to customize:

```env
# API Key from newsapi.org
NEWSAPI_KEY=your_key_here

# Cache duration in minutes (default: 15)
# Lower = more API calls, Higher = stale data
NEWSAPI_CACHE_MINUTES=15
```

## Migration from Mock Data

Your app automatically:
1. Tries to fetch from NewsAPI.org (with your key)
2. Caches results for 15 minutes
3. Falls back to mock data if API fails
4. Shows message indicating data source in response

## FAQ

**Q: Do I need to pay for NewsAPI?**
A: No, free tier is available (100 requests/day). Paid plans available for higher usage.

**Q: Can I use other news APIs?**
A: Yes! Modify `fetch_newsapi_data()` function in `backend/app.py` to use alternative APIs.

**Q: How often do I need to renew the API key?**
A: API keys don't expire. You can use one key indefinitely.

**Q: Can I get more than 100 requests/day?**
A: Yes, upgrade to a paid plan on newsapi.org.

## Support

- NewsAPI Documentation: https://newsapi.org/docs
- Get API Key: https://newsapi.org/
- Issues? Check Flask console logs for error messages

---

**You're all set! Enjoy live market news! 📰**
