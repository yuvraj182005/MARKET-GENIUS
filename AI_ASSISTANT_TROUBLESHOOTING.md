# 🔧 AI Assistant Troubleshooting Guide

## Issue: AI Assistant Not Responding

### Root Cause: Gemini API Quota Exceeded

**Error**: `429 You exceeded your current quota, please check your plan and billing details`

The Gemini API free tier has **very limited quotas**:
- **Daily limit**: 10 requests per day per model
- **Minute limit**: 2 requests per minute per model  
- **Input tokens**: 1 million tokens per day (free tier)

Once these limits are hit, all requests are rejected with a 429 error.

---

## Problem Details

### What Happened?
1. ✅ Backend is working correctly
2. ✅ API endpoint (`/chat`) is correctly configured
3. ✅ Gemini model updated from deprecated `gemini-pro` to `gemini-2.0-flash`
4. ❌ **Gemini API free tier quota exhausted**

### Error Messages
```
429 You exceeded your current quota, please check your plan and billing details.
Limit: 0 requests remaining per day
```

---

## Solutions

### ✅ Solution 1: Wait for Quota Reset (Recommended for Free Tier)

**Timeline**: 
- Daily quota resets at midnight UTC
- Per-minute quota resets after 60 seconds

**Steps**:
1. Wait until tomorrow (quota resets at 12:00 AM UTC)
2. Try the AI Assistant again
3. **Tip**: Wait a minute between multiple consecutive requests

---

### ✅ Solution 2: Upgrade to Paid Plan

**Best for**: Regular/production use

**Steps**:
1. Go to [Google AI Studio](https://aistudio.google.com/app/billing/overview)
2. Click "Upgrade" button
3. Add a valid payment method
4. Enable billing for your project
5. New quotas will apply immediately:
   - **Paid tier**: 1,500 requests per minute (much higher!)
   - **Cost**: ~$0.075 per 1 million input tokens for gemini-2.0-flash
   - **Estimate**: For stock analysis, likely $1-5/month depending on usage

**Pricing** (as of Jan 2026):
- `gemini-2.0-flash`: $0.075/M input tokens, $0.30/M output tokens
- `gemini-2.5-flash`: $0.075/M input tokens, $0.30/M output tokens (similar cost)

---

### ✅ Solution 3: Use Alternative Model (Less Capable)

**Alternative**: Use `gemini-flash-latest` or `gemini-pro-latest` if they have separate quotas

**Steps**:
1. Edit `backend/app.py` line ~1637
2. Change: `model = genai.GenerativeModel('gemini-2.0-flash')`
3. To: `model = genai.GenerativeModel('gemini-flash-latest')`
4. Save and restart backend

**Note**: Each model may have separate quotas on free tier

---

### ✅ Solution 4: Implement Caching/Rate Limiting

**For Development**: Cache responses to reduce API calls

**Implementation**:
```python
# Pseudo-code to add to backend
conversation_cache = {}

@app.route('/chat', methods=['POST'])
def chat_with_gemini():
    # ... existing code ...
    
    # Create cache key from symbol + message
    cache_key = f"{symbol}:{user_message}"
    
    # Check cache first
    if cache_key in conversation_cache:
        return conversation_cache[cache_key]
    
    # ... call Gemini API ...
    
    # Store in cache
    conversation_cache[cache_key] = response
    return response
```

---

## Status Check

### ✅ What's Working
- [x] Flask backend server running on port 5000
- [x] `/chat` endpoint configured and active
- [x] Market data loading (CSV files)
- [x] Stock validation keywords (30+)
- [x] System prompt enforcing stock-market-only responses
- [x] Frontend UI and input validation
- [x] Gemini model updated to current API (`gemini-2.0-flash`)
- [x] API key configured in `.env`

### ❌ What's Not Working
- [ ] Gemini API calls (quota exceeded)
  - Reason: Free tier limit reached (0 requests remaining)
  - Status: Waiting for quota reset or upgrade to paid

---

## Current Configuration

**Gemini Settings**:
- API Key: ✅ Configured in `.env`
- Model: `gemini-2.0-flash` (latest stable model)
- Endpoint: `/chat` (POST)
- Stock validation: 30+ keywords
- System prompt: Stock-market-only restriction

**Free Tier Quotas** (as of Jan 2026):
```
Per Day:        10 requests per day per model
Per Minute:     2 requests per minute per model
Input Tokens:   1 million per day total
```

---

## Testing After Quota Reset

### Test 1: Backend Health
```bash
curl http://127.0.0.1:5000/available_symbols
# Should return list of available stocks
```

### Test 2: Chat Endpoint
```bash
curl -X POST http://127.0.0.1:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"What is the price of NIFTY?","symbol":"NIFTY"}'
# Should return Gemini response
```

### Test 3: UI Test
1. Open dashboard: http://127.0.0.1:5000
2. Go to "AI Assistant" tab
3. Select a stock (e.g., TCS)
4. Type: "Should I buy this stock?"
5. Expected: AI response with analysis

---

## Prevention Tips

### For Future Use:
1. **Monitor Quota**: Check [AI Studio Billing](https://aistudio.google.com/app/billing)
2. **Implement Caching**: Cache similar questions
3. **Add Rate Limiting**: Limit requests per user per minute
4. **Upgrade Early**: Switch to paid plan for consistent access
5. **Use Efficient Prompts**: Shorter prompts = fewer tokens = lower cost

---

## Resources

- [Gemini API Quotas](https://ai.google.dev/gemini-api/docs/rate-limits)
- [Google AI Studio Billing](https://aistudio.google.com/app/billing/overview)
- [Gemini API Pricing](https://ai.google.dev/pricing)
- [Rate Limiting Guide](https://ai.google.dev/gemini-api/docs/rate-limits)

---

## Next Steps

**Recommended Action**:
1. ⏰ **Now**: Wait for quota reset (midnight UTC) OR
2. 💳 **Upgrade**: Enable paid plan on Google Cloud for unlimited access
3. 🧪 **Test**: Verify AI Assistant works after quota reset
4. 💾 **Optimize**: Implement response caching to reduce API calls

---

## Summary

| Aspect | Status | Details |
|--------|--------|---------|
| Backend Server | ✅ Running | Port 5000, all endpoints active |
| Chat Endpoint | ✅ Working | `/chat` endpoint configured |
| Stock Validation | ✅ Active | 30+ keywords, dual-layer validation |
| Gemini Integration | ✅ Configured | API key set, model updated to `gemini-2.0-flash` |
| API Quota | ❌ Exceeded | Free tier limit reached (0/10 requests today) |
| **Overall Status** | 🟡 Waiting | **Ready to use after quota reset or upgrade** |

---

**Last Updated**: January 4, 2026  
**Issue Type**: API Quota Exhaustion  
**Severity**: Medium (temporary, resolves after quota reset)
