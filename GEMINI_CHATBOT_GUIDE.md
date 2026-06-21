# 🤖 Gemini AI Chatbot - Stock Market Assistant Guide

## Overview

The Gemini AI Chatbot is now fully integrated into your Nifty Stock Predictor application. It provides **stock market-only** responses using real-time market data, technical indicators, and AI-powered analysis.

## ✅ Features Implemented

### 1. **Strict Stock Market Focus**
- ✅ Frontend validation: Questions are validated against stock market keywords before sending
- ✅ Backend system prompt: Gemini is instructed to refuse non-financial questions
- ✅ User-friendly feedback: Clear messages guide users to ask stock-related questions
- ✅ Keyword detection: 30+ stock market keywords for intelligent filtering

### 2. **Rich Market Context**
When you ask a question, Gemini receives:

**Price & Volume Data:**
- Latest close, high, low prices
- Daily price changes with percentage
- Trading volume
- Trend direction (📈 UP / 📉 DOWN)

**Technical Indicators:**
- 20-day and 50-day Simple Moving Averages (SMA)
- Bullish/Bearish alignment analysis
- RSI (14) with interpretation:
  - **Overbought** (>70): Potential pullback
  - **Strong uptrend** (60-70): Cautious entry
  - **Oversold** (<30): Potential bounce
  - **Weak downtrend** (30-40): Wait for reversal
  - **Neutral** (40-60): Consolidation

**Price Levels:**
- 52-week highs and lows
- Recent 20-day highs and lows
- Support & resistance levels

**Volatility:**
- Standard deviation of returns
- Risk assessment metrics

**Model Predictions:**
- AI ensemble model forecasts
- Confidence levels
- Predicted price movements

### 3. **Dynamic Symbol Context**
- When you change the selected stock, the chatbot updates its focus
- Input placeholder dynamically shows current symbol
- Welcome message confirms the stock you're analyzing
- All responses are tailored to the selected symbol

### 4. **Meaningful Interactions**

**What the bot CAN do:**
```
"Should I buy INFY?" → Analyzes price, RSI, trends, gives buy/sell signals
"Is TCS overbought?" → Checks RSI level, gives technical assessment
"What's the support level for RELIANCE?" → Provides support/resistance analysis
"Compare NIFTY trend over 20 days" → Analyzes SMA alignment and price position
"Give me a trading strategy for SENSEX" → Suggests entry/exit points, risk management
```

**What the bot WILL REJECT:**
```
"What's the weather?" → ❌ Not stock market related
"Tell me a joke" → ❌ Not stock market related
"How do I cook rice?" → ❌ Not stock market related
→ Guidance: "I can only help with stock market analysis. Please ask about..."
```

## 🚀 How to Use

### Step 1: Open the AI Assistant Tab
1. Navigate to the "AI Assistant" tab in the navigation menu
2. You'll see the chatbot interface with welcome information

### Step 2: Select a Stock Symbol
1. Go to any other tab (Dashboard, Analysis, etc.)
2. Select a stock symbol from the dropdown (e.g., NIFTY, TCS, INFY, RELIANCE)
3. Return to the "AI Assistant" tab
4. Chatbot updates with focus on the selected symbol

### Step 3: Ask Stock Market Questions
**Example questions:**
- "What's the current trend for [SYMBOL]?"
- "Is [SYMBOL] a good buy at current levels?"
- "Give me entry and exit points for [SYMBOL]"
- "What are the support and resistance levels?"
- "Should I go long or short on [SYMBOL]?"
- "Analyze the technical setup for [SYMBOL]"
- "What's the RSI telling us about [SYMBOL]?"
- "Compare [SYMBOL] to its 50-day average"

### Step 4: Get Intelligent Responses
Gemini analyzes:
- Real-time price data from CSV files
- Technical indicators (SMA, RSI, etc.)
- Price levels and volatility
- AI model predictions
- Provides specific trading advice with entry/exit points

## 🔧 Configuration

### Requirements Met ✓
1. **Gemini API Key**: ✓ Added to `.env` file
2. **google-generativeai package**: ✓ Installed
3. **Backend /chat endpoint**: ✓ Created with full market context
4. **Frontend validation**: ✓ Client-side keyword filtering
5. **Rich UI**: ✓ Beautiful chat interface with styling

### Environment Variables
```
FINNHUB_API_KEY=your_finnhub_key
GEMINI_API_KEY=AIzaSyDQxy01Kwr1deO_zHbEoMc82Dwie5G7lYg
```

## 📊 Technical Implementation

### Frontend (`frontend/gemini.js`)
```javascript
// Stock market keyword validation
function isStockMarketRelated(message)
  - Checks 30+ keywords
  - Case-insensitive matching
  - Instant feedback to user

// Symbol context updates
function setGeminiSymbol(symbol)
  - Updates placeholder
  - Shows welcome message
  - Focuses on selected stock

// Message sending
function sendGeminiMessage()
  - Validates question relevance
  - Sends to backend with context
  - Displays response with formatting
```

### Backend (`backend/app.py`)
```python
@app.route('/chat', methods=['POST'])
def chat_with_gemini()
  - Validates Gemini API availability
  - Gathers market context
  - Sends system prompt enforcing stock-market-only
  - Returns AI response

def _get_market_context(symbol)
  - Fetches current prices
  - Calculates technical indicators
  - Prepares rich context string
  - Includes model predictions
```

### System Prompt
The backend sends Gemini a strict system prompt:
- **ONLY stock market questions allowed**
- **Must refuse non-financial questions**
- **Real-time data provided for context**
- **SPECIFIC, ACTIONABLE, RISK-AWARE responses required**

## 💡 Example Interactions

### Example 1: Price Analysis
```
User: "What's the NIFTY trend?"

AI Response:
The NIFTY index shows a bullish setup:
- Current Close: ₹21,450.50 (↑2.15%)
- SMA-20: ₹21,100 (price above MA - positive)
- SMA-50: ₹20,800 (bullish alignment)
- RSI: 62.5 - Strong uptrend, cautious entry on dips
- Next Resistance: ₹21,600
- Support: ₹21,200

Recommendation: Wait for pullback to SMA-20 for better entry. Target: ₹22,000
Risk Management: Stop-loss at ₹20,900
```

### Example 2: Technical Analysis
```
User: "Is TCS overbought?"

AI Response:
TCS Technical Status:
- Price: ₹4,325.50
- RSI(14): 72.8 - OVERBOUGHT (>70)
- Warning: Potential pullback coming
- Observation: Price is above SMA-20 and SMA-50

Action: 
- SELL signal from overbought levels
- Expect resistance around ₹4,350
- Support at ₹4,200 (SMA-20)
- Consider taking profits at current levels
```

## 🔐 Security Features

1. **Client-side validation**: Questions filtered before API call
2. **Server-side enforcement**: Backend system prompt enforces stock-market-only
3. **Graceful rejection**: Non-relevant questions get helpful guidance
4. **No data leakage**: Only market data used, no personal information

## 📈 Performance

- **Response Time**: ~2-3 seconds (depends on Gemini API)
- **Context Size**: Optimized to include max relevant data without token bloat
- **Error Handling**: Graceful degradation if API unavailable
- **Rate Limiting**: Respects Gemini free tier limits

## 🐛 Troubleshooting

### Issue: "Gemini API not configured"
**Solution**: Ensure `GEMINI_API_KEY` is in `.env` file
```
GEMINI_API_KEY=your_actual_key
```

### Issue: Bot responding to non-stock questions
**Solution**: 
- Check browser console for validation logs
- Ensure gemini.js is loaded (check Network tab)
- Verify keyword list includes question terms

### Issue: Slow responses
**Solution**:
- Gemini API may be rate-limited
- Check your API usage at https://aistudio.google.com/
- Try simpler questions

### Issue: Technical indicators not showing
**Solution**:
- Data might not be available for the selected symbol
- Try switching to major indices (NIFTY, SENSEX) which have full data
- Check browser console for errors

## 🎯 Best Practices

1. **Be Specific**: "What should I do for TCS?" → Better: "Should I buy TCS at 4300?"
2. **Use Technical Terms**: RSI, SMA, support, resistance for better responses
3. **Ask Trading Questions**: Entry points, exit signals, risk management
4. **Change Symbols**: Test different stocks to see contextual analysis
5. **Check Indicators**: Reference the provided data in AI responses

## 📝 Notes

- ✅ Stock market questions only (strictly enforced)
- ✅ Real-time data from your CSV files
- ✅ Technical analysis with 3 indicators (SMA, RSI, price levels)
- ✅ AI model predictions included
- ✅ Beautiful, responsive chat UI
- ✅ Conversation history maintained
- ✅ Symbol-specific context updates

## 🚀 Future Enhancements

Possible additions:
- MACD indicator in market context
- Bollinger Bands analysis
- Volume analysis
- Sector comparison
- News sentiment analysis
- Portfolio recommendations
- Risk scoring

---

**Version**: 1.0  
**Last Updated**: January 4, 2026  
**Status**: ✅ Fully Operational
