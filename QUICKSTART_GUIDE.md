# 🚀 Quick Start - Gemini Chatbot

## What's New?

✅ **Stock Market Only**: Chatbot rejects non-financial questions  
✅ **Rich Context**: 15+ market data points (prices, RSI, SMA, levels)  
✅ **Smart Validation**: 30+ keyword detection on client & server  
✅ **Symbol Focus**: Dynamic context when you change stocks  

---

## How to Use (3 Steps)

### 1️⃣ Select Stock
Go to any tab, pick a symbol (e.g., NIFTY, TCS, INFY)

### 2️⃣ Open AI Assistant Tab
Click "AI Assistant" in navigation menu

### 3️⃣ Ask Stock Market Question
Example:
```
"Should I buy INFY?"
"What's the RSI for TCS?"
"Is NIFTY overbought?"
"Support level for RELIANCE?"
```

---

## ✅ What Works

- "What's the trend for NIFTY?" → ✓ Detailed analysis
- "Is TCS a good buy?" → ✓ Buy/sell signal
- "Support/resistance for SENSEX?" → ✓ Price levels
- "Technical setup for INFY?" → ✓ RSI, SMA, levels
- "Should I go long?" → ✓ Trading advice

---

## ❌ What Doesn't Work

- "What's the weather?" → ✗ Rejected
- "Tell me a joke" → ✗ Rejected
- "How to cook rice?" → ✗ Rejected
- Any non-stock question → ✗ Helpful guidance shown

---

## Market Data Included

When you ask, Gemini gets:

📊 **Prices**
- Current close, high, low
- Daily change with %

📈 **Moving Averages**
- SMA-20, SMA-50
- Bullish/Bearish alignment

📊 **RSI Indicator**
- Overbought (>70)
- Oversold (<30)
- Interpretation & signals

📏 **Price Levels**
- 52-week highs/lows
- Recent support/resistance

⚡ **Volatility**
- Risk metrics

🤖 **AI Predictions**
- Model forecasts
- Confidence scores

---

## Example Questions

**Best Questions to Ask:**
```
1. "Is [STOCK] oversold at current levels?"
2. "What's the 20-day trend for [STOCK]?"
3. "Give me entry and exit points for [STOCK]"
4. "Technical analysis for [STOCK]"
5. "Should I buy or sell [STOCK]?"
6. "What are support and resistance levels?"
7. "Is the RSI signaling a reversal?"
8. "Compare [STOCK] to its 50-day average"
```

---

## 📱 Chat Tips

1. **Switch Symbols**: Chatbot updates focus automatically
2. **Ask Clearly**: "Buy signal for NIFTY?" beats "NIFTY?"
3. **Use Tech Terms**: RSI, SMA, support, resistance
4. **Be Specific**: Price targets, timeframes
5. **Multi-turn**: Follow-up questions work too

---

## ⚙️ Configuration

✅ **GEMINI_API_KEY** is set in `.env`  
✅ **Backend running** on http://127.0.0.1:5000  
✅ **All modules loaded** (gemini.js active)  

---

## 🔗 Documentation

- **Full Guide**: `GEMINI_CHATBOT_GUIDE.md`
- **Enhancement Details**: `ENHANCEMENT_SUMMARY.md`
- **Technical Changes**: `TECHNICAL_CHANGES.md`

---

## 🆘 Quick Troubleshoot

**Q: Bot not responding?**  
A: Ensure Gemini API key is in `.env`, restart backend

**Q: Rejecting valid stock questions?**  
A: Try using technical terms (RSI, trend, price)

**Q: Slow responses?**  
A: Normal for Gemini (2-3 sec), check API usage

**Q: Symbol not updating?**  
A: Go to another tab, select symbol, return to AI tab

---

## 🎯 Perfect For

✅ Quick price analysis  
✅ Technical indicator interpretation  
✅ Buy/sell decision making  
✅ Risk assessment  
✅ Trading strategy planning  
✅ Market trend analysis  

---

## 🚫 NOT For

❌ Non-financial questions  
❌ General chatbot conversations  
❌ News or weather  
❌ Jokes or entertainment  

---

## 📊 Example Session

```
User: Select "TCS" from dropdown, go to AI Assistant tab
Bot: 📈 Switched to TCS. Analyzing with real-time data...

User: "Is TCS overbought?"
Bot: TCS Analysis:
     - Price: ₹4,325
     - RSI: 72.8 - OVERBOUGHT
     - Action: SELL signal
     - Target: ₹4,200
     - Risk: ₹4,350

User: "When should I enter?"
Bot: Wait for pullback to SMA-20 (₹4,200)
     Use stop-loss at ₹4,100
     Target: ₹4,500

User: "What about INFY?"
[Select INFY, symbol context updates]
Bot: 📈 Switched to INFY...
```

---

## ✨ Key Features

| Feature | Status |
|---------|--------|
| Stock market questions only | ✅ Active |
| Real-time market data | ✅ Active |
| Technical indicators | ✅ Active |
| Price targets | ✅ Active |
| Entry/exit signals | ✅ Active |
| Risk analysis | ✅ Active |
| Symbol context switching | ✅ Active |
| Conversation history | ✅ Active |

---

**Ready to trade with AI insights!** 🚀

---

Last Updated: January 4, 2026  
Version: 1.0 - Stable
