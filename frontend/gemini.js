// Gemini AI Chatbot Integration Module
// Handles communication with Gemini API via backend /chat endpoint

const GEMINI_CONFIG = {
    apiBaseUrl: 'http://127.0.0.1:5000',
    maxMessages: 50
};

let geminiState = {
    messages: [],
    currentSymbol: 'NIFTY',
    isLoading: false
};

/**
 * Initialize Gemini chat UI event listeners
 */
function initGeminiChat() {
    console.log('Initializing Gemini Chat...');
    
    const inputEl = document.getElementById('geminiInput');
    const sendBtn = document.getElementById('geminiSendBtn');
    
    if (!inputEl || !sendBtn) {
        console.warn('Gemini chat input elements not found');
        return;
    }
    
    // Send message on button click
    sendBtn.addEventListener('click', () => sendGeminiMessage());
    
    // Send message on Enter key
    inputEl.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !geminiState.isLoading) {
            sendGeminiMessage();
        }
    });
    
    // Add placeholder hint
    inputEl.placeholder = `Ask about ${geminiState.currentSymbol} trends, risks, predictions...`;
    
    console.log('Gemini Chat initialized');
}

/**
 * Check if message is stock market related
 */
function isStockMarketRelated(message) {
    const stockKeywords = [
        'stock', 'market', 'price', 'buy', 'sell', 'invest', 'trade',
        'trend', 'analysis', 'technical', 'rsi', 'sma', 'macd', 'forecast',
        'prediction', 'bullish', 'bearish', 'support', 'resistance', 'volume',
        'dividend', 'earnings', 'pe ratio', 'profit', 'loss', 'portfolio',
        'risk', 'return', 'performance', 'chart', 'candle', 'indicator',
        'volatility', 'momentum', 'uptrend', 'downtrend', 'profit target',
        'nifty', 'sensex', 'infy', 'tcs', 'reliance', 'bank', 'sector',
        'value', 'growth', 'dividend yield', 'trading', 'swing', 'intraday'
    ];
    const lowerMessage = message.toLowerCase();
    return stockKeywords.some(keyword => lowerMessage.includes(keyword));
}

/**
 * Send user message to Gemini via backend /chat endpoint
 */
async function sendGeminiMessage() {
    const inputEl = document.getElementById('geminiInput');
    const sendBtn = document.getElementById('geminiSendBtn');
    
    if (!inputEl || geminiState.isLoading) return;
    
    const userMessage = inputEl.value.trim();
    if (!userMessage) return;
    
    // Validate that message is stock market related
    if (!isStockMarketRelated(userMessage)) {
        addGeminiMessageToUI('assistant', 
            '📊 <strong>Stock Market Questions Only</strong>\n\n' +
            'I can only help with stock market-related questions. Please ask about:\n' +
            '• Stock prices, trends & historical performance\n' +
            '• Technical analysis (RSI, SMA, MACD, support/resistance)\n' +
            '• Buy/sell signals and entry/exit points\n' +
            '• Market predictions and forecasts\n' +
            '• Portfolio strategies for ' + geminiState.currentSymbol + '\n' +
            '• Risk assessment and volatility analysis\n' +
            '• Sector analysis and comparisons'
        );
        inputEl.value = '';
        return;
    }
    
    // Add user message to UI
    addGeminiMessageToUI('user', userMessage);
    inputEl.value = '';
    inputEl.focus();
    
    // Add to state
    geminiState.messages.push({ role: 'user', content: userMessage });
    
    // Disable input while waiting
    geminiState.isLoading = true;
    if (sendBtn) sendBtn.disabled = true;
    if (inputEl) inputEl.disabled = true;
    
    try {
        // Call backend /chat endpoint
        const response = await fetch(`${GEMINI_CONFIG.apiBaseUrl}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message: userMessage,
                symbol: geminiState.currentSymbol,
                conversationHistory: geminiState.messages.slice(0, -1) // Exclude current user msg to avoid duplication
            })
        });
        
        if (!response.ok) {
            throw new Error(`Backend error: ${response.status}`);
        }
        
        const data = await response.json();
        
        if (data.error) {
            addGeminiMessageToUI('assistant', `Error: ${data.error}`);
        } else {
            const assistantMessage = data.response || 'No response received';
            addGeminiMessageToUI('assistant', assistantMessage);
            geminiState.messages.push({ role: 'assistant', content: assistantMessage });
        }
    } catch (error) {
        console.error('Error sending Gemini message:', error);
        addGeminiMessageToUI('assistant', `Connection error: ${error.message}`);
    } finally {
        geminiState.isLoading = false;
        if (sendBtn) sendBtn.disabled = false;
        if (inputEl) inputEl.disabled = false;
        if (inputEl) inputEl.focus();
    }
}

/**
 * Add message to chat UI
 */
function addGeminiMessageToUI(role, text) {
    const messagesContainer = document.querySelector('.gemini-messages');
    if (!messagesContainer) return;
    
    const messageEl = document.createElement('div');
    messageEl.className = `gemini-message ${role}`;
    messageEl.innerHTML = `<p>${escapeHtml(text)}</p>`;
    
    messagesContainer.appendChild(messageEl);
    
    // Auto-scroll to bottom
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

/**
 * Escape HTML to prevent injection
 */
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

/**
 * Update current symbol context for Gemini
 */
function setGeminiSymbol(symbol) {
    geminiState.currentSymbol = symbol;
    const inputEl = document.getElementById('geminiInput');
    if (inputEl) {
        inputEl.placeholder = `Ask about ${symbol}: trends, risks, buy signals...`;
    }
    console.log(`Gemini context symbol: ${symbol}`);
    // Show notification that symbol context updated
    addGeminiMessageToUI('assistant', 
        `📈 <strong>Switched to ${symbol}</strong>\n\n` +
        `I'm now analyzing ${symbol} with real-time prices, technical indicators, and predictions. ` +
        `Ask me:\n` +
        `• Should I buy/sell ${symbol}?\n` +
        `• What's the ${symbol} trend?\n` +
        `• Technical analysis for ${symbol}\n` +
        `• Risk factors for ${symbol}`
    );
}

/**
 * Clear chat history
 */
function clearGeminiChat() {
    geminiState.messages = [];
    const messagesContainer = document.querySelector('.gemini-messages');
    if (messagesContainer) {
        messagesContainer.innerHTML = `
            <div class="gemini-message assistant">
                <p>📊 <strong>Stock Market AI Assistant</strong><br><br>
                I analyze Indian stock market data to help you make informed trading decisions.<br><br>
                <strong>What I can help with:</strong><br>
                • Real-time price analysis for any stock<br>
                • Technical analysis (RSI, SMA, MACD patterns)<br>
                • Buy/sell signals with entry/exit points<br>
                • Risk assessment & volatility analysis<br>
                • Market trend predictions<br>
                • Portfolio strategy recommendations<br><br>
                Select a stock above and ask away! 🚀</p>
            </div>
        `;
    }
}

// Initialize on DOM load
document.addEventListener('DOMContentLoaded', initGeminiChat);

// Export functions
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        initGeminiChat,
        sendGeminiMessage,
        setGeminiSymbol,
        clearGeminiChat,
        geminiState
    };
}
