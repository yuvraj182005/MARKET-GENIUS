// Finnhub API Integration Module
// Handles real-time market data from Finnhub API

const FINNHUB_CONFIG = {
    apiBaseUrl: 'http://127.0.0.1:5000',
    refreshInterval: 5000, // 5 seconds
    statusCheckInterval: 30000 // 30 seconds
};

let finnhubData = {
    status: 'disconnected',
    authenticated: false,
    profile: {},
    quotes: {},
    news: []
};

/**
 * Initialize Finnhub connection
 */
async function initFinnhub() {
    console.log('Initializing Finnhub API...');
    await checkFinnhubStatus();
    startFinnhubRefresh();
}

/**
 * Check Finnhub API connection status
 */
async function checkFinnhubStatus() {
    try {
        const response = await fetch(`${FINNHUB_CONFIG.apiBaseUrl}/finnhub-status`);
        const data = await response.json();
        
        finnhubData.status = data.status;
        finnhubData.authenticated = data.authenticated;
        
        console.log(`Finnhub Status: ${data.status} | Features: ${data.features?.join(', ')}`);
        updateFinnhubStatusUI();
        
        return data.authenticated;
    } catch (error) {
        console.error('Error checking Finnhub status:', error);
        finnhubData.status = 'error';
        updateFinnhubStatusUI();
        return false;
    }
}

/**
 * Fetch real-time quotes from Finnhub
 */
async function fetchFinnhubQuotes(symbols = 'NIFTY,SENSEX,TCS,RELIANCE,INFY') {
    try {
        const response = await fetch(
            `${FINNHUB_CONFIG.apiBaseUrl}/finnhub-quotes?symbols=${encodeURIComponent(symbols)}`
        );
        const data = await response.json();
        
        if (data.quotes) {
            finnhubData.quotes = data.quotes;
            updateFinnhubQuotesUI();
            return data.quotes;
        }
    } catch (error) {
        console.error('Error fetching Finnhub quotes:', error);
    }
}

/**
 * Fetch company profile information
 */
async function fetchFinnhubProfile(symbol = 'TCS') {
    try {
        const response = await fetch(`${FINNHUB_CONFIG.apiBaseUrl}/finnhub-profile?symbol=${symbol}`);
        const data = await response.json();
        
        if (data.profile) {
            finnhubData.profile = data.profile;
            updateFinnhubProfileUI();
            return data.profile;
        }
    } catch (error) {
        console.error('Error fetching Finnhub profile:', error);
    }
}

/**
 * Fetch latest market news
 */
async function fetchFinnhubNews(symbol = '', limit = 10) {
    try {
        const response = await fetch(
            `${FINNHUB_CONFIG.apiBaseUrl}/finnhub-news?symbol=${symbol}&limit=${limit}`
        );
        const data = await response.json();
        
        if (data.news) {
            finnhubData.news = data.news;
            updateFinnhubNewsUI();
            return data.news;
        }
    } catch (error) {
        console.error('Error fetching Finnhub news:', error);
    }
}

/**
 * Update Finnhub status in UI
 */
function updateFinnhubStatusUI() {
    const statusElement = document.getElementById('finnhubStatus') || document.getElementById('angelOneStatus');
    if (!statusElement) return;

    // Show three states:
    // - authenticated (live): green
    // - connected but unauthenticated (mock fallback): yellow and labeled "Connected (mock)"
    // - disconnected/error: red
    let statusClass = 'status-disconnected';
    let statusText = 'Disconnected';
    let statusIcon = '🔴';

    if (finnhubData.status === 'connected') {
        if (finnhubData.authenticated) {
            statusClass = 'status-connected';
            statusText = 'Connected';
            statusIcon = '🟢';
        } else {
            statusClass = 'status-mock';
            statusText = 'Connected (mock)';
            statusIcon = '🟡';
        }
    }

    statusElement.className = `finnhub-status ${statusClass}`;
    statusElement.innerHTML = `${statusIcon} Finnhub: ${statusText}`;
}

/**
 * Update quotes in UI
 */
function updateFinnhubQuotesUI() {
    const quotesContainer = document.getElementById('finnhubQuotes') || document.getElementById('angelOneQuotes');
    if (!quotesContainer) return;
    
    let html = '<div class="finnhub-quotes-grid">';
    
    Object.entries(finnhubData.quotes).forEach(([symbol, quote]) => {
        const change = ((quote.close - quote.open) / quote.open * 100).toFixed(2);
        const changeClass = change >= 0 ? 'up' : 'down';
        const changeSymbol = change >= 0 ? '▲' : '▼';
        
        html += `
            <div class="quote-card">
                <div class="quote-symbol">${symbol}</div>
                <div class="quote-price">₹${quote.ltp.toFixed(2)}</div>
                <div class="quote-change ${changeClass}">${changeSymbol} ${change}%</div>
                <div class="quote-range">
                    <small>H: ${quote.high.toFixed(2)} L: ${quote.low.toFixed(2)}</small>
                </div>
            </div>
        `;
    });
    
    html += '</div>';
    quotesContainer.innerHTML = html;
}

/**
 * Update profile in UI
 */
function updateFinnhubProfileUI() {
    const profileContainer = document.getElementById('finnhubProfile') || document.getElementById('angelOnePortfolio');
    if (!profileContainer) return;
    
    const profile = finnhubData.profile;
    const html = `
        <div class="profile-grid">
            <div class="profile-item">
                <span>Company:</span>
                <strong>${profile.name || 'N/A'}</strong>
            </div>
            <div class="profile-item">
                <span>Industry:</span>
                <strong>${profile.industry || 'N/A'}</strong>
            </div>
            <div class="profile-item">
                <span>Country:</span>
                <strong>${profile.country || 'N/A'}</strong>
            </div>
            <div class="profile-item">
                <span>Market Cap:</span>
                <strong>${profile.marketCap ? `$${(profile.marketCap / 1e9).toFixed(2)}B` : 'N/A'}</strong>
            </div>
        </div>
    `;
    
    profileContainer.innerHTML = html;
}

/**
 * Update news in UI
 */
function updateFinnhubNewsUI() {
    const newsContainer = document.getElementById('finnhubNews') || document.getElementById('angelOneOrderBook');
    if (!newsContainer) return;
    
    if (!finnhubData.news || finnhubData.news.length === 0) {
        newsContainer.innerHTML = '<p>No news available</p>';
        return;
    }
    
    let html = '<div class="news-list">';
    
    finnhubData.news.forEach(article => {
        html += `
            <div class="news-item">
                <div class="news-title">${article.headline || article.title || 'Untitled'}</div>
                <div class="news-source">${article.source || 'Finnhub'} • ${new Date(article.datetime * 1000).toLocaleDateString()}</div>
                ${article.summary ? `<div class="news-summary">${article.summary}</div>` : ''}
                ${article.url ? `<a href="${article.url}" target="_blank" class="news-link">Read More →</a>` : ''}
            </div>
        `;
    });
    
    html += '</div>';
    newsContainer.innerHTML = html;
}

/**
 * Start periodic refresh of Finnhub data
 */
function startFinnhubRefresh() {
    // Refresh quotes every 5 seconds (always refresh so mock data is kept current)
    setInterval(() => {
        fetchFinnhubQuotes();
    }, FINNHUB_CONFIG.refreshInterval);
    
    // Check status every 30 seconds
    setInterval(() => {
        checkFinnhubStatus();
    }, FINNHUB_CONFIG.statusCheckInterval);
    
    // Fetch initial data
    fetchFinnhubQuotes();
    fetchFinnhubProfile();
    fetchFinnhubNews();
}

/**
 * Display Finnhub trading panel
 */
function showFinnhubPanel() {
    const panel = document.getElementById('finnhubPanel') || document.getElementById('angelOnePanel');
    if (!panel) {
        console.error('Finnhub Panel element not found');
        return;
    }
    
    panel.style.display = 'block';
    checkFinnhubStatus();
    fetchFinnhubQuotes();
    fetchFinnhubProfile();
    fetchFinnhubNews();
}

/**
 * Hide Finnhub trading panel
 */
function hideFinnhubPanel() {
    const panel = document.getElementById('finnhubPanel') || document.getElementById('angelOnePanel');
    if (panel) {
        panel.style.display = 'none';
    }
}

/**
 * Get formatted quote data for display
 */
function getFinnhubQuoteFormatted(symbol) {
    const quote = finnhubData.quotes[symbol];
    if (!quote) return null;
    
    return {
        symbol,
        ltp: `₹${quote.ltp.toFixed(2)}`,
        open: `₹${quote.open.toFixed(2)}`,
        high: `₹${quote.high.toFixed(2)}`,
        low: `₹${quote.low.toFixed(2)}`,
        change: `${(((quote.close - quote.open) / quote.open) * 100).toFixed(2)}%`
    };
}

// Export functions
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        initFinnhub,
        checkFinnhubStatus,
        fetchFinnhubQuotes,
        fetchFinnhubProfile,
        fetchFinnhubNews,
        showFinnhubPanel,
        hideFinnhubPanel,
        getFinnhubQuoteFormatted,
        finnhubData
    };
}
