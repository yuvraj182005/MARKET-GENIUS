const API_URL = "http://localhost:5000";
let mainChart = null;
let volumeChart = null;
let sentimentGauge = null;

// Technical indicators configuration
const indicators = {
  sma: { periods: [20, 50, 200], enabled: false },
  ema: { periods: [9, 21], enabled: false },
  rsi: { period: 14, enabled: false },
  macd: { enabled: false }
};

// Initialize page
async function initializePage() {
  await loadSymbols();
  initializeCharts();
  setupEventListeners();
  updateMarketStats();
  updateWatchlist();
  loadMarketSentiment();
}

async function loadSymbols() {
  const res = await fetch(`${API_URL}/available_symbols`);
  const data = await res.json();
  
  const select = document.getElementById('symbolSelect');
  const indicesGroup = select.querySelector('optgroup[label="Market Indices"]');
  const stocksGroup = select.querySelector('optgroup[label="Popular Stocks"]');
  
  data.indices.forEach(symbol => {
    const option = document.createElement('option');
    option.value = symbol;
    option.text = symbol;
    indicesGroup.appendChild(option);
  });
  
  data.stocks.forEach(symbol => {
    const option = document.createElement('option');
    option.value = symbol;
    option.text = symbol;
    stocksGroup.appendChild(option);
  });
  
  await loadHistory(data.indices[0]);
}

function setupEventListeners() {
  // Symbol change
  document.getElementById('symbolSelect').addEventListener('change', () => {
    loadHistory();
    document.getElementById('prediction').innerHTML = '';
  });

  // Time frame buttons
  document.querySelectorAll('.time-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
      document.querySelectorAll('.time-btn').forEach(b => b.classList.remove('active'));
      e.target.classList.add('active');
      loadHistory();
    });
  });

  // Technical indicators
  document.querySelectorAll('.indicator-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
      const indicator = e.target.dataset.indicator;
      indicators[indicator].enabled = !indicators[indicator].enabled;
      e.target.classList.toggle('active');
      updateChart();
    });
  });

  // Chart type
  document.querySelectorAll('.chart-type-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
      document.querySelectorAll('.chart-type-btn').forEach(b => b.classList.remove('active'));
      e.target.classList.add('active');
      updateChartType(e.target.dataset.type);
    });
  });
}

async function predictStock() {
  const date = document.getElementById('dateInput').value;
  const symbol = document.getElementById('symbolSelect').value;
  
  if (!date) return alert("Please select a date");

  const res = await fetch(`${API_URL}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ date, symbol })
  });

  const data = await res.json();
  if (data.predicted_price) {
    const predictedPrice = parseFloat(data.predicted_price);
    const currentPrice = mainChart.data.datasets[0].data.slice(-1)[0];
    const changePercent = ((predictedPrice - currentPrice) / currentPrice * 100).toFixed(2);
    const trend = changePercent >= 0 ? 'up' : 'down';
    
    document.getElementById('prediction').innerHTML = `
      <div class="prediction-card ${trend}">
        <div class="prediction-header">
          <i class="fas fa-chart-line"></i>
          <h4>${symbol} Prediction</h4>
        </div>
        <div class="prediction-body">
          <div class="price">₹${data.predicted_price}</div>
          <div class="date">${date}</div>
          <div class="change ${trend}">
            <i class="fas fa-caret-${trend}"></i>
            ${Math.abs(changePercent)}%
          </div>
        </div>
      </div>
    `;
  } else {
    alert("Error: " + data.error);
  }
}

async function loadHistory(symbol = null) {
  if (!symbol) {
    symbol = document.getElementById('symbolSelect').value;
  }
  
  const res = await fetch(`${API_URL}/history?symbol=${symbol}`);
  const data = await res.json();
  
  if (data.error) {
    alert("Error: " + data.error);
    return;
  }

  updateCharts(data, symbol);
  updateTechnicalIndicators(data);
}

function initializeCharts() {
  const mainCtx = document.getElementById('mainChart').getContext('2d');
  const volumeCtx = document.getElementById('volumeChart').getContext('2d');
  const sentimentCtx = document.getElementById('sentimentGauge').getContext('2d');

  // Initialize main price chart
  mainChart = new Chart(mainCtx, {
    type: 'line',
    data: { datasets: [] },
    options: {
      responsive: true,
      interaction: {
        intersect: false,
        mode: 'index'
      },
      plugins: {
        legend: {
          position: 'top',
          labels: {
            color: '#94a3b8'
          }
        },
        tooltip: {
          mode: 'index',
          intersect: false
        }
      },
      scales: {
        x: {
          grid: {
            color: 'rgba(148, 163, 184, 0.1)'
          },
          ticks: {
            color: '#94a3b8'
          }
        },
        y: {
          grid: {
            color: 'rgba(148, 163, 184, 0.1)'
          },
          ticks: {
            color: '#94a3b8'
          }
        }
      }
    }
  });

  // Initialize volume chart
  volumeChart = new Chart(volumeCtx, {
    type: 'bar',
    data: { datasets: [] },
    options: {
      responsive: true,
      plugins: {
        legend: {
          display: false
        }
      },
      scales: {
        x: {
          display: false
        },
        y: {
          grid: {
            color: 'rgba(148, 163, 184, 0.1)'
          },
          ticks: {
            color: '#94a3b8'
          }
        }
      }
    }
  });

  // Initialize sentiment gauge
  sentimentGauge = new Chart(sentimentCtx, {
    type: 'doughnut',
    data: {
      labels: ['Bullish', 'Neutral', 'Bearish'],
      datasets: [{
        data: [60, 25, 15],
        backgroundColor: ['#22c55e', '#3b82f6', '#ef4444']
      }]
    },
    options: {
      responsive: true,
      plugins: {
        legend: {
          display: false
        }
      },
      cutout: '70%'
    }
  });
}

function updateCharts(data, symbol) {
  // Update main chart
  mainChart.data = {
    labels: data.dates,
    datasets: [{
      label: `${symbol} Price`,
      data: data.prices,
      borderColor: '#3b82f6',
      backgroundColor: 'rgba(59, 130, 246, 0.1)',
      fill: true,
      tension: 0.4
    }]
  };
  mainChart.update();

  // Update volume chart with mock data (replace with real data when available)
  const volumes = data.prices.map(price => price * (0.8 + Math.random() * 0.4));
  volumeChart.data = {
    labels: data.dates,
    datasets: [{
      label: 'Volume',
      data: volumes,
      backgroundColor: 'rgba(59, 130, 246, 0.3)'
    }]
  };
  volumeChart.update();
}

function updateTechnicalIndicators(data) {
  const prices = data.prices;
  const indicatorValues = document.getElementById('indicator-values');
  indicatorValues.innerHTML = '';

  // Calculate and display RSI
  const rsi = calculateRSI(prices, 14);
  const lastRSI = rsi[rsi.length - 1].toFixed(2);
  
  // Calculate and display SMA
  const sma20 = calculateSMA(prices, 20);
  const lastSMA20 = sma20[sma20.length - 1].toFixed(2);

  // Calculate and display EMA
  const ema9 = calculateEMA(prices, 9);
  const lastEMA9 = ema9[ema9.length - 1].toFixed(2);

  const indicators = [
    { name: 'RSI (14)', value: lastRSI, type: getRSISignal(lastRSI) },
    { name: 'SMA (20)', value: lastSMA20, type: getPriceSignal(prices[prices.length - 1], lastSMA20) },
    { name: 'EMA (9)', value: lastEMA9, type: getPriceSignal(prices[prices.length - 1], lastEMA9) }
  ];

  indicators.forEach(ind => {
    indicatorValues.innerHTML += `
      <div class="indicator-card ${ind.type}">
        <span class="indicator-name">${ind.name}</span>
        <span class="indicator-value">${ind.value}</span>
        <span class="indicator-signal">${ind.type}</span>
      </div>
    `;
  });
}

// Helper functions for technical analysis
function calculateSMA(prices, period) {
  const sma = [];
  for (let i = period - 1; i < prices.length; i++) {
    const sum = prices.slice(i - period + 1, i + 1).reduce((a, b) => a + b, 0);
    sma.push(sum / period);
  }
  return sma;
}

function calculateEMA(prices, period) {
  const multiplier = 2 / (period + 1);
  const ema = [prices[0]];
  for (let i = 1; i < prices.length; i++) {
    ema.push((prices[i] - ema[i - 1]) * multiplier + ema[i - 1]);
  }
  return ema;
}

function calculateRSI(prices, period) {
  const changes = [];
  for (let i = 1; i < prices.length; i++) {
    changes.push(prices[i] - prices[i - 1]);
  }

  const rsi = [];
  let avgGain = 0;
  let avgLoss = 0;

  // Calculate first averages
  for (let i = 0; i < period; i++) {
    if (changes[i] > 0) avgGain += changes[i];
    if (changes[i] < 0) avgLoss += Math.abs(changes[i]);
  }
  avgGain /= period;
  avgLoss /= period;

  // Calculate RSI
  for (let i = period; i < changes.length; i++) {
    const rs = avgGain / avgLoss;
    rsi.push(100 - (100 / (1 + rs)));

    // Update averages
    avgGain = ((avgGain * (period - 1)) + (changes[i] > 0 ? changes[i] : 0)) / period;
    avgLoss = ((avgLoss * (period - 1)) + (changes[i] < 0 ? Math.abs(changes[i]) : 0)) / period;
  }

  return rsi;
}

function getRSISignal(rsi) {
  if (rsi > 70) return 'overbought';
  if (rsi < 30) return 'oversold';
  return 'neutral';
}

function getPriceSignal(price, indicator) {
  if (price > indicator) return 'bullish';
  if (price < indicator) return 'bearish';
  return 'neutral';
}

// Market stats and watchlist updates
function updateMarketStats() {
  const marketStats = document.getElementById('market-stats');
  marketStats.innerHTML = `
    <div class="market-stat">
      <span>Market Status</span>
      <span class="active">Open</span>
    </div>
    <div class="market-stat">
      <span>Market Trend</span>
      <span class="up">Bullish</span>
    </div>
    <div class="market-stat">
      <span>Volatility Index</span>
      <span>16.24</span>
    </div>
  `;
}

function updateWatchlist() {
  const watchlist = document.getElementById('watchlist-items');
  const mockData = [
    { symbol: 'NIFTY', price: '19,674.25', change: '+1.2%' },
    { symbol: 'SENSEX', price: '65,970.10', change: '+0.9%' },
    { symbol: 'TCS', price: '3,456.75', change: '-0.5%' }
  ];

  watchlist.innerHTML = mockData.map(item => `
    <div class="watchlist-item">
      <span class="symbol">${item.symbol}</span>
      <span class="price">₹${item.price}</span>
      <span class="change ${item.change.startsWith('+') ? 'up' : 'down'}">${item.change}</span>
    </div>
  `).join('');
}

function loadMarketSentiment() {
  // Mock sentiment data - replace with real data when available
  const tradingSignals = document.querySelector('.trading-signals');
  const signals = [
    { name: 'Moving Averages', signal: 'Buy', strength: 'Strong' },
    { name: 'Technical Indicators', signal: 'Buy', strength: 'Moderate' },
    { name: 'Pivot Points', signal: 'Neutral', strength: 'Weak' }
  ];

  tradingSignals.innerHTML = signals.map(signal => `
    <div class="signal-card">
      <div class="signal-name">${signal.name}</div>
      <div class="signal-value ${signal.signal.toLowerCase()}">${signal.signal}</div>
      <div class="signal-strength">${signal.strength}</div>
    </div>
  `).join('');
}

// Initialize the page
initializePage();