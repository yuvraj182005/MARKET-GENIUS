// Use the same origin the page was served from to avoid CORS/host issues
// But if the frontend is served by a dev server (e.g. Live Server) on another port
// we fall back to the backend running at DEFAULT_API.
const DEFAULT_API = 'http://127.0.0.1:5000';
let API_URL = window.location.origin;
// ===== CANDLESTICK CHART GLOBALS =====
let candleChart = null;
let candleSeries = null;

// Fetch helpers that try the current origin first then fall back to the backend
async function fetchJSONWithFallback(path) {
  // Try the backend API first (DEFAULT_API). If the page is served from a
  // different origin (e.g. Live Server at :5501), use that as a fallback.
  const bases = [DEFAULT_API];
  if (API_URL && API_URL !== DEFAULT_API) {
    bases.push(API_URL);
  }
  
  for (const base of bases) {
    try {
      const url = `${base}${path}`;
      console.log(`Trying fetch: ${url}`);
      const res = await fetch(url, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json'
        }
      });
      
      const text = await res.text();
      
      console.log(`Response from ${base}: status=${res.status}, content-length=${text.length}, starts with: ${text.substring(0, 50)}`);
      
      // If server returned HTML (e.g. index.html) or a 404 page, skip
      if (!res.ok) {
        console.log(`Skipping ${base}: response not ok (${res.status})`);
        continue;
      }
      if (text && text.trim().startsWith('<')) {
        console.log(`Skipping ${base}: response is HTML`);
        continue;
      }
      
      const parsed = JSON.parse(text);
      console.log(`Successfully parsed JSON from ${base}:`, parsed);
      return parsed;
    } catch (e) {
      console.log(`Error with ${base}:`, e.message);
      // try next base
      continue;
    }
  }
  throw new Error(`Failed to fetch JSON for ${path}`);
}

function initCandleChart() {
  const container = document.getElementById('candleChart');
  if (!container || !window.LightweightCharts) return;

  if (candleChart) {
    // already created
    return;
  }

  candleChart = LightweightCharts.createChart(container, {
    width: container.clientWidth,
    height: 400,
    layout: {
      background: { color: '#0f172a' },
      textColor: '#f8fafc'
    },
    rightPriceScale: { 
      borderVisible: false,
      textColor: '#f8fafc'
    },
    timeScale: {
      borderVisible: false,
      timeVisible: true,
      secondsVisible: false,
      textColor: '#f8fafc'
    },
    grid: {
      vertLines: { visible: true, color: 'rgba(148, 163, 184, 0.1)' },
      horzLines: { visible: true, color: 'rgba(148, 163, 184, 0.1)' }
    }
  });

  candleSeries = candleChart.addCandlestickSeries({
    upColor: '#22c55e',
    downColor: '#ef4444',
    wickUpColor: '#22c55e',
    wickDownColor: '#ef4444',
    borderUpColor: '#22c55e',
    borderDownColor: '#ef4444'
  });
  
  window.addEventListener('resize', () => {
    const rect = container.getBoundingClientRect();
    candleChart.applyOptions({ width: rect.width });
  });
  
  candleChart.timeScale().fitContent();
  console.log('Candlestick chart initialized');
  const statusEl = document.getElementById('candleStatus');
  if (statusEl) statusEl.innerText = 'Candlestick initialized';
}


async function postJSONWithFallback(path, body) {
  const bases = [DEFAULT_API]; // Try backend first
  
  if (API_URL !== DEFAULT_API) {
    bases.unshift(API_URL);
  }
  
  for (const base of bases) {
    try {
      const url = `${base}${path}`;
      console.log(`Trying POST: ${url}`);
      const res = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      });
      const text = await res.text();
      
      console.log(`POST response from ${base}: status=${res.status}`);
      
      if (!res.ok) {
        console.log(`Skipping POST ${base}: response not ok (${res.status})`);
        continue;
      }
      if (text && text.trim().startsWith('<')) {
        console.log(`Skipping POST ${base}: response is HTML`);
        continue;
      }
      
      const parsed = JSON.parse(text);
      console.log(`Successfully parsed POST response from ${base}:`, parsed);
      return parsed;
    } catch (e) {
      console.log(`Error with POST ${base}:`, e.message);
      continue;
    }
  }
  throw new Error(`Failed to POST JSON to ${path}`);
}

let mainChart = null;
let volumeChart = null;
let rsiChart = null;
let macdChart = null;
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
  try {
    console.log('Starting page initialization...');
    
    try {
      initializeCharts();  // Initialize charts FIRST before loading data
      console.log('Charts initialized successfully');
    } catch (e) {
      console.error('Error initializing charts:', e);
    }
    
    try {
      setupEventListeners();
      console.log('Event listeners set up');
    } catch (e) {
      console.error('Error setting up event listeners:', e);
    }
    
    try {
      setupDateInput();
      console.log('Date input set up');
    } catch (e) {
      console.error('Error setting up date input:', e);
    }
    
    try {
      await loadSymbols();
      console.log('Symbols loaded');
    } catch (e) {
      console.error('Error loading symbols:', e);
    }
    
    try {
      updateMarketStats();
      console.log('Market stats updated');
    } catch (e) {
      console.error('Error updating market stats:', e);
    }
    
    try {
      updateWatchlist();
      console.log('Watchlist updated');
    } catch (e) {
      console.error('Error updating watchlist:', e);
    }
    
    try {
      loadMarketSentiment();
      console.log('Market sentiment loaded');
    } catch (e) {
      console.error('Error loading market sentiment:', e);
    }
    
    // Initialize Finnhub API integration
    try {
      console.log('Initializing Finnhub API...');
      if (typeof initFinnhub === 'function') {
        initFinnhub().catch(error => {
          console.warn('Finnhub initialization error:', error);
        });
      } else {
        console.warn('Finnhub module not loaded (finnhub.js)');
      }
    } catch (e) {
      console.warn('Finnhub initialization failed:', e);
    }
    
    console.log('Page initialization complete');
  } catch (error) {
    console.error('Unexpected error during page initialization:', error);
  }
}

// Setup date input to prevent past dates and warn about weekends
function setupDateInput() {
  const dateInput = document.getElementById('dateInput');
  if (dateInput) {
    // Set minimum date to today
    const today = new Date();
    const todayString = today.toISOString().split('T')[0];
    dateInput.setAttribute('min', todayString);
    
    // Warn user if they select a weekend
    dateInput.addEventListener('change', function() {
      const selectedDate = this.value;
      if (selectedDate && isWeekend(selectedDate)) {
        const nextWeekday = getNextWeekday(selectedDate);
        const dayName = new Date(selectedDate).toLocaleDateString('en-US', { weekday: 'long' });
        alert(`Warning: ${dayName} is a market holiday. Markets are closed on weekends. Please select a weekday (Monday-Friday).`);
        // Optionally auto-adjust to next weekday
        // this.value = nextWeekday;
      }
    });
  }
}

// Helper function to populate a select element with symbols
function populateSelect(selectId, indices, stocks) {
  const select = document.getElementById(selectId);
  if (!select) {
    console.warn(`Select element ${selectId} not found`);
    return;
  }

  const indicesGroup = select.querySelector('optgroup[label="Market Indices"]');
  const stocksGroup = select.querySelector('optgroup[label="Popular Stocks"]');

  if (!indicesGroup || !stocksGroup) {
    console.warn(`Could not find optgroups in ${selectId}`);
    return;
  }

  // Clear existing options
  indicesGroup.innerHTML = '';
  stocksGroup.innerHTML = '';

  // Add indices
  indices.forEach(symbol => {
    const option = document.createElement('option');
    option.value = symbol;
    option.text = symbol;
    indicesGroup.appendChild(option);
  });

  // Add stocks
  stocks.forEach(symbol => {
    const option = document.createElement('option');
    option.value = symbol;
    option.text = symbol;
    stocksGroup.appendChild(option);
  });

  // Select NIFTY by default if present
  const niftyOption = Array.from(select.options).find(o => o.value === 'NIFTY');
  if (niftyOption) {
    select.value = 'NIFTY';
  } else if (select.options.length > 0) {
    select.selectedIndex = 0;
  }
}

async function loadSymbols() {
  try {
    console.log('Fetching available symbols...');
    const data = await fetchJSONWithFallback('/available_symbols');
    console.log('Received symbols:', data);

    // Verify data structure
    if (!data || typeof data !== 'object') {
      throw new Error('Invalid response: data is not an object');
    }

    if (!data.indices || !Array.isArray(data.indices)) {
      console.warn('Warning: data.indices is not an array:', data.indices);
      data.indices = [];
    }

    if (!data.stocks || !Array.isArray(data.stocks)) {
      console.warn('Warning: data.stocks is not an array:', data.stocks);
      data.stocks = [];
    }

    // Populate all select elements
    populateSelect('symbolSelect', data.indices, data.stocks);
    populateSelect('analysisSymbolSelect', data.indices, data.stocks);
    populateSelect('predictionSymbolSelect', data.indices, data.stocks);
    populateSelect('investmentSymbolSelect', data.indices, data.stocks);
    populateSelect('sentimentSymbolSelect', data.indices, data.stocks);

    console.log('Populated dropdowns with', data.indices.length, 'indices and', data.stocks.length, 'stocks');

    // Load initial history for dashboard
    const select = document.getElementById('symbolSelect');
    if (select && select.value) {
      await loadHistory(select.value);
    }
  } catch (error) {
    console.error('Error loading symbols:', error);
  }
}

function setupEventListeners() {
  // Symbol change (dashboard)
  const symbolSelect = document.getElementById('symbolSelect');
  if (symbolSelect) {
    symbolSelect.addEventListener('change', () => {
      loadHistory();
      document.getElementById('prediction').innerHTML = '';
      // Update Gemini symbol context if function exists
      if (typeof setGeminiSymbol === 'function') {
        setGeminiSymbol(symbolSelect.value);
      }
    });
  }

  // Symbol change (analysis tab)
  const analysisSymbolSelect = document.getElementById('analysisSymbolSelect');
  if (analysisSymbolSelect) {
    analysisSymbolSelect.addEventListener('change', async () => {
      const symbol = analysisSymbolSelect.value;
      try {
        const data = await fetchJSONWithFallback(`/history?symbol=${encodeURIComponent(symbol)}`);
        if (!data.error) {
          updateTechnicalIndicators(data);
        }
      } catch (error) {
        console.error('Error loading analysis data:', error);
      }
    });
  }

  // Symbol change (investment analysis tab)
  const investmentSymbolSelect = document.getElementById('investmentSymbolSelect');
  if (investmentSymbolSelect) {
    investmentSymbolSelect.addEventListener('change', async () => {
      const symbol = investmentSymbolSelect.value;
      try {
        const data = await fetchJSONWithFallback(`/investment_analysis?symbol=${encodeURIComponent(symbol)}`);
        if (!data.error) {
          updateInvestmentAnalysis(data);
        } else {
          showInvestmentAnalysisError(data.error);
        }
      } catch (error) {
        console.error('Error loading investment analysis data:', error);
        showInvestmentAnalysisError('Failed to load investment analysis data');
      }
    });
  }

  // Symbol change (market sentiment tab)
  const sentimentSymbolSelect = document.getElementById('sentimentSymbolSelect');
  if (sentimentSymbolSelect) {
    sentimentSymbolSelect.addEventListener('change', async () => {
      const symbol = sentimentSymbolSelect.value;
      try {
        const data = await fetchJSONWithFallback(`/history?symbol=${encodeURIComponent(symbol)}`);
        if (!data.error) {
          updateSentimentGauge(data);
          // Update the label
          const sentimentLabel = document.getElementById('sentimentLabel');
          if (sentimentLabel) {
            sentimentLabel.textContent = `${symbol} Sentiment`;
          }
        }
      } catch (error) {
        console.error('Error loading sentiment data:', error);
      }
    });
  }

  // Time frame buttons
  document.querySelectorAll('.time-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
      document.querySelectorAll('.time-btn').forEach(b => b.classList.remove('active'));
      e.target.classList.add('active');
      const days = e.target.getAttribute('data-days');
      loadHistory(null, days);
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

  // Refresh news button
  const refreshNewsBtn = document.getElementById('refreshNewsBtn');
  if (refreshNewsBtn) {
    refreshNewsBtn.addEventListener('click', async () => {
      refreshNewsBtn.disabled = true;
      refreshNewsBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Refreshing...';
      try {
        await loadMarketNews();
      } catch (error) {
        console.error('Error refreshing news:', error);
      } finally {
        refreshNewsBtn.disabled = false;
        refreshNewsBtn.innerHTML = '<i class="fas fa-sync-alt"></i> Refresh';
      }
    });
  }
}

// Helper function to check if a date is a weekend (Saturday or Sunday)
function isWeekend(dateString) {
  const date = new Date(dateString);
  const dayOfWeek = date.getDay(); // 0 = Sunday, 6 = Saturday
  return dayOfWeek === 0 || dayOfWeek === 6;
}

// Helper function to get next weekday (skip weekends)
function getNextWeekday(dateString) {
  const date = new Date(dateString);
  let dayOfWeek = date.getDay();
  
  // If it's Sunday (0), move to Monday
  if (dayOfWeek === 0) {
    date.setDate(date.getDate() + 1);
  }
  // If it's Saturday (6), move to Monday
  else if (dayOfWeek === 6) {
    date.setDate(date.getDate() + 2);
  }
  
  return date.toISOString().split('T')[0];
}

async function predictStock() {
  const dateInput = document.getElementById('dateInput');
  const date = dateInput.value;
  // Use predictionSymbolSelect if on predictions tab, otherwise fall back to symbolSelect
  const predictionSelect = document.getElementById('predictionSymbolSelect');
  const dashboardSelect = document.getElementById('symbolSelect');
  const symbol = (predictionSelect && predictionSelect.value) || (dashboardSelect && dashboardSelect.value) || 'NIFTY';

  if (!date) {
    alert("Please select a date");
    return;
  }

  // Check if the selected date is a weekend
  if (isWeekend(date)) {
    const nextWeekday = getNextWeekday(date);
    const confirmMessage = `Markets are closed on weekends. Would you like to predict for the next weekday (${nextWeekday}) instead?`;
    if (confirm(confirmMessage)) {
      dateInput.value = nextWeekday;
      // Recursively call with the adjusted date
      return predictStock();
    } else {
      alert("Please select a weekday (Monday-Friday) for prediction. Markets are closed on weekends.");
      return;
    }
  }

  try {
    const data = await postJSONWithFallback('/predict', { date, symbol });
    if (data.predicted_price) {
      const predictedPrice = parseFloat(data.predicted_price);
      const currentPrice = parseFloat(data.current_price);
      const changePercent = data.change_percent || ((predictedPrice - currentPrice) / currentPrice * 100).toFixed(2);
      const trend = changePercent >= 0 ? 'up' : 'down';

      const predictionEl = document.getElementById('prediction');
      if (predictionEl) {
        const accuracyDisplay = data.model_accuracy !== undefined 
          ? `<div class="metric-item">
               <span class="metric-label"><i class="fas fa-bullseye"></i> Accuracy:</span>
               <span class="metric-value accuracy-${data.model_accuracy >= 70 ? 'high' : data.model_accuracy >= 50 ? 'medium' : 'low'}">${data.model_accuracy}%</span>
             </div>`
          : '';
        
        const f1Display = data.model_f1 !== undefined
          ? `<div class="metric-item">
               <span class="metric-label"><i class="fas fa-chart-bar"></i> F1 Score:</span>
               <span class="metric-value">${data.model_f1}%</span>
             </div>`
          : '';
        
        const r2Display = data.model_r2 !== undefined
          ? `<div class="metric-item">
               <span class="metric-label"><i class="fas fa-square-root-alt"></i> R² Score:</span>
               <span class="metric-value">${data.model_r2}%</span>
             </div>`
          : '';
        
        const confidenceLevelDisplay = data.confidence_level !== undefined
          ? `<div class="metric-item confidence-badge confidence-${data.confidence_level.toLowerCase().replace(' ', '-')}">
               <span class="metric-label"><i class="fas fa-shield-alt"></i> Confidence:</span>
               <span class="metric-value">${data.confidence_level}</span>
             </div>`
          : '';
        
        const priceRangeDisplay = (data.price_lower_bound !== undefined && data.price_upper_bound !== undefined)
          ? `<div class="price-range-section">
               <div class="range-label">Price Range (95% Confidence):</div>
               <div class="range-values">
                 <span class="lower-bound">₹${data.price_lower_bound}</span>
                 <span class="separator">–</span>
                 <span class="upper-bound">₹${data.price_upper_bound}</span>
               </div>
             </div>`
          : '';
        
        const metricsSection = (accuracyDisplay || f1Display || r2Display || confidenceLevelDisplay)
          ? `<div class="prediction-metrics">
               ${accuracyDisplay}
               ${f1Display}
               ${r2Display}
               ${confidenceLevelDisplay}
             </div>`
          : '';
        
        predictionEl.innerHTML = `
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
              ${metricsSection}
              ${priceRangeDisplay}
            </div>
          </div>
        `;
      }
    } else if (data.error) {
      // Check if error is about weekend
      if (data.error.includes('weekend') || data.error.includes('closed')) {
        alert(data.error);
        // Clear the date input
        const dateInputEl = document.getElementById('dateInput');
        if (dateInputEl) {
          dateInputEl.value = '';
        }
      } else {
        alert("Error: " + data.error);
      }
    }
  } catch (error) {
    console.error('Error predicting stock:', error);
    alert("Error making prediction. Please try again.");
  }
}

async function loadHistory(symbol = null, days = null) {
  try {
    if (!symbol) {
      symbol = document.getElementById('symbolSelect').value;
    }

    // Get days from active time button if not provided
    if (!days) {
      const activeTimeBtn = document.querySelector('.time-btn.active');
      if (activeTimeBtn) {
        days = activeTimeBtn.getAttribute('data-days') || '60';
      } else {
        days = '60'; // Default to 60 days
      }
    }

    const data = await fetchJSONWithFallback(`/history?symbol=${encodeURIComponent(symbol)}&days=${days}`);

    if (data.error) {
      alert("Error: " + data.error);
      return;
    }

    updateCharts(data, symbol);
    updateTechnicalIndicators(data);
    updateSentimentGauge(data);
  } catch (error) {
    console.error('Error loading history:', error);
  }
}

function initializeCharts() {
  const mainCtx = document.getElementById('mainChart').getContext('2d');
  const volumeCtx = document.getElementById('volumeChart').getContext('2d');
  const rsiCanvas = document.getElementById('rsiChart');
  const macdCanvas = document.getElementById('macdChart');
  
  // Initialize sentiment gauge if the element exists (it might be in a hidden section)
  const sentimentCanvas = document.getElementById('sentimentGauge');
  let sentimentCtx = null;
  if (sentimentCanvas) {
    sentimentCtx = sentimentCanvas.getContext('2d');
  }

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

  // Initialize RSI chart
  if (rsiCanvas) {
    const rsiCtx = rsiCanvas.getContext('2d');
    rsiChart = new Chart(rsiCtx, {
      type: 'line',
      data: { datasets: [] },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            display: true,
            position: 'top',
            labels: {
              color: '#94a3b8'
            }
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
            min: 0,
            max: 100,
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
  }

  // Initialize MACD chart
  if (macdCanvas) {
    const macdCtx = macdCanvas.getContext('2d');
    macdChart = new Chart(macdCtx, {
      type: 'line',
      data: { datasets: [] },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            display: true,
            position: 'top',
            labels: {
              color: '#94a3b8'
            }
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
  }

  // Initialize sentiment gauge (only if canvas exists)
  if (sentimentCtx) {
    sentimentGauge = new Chart(sentimentCtx, {
      type: 'doughnut',
      data: {
        labels: ['Bullish', 'Neutral', 'Bearish'],
        datasets: [{
          data: [60, 25, 15],
          backgroundColor: ['#22c55e', '#3b82f6', '#ef4444'],
          borderColor: ['#16a34a', '#2563eb', '#dc2626'],
          borderWidth: 2
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: true,
        plugins: {
          legend: {
            display: true,
            position: 'bottom',
            labels: {
              color: '#94a3b8',
              padding: 15,
              font: { size: 12 }
            }
          },
          tooltip: {
            callbacks: {
              label: function(context) {
                const value = context.parsed || 0;
                return context.label + ': ' + value.toFixed(1) + '%';
              }
            },
            backgroundColor: 'rgba(0,0,0,0.7)',
            titleColor: '#fff',
            bodyColor: '#fff',
            borderColor: '#94a3b8',
            borderWidth: 1
          }
        },
        cutout: '65%'
      }
    });
  } else {
    console.warn('Sentiment gauge canvas not found, initialization deferred');
  }
}

function updateCharts(data, symbol) {
  try {
    // Start with price dataset
    const datasets = [{
      label: `${symbol} Price`,
      data: data.prices,
      borderColor: '#3b82f6',
      backgroundColor: 'rgba(59, 130, 246, 0.1)',
      fill: true,
      tension: 0.4,
      yAxisID: 'y'
    }];

    // Add SMA indicators if enabled
    if (indicators.sma.enabled && data.indicators) {
      if (data.indicators.sma20) {
        datasets.push({
          label: 'SMA 20',
          data: data.indicators.sma20,
          borderColor: '#f59e0b',
          backgroundColor: 'transparent',
          borderWidth: 1.5,
          borderDash: [],
          fill: false,
          tension: 0.4,
          yAxisID: 'y'
        });
      }
      if (data.indicators.sma50) {
        datasets.push({
          label: 'SMA 50',
          data: data.indicators.sma50,
          borderColor: '#ef4444',
          backgroundColor: 'transparent',
          borderWidth: 1.5,
          borderDash: [],
          fill: false,
          tension: 0.4,
          yAxisID: 'y'
        });
      }
    }

    // Add EMA indicators if enabled
    if (indicators.ema.enabled && data.indicators) {
      if (data.indicators.ema9) {
        datasets.push({
          label: 'EMA 9',
          data: data.indicators.ema9,
          borderColor: '#10b981',
          backgroundColor: 'transparent',
          borderWidth: 1.5,
          borderDash: [5, 5],
          fill: false,
          tension: 0.4,
          yAxisID: 'y'
        });
      }
    }

    // Update main chart
    mainChart.data = {
      labels: data.dates,
      datasets: datasets
    };
    mainChart.update();

    // Update volume chart
    if (data.volumes) {
      volumeChart.data = {
        labels: data.dates,
        datasets: [{
          label: 'Volume',
          data: data.volumes,
          backgroundColor: 'rgba(59, 130, 246, 0.3)'
        }]
      };
      volumeChart.update();
    }

    // Update candlestick chart if candle data available
    try {
      if (data.candles && data.candles.length > 0) {
        if (!candleChart || !candleSeries) {
          initCandleChart();
        }
        if (candleSeries && typeof candleSeries.setData === 'function') {
          // Ensure data is in the expected format for LightweightCharts
          const formatted = data.candles.map(c => ({
            time: c.time,
            open: c.open,
            high: c.high,
            low: c.low,
            close: c.close
          }));
          console.log('Setting candlestick data with', formatted.length, 'points');
          const statusEl = document.getElementById('candleStatus');
          if (statusEl) statusEl.innerText = `Updating candlestick (${formatted.length} points)`;
          try {
            candleSeries.setData(formatted);
            console.log('Candlestick series updated successfully');
            if (statusEl) statusEl.innerText = `Candlestick updated (${formatted.length} pts)`;
          } catch (err) {
            console.error('Error calling candleSeries.setData():', err);
            if (statusEl) statusEl.innerText = `Candlestick error: ${err.message || err}`;
          }
          if (candleChart && candleChart.timeScale) {
            // Delay slightly to ensure DOM changes applied before fitting content
            setTimeout(() => {
              try {
                candleChart.timeScale().fitContent();
              } catch (err) {
                console.error('Error fitting candlestick time scale:', err);
              }
            }, 50);
          }
        }
      }
    } catch (e) {
      console.error('Error updating candlestick chart:', e);
    }

    // Update RSI chart if enabled
    const rsiCanvas = document.getElementById('rsiChart');
    if (indicators.rsi.enabled && data.indicators && data.indicators.rsi && rsiChart && rsiCanvas) {
      rsiCanvas.style.display = 'block';
      rsiChart.data = {
        labels: data.dates,
        datasets: [{
          label: 'RSI (14)',
          data: data.indicators.rsi,
          borderColor: '#8b5cf6',
          backgroundColor: 'rgba(139, 92, 246, 0.1)',
          borderWidth: 2,
          fill: true,
          tension: 0.4
        }]
      };
      rsiChart.update();
    } else if (rsiCanvas) {
      rsiCanvas.style.display = 'none';
    }

    // Update MACD chart if enabled
    const macdCanvas = document.getElementById('macdChart');
    if (indicators.macd.enabled && data.indicators && macdChart && macdCanvas) {
      if (data.indicators.macd && data.indicators.macd_signal) {
        macdCanvas.style.display = 'block';
        macdChart.data = {
          labels: data.dates,
          datasets: [
            {
              label: 'MACD',
              data: data.indicators.macd,
              borderColor: '#3b82f6',
              backgroundColor: 'transparent',
              borderWidth: 2,
              tension: 0.4
            },
            {
              label: 'Signal Line',
              data: data.indicators.macd_signal,
              borderColor: '#ef4444',
              backgroundColor: 'transparent',
              borderWidth: 2,
              borderDash: [5, 5],
              tension: 0.4
            }
          ]
        };
        macdChart.update();
      }
    } else if (macdCanvas) {
      macdCanvas.style.display = 'none';
    }
  } catch (error) {
    console.error('Error updating charts:', error);
  }
}

// Backwards-compatible small helpers
// Called by UI controls; these keep existing behavior but avoid ReferenceErrors
function updateChart() {
  // Refresh current symbol's history which will update charts/indicators/gauge
  try {
    const select = document.getElementById('symbolSelect');
    if (select && select.value) {
      loadHistory(select.value).catch(e => console.error('updateChart loadHistory failed', e));
    } else {
      // If no symbol selected, try reloading the currently displayed chart by reusing last symbol
      loadHistory().catch(e => console.error('updateChart loadHistory failed', e));
    }
  } catch (err) {
    console.error('updateChart error:', err);
  }
}

function updateChartType(type) {
  try {
    const mainChartCanvas = document.getElementById('mainChart');
    const candleChartDiv = document.getElementById('candleChart');
    
    if (type === 'candlestick') {
      // Switch to candlestick view
      if (mainChartCanvas) mainChartCanvas.style.display = 'none';
      if (candleChartDiv) {
        candleChartDiv.style.display = 'block';
        console.log('Switched chart type to candlestick; ensuring chart initialized');
        if (!candleChart) {
          initCandleChart();
        }
        // Fit content after ensuring chart is visible
        setTimeout(() => {
          try { if (candleChart && candleChart.timeScale) candleChart.timeScale().fitContent(); } catch (err) { console.error('Error fitting candlestick scale after type switch:', err); }
        }, 50);
      }
    } else {
      // Switch back to line/bar view
      if (mainChartCanvas) mainChartCanvas.style.display = 'block';
      if (candleChartDiv) candleChartDiv.style.display = 'none';
      
      // Ensure mainChart is properly displayed
      if (mainChart) {
        mainChart.resize();
      }
    }
  } catch (err) {
    console.error('updateChartType error:', err);
  }

}

function updateTechnicalIndicators(data) {
  const prices = data.prices;
  const indicatorValues = document.getElementById('indicator-values');
  indicatorValues.innerHTML = '';

  try {
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
  } catch (error) {
    console.error('Error updating technical indicators:', error);
  }
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
  
  // Fetch real market data for indices
  const fetchMarketData = async () => {
    try {
      // Get NIFTY data
      const niftyData = await fetchJSONWithFallback('/history?symbol=NIFTY&days=2');
      const sensexData = await fetchJSONWithFallback('/history?symbol=SENSEX&days=2');
      
      if (niftyData && niftyData.prices && niftyData.prices.length >= 2 &&
          sensexData && sensexData.prices && sensexData.prices.length >= 2) {
        
        const niftyLast = niftyData.prices[niftyData.prices.length - 1];
        const niftyPrev = niftyData.prices[niftyData.prices.length - 2];
        const niftyChange = ((niftyLast - niftyPrev) / niftyPrev) * 100;
        
        const sensexLast = sensexData.prices[sensexData.prices.length - 1];
        const sensexPrev = sensexData.prices[sensexData.prices.length - 2];
        const sensexChange = ((sensexLast - sensexPrev) / sensexPrev) * 100;
        
        const trend = (niftyChange + sensexChange) / 2 > 0 ? 'Bullish' : 'Bearish';
        const trendClass = trend === 'Bullish' ? 'up' : 'down';
        
        marketStats.innerHTML = `
          <div class="market-stat">
            <span>Market Status</span>
            <span class="active">Open</span>
          </div>
          <div class="market-stat">
            <span>Market Trend</span>
            <span class="${trendClass}">${trend}</span>
          </div>
          <div class="market-stat">
            <span>NIFTY</span>
            <span class="${niftyChange > 0 ? 'up' : 'down'}">${niftyChange.toFixed(2)}%</span>
          </div>
          <div class="market-stat">
            <span>SENSEX</span>
            <span class="${sensexChange > 0 ? 'up' : 'down'}">${sensexChange.toFixed(2)}%</span>
          </div>
        `;
      } else {
        throw new Error('Invalid data');
      }
    } catch (error) {
      console.error('Error fetching market stats:', error);
      // Fallback to static data
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
  };
  
  fetchMarketData();
}

function updateWatchlist() {
  const watchlist = document.getElementById('watchlist-items');
  
  const fetchWatchlistData = async () => {
    try {
      const symbols = await fetchJSONWithFallback('/available_symbols');
      const watchlistSymbols = ['NIFTY', 'SENSEX', 'TCS', 'RELIANCE', 'INFY'];
      
      const watchlistData = [];
      for (const symbol of watchlistSymbols) {
        if (symbols.indices.includes(symbol) || symbols.stocks.includes(symbol)) {
          try {
            const data = await fetchJSONWithFallback(`/history?symbol=${symbol}&days=2`);
            if (data.prices && data.prices.length >= 2) {
              const lastPrice = data.prices[data.prices.length - 1];
              const prevPrice = data.prices[data.prices.length - 2];
              const change = ((lastPrice - prevPrice) / prevPrice) * 100;
              watchlistData.push({
                symbol,
                price: lastPrice.toFixed(2),
                change: change.toFixed(2)
              });
            }
          } catch (e) {
            console.warn(`Could not fetch data for ${symbol}`, e);
          }
        }
      }
      
      if (watchlistData.length > 0) {
        watchlist.innerHTML = watchlistData.map(item => `
          <div class="watchlist-item">
            <span class="symbol">${item.symbol}</span>
            <span class="price">₹${item.price}</span>
            <span class="change ${parseFloat(item.change) >= 0 ? 'up' : 'down'}">${parseFloat(item.change) >= 0 ? '+' : ''}${item.change}%</span>
          </div>
        `).join('');
      } else {
        throw new Error('No watchlist data');
      }
    } catch (error) {
      console.error('Error fetching watchlist:', error);
      // Fallback to mock data
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
  };
  
  fetchWatchlistData();
}

function updateSentimentGauge(data) {
  console.log('updateSentimentGauge called with data:', data);
  if (!sentimentGauge || !data || !data.prices) {
    console.warn('updateSentimentGauge: Missing sentimentGauge, data, or prices', { sentimentGauge: !!sentimentGauge, data: !!data, prices: data?.prices?.length });
    return;
  }

  const prices = data.prices;
  console.log('Prices array length:', prices.length);
  if (prices.length < 2) {
    console.warn('Not enough price data');
    // Set default neutral sentiment
    sentimentGauge.data.datasets[0].data = [33.3, 33.4, 33.3];
    sentimentGauge.update();
    return;
  }

  // Calculate simple sentiment metrics
  const lastPrice = prices[prices.length - 1];
  const prevPrice = prices[prices.length - 2];
  const priceChange = ((lastPrice - prevPrice) / prevPrice) * 100;
  console.log('Price change:', priceChange, 'Last:', lastPrice, 'Prev:', prevPrice);

  // Calculate trend strength using available data
  let bullishScore = 0;
  let neutralScore = 0;
  let bearishScore = 0;

  // Use 20-day average if available, otherwise use available data
  if (prices.length >= 40) {
    const recentAvg = prices.slice(-20).reduce((a, b) => a + b) / 20;
    const olderAvg = prices.slice(-40, -20).reduce((a, b) => a + b) / 20;
    const trend = ((recentAvg - olderAvg) / olderAvg) * 100;
    console.log('Trend:', trend, 'Recent:', recentAvg, 'Older:', olderAvg);

    // Score based on trend
    if (trend > 2) bullishScore += 40;
    else if (trend > 0.5) { bullishScore += 25; neutralScore += 15; }
    else if (trend > -0.5) neutralScore += 40;
    else if (trend > -2) { bearishScore += 25; neutralScore += 15; }
    else bearishScore += 40;
  } else if (prices.length >= 20) {
    // Use shorter period if 40 days not available
    const recentAvg = prices.slice(-10).reduce((a, b) => a + b) / 10;
    const olderAvg = prices.slice(-20, -10).reduce((a, b) => a + b) / 10;
    const trend = ((recentAvg - olderAvg) / olderAvg) * 100;
    
    if (trend > 2) bullishScore += 40;
    else if (trend > 0.5) { bullishScore += 25; neutralScore += 15; }
    else if (trend > -0.5) neutralScore += 40;
    else if (trend > -2) { bearishScore += 25; neutralScore += 15; }
    else bearishScore += 40;
  } else {
    // If not enough data for trend, give neutral score
    neutralScore += 40;
  }

  // Score based on recent price movement
  if (priceChange > 1) bullishScore += 30;
  else if (priceChange > 0.3) { bullishScore += 15; neutralScore += 15; }
  else if (priceChange > -0.3) neutralScore += 30;
  else if (priceChange > -1) { bearishScore += 15; neutralScore += 15; }
  else bearishScore += 30;

  // Calculate RSI-like metric for overbought/oversold
  if (prices.length >= 14) {
    const changes = [];
    for (let i = Math.max(0, prices.length - 14); i < prices.length; i++) {
      if (i > 0) changes.push(prices[i] - prices[i - 1]);
    }
    const gains = changes.filter(c => c > 0).reduce((a, b) => a + b, 0);
    const losses = Math.abs(changes.filter(c => c < 0).reduce((a, b) => a + b, 0));
    const ratio = gains / (losses || 1);
    console.log('RSI ratio:', ratio, 'Gains:', gains, 'Losses:', losses);

    if (ratio > 1.5) bullishScore += 15;
    else if (ratio < 0.7) bearishScore += 15;
    else neutralScore += 15;
  } else {
    // If not enough data for RSI, give neutral score
    neutralScore += 15;
  }

  // Ensure we have some scores
  if (bullishScore === 0 && neutralScore === 0 && bearishScore === 0) {
    bullishScore = 33.3;
    neutralScore = 33.4;
    bearishScore = 33.3;
  }

  // Normalize scores to ensure they add up to 100
  const total = bullishScore + neutralScore + bearishScore;
  const finalBullish = Math.max(0, Math.min(100, (bullishScore / total) * 100));
  const finalNeutral = Math.max(0, Math.min(100, (neutralScore / total) * 100));
  const finalBearish = Math.max(0, Math.min(100, (bearishScore / total) * 100));
  
  // Ensure they add up to exactly 100
  const sum = finalBullish + finalNeutral + finalBearish;
  const adjustment = 100 - sum;
  const adjustedNeutral = finalNeutral + adjustment;

  console.log('Sentiment scores - Bullish:', finalBullish, 'Neutral:', adjustedNeutral, 'Bearish:', finalBearish);

  // Update pie chart
  sentimentGauge.data.datasets[0].data = [finalBullish, adjustedNeutral, finalBearish];
  sentimentGauge.update('active');
  console.log('Sentiment gauge updated');
}

function loadMarketSentiment() {
  // Mock sentiment data - replace with real data when available
  const tradingSignals = document.querySelector('.trading-signals');
  if (tradingSignals) {
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
}

// Function to load analysis data when analysis tab is shown
window.loadAnalysisData = async function() {
  const analysisSelect = document.getElementById('analysisSymbolSelect');
  if (!analysisSelect || !analysisSelect.value) return;
  
  try {
    const data = await fetchJSONWithFallback(`/history?symbol=${encodeURIComponent(analysisSelect.value)}`);
    if (!data.error) {
      updateTechnicalIndicators(data);
    }
  } catch (error) {
    console.error('Error loading analysis data:', error);
  }
};

// Function to load market sentiment data when market-news tab is shown
window.loadMarketSentimentData = async function() {
  // Initialize sentiment gauge if it hasn't been initialized yet
  if (!sentimentGauge) {
    const sentimentCanvas = document.getElementById('sentimentGauge');
    if (sentimentCanvas) {
      const sentimentCtx = sentimentCanvas.getContext('2d');
      sentimentGauge = new Chart(sentimentCtx, {
        type: 'doughnut',
        data: {
          labels: ['Bullish', 'Neutral', 'Bearish'],
          datasets: [{
            data: [60, 25, 15],
            backgroundColor: ['#22c55e', '#3b82f6', '#ef4444'],
            borderColor: ['#16a34a', '#2563eb', '#dc2626'],
            borderWidth: 2
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: true,
          plugins: {
            legend: {
              display: true,
              position: 'bottom',
              labels: {
                color: '#94a3b8',
                padding: 15,
                font: { size: 12 }
              }
            },
            tooltip: {
              callbacks: {
                label: function(context) {
                  const value = context.parsed || 0;
                  return context.label + ': ' + value.toFixed(1) + '%';
                }
              },
              backgroundColor: 'rgba(0,0,0,0.7)',
              titleColor: '#fff',
              bodyColor: '#fff',
              borderColor: '#94a3b8',
              borderWidth: 1
            }
          },
          cutout: '65%'
        }
      });
    }
  }

  // Get the selected symbol from dropdown, default to NIFTY
  const sentimentSelect = document.getElementById('sentimentSymbolSelect');
  const symbol = (sentimentSelect && sentimentSelect.value) || 'NIFTY';

  // Load sentiment data for the selected symbol
  try {
    const data = await fetchJSONWithFallback(`/history?symbol=${encodeURIComponent(symbol)}`);
    if (!data.error) {
      updateSentimentGauge(data);
      // Update the label
      const sentimentLabel = document.getElementById('sentimentLabel');
      if (sentimentLabel) {
        sentimentLabel.textContent = `${symbol} Sentiment`;
      }
      loadMarketSentiment();
    }
  } catch (error) {
    console.error('Error loading market sentiment data:', error);
    // Still show mock data
    loadMarketSentiment();
  }
  
  await loadMarketNews();
};

// Function to load investment analysis data when investment-analysis tab is shown
window.loadInvestmentAnalysisData = async function() {
  const investmentSelect = document.getElementById('investmentSymbolSelect');
  if (!investmentSelect || !investmentSelect.value) return;

  try {
    const data = await fetchJSONWithFallback(`/investment_analysis?symbol=${encodeURIComponent(investmentSelect.value)}`);
    if (!data.error) {
      updateInvestmentAnalysis(data);
    } else {
      showInvestmentAnalysisError(data.error);
    }
  } catch (error) {
    console.error('Error loading investment analysis data:', error);
    showInvestmentAnalysisError('Failed to load investment analysis data');
  }
};

// Function to update investment analysis display
function updateInvestmentAnalysis(data) {
  // Update recommendation card
  const recommendationCard = document.getElementById('investment-recommendation');
  if (recommendationCard) {
    const recommendationClass = data.recommendation.toLowerCase().replace(' ', '-');
    recommendationCard.innerHTML = `
      <div class="recommendation-header ${recommendationClass}">
        <div class="recommendation-signal">${data.recommendation}</div>
        <div class="recommendation-confidence">Confidence: ${data.confidence}</div>
      </div>
      <div class="current-price">
        <span class="price">₹${data.current_price.toFixed(2)}</span>
        <span class="change ${data.daily_change >= 0 ? 'positive' : 'negative'}">
          ${data.daily_change >= 0 ? '+' : ''}${data.daily_change.toFixed(2)}%
        </span>
      </div>
    `;
  }

  // Update key metrics
  const keyMetrics = document.getElementById('key-metrics');
  if (keyMetrics) {
    keyMetrics.innerHTML = `
      <div class="metric-item">
        <span class="metric-label">1M Return</span>
        <span class="metric-value ${data.key_metrics.returns_1m >= 0 ? 'positive' : 'negative'}">${data.key_metrics.returns_1m.toFixed(2)}%</span>
      </div>
      <div class="metric-item">
        <span class="metric-label">3M Return</span>
        <span class="metric-value ${data.key_metrics.returns_3m >= 0 ? 'positive' : 'negative'}">${data.key_metrics.returns_3m.toFixed(2)}%</span>
      </div>
      <div class="metric-item">
        <span class="metric-label">6M Return</span>
        <span class="metric-value ${data.key_metrics.returns_6m >= 0 ? 'positive' : 'negative'}">${data.key_metrics.returns_6m.toFixed(2)}%</span>
      </div>
      <div class="metric-item">
        <span class="metric-label">1Y Return</span>
        <span class="metric-value ${data.key_metrics.returns_1y >= 0 ? 'positive' : 'negative'}">${data.key_metrics.returns_1y.toFixed(2)}%</span>
      </div>
      <div class="metric-item">
        <span class="metric-label">Volatility</span>
        <span class="metric-value">${data.key_metrics.volatility.toFixed(2)}%</span>
      </div>
      <div class="metric-item">
        <span class="metric-label">RSI (14)</span>
        <span class="metric-value ${data.key_metrics.rsi > 70 ? 'overbought' : data.key_metrics.rsi < 30 ? 'oversold' : ''}">${data.key_metrics.rsi.toFixed(1)}</span>
      </div>
      <div class="metric-item">
        <span class="metric-label">SMA 20</span>
        <span class="metric-value">₹${data.key_metrics.sma20.toFixed(2)}</span>
      </div>
      <div class="metric-item">
        <span class="metric-label">SMA 50</span>
        <span class="metric-value">₹${data.key_metrics.sma50.toFixed(2)}</span>
      </div>
      <div class="metric-item">
        <span class="metric-label">Volume Ratio</span>
        <span class="metric-value ${data.key_metrics.volume_ratio > 1.5 ? 'high-volume' : data.key_metrics.volume_ratio < 0.5 ? 'low-volume' : ''}">${data.key_metrics.volume_ratio.toFixed(1)}x</span>
      </div>
    `;
  }

  // Update technical signals
  const technicalSignals = document.getElementById('technical-signals');
  if (technicalSignals) {
    technicalSignals.innerHTML = data.technical_signals.map(signal => `
      <div class="signal-item ${signal.type}">
        <div class="signal-icon">
          <i class="fas ${signal.type === 'bullish' ? 'fa-arrow-up' : signal.type === 'bearish' ? 'fa-arrow-down' : 'fa-minus'}"></i>
        </div>
        <div class="signal-content">
          <div class="signal-name">${signal.signal}</div>
          <div class="signal-description">${signal.description}</div>
        </div>
      </div>
    `).join('');
  }

  // Update risk assessment
  const riskAssessment = document.getElementById('risk-assessment');
  if (riskAssessment) {
    const riskClass = data.risk_assessment.level.toLowerCase();
    riskAssessment.innerHTML = `
      <div class="risk-level ${riskClass}">
        <div class="risk-header">
          <span class="risk-label">Risk Level:</span>
          <span class="risk-value">${data.risk_assessment.level}</span>
        </div>
        <div class="risk-score">Score: ${data.risk_assessment.score}/10</div>
        <div class="risk-factors">
          <strong>Risk Factors:</strong>
          <ul>
            ${data.risk_assessment.factors.map(factor => `<li>${factor}</li>`).join('')}
          </ul>
        </div>
      </div>
    `;
  }

  // Update detailed insights
  const detailedInsights = document.getElementById('detailed-insights');
  if (detailedInsights) {
    detailedInsights.innerHTML = data.insights.map(insight => `
      <div class="insight-item">
        <i class="fas fa-lightbulb"></i>
        <span>${insight}</span>
      </div>
    `).join('');
  }
}

// Function to show error in investment analysis
function showInvestmentAnalysisError(error) {
  const elements = [
    'investment-recommendation',
    'key-metrics',
    'technical-signals',
    'risk-assessment',
    'detailed-insights'
  ];

  elements.forEach(id => {
    const element = document.getElementById(id);
    if (element) {
      element.innerHTML = `
        <div class="error-message">
          <i class="fas fa-exclamation-triangle"></i>
          <span>${error}</span>
        </div>
      `;
    }
  });
}

// Function to load and display market news
async function loadMarketNews() {
  const newsContainer = document.getElementById('newsContainer');
  if (!newsContainer) return;
  
  try {
    newsContainer.innerHTML = '<div class="loading-spinner"><i class="fas fa-spinner"></i> Loading news...</div>';
    
    const data = await fetchJSONWithFallback('/news');
    
    if (data && data.news && Array.isArray(data.news)) {
      if (data.news.length === 0) {
        newsContainer.innerHTML = '<div class="no-news-message">No market news available at the moment.</div>';
      } else {
        newsContainer.innerHTML = data.news.map(article => `
          <div class="news-card">
            <div class="news-header">
              <h4 class="news-title">${escapeHtml(article.title)}</h4>
              <span class="news-time">${formatTimeAgo(article.published_date || new Date())}</span>
            </div>
            <div class="news-source">
              <i class="fas fa-newspaper"></i> ${escapeHtml(article.source || 'Market News')}
            </div>
            <p class="news-description">${escapeHtml(article.description || article.content || 'No description available')}</p>
            ${article.url ? `<a href="${encodeURI(article.url)}" target="_blank" class="news-link"><i class="fas fa-external-link-alt"></i> Read More</a>` : ''}
          </div>
        `).join('');
      }
    } else {
      throw new Error('Invalid news data format');
    }
  } catch (error) {
    console.error('Error loading market news:', error);
    newsContainer.innerHTML = `
      <div class="error-message">
        <i class="fas fa-exclamation-triangle"></i>
        <span>Failed to load market news. Using sample data...</span>
      </div>
    `;
    displaySampleNews();
  }
}

// Function to display sample/mock news when real data is unavailable
function displaySampleNews() {
  const newsContainer = document.getElementById('newsContainer');
  if (!newsContainer) return;
  
  const sampleNews = [
    {
      title: 'Market Shows Strong Recovery',
      source: 'Economic Times',
      description: 'Stock indices bounce back with strong gains driven by IT and Banking sectors.',
      published_date: new Date(Date.now() - 2 * 60 * 60 * 1000)
    },
    {
      title: 'RBI Maintains Policy Rate',
      source: 'Business Today',
      description: 'Reserve Bank of India holds key policy rate steady amid inflationary concerns.',
      published_date: new Date(Date.now() - 4 * 60 * 60 * 1000)
    },
    {
      title: 'Tech Giants Report Earnings',
      source: 'Moneycontrol',
      description: 'Leading IT companies announce strong quarterly results with growth in cloud services.',
      published_date: new Date(Date.now() - 6 * 60 * 60 * 1000)
    },
    {
      title: 'Rupee Strengthens Against Dollar',
      source: 'LiveMint',
      description: 'Indian rupee gains ground as foreign investors continue to show confidence in markets.',
      published_date: new Date(Date.now() - 8 * 60 * 60 * 1000)
    }
  ];
  
  newsContainer.innerHTML = sampleNews.map(article => `
    <div class="news-card">
      <div class="news-header">
        <h4 class="news-title">${escapeHtml(article.title)}</h4>
        <span class="news-time">${formatTimeAgo(article.published_date)}</span>
      </div>
      <div class="news-source">
        <i class="fas fa-newspaper"></i> ${escapeHtml(article.source)}
      </div>
      <p class="news-description">${escapeHtml(article.description)}</p>
    </div>
  `).join('');
}

// Helper function to escape HTML to prevent XSS
function escapeHtml(text) {
  if (!text) return '';
  const map = {
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#039;'
  };
  return text.replace(/[&<>"']/g, m => map[m]);
}

// Helper function to format time as "X hours ago" style
function formatTimeAgo(date) {
  if (typeof date === 'string') {
    date = new Date(date);
  }
  const now = new Date();
  const seconds = Math.floor((now - date) / 1000);
  
  if (seconds < 60) return 'Just now';
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  if (days < 7) return `${days}d ago`;
  return date.toLocaleDateString();
}

// Wait for DOM to be fully loaded before initializing
document.addEventListener('DOMContentLoaded', () => {
  console.log('DOM loaded, initializing page...');
  initializePage().catch(error => {
    console.error('Error during page initialization:', error);
  });
});
