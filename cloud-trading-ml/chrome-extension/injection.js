// TradingView Data Extractor
// Injected into TradingView page to extract candle data

(function() {
    'use strict';
    
    console.log("✅ Cloud ML: Data extractor injected");
    
    // Send ready signal
    window.postMessage({
        type: 'TRADINGVIEW_READY',
        timestamp: new Date().toISOString()
    }, '*');
    
    // Function to extract candle data from TradingView
    function extractCandleData() {
        try {
            // Method 1: Try to access TradingView's internal data structure
            let candles = [];
            
            // Look for chart widget
            const chartWidget = window.tvWidget || 
                               window.TradingView?.widget || 
                               document.querySelector('[data-role="chart"]')?.__vue__;
            
            if (chartWidget) {
                try {
                    // Try different access methods
                    if (chartWidget.chart && chartWidget.chart().series) {
                        const series = chartWidget.chart().series();
                        if (series && series.bars) {
                            candles = series.bars();
                        }
                    } else if (chartWidget._chartApi) {
                        const api = chartWidget._chartApi();
                        if (api && api.getSeries) {
                            const series = api.getSeries();
                            if (series && series._series && series._series.bars) {
                                candles = series._series.bars();
                            }
                        }
                    }
                } catch (e) {
                    console.log("TV internal access failed:", e);
                }
            }
            
            // Method 2: Extract from DOM (fallback)
            if (!candles || candles.length === 0) {
                candles = extractFromDOM();
            }
            
            // Get the latest candle
            if (candles && candles.length > 0) {
                const latest = candles[candles.length - 1];
                
                // Create candle object
                const candleData = {
                    symbol: getCurrentSymbol(),
                    timeframe: getCurrentTimeframe(),
                    timestamp: new Date(latest.time * 1000).toISOString(),
                    open: latest.open,
                    high: latest.high,
                    low: latest.low,
                    close: latest.close,
                    volume: latest.volume || 0,
                    rsi: calculateRSI(candles.slice(-15))
                };
                
                // Send to extension
                window.postMessage({
                    type: 'TRADINGVIEW_CANDLE',
                    candle: candleData
                }, '*');
                
                return candleData;
            }
            
        } catch (error) {
            console.warn("Candle extraction error:", error);
        }
        
        return null;
    }
    
    function extractFromDOM() {
        // Try to find candle elements in DOM
        const candles = [];
        
        // Look for chart pane
        const chartPane = document.querySelector('.pane-legend');
        if (chartPane) {
            const text = chartPane.textContent;
            
            // Try to parse OHLC from text
            const matches = text.match(/O\s*([\d.]+)\s*H\s*([\d.]+)\s*L\s*([\d.]+)\s*C\s*([\d.]+)/i);
            if (matches) {
                candles.push({
                    time: Math.floor(Date.now() / 1000),
                    open: parseFloat(matches[1]),
                    high: parseFloat(matches[2]),
                    low: parseFloat(matches[3]),
                    close: parseFloat(matches[4]),
                    volume: 0
                });
            }
        }
        
        return candles;
    }
    
    function getCurrentSymbol() {
        // Extract symbol from page
        try {
            const symbolElement = document.querySelector('[data-symbol]') || 
                                 document.querySelector('.symbol');
            if (symbolElement) {
                return symbolElement.getAttribute('data-symbol') || 
                       symbolElement.textContent.trim();
            }
        } catch (e) {}
        
        return 'EURUSD'; // Default
    }
    
    function getCurrentTimeframe() {
        // Extract timeframe
        try {
            const tfElement = document.querySelector('.interval') || 
                             document.querySelector('[data-interval]');
            if (tfElement) {
                return tfElement.textContent.trim() || '1m';
            }
        } catch (e) {}
        
        return '1m'; // Default
    }
    
    function calculateRSI(candles) {
        if (!candles || candles.length < 14) return 50;
        
        let gains = 0;
        let losses = 0;
        
        for (let i = 1; i < Math.min(15, candles.length); i++) {
            const change = candles[i].close - candles[i-1].close;
            if (change > 0) {
                gains += change;
            } else {
                losses -= change;
            }
        }
        
        const avgGain = gains / 14;
        const avgLoss = losses / 14;
        
        if (avgLoss === 0) return 100;
        
        const rs = avgGain / avgLoss;
        return 100 - (100 / (1 + rs));
    }
    
    // Start monitoring
    let lastExtraction = 0;
    
    function monitorChart() {
        const now = Date.now();
        
        // Extract data every 5 seconds
        if (now - lastExtraction > 5000) {
            extractCandleData();
            lastExtraction = now;
        }
        
        // Also monitor for chart updates
        const observer = new MutationObserver((mutations) => {
            extractCandleData();
        });
        
        // Observe chart container for changes
        const chartContainer = document.querySelector('[data-role="chart-container"]') || 
                              document.querySelector('.chart-container');
        if (chartContainer) {
            observer.observe(chartContainer, {
                childList: true,
                subtree: true,
                attributes: true,
                characterData: true
            });
        }
    }
    
    // Wait for TradingView to load
    function waitForTradingView() {
        if (window.TradingView || document.querySelector('.chart-container')) {
            console.log("✅ TradingView detected, starting monitoring");
            monitorChart();
            
            // Also extract immediately
            setTimeout(extractCandleData, 1000);
        } else {
            setTimeout(waitForTradingView, 1000);
        }
    }
    
    // Start
    waitForTradingView();
    
})();