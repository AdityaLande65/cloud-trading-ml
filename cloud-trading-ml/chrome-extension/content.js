// Cloud ML Trading Extension - Content Script
// This runs on TradingView pages

console.log("✅ Cloud ML Trading Extension loaded");

class TradingSignalClient {
    constructor() {
        this.ws = null;
        this.serverUrl = null;
        this.connected = false;
        this.signals = [];
        this.lastCandle = null;
        
        // Initialize
        this.init();
    }
    
    async init() {
        // Load server URL from storage
        chrome.storage.local.get(['serverUrl'], (result) => {
            this.serverUrl = result.serverUrl || "ws://localhost:8000/ws";
            this.connectWebSocket();
        });
        
        // Inject the data extraction script
        this.injectDataExtractor();
        
        // Listen for messages from the injected script
        window.addEventListener('message', this.handleInjectedMessage.bind(this));
        
        // Listen for extension messages
        chrome.runtime.onMessage.addListener(this.handleExtensionMessage.bind(this));
        
        console.log("✅ TradingSignalClient initialized");
    }
    
    injectDataExtractor() {
        // Inject script into page context
        const script = document.createElement('script');
        script.src = chrome.runtime.getURL('injection.js');
        script.onload = () => script.remove();
        (document.head || document.documentElement).appendChild(script);
        
        console.log("✅ Data extractor injected");
    }
    
    connectWebSocket() {
        if (this.ws) {
            this.ws.close();
        }
        
        this.ws = new WebSocket(this.serverUrl);
        
        this.ws.onopen = () => {
            this.connected = true;
            console.log("✅ Connected to Cloud ML Server:", this.serverUrl);
            this.updateStatus("connected");
        };
        
        this.ws.onmessage = (event) => {
            try {
                const signal = JSON.parse(event.data);
                this.handleSignal(signal);
            } catch (e) {
                console.error("WebSocket parse error:", e);
            }
        };
        
        this.ws.onerror = (error) => {
            console.error("WebSocket error:", error);
            this.connected = false;
            this.updateStatus("error");
        };
        
        this.ws.onclose = () => {
            console.log("WebSocket disconnected, reconnecting...");
            this.connected = false;
            this.updateStatus("disconnected");
            
            // Reconnect after 5 seconds
            setTimeout(() => this.connectWebSocket(), 5000);
        };
    }
    
    handleInjectedMessage(event) {
        // Only accept messages from our own window
        if (event.source !== window) return;
        
        if (event.data.type === 'TRADINGVIEW_CANDLE') {
            this.processCandle(event.data.candle);
        } else if (event.data.type === 'TRADINGVIEW_READY') {
            console.log("✅ TradingView data extractor ready");
        }
    }
    
    handleExtensionMessage(request, sender, sendResponse) {
        if (request.action === 'getStatus') {
            sendResponse({
                connected: this.connected,
                serverUrl: this.serverUrl,
                lastSignal: this.signals[0]
            });
        } else if (request.action === 'setServerUrl') {
            this.serverUrl = request.url;
            chrome.storage.local.set({ serverUrl: request.url });
            this.connectWebSocket();
            sendResponse({ success: true });
        } else if (request.action === 'forceSignal') {
            this.createTestSignal();
            sendResponse({ success: true });
        }
        
        return true; // Keep message channel open for async response
    }
    
    processCandle(candle) {
        this.lastCandle = candle;
        
        // Store in buffer (last 50 candles)
        if (!window.candleBuffer) window.candleBuffer = [];
        window.candleBuffer.push(candle);
        if (window.candleBuffer.length > 50) {
            window.candleBuffer.shift();
        }
        
        // Extract features
        const features = this.extractFeatures(candle, window.candleBuffer);
        
        // Send to ML server if connected
        if (this.connected && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({
                candles: [candle],
                features: features
            }));
        }
        
        // Update UI
        this.updateCandleDisplay(candle);
    }
    
    extractFeatures(candle, buffer) {
        const features = [];
        
        // 1. RSI (normalized)
        features.push((candle.rsi || 50) / 100);
        
        // 2. Price change (last 2 candles)
        if (buffer.length >= 2) {
            const prevCandle = buffer[buffer.length - 2];
            const change = (candle.close - prevCandle.close) / prevCandle.close;
            features.push(Math.min(Math.max(change, -0.1), 0.1) / 0.1); // Normalize ±10%
        } else {
            features.push(0);
        }
        
        // 3. Volume ratio
        if (buffer.length >= 5) {
            const recentVolumes = buffer.slice(-5).map(c => c.volume);
            const avgVolume = recentVolumes.reduce((a, b) => a + b, 0) / recentVolumes.length;
            const volumeRatio = candle.volume / avgVolume;
            features.push(Math.min(volumeRatio, 3) / 3); // Cap at 3x
        } else {
            features.push(1);
        }
        
        // 4. Time feature
        const hour = new Date(candle.timestamp).getHours();
        features.push(hour / 23);
        
        return features;
    }
    
    handleSignal(signal) {
        console.log("📡 ML Signal:", signal);
        
        // Add to history
        this.signals.unshift({
            ...signal,
            receivedAt: new Date().toISOString()
        });
        
        if (this.signals.length > 10) {
            this.signals.pop();
        }
        
        // Draw on chart
        this.drawSignalOnChart(signal);
        
        // Update popup
        this.updatePopup(signal);
        
        // Play sound (optional)
        if (signal.confidence > 0.6 && signal.signal !== 'HOLD') {
            this.playSignalSound(signal.signal);
        }
    }
    
    drawSignalOnChart(signal) {
        // Create signal indicator
        const indicatorId = `ml-signal-${Date.now()}`;
        const indicator = document.createElement('div');
        
        indicator.id = indicatorId;
        indicator.className = 'ml-trading-signal';
        indicator.style.cssText = `
            position: absolute;
            top: 100px;
            right: 20px;
            background: ${signal.signal === 'BUY' ? 'rgba(0,255,0,0.15)' : 
                        signal.signal === 'SELL' ? 'rgba(255,0,0,0.15)' : 
                        'rgba(255,255,0,0.15)'};
            border: 2px solid ${signal.arrow_color || 
                              (signal.signal === 'BUY' ? '#00ff00' : 
                               signal.signal === 'SELL' ? '#ff0000' : '#ffff00')};
            border-radius: 8px;
            padding: 12px 15px;
            color: white;
            font-family: Arial, sans-serif;
            font-size: 14px;
            font-weight: bold;
            z-index: 99999;
            pointer-events: none;
            backdrop-filter: blur(5px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            min-width: 150px;
        `;
        
        indicator.innerHTML = `
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 5px;">
                <span style="font-size: 16px;">
                    ${signal.signal === 'BUY' ? '🟢 BUY' : 
                      signal.signal === 'SELL' ? '🔴 SELL' : '🟡 HOLD'}
                </span>
                <span style="font-size: 16px;">${(signal.confidence * 100).toFixed(1)}%</span>
            </div>
            <div style="font-size: 11px; color: #ccc; text-align: center;">
                ${new Date(signal.timestamp).toLocaleTimeString()}
            </div>
        `;
        
        // Remove old indicators
        document.querySelectorAll('.ml-trading-signal').forEach(el => {
            if (el.id !== indicatorId) el.remove();
        });
        
        // Add to chart container
        const chartContainer = document.querySelector('[data-role="chart-container"]') || 
                              document.querySelector('.chart-container') || 
                              document.body;
        chartContainer.appendChild(indicator);
        
        // Auto-remove after 30 seconds
        setTimeout(() => {
            const el = document.getElementById(indicatorId);
            if (el) el.remove();
        }, 30000);
    }
    
    updatePopup(signal) {
        // Send message to popup
        chrome.runtime.sendMessage({
            type: 'newSignal',
            signal: signal
        });
    }
    
    updateCandleDisplay(candle) {
        // Update candle info display
        let display = document.getElementById('ml-candle-display');
        
        if (!display) {
            display = document.createElement('div');
            display.id = 'ml-candle-display';
            display.style.cssText = `
                position: fixed;
                bottom: 20px;
                right: 20px;
                background: rgba(30,30,30,0.9);
                color: white;
                padding: 10px;
                border-radius: 5px;
                font-family: Arial;
                font-size: 11px;
                z-index: 99998;
                border: 1px solid #444;
                min-width: 150px;
            `;
            document.body.appendChild(display);
        }
        
        display.innerHTML = `
            <div style="margin-bottom: 5px; color: #aaa;">Last Candle:</div>
            <div>O: ${candle.open.toFixed(5)}</div>
            <div>H: ${candle.high.toFixed(5)}</div>
            <div>L: ${candle.low.toFixed(5)}</div>
            <div>C: <span style="color: ${candle.close >= candle.open ? '#0f0' : '#f00'}">
                ${candle.close.toFixed(5)}
            </span></div>
            <div>RSI: ${candle.rsi ? candle.rsi.toFixed(1) : 'N/A'}</div>
            <div style="font-size: 9px; color: #888; margin-top: 5px;">
                ${new Date(candle.timestamp).toLocaleTimeString()}
            </div>
        `;
    }
    
    updateStatus(status) {
        // Update connection status display
        let statusEl = document.getElementById('ml-connection-status');
        
        if (!statusEl) {
            statusEl = document.createElement('div');
            statusEl.id = 'ml-connection-status';
            statusEl.style.cssText = `
                position: fixed;
                top: 20px;
                right: 20px;
                padding: 8px 12px;
                border-radius: 5px;
                font-family: Arial;
                font-size: 12px;
                font-weight: bold;
                z-index: 99997;
                background: #333;
                color: white;
            `;
            document.body.appendChild(statusEl);
        }
        
        const statusMap = {
            connected: { text: '✅ ML Connected', color: '#0f0' },
            disconnected: { text: '❌ ML Disconnected', color: '#f00' },
            error: { text: '⚠️ ML Error', color: '#ff0' }
        };
        
        const statusInfo = statusMap[status] || { text: 'ML Unknown', color: '#888' };
        statusEl.textContent = statusInfo.text;
        statusEl.style.border = `2px solid ${statusInfo.color}`;
        statusEl.style.background = statusInfo.color.replace(')', ', 0.1)').replace('rgb', 'rgba');
    }
    
    playSignalSound(type) {
        // Create audio context for beep sounds
        try {
            const audioContext = new (window.AudioContext || window.webkitAudioContext)();
            const oscillator = audioContext.createOscillator();
            const gainNode = audioContext.createGain();
            
            oscillator.connect(gainNode);
            gainNode.connect(audioContext.destination);
            
            oscillator.frequency.value = type === 'BUY' ? 800 : 400; // Higher for BUY
            oscillator.type = 'sine';
            
            gainNode.gain.setValueAtTime(0.1, audioContext.currentTime);
            gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.3);
            
            oscillator.start(audioContext.currentTime);
            oscillator.stop(audioContext.currentTime + 0.3);
        } catch (e) {
            console.log("Audio not supported");
        }
    }
    
    createTestSignal() {
        // For testing without real data
        const testSignal = {
            signal: Math.random() > 0.5 ? 'BUY' : 'SELL',
            confidence: Math.random() * 0.3 + 0.6,
            timestamp: new Date().toISOString(),
            model_version: 'test',
            arrow_color: Math.random() > 0.5 ? '#00ff00' : '#ff0000'
        };
        
        this.handleSignal(testSignal);
    }
}

// Initialize when page is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        new TradingSignalClient();
    });
} else {
    new TradingSignalClient();
}