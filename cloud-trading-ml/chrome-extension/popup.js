document.addEventListener('DOMContentLoaded', function() {
    // Elements
    const statusEl = document.getElementById('status');
    const predictionsEl = document.getElementById('predictions');
    const modelEl = document.getElementById('model');
    const signalCard = document.getElementById('signalCard');
    const signalText = document.getElementById('signalText');
    const signalConfidence = document.getElementById('signalConfidence');
    const signalTime = document.getElementById('signalTime');
    const signalsList = document.getElementById('signalsList');
    const connectBtn = document.getElementById('connectBtn');
    const testBtn = document.getElementById('testBtn');
    const uploadBtn = document.getElementById('uploadBtn');
    const serverUrlInput = document.getElementById('serverUrl');
    const saveUrlBtn = document.getElementById('saveUrl');
    const versionEl = document.getElementById('version');
    
    // State
    let connected = false;
    let signals = [];
    let predictionCount = 0;
    let currentModel = 'v1.0';
    
    // Initialize
    init();
    
    async function init() {
        // Load saved server URL
        chrome.storage.local.get(['serverUrl'], (result) => {
            if (result.serverUrl) {
                serverUrlInput.value = result.serverUrl;
            } else {
                // Default to localhost for development
                serverUrlInput.value = 'ws://localhost:8000/ws';
            }
        });
        
        // Get initial status from content script
        chrome.tabs.query({active: true, currentWindow: true}, (tabs) => {
            if (tabs[0] && tabs[0].url.includes('tradingview.com')) {
                chrome.tabs.sendMessage(tabs[0].id, {action: 'getStatus'}, (response) => {
                    if (response) {
                        updateStatus(response.connected);
                        if (response.lastSignal) {
                            updateCurrentSignal(response.lastSignal);
                        }
                        if (response.serverUrl) {
                            serverUrlInput.value = response.serverUrl;
                        }
                    }
                });
            }
        });
        
        // Listen for new signals from content script
        chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
            if (message.type === 'newSignal') {
                addSignal(message.signal);
                updateCurrentSignal(message.signal);
            }
        });
        
        // Set version
        versionEl.textContent = `v${chrome.runtime.getManifest().version}`;
        
        // Setup event listeners
        setupEventListeners();
    }
    
    function setupEventListeners() {
        connectBtn.addEventListener('click', () => {
            if (connected) {
                disconnect();
            } else {
                connect();
            }
        });
        
        testBtn.addEventListener('click', () => {
            // Send test signal request to content script
            chrome.tabs.query({active: true, currentWindow: true}, (tabs) => {
                if (tabs[0] && tabs[0].url.includes('tradingview.com')) {
                    chrome.tabs.sendMessage(tabs[0].id, {action: 'forceSignal'});
                }
            });
            
            // Also create a test signal locally
            const testSignal = {
                signal: Math.random() > 0.5 ? 'BUY' : 'SELL',
                confidence: Math.random() * 0.3 + 0.6,
                timestamp: new Date().toISOString(),
                model_version: 'test',
                arrow_color: Math.random() > 0.5 ? '#00ff00' : '#ff0000'
            };
            addSignal(testSignal);
            updateCurrentSignal(testSignal);
        });
        
        uploadBtn.addEventListener('click', () => {
            // Create file input for uploading historical trades
            const input = document.createElement('input');
            input.type = 'file';
            input.accept = '.json';
            
            input.onchange = async (e) => {
                const file = e.target.files[0];
                if (!file) return;
                
                try {
                    const text = await file.text();
                    const trades = JSON.parse(text);
                    
                    // Send to backend
                    const serverUrl = serverUrlInput.value.replace('ws://', 'https://').replace('/ws', '');
                    
                    const response = await fetch(`${serverUrl}/upload/historical`, {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: text
                    });
                    
                    const result = await response.json();
                    
                    if (result.status === 'success') {
                        alert(`✅ Uploaded ${trades.length} trades!\nTraining started...`);
                    } else {
                        alert(`⚠️ Upload failed: ${result.message}`);
                    }
                    
                } catch (error) {
                    alert('❌ Error uploading file: ' + error.message);
                }
            };
            
            input.click();
        });
        
        saveUrlBtn.addEventListener('click', () => {
            const url = serverUrlInput.value.trim();
            
            if (!url.startsWith('ws://') && !url.startsWith('wss://')) {
                alert('❌ URL must start with ws:// or wss://');
                return;
            }
            
            // Save to storage
            chrome.storage.local.set({ serverUrl: url }, () => {
                alert('✅ Server URL saved!');
                
                // Update in content script
                chrome.tabs.query({active: true, currentWindow: true}, (tabs) => {
                    if (tabs[0] && tabs[0].url.includes('tradingview.com')) {
                        chrome.tabs.sendMessage(tabs[0].id, {
                            action: 'setServerUrl',
                            url: url
                        }, (response) => {
                            if (response && response.success) {
                                updateStatus(false); // Will reconnect automatically
                            }
                        });
                    }
                });
            });
        });
        
        // Auto-save on Enter
        serverUrlInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                saveUrlBtn.click();
            }
        });
    }
    
    function connect() {
        const url = serverUrlInput.value.trim();
        
        if (!url) {
            alert('Please enter a server URL');
            return;
        }
        
        // Update content script with new URL
        chrome.tabs.query({active: true, currentWindow: true}, (tabs) => {
            if (tabs[0] && tabs[0].url.includes('tradingview.com')) {
                chrome.tabs.sendMessage(tabs[0].id, {
                    action: 'setServerUrl',
                    url: url
                }, (response) => {
                    if (response && response.success) {
                        updateStatus(true);
                        connectBtn.textContent = 'Disconnect';
                    }
                });
            }
        });
    }
    
    function disconnect() {
        // Just update UI, content script will handle disconnection
        updateStatus(false);
        connectBtn.textContent = 'Connect';
    }
    
    function updateStatus(isConnected) {
        connected = isConnected;
        statusEl.textContent = isConnected ? 'Connected' : 'Disconnected';
        statusEl.className = isConnected ? 'status-value connected' : 'status-value disconnected';
        connectBtn.textContent = isConnected ? 'Disconnect' : 'Connect';
        
        // Update predictions count
        if (isConnected) {
            predictionCount++;
            predictionsEl.textContent = predictionCount;
        }
    }
    
    function updateCurrentSignal(signal) {
        // Update signal card
        signalText.textContent = signal.signal;
        signalConfidence.textContent = `${(signal.confidence * 100).toFixed(1)}%`;
        signalTime.textContent = new Date(signal.timestamp).toLocaleTimeString();
        
        // Update colors and class
        signalCard.className = `signal-card signal-${signal.signal.toLowerCase()}`;
        signalText.style.color = signal.arrow_color || 
            (signal.signal === 'BUY' ? '#4ade80' : 
             signal.signal === 'SELL' ? '#f87171' : '#fbbf24');
        
        // Update model version
        if (signal.model_version) {
            modelEl.textContent = signal.model_version;
            currentModel = signal.model_version;
        }
    }
    
    function addSignal(signal) {
        // Add to beginning of array
        signals.unshift(signal);
        
        // Keep only last 10
        if (signals.length > 10) {
            signals.pop();
        }
        
        // Update list
        updateSignalsList();
    }
    
    function updateSignalsList() {
        signalsList.innerHTML = '';
        
        if (signals.length === 0) {
            signalsList.innerHTML = `
                <div class="signal-item signal-item-hold">
                    <div>No signals yet...</div>
                </div>
            `;
            return;
        }
        
        signals.forEach((signal, index) => {
            const item = document.createElement('div');
            item.className = `signal-item signal-item-${signal.signal.toLowerCase()}`;
            
            const time = new Date(signal.timestamp);
            const timeStr = time.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
            
            item.innerHTML = `
                <div style="display: flex; justify-content: space-between;">
                    <span style="font-weight: bold;">${signal.signal}</span>
                    <span style="color: #cbd5e1;">${(signal.confidence * 100).toFixed(1)}%</span>
                </div>
                <div style="font-size: 10px; color: #94a3b8; margin-top: 2px;">
                    ${timeStr} • ${signal.model_version || 'v1.0'}
                </div>
            `;
            
            signalsList.appendChild(item);
        });
    }
    
    // Auto-refresh every 5 seconds
    setInterval(() => {
        if (connected) {
            // Update time for current signal
            if (signals.length > 0) {
                const currentSignal = signals[0];
                signalTime.textContent = new Date(currentSignal.timestamp).toLocaleTimeString();
            }
        }
    }, 5000);
});