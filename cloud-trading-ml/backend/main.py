"""
CLOUD ML TRADING SERVER - 100% Cloud Based
Deploy to Render.com (Free Forever)
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import asyncio
import logging
import hashlib
from sklearn.preprocessing import StandardScaler
# ===================== CONFIGURATION =====================
class Config:
    # Cloud paths (Render provides /tmp for free tier)
    MODEL_DIR = "/tmp/models"
    DATA_DIR = "/tmp/data"
    PORT = int(os.getenv("PORT", 8000))
    
    # ML Settings
    ML_CONFIDENCE_THRESHOLD = 0.52
    MIN_TRADES_FOR_TRAINING = 50
    
    # WebSocket settings
    MAX_CONNECTIONS = 1000
    
    # Signal colors
    COLORS = {
        "BUY": "#00ff00",
        "SELL": "#ff0000",
        "HOLD": "#ffff00"
    }

# Create directories
os.makedirs(Config.MODEL_DIR, exist_ok=True)
os.makedirs(Config.DATA_DIR, exist_ok=True)

# ===================== DATA MODELS =====================
class CandleData(BaseModel):
    symbol: str = "EURUSD"
    timeframe: str = "1m"
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float = 0
    rsi: Optional[float] = None
    indicators: Optional[Dict] = {}

class HistoricalTrade(BaseModel):
    timestamp: str
    symbol: str
    action: str
    result: str
    profit: float
    rsi: Optional[float] = None
    features: Optional[List[float]] = []

class SignalResponse(BaseModel):
    signal: str  # "BUY", "SELL", "HOLD"
    confidence: float
    timestamp: str
    model_version: str
    features_used: List[str]
    arrow_color: str

# ===================== ML ENGINE =====================
class CloudTradingML:
    """Cloud-based trading ML model"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.model_version = "v1.0-cloud"
        self.feature_names = ['rsi', 'price_change', 'volume_ratio', 'hour']
        
        # Statistics
        self.stats = {
            "total_predictions": 0,
            "total_trades": 0,
            "win_rate": 0.0,
            "last_trained": None,
            "uptime": datetime.now().isoformat()
        }
        
        # Load or initialize model
        self.load_or_initialize_model()
        
        # WebSocket connections
        self.active_connections = []
        
        print(f"✅ Cloud Trading ML v{self.model_version} initialized")
    
    def load_or_initialize_model(self):
        """Load model or create new one"""
        model_path = os.path.join(Config.MODEL_DIR, "model.pkl")
        scaler_path = os.path.join(Config.MODEL_DIR, "scaler.pkl")
        
        try:
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                self.model = joblib.load(model_path)
                self.scaler = joblib.load(scaler_path)
                print(f"✅ Loaded existing model from {model_path}")
                return True
        except Exception as e:
            print(f"⚠️ Error loading model: {e}")
        
        # Initialize new model
        print("🔄 Initializing new model...")
        self.initialize_model()
        return False
    
    def initialize_model(self):
        """Create initial model"""
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.preprocessing import StandardScaler
        
        # Create dummy training data
        np.random.seed(42)
        X_train = np.random.randn(100, 4)
        y_train = np.random.randint(0, 2, 100)
        
        # Initialize and train
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_train)
        
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=42
        )
        self.model.fit(X_scaled, y_train)
        
        # Save
        self.save_model()
        print("✅ New model created and saved")
    
    def save_model(self):
        """Save model to cloud storage"""
        model_path = os.path.join(Config.MODEL_DIR, "model.pkl")
        scaler_path = os.path.join(Config.MODEL_DIR, "scaler.pkl")
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        
        # Save versioned copy
        version_path = os.path.join(
            Config.MODEL_DIR, 
            f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        )
        joblib.dump(self.model, version_path)
    
    def extract_features(self, candle: CandleData, previous_candles: List[CandleData]) -> List[float]:
        """Extract features for ML prediction"""
        features = []
        
        # 1. RSI (normalized 0-1)
        rsi = candle.rsi if candle.rsi else 50.0
        features.append(rsi / 100.0)  # Normalize to 0-1
        
        # 2. Price change percentage
        if previous_candles and len(previous_candles) > 0:
            prev_close = previous_candles[-1].close
            price_change = ((candle.close - prev_close) / prev_close) * 100
            # Normalize: assume max ±10% change
            features.append(min(max(price_change / 10.0, -1.0), 1.0))
        else:
            features.append(0.0)
        
        # 3. Volume ratio (current vs average of last 5)
        if len(previous_candles) >= 5:
            recent_volumes = [c.volume for c in previous_candles[-5:]]
            avg_volume = np.mean(recent_volumes) if recent_volumes else 1.0
            volume_ratio = candle.volume / avg_volume if avg_volume > 0 else 1.0
            # Cap at 3x for normalization
            features.append(min(volume_ratio / 3.0, 1.0))
        else:
            features.append(1.0)
        
        # 4. Time feature (market session)
        try:
            hour = datetime.fromisoformat(candle.timestamp.replace('Z', '+00:00')).hour
            # Normalize hour (0-23 to 0-1)
            features.append(hour / 23.0)
        except:
            features.append(0.5)
        
        return features
    
    def predict(self, features: List[float]) -> Dict:
        """Make trading prediction"""
        try:
            # Update stats
            self.stats["total_predictions"] += 1
            
            # Scale features
            X = np.array(features).reshape(1, -1)
            X_scaled = self.scaler.transform(X)
            
            # Predict
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(X_scaled)[0]
                confidence = max(probabilities)
                
                # Class 1 is BUY, Class 0 is SELL
                if probabilities[1] > probabilities[0]:
                    signal = "BUY" if confidence > Config.ML_CONFIDENCE_THRESHOLD else "HOLD"
                else:
                    signal = "SELL" if confidence > Config.ML_CONFIDENCE_THRESHOLD else "HOLD"
                
                # Adjust confidence for HOLD signals
                if signal == "HOLD":
                    confidence = 1.0 - confidence
            else:
                prediction = self.model.predict(X_scaled)[0]
                signal = "BUY" if prediction == 1 else "SELL"
                confidence = 0.6  # Default
            
            return {
                "signal": signal,
                "confidence": float(confidence),
                "arrow_color": Config.COLORS.get(signal, "#ffffff")
            }
            
        except Exception as e:
            print(f"⚠️ Prediction error: {e}")
            return {
                "signal": "HOLD",
                "confidence": 0.0,
                "arrow_color": Config.COLORS["HOLD"]
            }
    
    async def retrain_with_data(self, trades: List[HistoricalTrade]):
        """Retrain model with new trade data"""
        if len(trades) < Config.MIN_TRADES_FOR_TRAINING:
            return {"status": "insufficient_data", "trades": len(trades)}
        
        try:
            # Prepare training data
            X = []
            y = []
            
            for trade in trades:
                if trade.features and len(trade.features) == 4:  # Our 4 features
                    X.append(trade.features)
                    # Convert WIN/LOSS to 1/0
                    y.append(1 if trade.result == "WIN" else 0)
            
            if len(X) < Config.MIN_TRADES_FOR_TRAINING:
                return {"status": "insufficient_features", "samples": len(X)}
            
            # Convert to numpy
            X_array = np.array(X)
            y_array = np.array(y)
            
            # Scale features
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X_array)
            
            # Train new model
            from sklearn.ensemble import GradientBoostingClassifier
            self.model = GradientBoostingClassifier(
                n_estimators=150,
                max_depth=4,
                learning_rate=0.05,
                random_state=42
            )
            self.model.fit(X_scaled, y_array)
            
            # Update version and save
            self.model_version = f"v1.{self.stats['total_trades']}-retrained"
            self.save_model()
            
            # Update stats
            self.stats["last_trained"] = datetime.now().isoformat()
            
            return {
                "status": "success",
                "trades_used": len(X),
                "model_version": self.model_version,
                "win_rate": float(np.mean(y_array))
            }
            
        except Exception as e:
            print(f"⚠️ Training error: {e}")
            return {"status": "error", "message": str(e)}
    
    async def broadcast_signal(self, signal: Dict):
        """Broadcast signal to all connected clients"""
        for connection in self.active_connections:
            try:
                await connection.send_json(signal)
            except:
                # Remove disconnected clients
                self.active_connections.remove(connection)

# ===================== FASTAPI APP =====================
app = FastAPI(
    title="Cloud Trading ML Server",
    description="100% Cloud-Based Trading Intelligence System",
    version="1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global ML Engine
ml_engine = CloudTradingML()

# ===================== API ENDPOINTS =====================
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Cloud Trading ML Server",
        "status": "active",
        "version": "1.0",
        "model_version": ml_engine.model_version,
        "uptime": ml_engine.stats["uptime"],
        "docs": "/docs",
        "websocket": "/ws",
        "upload": "/upload/historical"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model": "loaded" if ml_engine.model else "not_loaded",
        "connections": len(ml_engine.active_connections)
    }

@app.get("/stats")
async def get_stats():
    """Get ML engine statistics"""
    return {
        "model_version": ml_engine.model_version,
        "stats": ml_engine.stats,
        "feature_names": ml_engine.feature_names,
        "confidence_threshold": Config.ML_CONFIDENCE_THRESHOLD
    }

@app.post("/predict")
async def predict(request: Dict):
    """HTTP endpoint for predictions"""
    try:
        candles = [CandleData(**c) for c in request.get("candles", [])]
        features = request.get("features", [])
        
        if not features and candles:
            features = ml_engine.extract_features(candles[-1], candles[:-1])
        
        prediction = ml_engine.predict(features)
        
        response = SignalResponse(
            signal=prediction["signal"],
            confidence=prediction["confidence"],
            timestamp=datetime.now().isoformat(),
            model_version=ml_engine.model_version,
            features_used=ml_engine.feature_names,
            arrow_color=prediction["arrow_color"]
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload/historical")
async def upload_historical(file: UploadFile = File(...)):
    """Upload historical trades (your 6000+ trades)"""
    try:
        content = await file.read()
        trades_data = json.loads(content)
        
        # Validate and convert
        trades = []
        for item in trades_data:
            trade = HistoricalTrade(**item)
            trades.append(trade)
        
        # Save to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(Config.DATA_DIR, f"trades_{timestamp}.json")
        
        with open(file_path, 'w') as f:
            json.dump([t.dict() for t in trades], f)
        
        # Start retraining in background
        asyncio.create_task(ml_engine.retrain_with_data(trades))
        
        return {
            "status": "success",
            "message": f"Uploaded {len(trades)} trades",
            "file": file_path,
            "training_started": True
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train")
async def trigger_training():
    """Manually trigger model training"""
    # Load latest trades
    trade_files = [f for f in os.listdir(Config.DATA_DIR) if f.endswith('.json')]
    
    if not trade_files:
        return {"status": "error", "message": "No trade data available"}
    
    # Load most recent file
    latest_file = max(trade_files)
    with open(os.path.join(Config.DATA_DIR, latest_file), 'r') as f:
        trades_data = json.load(f)
    
    trades = [HistoricalTrade(**item) for item in trades_data]
    
    # Train
    result = await ml_engine.retrain_with_data(trades)
    
    return result

# ===================== WEBSOCKET ENDPOINT =====================
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time signals"""
    await websocket.accept()
    ml_engine.active_connections.append(websocket)
    
    try:
        while True:
            # Wait for data from client
            data = await websocket.receive_json()
            
            # Extract candles
            candles = [CandleData(**c) for c in data.get("candles", [])]
            features = data.get("features", [])
            
            if not features and candles:
                features = ml_engine.extract_features(candles[-1], candles[:-1])
            
            # Get prediction
            prediction = ml_engine.predict(features)
            
            # Prepare response
            response = {
                "signal": prediction["signal"],
                "confidence": prediction["confidence"],
                "timestamp": datetime.now().isoformat(),
                "model_version": ml_engine.model_version,
                "arrow_color": prediction["arrow_color"],
                "features": features
            }
            
            # Send back to client
            await websocket.send_json(response)
            
            # Broadcast to all other connections (for dashboards)
            await ml_engine.broadcast_signal({
                "type": "broadcast_signal",
                "data": response,
                "timestamp": datetime.now().isoformat()
            })
            
    except WebSocketDisconnect:
        ml_engine.active_connections.remove(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        ml_engine.active_connections.remove(websocket)

# ===================== STARTUP EVENT =====================
@app.on_event("startup")
async def startup_event():
    """Startup event"""
    print("🚀 Cloud Trading ML Server Starting...")
    print(f"✅ Model Version: {ml_engine.model_version}")
    print(f"✅ Features: {ml_engine.feature_names}")
    print(f"✅ WebSocket: /ws")
    print(f"✅ API Docs: /docs")
    print(f"✅ Ready to accept connections!")

# ===================== MAIN =====================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=Config.PORT,
        reload=True
    )