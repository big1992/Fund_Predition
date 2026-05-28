# 🏦 Thai Equity Fund Prediction System

ระบบวิเคราะห์และพยากรณ์กองทุนรวมหุ้นไทย ด้วย Machine Learning

## 🏗️ Architecture

| Layer | Stack |
|-------|-------|
| **Backend** | Python, FastAPI, SQLite, TensorFlow, XGBoost, AutoGluon |
| **Frontend** | React 19, Vite, Plotly.js |

## ⚡ Quick Start

### 1. Backend

```bash
cd backend

# Create virtual environment
python -m venv venv
venv\Scripts\activate    # Windows
# source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Start server
python -m uvicorn api.main:app --reload --port 8000
```

### 2. Frontend

```bash
cd frontend

# Install dependencies
npm install

# Start dev server
npm run dev
```

### 3. เปิดเบราว์เซอร์

- **Frontend**: http://localhost:5173
- **Backend API Docs**: http://localhost:8000/docs

## 📖 วิธีใช้งาน

1. **Dashboard** → กดปุ่ม "ดาวน์โหลดข้อมูลหุ้น" เพื่อดึงข้อมูลย้อนหลัง 5 ปี
2. **Fund Analysis** → เลือกหุ้น → ดูกราฟ Candlestick + RSI, MACD, Bollinger
3. **Prediction** → กด "Train Models" → จากนั้นกด "Predict" → ดูสัญญาณ BUY/SELL/HOLD
4. **Portfolio** → เลือกหุ้นเข้าพอร์ต → เลือก Method → กด "Optimize"
5. **Model Performance** → ดูผลประเมิน RMSE, MAPE → กด "Run Backtest" → ดู AutoGluon Leaderboard

## 🤖 ML Models

| Model | Type | Features |
|-------|------|----------|
| **LSTM** | Deep Learning (Time Series) | 60-day price sequences |
| **XGBoost** | Gradient Boosting (Tabular) | 33 technical indicators |
| **AutoGluon** | AutoML Ensemble (Tabular) | 33 technical indicators |
| **Ensemble** | Weighted Average | 0.3×LSTM + 0.35×XGBoost + 0.35×AutoGluon |

### AutoGluon — AutoML

AutoGluon จะทดลองหลาย model อัตโนมัติ แล้วเลือก model ที่ดีที่สุด:

| สิ่งที่ AutoGluon ทำ | รายละเอียด |
|---|---|
| **Model Selection** | ทดลอง LightGBM, CatBoost, XGBoost, RandomForest, ExtraTrees, KNN, Neural Net |
| **Hyperparameter Tuning** | ค้นหา hyperparameters ที่ดีที่สุดอัตโนมัติ |
| **Stacking/Ensembling** | สร้าง multi-layer stacking ensemble ภายใน |
| **Leaderboard** | แสดงผลเปรียบเทียบทุก model ที่ลอง |

## 📊 Technical Indicators (33 features)

- **Trend**: SMA, EMA, MACD, ADX
- **Momentum**: RSI, Stochastic, Williams %R, ROC
- **Volatility**: Bollinger Bands, ATR, Historical Volatility
- **Volume**: OBV, VWAP, Volume Ratio
- **Lag Features**: Close/Return lags (1,3,5,10 days)

## 📡 API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /api/health` | Health check |
| `GET /api/stocks` | List stocks |
| `POST /api/stocks/collect` | Download data |
| `GET /api/predictions/{symbol}` | Get prediction |
| `POST /api/predictions/train` | Train models |
| `GET /api/predictions/models/autogluon-leaderboard/{symbol}` | AutoGluon leaderboard |
| `POST /api/portfolio/optimize` | Optimize portfolio |
| `POST /api/backtest/run` | Run backtest |
| `POST /api/ai/explain-prediction` | AI explain prediction |
| `POST /api/ai/explain-performance` | AI explain model metrics |

## 🔑 Environment Variables

Create a `.env` file in the `backend/` directory:

```env
OPENAI_API_KEY=your-api-key-here
```

## Docker Compose

Run full stack with containers:

```bash
docker compose up --build
```

Endpoints:
- Frontend: http://localhost:5173
- Backend: http://localhost:8000
- API Docs: http://localhost:8000/docs

## Environment Template

Create `.env` from template:

```bash
copy backend\.env.example backend\.env
```

## CI Status

[![CI](https://github.com/big1992/Fund_Predition/actions/workflows/ci.yml/badge.svg)](https://github.com/big1992/Fund_Predition/actions/workflows/ci.yml)

## Smoke Test (Compose)

After `docker compose up --build`, run:

```bash
pwsh ./scripts/smoke-compose.ps1
```

Rollback reference:
- `rollback_notes.md`
