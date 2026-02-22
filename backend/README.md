# Feelow 🦈

**Cross-Market Intelligence Platform — Prediction Markets × Financial Sentiment**

Feelow detects mispricings between what prediction markets (Polymarket) anticipate and what real financial markets (stocks, crypto) reflect. It fuses FinBERT sentiment, technical indicators, and Polymarket signals into a unified **Market Mispricing Score** displayed in a live Next.js dashboard.

---

## 📊 Features

| Feature | Description |
|---------|-------------|
| **FinBERT Ensemble Sentiment** | 3-model voting (ProsusAI/finbert, DistilRoBERTa Financial, Sigma Financial SA) |
| **Reddit Finance Sentiment** | FinBERT scoring on reddit-finance dataset via HuggingFace |
| **Real-Time RSS Ingestion** | Yahoo Finance headlines per ticker |
| **Technical Indicators** | SMA, EMA, RSI, MACD, Bollinger Bands |
| **Polymarket Agent Search** | Gemini LLM searches Polymarket for prediction markets related to any company |
| **Polymarket Scoring** | Momentum, volatility, concentration, composite signal, cross-market correlation |
| **Next.js Dashboard** | Live KPIs, interactive price charts, news table with sentiment, Polymarket panel |

---

## 🏗️ Architecture

```
feelow/
├── backend/                          # FastAPI unified API (port 8000)
│   ├── src/
│   │   ├── main.py                   # FastAPI app — all endpoints
│   │   ├── config.py                 # Central config (models, tickers, thresholds)
│   │   ├── full_pipeline.py          # Polymarket pipeline glue (agent-search → scoring)
│   │   ├── finance-data/             # Core financial modules
│   │   │   ├── sentiment_engine.py   # Multi-model FinBERT ensemble
│   │   │   ├── news_ingestor.py      # RSS headline fetching
│   │   │   ├── market_data.py        # yfinance price data loader
│   │   │   ├── technicals.py         # RSI, MACD, Bollinger, SMA, EMA
│   │   │   ├── gemini_agent.py       # Google Gemini search grounding agent
│   │   │   └── agent_orchestrator.py # Multi-step agentic pipeline orchestrator
│   │   ├── agent_search/             # Polymarket LLM search
│   │   │   ├── polymarket_pipeline.py
│   │   │   ├── orchestrator.py
│   │   │   └── scoring/              # Relevance, impact, novelty, sentiment, reliability
│   │   ├── polymarket-analysis/      # Advanced market scoring
│   │   │   └── market_scorer.py      # Momentum, volatility, concentration, composite signal
│   │   └── stock_analysis/           # Reddit-based FinBERT sentiment
│   │       └── api_finbert_transformer.py
│   └── tests/
└── webapp/
    └── UI-fr/                        # Next.js 15 dashboard (port 3000)
        ├── app/dashboard/page.tsx    # Main dashboard page
        ├── lib/ticker-context.tsx    # Global ticker state + API calls
        └── components/
            ├── section-cards.tsx           # KPI cards (price, sentiment, RSI, signal)
            ├── chart-area-interactive.tsx  # OHLCV price chart + Polymarket panel
            ├── data-table.tsx              # News headlines with sentiment badges
            └── app-sidebar.tsx             # Ticker selector (Tech / Finance / Crypto)
```

---

## 🔄 Webapp Data Pipeline

When the user selects a ticker in the sidebar, the Next.js frontend triggers parallel fetches to four backend endpoints:

```
User selects ticker
        │
        ├─→ GET /api/kpis?ticker=X
        │       MarketDataLoader → yfinance price + 7d change
        │       NewsIngestor → RSS headlines
        │       MultiModelSentimentEngine → avg_sentiment, signal
        │       TechnicalIndicators → RSI
        │       → SectionCards: price, Δ%, news volume, sentiment score, RSI, signal
        │
        ├─→ GET /api/price-history?ticker=X
        │       MarketDataLoader → OHLCV (yfinance)
        │       TechnicalIndicators → SMA20, SMA50, RSI, MACD, Bollinger
        │       → ChartAreaInteractive: area chart with period selector (7d/1mo/3mo/1y)
        │
        ├─→ GET /api/news?ticker=X
        │       NewsIngestor → RSS headlines
        │       MultiModelSentimentEngine → per-headline label + confidence
        │       → DataTable: sortable news feed with sentiment badges
        │
        └─→ GET /api/polymarket?ticker=X
                ticker → company name mapping
                PolymarketPipeline (Gemini) → relevant markets (pertinence 0–100)
                market_scorer → momentum, volatility, concentration, composite signal
                → ChartAreaInteractive: Polymarket panel with top markets + global score
```

`SectionCards` also fires a separate call for community sentiment:
```
POST /api/sentiment/score  { company }
        stock_analysis (Reddit FinBERT) → community sentiment score
        → displayed as "Reddit Sentiment" gauge card
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- `GEMINI_API_KEY` (required)

### 1. Backend (FastAPI)

```bash
cd backend/src
pip install -r requirements.txt

echo "GEMINI_API_KEY=your_key" > ../.env

uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

API docs: [http://localhost:8000/docs](http://localhost:8000/docs)

### 2. Next.js Dashboard

```bash
cd webapp/UI-fr
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

---

## 📡 API Endpoints (used by the dashboard)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/kpis?ticker=X` | Price, pct_change, avg_sentiment, RSI, signal |
| `GET` | `/api/news?ticker=X` | Headlines with per-headline FinBERT sentiment |
| `GET` | `/api/price-history?ticker=X` | OHLCV + SMA, EMA, RSI, MACD, Bollinger |
| `GET` | `/api/polymarket?ticker=X` | Polymarket scored markets for the ticker |
| `POST` | `/api/sentiment/score` | Reddit-based FinBERT community sentiment score |

---

## 🤖 Sentiment Models

| Model | HuggingFace ID | Best For |
|-------|---------------|----------|
| **FinBERT (ProsusAI)** | `ProsusAI/finbert` | General financial sentiment |
| **DistilRoBERTa Financial** | `mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis` | Financial news tone |
| **Sigma Financial SA** | `Sigma/financial-sentiment-analysis` | High-accuracy classification |

---

## 🧪 Tests

```bash
cd backend
python -m pytest tests/ -v
```

| Test file | What's covered |
|-----------|----------------|
| `test_score_polymarket.py` | Helpers, Market class, correlation, scoring pipeline |
| `test_full_pipeline.py` | Format bridge, pertinence normalisation, mocked flow |
| `test_backend.py` | HTTP endpoints, validation, full E2E |

---

## 🏆 Hackathon Tracks

- **Best Use of Data (Susquehanna €7K)** — Fuses RSS news, price data, and prediction markets into trading signals
- **Best Use of Gemini (€50K credits)** — Gemini visual chart analysis + search grounding agent
- **Fintech Track (€1K)** — Cross-market mispricing detection platform

---

## 👥 Team

- **Gabriel Dupuis** — ML Engineer @ Deezer, ENSTA Paris, Stanford
- **Adrien Scazzola** — Security & AI, Microsoft
- **Amine Ould** — Development, ENS-MVA
- **Tristan Lecourtois** — NASA, Systems Engineering, ENS-MVA

---

## License

MIT — Built for HackEurope 2026