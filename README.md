# Feelow 🦈


https://github.com/user-attachments/assets/724ae9c5-8c7e-4534-89c7-e4a3f3e6808f


Personal Finance Agent based on Polymarket Monitoring

## Overview

Feelow analyses prediction markets on [Polymarket](https://polymarket.com) to generate financial insights for any publicly traded company. It combines LLM-powered search with quantitative scoring to surface the most relevant and active markets.

## Project Structure

```
feelow/
├── backend/          # FastAPI server + analysis pipeline
│   ├── src/          # Source code
│   └── tests/        # Unit & integration tests
└── frontend/         # (coming soon)
```

## Backend

The backend exposes a REST API that runs a two-step pipeline:

1. **Agent Search** — Gemini LLM searches Polymarket for prediction markets related to a company
2. **Advanced Scoring** — computes momentum, volatility, concentration, composite signal, and generates LLM-ready summaries

### Quick Start

```bash
# Install dependencies
pip install fastapi uvicorn google-genai mcp pydantic numpy requests

# Run the server
cd backend/src
GEMINI_API_KEY=your_key uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Usage

```bash
curl -X POST http://localhost:8000/get_polymarket \
  -H "Content-Type: application/json" \
  -d '{"company": "NVIDIA", "date": "February 2026", "top_k": 3}'
```

### Tests

```bash
cd backend
python -m pytest tests/ -v
```

See [backend/README.md](backend/README.md) for full API reference and architecture details.


------------ readmAD
# Feelow 🦈
Personal Finance Agent based on Polymarket Monitoring

pour toi le goat : 

cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

cd frontend
pip install -r requirements.txt
streamlit run app.py


Feelow est une plateforme d’intelligence de marché cross-market qui détecte les écarts entre ce que “prédit” le collectif sur les prediction markets (ex. Polymarket) et ce que reflètent les marchés financiers réels (prix actions, volatilité, indicateurs techniques).

L’idée centrale : les prediction markets condensent des croyances et des anticipations (probabilités, volumes, variations rapides). En parallèle, les marchés actions intègrent ces informations avec latence, bruit, ou biais. Feelow fusionne ces signaux pour produire un Market Mispricing Score : l’action semble-t-elle sur-valorisée ou sous-valorisée par rapport à l’engouement et aux attentes implicites du marché “événementiel” ?

## 📊 Features

| Feature | Description | Source Repo |
|---------|-------------|-------------|
| FinBERT Sentiment | Financial text sentiment classification | ProsusAI/finBERT |
| Multi-Model Ensemble | 3 models voting for robust predictions | nickmuchi/finbert-tone, Sigma/financial-SA |
| Real-Time RSS Ingestion | Yahoo Finance + Finviz headlines | nlp-sentiment-quant-monitor |
| Candlestick + Overlay | Price chart with sentiment scatter | nlp-sentiment-quant-monitor |
| Technical Indicators | SMA, EMA, RSI, MACD, Bollinger | nlp-finance-forecast |
| Claude AI Reasoning | Deep analysis combining all signals | Anthropic Claude API |
| Model Comparison | Side-by-side model benchmarking | Custom |

---

## expert Models Used

| Model | HuggingFace ID | F1 Score | Best For |
|-------|---------------|----------|----------|
| **FinBERT (ProsusAI)** | `ProsusAI/finbert` | ~87% | General financial sentiment |
| **FinBERT-Tone** | `nickmuchi/finbert-tone` | ~90% | Tone detection (analyst reports) |
| **Sigma Financial SA** | `Sigma/financial-sentiment-analysis` | ~98% | High-accuracy classification |

---

## ancien Project Structure

```
feelow/
├── app.py                    # Main Streamlit application (5 tabs)
├── config.py                 # Central configuration
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── src/
    ├── __init__.py
    ├── sentiment_engine.py   # Multi-model FinBERT ensemble
    ├── news_ingestor.py      # RSS + Finviz news fetching
    ├── market_data.py        # yfinance price data
    ├── technicals.py         # RSI, MACD, Bollinger, SMA
    ├── visualizer.py         # Plotly charts (8 chart types)
    └── claude_analyst.py     # Claude API integration
```

## 🏆 Hackathon Prize Targeting

- **Best Use of Data (Susquehanna €7K)** — Turns raw news + price data into trading signals
- **Best Use of Gemini (€50K credits)** — Can extend with Gemini multimodal (video/image analysis)
- **Best Stripe Integration (€3K)** — Ready for Stripe Agent Toolkit monetisation layer
- **Fintech Track (€1K)** —

---

## 👥 Team

- **Gabriel Dupuis** — ML Engineer @ Deezer, ENSTA, Stanford
- **Adrien Scazzola** — Security & AI, Microsoft, 
- **Amine Ould** — Development ENS-MVA
- **Tristan Lecourtois** — NASA, Systems Engineering- ENS MVA

---

## License

MIT — Built for HackEurope 2026 with love
