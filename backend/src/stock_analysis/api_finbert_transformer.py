from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
from typing import Optional

# ─── Load Reddit finance dataset ─────────────────────────────
ds = load_dataset("aurelio-ai/reddit-finance", split="train")
df = pd.DataFrame(ds)
df = df[['selftext']].dropna()  # only keep selftext

# ─── Load FinBERT model ──────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(
    "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
)
model = AutoModelForSequenceClassification.from_pretrained(
    "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
)
model.eval()  # evaluation mode

# ─── Function to compute sentiment score ─────────────────────
def compute_sentiment_score(company_name: str, top_k: Optional[int] = 200) -> float:
    """
    Compute average sentiment score for news mentioning the company_name.
    Returns a score between -1 (negative) and 1 (positive).
    """
    # Filter news mentioning the company
    headlines = df[df['selftext'].str.contains(company_name, case=False, na=False)]['selftext'].tolist()
    if not headlines:
        return 0.0

    headlines = headlines[:top_k]
    scores = []
    for text in headlines:
        inputs = tokenizer(text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            logits = model(**inputs).logits
            predicted_class = torch.argmax(logits, dim=1).item()
        # Map 0->-1, 1->0, 2->1
        score = predicted_class - 1
        scores.append(score)

    return float(sum(scores) / len(scores))