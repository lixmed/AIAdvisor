"""
main.py — FastAPI service for the Buy-Signal model (ORIGINAL Phase 2 model,
no Phase-3 experimental changes included).

Run locally:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    GET  /health           -> service health check
    POST /predict          -> given a list of tickers, returns buy probability
                               + conviction level for each, using the LATEST
                               available trading day of data.
"""

from typing import List, Optional
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from model_utils import build_features, load_artifacts, predict_buy_probability, SECTOR_MAP

app = FastAPI(
    title="Buy-Signal Model API",
    description="Serves buy-probability predictions from the trained XGBoost ensemble (Phase 2 / original model).",
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# Load model artifacts once, at startup (not per-request).
# ---------------------------------------------------------------------------
ENSEMBLE, FEATURES, THRESH_HP, THRESH_BAL = load_artifacts(model_dir=".")

# Minimum trading days of history needed so all rolling features (longest = 200d SMA) are valid.
MIN_HISTORY_DAYS = 400  # buffer above 200 trading days to be safe (weekends/holidays)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    tickers: List[str] = Field(..., example=["AAPL", "MSFT", "TSLA"], description="List of stock tickers to score.")


class TickerPrediction(BaseModel):
    ticker: str
    sector: int
    date: str
    buy_probability: float
    conviction: str  # "high" | "moderate" | "none"


class PredictResponse(BaseModel):
    predictions: List[TickerPrediction]
    model_version: str = "phase2_original"
    threshold_high_conviction: float
    threshold_moderate_conviction: float
    notes: str = (
        "buy_probability estimates the likelihood the stock rises more than 5% "
        "within the next 10 trading days. This is a probabilistic signal, not a "
        "guarantee. Model Test AUC ~0.63 — treat as one input among many, not a "
        "standalone investment decision."
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fetch_price_history(tickers: List[str]) -> pd.DataFrame:
    start_date = (datetime.today() - timedelta(days=int(MIN_HISTORY_DAYS * 1.6))).strftime("%Y-%m-%d")
    frames = []
    for ticker in tickers:
        t = yf.download(ticker, start=start_date, auto_adjust=True, progress=False)
        if t.empty:
            continue
        t = t.reset_index()
        if isinstance(t.columns, pd.MultiIndex):
            t.columns = t.columns.get_level_values(0)
        t['Ticker'] = ticker
        t['Sector'] = SECTOR_MAP.get(ticker, 7)
        frames.append(t)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df.columns = [str(c).capitalize() if c not in ['Ticker', 'Sector'] else c for c in df.columns]
    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
    df = df.drop_duplicates(['Ticker', 'Date'])
    df = df[(df['Low'] <= df['Open']) & (df['Open'] <= df['High']) &
            (df['Low'] <= df['Close']) & (df['Close'] <= df['High']) & (df['Volume'] >= 0)]
    df.sort_values(['Ticker', 'Date'], inplace=True)
    return df


def fetch_market_data() -> list:
    start_date = (datetime.today() - timedelta(days=int(MIN_HISTORY_DAYS * 1.6))).strftime("%Y-%m-%d")
    mkt_frames = []
    for sym, col in [('^GSPC', 'spy_close'), ('^VIX', 'vix_level')]:
        tmp = yf.download(sym, start=start_date, auto_adjust=True, progress=False)
        tmp = tmp.reset_index()
        if isinstance(tmp.columns, pd.MultiIndex):
            tmp.columns = tmp.columns.get_level_values(0)
        tmp.columns = [str(c).lower() for c in tmp.columns]
        tmp['date'] = pd.to_datetime(tmp['date']).dt.tz_localize(None)
        tmp = tmp[['date', 'close']].rename(columns={'date': 'Date', 'close': col}).sort_values('Date')
        tmp[col] = tmp[col].ffill()
        mkt_frames.append(tmp)
    return mkt_frames


def conviction_label(prob: float) -> str:
    if prob >= THRESH_HP:
        return "high"
    elif prob >= THRESH_BAL:
        return "moderate"
    return "none"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "model_version": "phase2_original", "features_count": len(FEATURES)}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not req.tickers:
        raise HTTPException(status_code=400, detail="tickers list cannot be empty")

    price_df = fetch_price_history(req.tickers)
    if price_df.empty:
        raise HTTPException(status_code=404, detail="No price data found for the given tickers")

    mkt_frames = fetch_market_data()
    feat_df = build_features(price_df, mkt_frames)
    feat_df.dropna(subset=FEATURES, inplace=True)

    if feat_df.empty:
        raise HTTPException(status_code=422, detail="Not enough history to compute features for the given tickers")

    feat_df['Ticker'] = feat_df['Ticker'].astype('category')
    feat_df['Sector'] = feat_df['Sector'].astype('category')

    latest = feat_df.groupby('Ticker').tail(1).copy()
    probs = predict_buy_probability(ENSEMBLE, latest[FEATURES])
    latest['Buy_Probability'] = probs

    predictions = [
        TickerPrediction(
            ticker=row['Ticker'],
            sector=int(row['Sector']),
            date=row['Date'].strftime("%Y-%m-%d"),
            buy_probability=round(float(row['Buy_Probability']), 4),
            conviction=conviction_label(row['Buy_Probability']),
        )
        for _, row in latest.iterrows()
    ]

    return PredictResponse(
        predictions=predictions,
        threshold_high_conviction=round(float(THRESH_HP), 4),
        threshold_moderate_conviction=round(float(THRESH_BAL), 4),
    )
