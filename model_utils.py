"""
model_utils.py
---------------
Feature engineering + model loading utilities for the Buy-Signal model.

IMPORTANT: build_features() below is copied AS-IS from the original notebook
(no modifications). This guarantees the live features match exactly what the
model was trained on.
"""

import os
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb

# ---------------------------------------------------------------------------
# Same feature list used at training time (Phase 2, ablated feature set).
# Loaded dynamically from the saved .pkl at runtime (see load_artifacts),
# kept here only as a fallback / reference.
# ---------------------------------------------------------------------------

SECTOR_MAP = {
    'AAPL': 0, 'MSFT': 0, 'GOOGL': 0, 'AMZN': 0, 'NVDA': 0, 'META': 0, 'AVGO': 0,
    'CSCO': 0, 'CRM': 0, 'NFLX': 0, 'AMD': 0, 'INTC': 0, 'QCOM': 0, 'TXN': 0, 'ACN': 0,
    'TSLA': 1, 'MCD': 1, 'HD': 1, 'COST': 1,
    'JPM': 2, 'BAC': 2, 'V': 2, 'MA': 2, 'BRK-B': 2,
    'JNJ': 3, 'UNH': 3, 'LLY': 3, 'ABBV': 3, 'MRK': 3, 'TMO': 3, 'ABT': 3,
    'WMT': 4, 'PG': 4, 'PEP': 4, 'PM': 4,
    'CAT': 5, 'DE': 5, 'UPS': 5,
    'CVX': 6, 'XOM': 6
}


def build_features(price_df, market_df):
    """
    Copied unchanged from the training notebook.
    price_df: concatenated OHLCV dataframe with columns
              ['Date','Open','High','Low','Close','Volume','Ticker','Sector']
    market_df: list of two dataframes [spy_df, vix_df] each with columns ['Date', <col>]
               spy_df col name 'spy_close', vix_df col name 'vix_level'
    """
    mkt = market_df[0].merge(market_df[1], on='Date', how='outer').sort_values('Date')
    mkt['spy_return_20d'] = mkt['spy_close'].pct_change(20).ffill()

    out = price_df.merge(mkt[['Date', 'spy_return_20d', 'vix_level']], on='Date', how='left')
    out['spy_return_20d'] = out['spy_return_20d'].ffill()
    out['vix_level'] = out['vix_level'].ffill()

    out.dropna(subset=['Close', 'Open', 'High', 'Low'], inplace=True)
    out['day_of_week'] = out['Date'].dt.dayofweek
    out['month'] = out['Date'].dt.month

    g = out.groupby('Ticker')

    rm20 = g['Close'].transform(lambda x: x.rolling(20).mean())
    for col in ['Open', 'High', 'Low', 'Close']:
        out[f'{col}_Norm'] = (out[col] - rm20) / rm20

    sma_10 = g['Close'].transform(lambda x: x.rolling(10).mean())
    sma_50 = g['Close'].transform(lambda x: x.rolling(50).mean())
    sma_200 = g['Close'].transform(lambda x: x.rolling(200).mean())
    out['sma_ratio'] = sma_10 / sma_50
    out['sma_50_200'] = sma_50 / sma_200

    hl = out['High'] - out['Low']
    hc = (out['High'] - g['Close'].shift(1)).abs()
    lc = (out['Low'] - g['Close'].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    out['atr_pct'] = tr.groupby(out['Ticker']).transform(lambda x: x.ewm(span=14, adjust=False).mean()) / out['Close']

    bb_m = g['Close'].transform(lambda x: x.rolling(20).mean())
    bb_s = g['Close'].transform(lambda x: x.rolling(20).std())
    bb_upper = bb_m + 2 * bb_s
    bb_lower = bb_m - 2 * bb_s
    out['bb_pct'] = (out['Close'] - bb_lower) / (bb_upper - bb_lower).replace(0, np.nan)
    out['bb_width'] = (bb_upper - bb_lower) / bb_m

    delta = g['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    ag = gain.groupby(out['Ticker']).transform(lambda x: x.rolling(14).mean())
    al = loss.groupby(out['Ticker']).transform(lambda x: x.rolling(14).mean())
    out['RSI_14'] = 100 - (100 / (1 + (ag / al.replace(0, np.nan)).fillna(0)))

    e12 = g['Close'].transform(lambda x: x.ewm(span=12, adjust=False).mean())
    e26 = g['Close'].transform(lambda x: x.ewm(span=26, adjust=False).mean())
    out['MACD'] = e12 - e26
    out['MACD_Signal'] = out.groupby('Ticker')['MACD'].transform(lambda x: x.ewm(span=9, adjust=False).mean())
    out['MACD_Hist'] = out['MACD'] - out['MACD_Signal']

    for lag in [1, 2, 3, 5, 10]:
        out[f'return_lag{lag}'] = g['Close'].pct_change(lag)

    out['volume_sma_20'] = g['Volume'].transform(lambda x: x.rolling(20).mean())
    out['volume_ratio'] = out['Volume'] / out['volume_sma_20']

    def calc_obv(d):
        return (np.sign(d['Close'].diff()).fillna(0) * d['Volume']).cumsum()

    out['obv'] = out.groupby('Ticker', group_keys=False).apply(calc_obv, include_groups=False)
    out['obv_sma_20'] = out.groupby('Ticker')['obv'].transform(lambda x: x.rolling(20).mean())
    out['obv_ratio'] = out['obv'] / out['obv_sma_20']

    out['Daily_Return'] = g['Close'].pct_change()
    out['Vol_5d'] = g['Daily_Return'].transform(lambda x: x.rolling(5).std())
    out['Vol_10d'] = g['Daily_Return'].transform(lambda x: x.rolling(10).std())
    out['Vol_20d'] = g['Daily_Return'].transform(lambda x: x.rolling(20).std())
    out['Momentum_10d'] = g['Close'].pct_change(10)
    out['stock_return_20d'] = g['Close'].pct_change(20)
    out['relative_strength_20d'] = out['stock_return_20d'] - out['spy_return_20d']
    out['vol_regime'] = (out['Vol_10d'] / out['Vol_20d']).clip(0.5, 2.0)

    out['rsi_volume'] = out['RSI_14'] * out['volume_ratio']
    out['macd_bb'] = out['MACD_Hist'] * out['bb_pct']
    out['momentum_vol'] = out['Momentum_10d'] / out['Vol_10d'].replace(0, np.nan)
    out['rsi_zone'] = pd.cut(out['RSI_14'], bins=[0, 30, 70, 100], labels=[0, 1, 2]).astype(float)

    date_g = out.groupby('Date')
    out['sector_rel_return_20d'] = out['stock_return_20d'] - out.groupby(['Date', 'Sector'])['stock_return_20d'].transform('mean')
    out['rsi_rank_pct'] = date_g['RSI_14'].rank(pct=True)
    out['momentum_rank_pct'] = date_g['Momentum_10d'].rank(pct=True)

    out['target_price_10d'] = g['Close'].shift(-10)
    out['price_change_10d'] = (out['target_price_10d'] - out['Close']) / out['Close']
    out['Target'] = (out['price_change_10d'] > 0.05).astype(int)

    expected_move_10d = out['Vol_20d'] * np.sqrt(10)
    out['Target_VolAdj'] = (out['price_change_10d'] > expected_move_10d).astype(int)

    return out


def load_artifacts(model_dir="."):
    """
    Load the ORIGINAL Phase 2 model artifacts (no Phase-3 changes).

    Loads each booster from XGBoost's native JSON format (model_native/xgb_model_*.json),
    which is version-independent — this avoids the joblib/pickle cross-version
    XGBoostError ("`i` is not supported for typed array") that happens when the
    XGBoost version used to train differs from the version used to serve.
    """
    native_dir = f"{model_dir}/model_native"
    ensemble = []
    if os.path.isdir(native_dir):
        json_files = sorted(f for f in os.listdir(native_dir) if f.endswith(".json"))
        for fname in json_files:
            clf = xgb.XGBClassifier()
            clf.load_model(f"{native_dir}/{fname}")
            ensemble.append(clf)
    else:
        # Fallback: old joblib pickle (only works if xgboost version matches training env exactly)
        ensemble = joblib.load(f"{model_dir}/phase2_ensemble.pkl")

    features = joblib.load(f"{model_dir}/phase2_features.pkl")
    threshold_hp = joblib.load(f"{model_dir}/phase2_threshold_hp.pkl")
    threshold_bal = joblib.load(f"{model_dir}/phase2_threshold_bal.pkl")
    return ensemble, features, threshold_hp, threshold_bal


def predict_buy_probability(ensemble, X):
    """Average predict_proba across all models in the ensemble."""
    return np.mean([m.predict_proba(X)[:, 1] for m in ensemble], axis=0)
