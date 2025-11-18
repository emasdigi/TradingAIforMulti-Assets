import logging
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pytz 

from . import clients
from config import config
from .indicators import *

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.common.exceptions import APIError


data_client = StockHistoricalDataClient(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY)

def get_latest_realtime_candles(symbol: str, 
                                client: StockHistoricalDataClient, 
                                timeframe: TimeFrame,
                                limit: int) -> pd.DataFrame: # <-- CHANGED RETURN TYPE
    """
    Tries to get the LATEST 'n' candles from the 'iex' REAL-TIME feed.
    Returns a Pandas DataFrame.
    """
    
    start_time = datetime.now(pytz.utc) - timedelta(days=7)
    if timeframe.unit == TimeFrameUnit.Minute:
        start_time = datetime.now(pytz.utc) - timedelta(days=7)
    elif timeframe.unit == TimeFrameUnit.Hour:
        start_time = datetime.now(pytz.utc) - timedelta(days=90)

    # --- Attempt 1: Try Real-Time (IEX) ---
    try:
        req_realtime = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=timeframe,
            start=start_time,
            end=None, 
            feed='iex'
        )
        bars_realtime = client.get_stock_bars(req_realtime)
        if symbol in bars_realtime.data and len(bars_realtime.data[symbol]) > 0:
            # Convert list of bar objects to DataFrame
            return pd.DataFrame(
                [bar.__dict__ for bar in bars_realtime.data[symbol]]
            ).tail(limit)

    except (APIError, Exception) as e:
        print(f"[Info] Real-time 'iex' failed (market likely closed): {e}")

    # --- Attempt 2: Fallback to Delayed (SIP) ---
    try:
        req_historical = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=timeframe,
            end=datetime.now(pytz.utc) - timedelta(minutes=16),
            limit=limit,
            feed='sip' 
        )
        bars_historical = client.get_stock_bars(req_historical)
        
        if symbol in bars_historical.data and len(bars_historical.data[symbol]) > 0:
            # Convert list of bar objects to DataFrame
            return pd.DataFrame(
                [bar.__dict__ for bar in bars_historical.data[symbol]]
            )
        else:
            print(f"[Error] No data found for {symbol} on 'sip' feed.")
            return pd.DataFrame() # Return empty DataFrame on failure

    except (APIError, Exception) as e:
        print(f"[Error] Error on 'sip' fallback: {e}")
        return pd.DataFrame() # Return empty DataFrame on failure

def collect_market_data(symbol: str) -> Optional[Dict[str, Any]]:
    """Return rich market snapshot for a given symbol."""

    try:
        timeframe_3_min = TimeFrame(3, TimeFrameUnit.Minute)
        df = get_latest_realtime_candles(
            symbol=symbol, 
            client=data_client, 
            timeframe=timeframe_3_min, 
            limit=100
        )
        df_intraday = df.copy()
        numeric_cols = ["open", "high", "low", "close", "volume", "vwap"]
        df_intraday[numeric_cols] = df_intraday[numeric_cols].astype(float)
        df_intraday["mid_price"] = (df_intraday["high"] + df_intraday["low"]) / 2
        df_intraday = add_indicator_columns(
            df_intraday,
            ema_lengths=(config.EMA_LEN,),
            rsi_periods=(7, config.RSI_LEN),
            macd_params=(config.MACD_FAST, config.MACD_SLOW, config.MACD_SIGNAL),
        )

        
        timeframe_1_hour = TimeFrame(1, TimeFrameUnit.Hour)
        df_long = get_latest_realtime_candles(
            symbol=symbol, 
            client=data_client, 
            timeframe=timeframe_1_hour, 
            limit=200
        )
        df_long[numeric_cols] = df_long[numeric_cols].astype(float)
        df_long = add_indicator_columns(
            df_long,
            ema_lengths=(20, 50),
            rsi_periods=(14,),
            macd_params=(config.MACD_FAST, config.MACD_SLOW, config.MACD_SIGNAL),
        )

        df_long["atr3"] = calculate_atr_series(df_long, 3)
        df_long["atr14"] = calculate_atr_series(df_long, 14)

        # try:
        #     oi_hist = binance_client.futures_open_interest_hist(symbol=symbol, period="5m", limit=30)
        #     open_interest_values = [float(entry["sumOpenInterest"]) for entry in oi_hist]
        # except Exception:
        #     open_interest_values = []

        # try:
        #     funding_hist = binance_client.futures_funding_rate(symbol=symbol, limit=30)
        #     funding_rates = [float(entry["fundingRate"]) for entry in funding_hist]
        # except Exception:
        #     funding_rates = []

        intraday_tail = df_intraday.tail(10)
        long_tail = df_long.tail(10)

        return {
            "symbol": symbol,
            "coin": config.SYMBOL_TO_COIN[symbol],
            "price": float(df_intraday["close"].iloc[-1]),
            "ema20": float(df_intraday["ema20"].iloc[-1]),
            "rsi": float(df_intraday[f"rsi{config.RSI_LEN}"].iloc[-1]),
            "rsi7": float(df_intraday["rsi7"].iloc[-1]),
            "macd": float(df_intraday["macd"].iloc[-1]),
            "macd_signal": float(df_intraday["macd_signal"].iloc[-1]),
            "vwap": float(df_intraday["vwap"].iloc[-1]),
            # "funding_rate": funding_rates[-1] if funding_rates else 0.0,
            # "funding_rates": funding_rates,
            # "open_interest": {
            #     "latest": open_interest_values[-1] if open_interest_values else None,
            #     "average": float(np.mean(open_interest_values)) if open_interest_values else None,
            # },
            "intraday_series": {
                "mid_prices": round_series(intraday_tail["mid_price"], 3),
                "ema20": round_series(intraday_tail["ema20"], 3),
                "macd": round_series(intraday_tail["macd"], 3),
                "rsi7": round_series(intraday_tail["rsi7"], 3),
                "rsi14": round_series(intraday_tail[f"rsi{config.RSI_LEN}"], 3),
                "vwap": round_series(intraday_tail["vwap"], 3),
            },
            "long_term": {
                "ema20": float(df_long["ema20"].iloc[-1]),
                "ema50": float(df_long["ema50"].iloc[-1]),
                "atr3": float(df_long["atr3"].iloc[-1]),
                "atr14": float(df_long["atr14"].iloc[-1]),
                "current_volume": float(df_long["volume"].iloc[-1]),
                "average_volume": float(df_long["volume"].mean()),
                "macd": round_series(long_tail["macd"], 3),
                "rsi14": round_series(long_tail[f"rsi{config.RSI_LEN}"], 3),
                "vwap": round_series(long_tail["vwap"], 3),
            },
        }
    except Exception as exc:
        logging.error("Failed to build market snapshot for %s: %s", symbol, exc, exc_info=True)
        return None
