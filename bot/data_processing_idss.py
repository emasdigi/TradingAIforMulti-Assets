import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from pymongo import MongoClient, DESCENDING
from pymongo.server_api import ServerApi

from config import config
from .indicators import *


class IDSStockMongoClient:
    """MongoDB client for fetching Indonesian stock (IDSS) OHLC data."""
    
    MONGO_DB_NAME="pluang_indo_stock_static_data_v2"
    COLLECTION_HOURLY_OHLC_NAME = "indo_stock_one_hour_ohlc_price_stats"
    COLLECTION_MINS_OHLC_NAME = "indo_stock_five_minutes_ohlc_price_stats"

    def __init__(self):
        uri = f"mongodb+srv://{config.MONGO_DB_USERNAME}:{config.MONGO_DB_PASSWORD}@{config.MONGO_DB_HOST}/?appName={config.MONGO_APP_NAME}"
        # Create a new client and connect to the server
        client = MongoClient(uri, server_api=ServerApi('1'))
        try:
            client.admin.command('ping')
            logging.info("Successfully connected to MongoDB for IDSS data!")
        except Exception as e:
            logging.error(f"Failed to connect to MongoDB: {e}")

        self.db = client[self.MONGO_DB_NAME]

    def fetch_hourly_ohlc(self, symbol: str, limit: int = 200) -> List[Dict[str, Any]]:
        """Fetch hourly OHLC data for a symbol."""
        collection_hourly_ohlc = self.db[self.COLLECTION_HOURLY_OHLC_NAME]
        data = collection_hourly_ohlc.find({"sc": symbol}).sort("psd", DESCENDING).limit(limit)
        return list(data)

    def fetch_mins_ohlc(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Fetch 5-minute OHLC data for a symbol."""
        collection_mins_ohlc = self.db[self.COLLECTION_MINS_OHLC_NAME]
        data = collection_mins_ohlc.find({"sc": symbol}).sort("psd", DESCENDING).limit(limit)
        return list(data)


# Initialize MongoDB client
mongo_client = IDSStockMongoClient()


def mongo_data_to_dataframe(mongo_data: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert MongoDB OHLC data to a pandas DataFrame.
    
    Expected MongoDB fields:
    - sc: symbol code
    - psd: period start date (timestamp)
    - o: open
    - h: high
    - l: low
    - c: close
    - v: volume
    """
    if not mongo_data:
        return pd.DataFrame()
    
    # Convert to DataFrame and rename columns
    df = pd.DataFrame(mongo_data)
    
    # Reverse the order (oldest first) for proper indicator calculation
    df = df.iloc[::-1].reset_index(drop=True)
    
    # Rename columns to standard format
    column_mapping = {
        "op": "open",
        "hip": "high",
        "lop": "low",
        "clp": "close",
        "vol": "volume",
        "cst": "timestamp"
    }
    
    df = df.rename(columns=column_mapping)
    
    # Ensure numeric columns are float
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Calculate mid_price
    df["mid_price"] = (df["high"] + df["low"]) / 2
    
    # Calculate VWAP (Volume Weighted Average Price)
    if "volume" in df.columns and df["volume"].sum() > 0:
        df["vwap"] = (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()
    else:
        df["vwap"] = df["close"]
    
    return df


def collect_market_data(symbol: str) -> Optional[Dict[str, Any]]:
    """
    Collect and process market data for Indonesian stocks (IDSS).
    Returns a market snapshot compatible with the trading workflow.
    
    Args:
        symbol: The stock symbol (e.g., "BBCA", "GOTO")
        
    Returns:
        Dictionary with market data and indicators, or None if data fetch fails
    """
    try:
        # Fetch 5-minute data for intraday analysis (last 100 candles = ~8.3 hours)
        mins_data = mongo_client.fetch_mins_ohlc(symbol, limit=100)
        if not mins_data:
            logging.warning(f"No 5-minute data found for {symbol}")
            return None
         
        df_intraday = mongo_data_to_dataframe(mins_data)
        
        # Add indicators for intraday data
        df_intraday = add_indicator_columns(
            df_intraday,
            ema_lengths=(config.EMA_LEN,),
            rsi_periods=(7, config.RSI_LEN),
            macd_params=(config.MACD_FAST, config.MACD_SLOW, config.MACD_SIGNAL),
        )
        
        # Fetch hourly data for long-term analysis (last 200 candles = ~8.3 days)
        hourly_data = mongo_client.fetch_hourly_ohlc(symbol, limit=200)
        if not hourly_data:
            logging.warning(f"No hourly data found for {symbol}")
            return None
        
        df_long = mongo_data_to_dataframe(hourly_data)
        
        # Add indicators for long-term data
        df_long = add_indicator_columns(
            df_long,
            ema_lengths=(20, 50),
            rsi_periods=(14,),
            macd_params=(config.MACD_FAST, config.MACD_SLOW, config.MACD_SIGNAL),
        )
        
        # Calculate ATR
        df_long["atr3"] = calculate_atr_series(df_long, 3)
        df_long["atr14"] = calculate_atr_series(df_long, 14)
        
        # Get tail data for series
        intraday_tail = df_intraday.tail(10)
        long_tail = df_long.tail(10)
        
        # Build the market snapshot in the same format as data_processing_stock
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
        logging.error(f"Failed to build market snapshot for {symbol}: {exc}", exc_info=True)
        return None


if __name__ == "__main__":
    # Test the data collection
    logging.basicConfig(level=logging.INFO)
    test_symbol = "BBCA"
    snapshot = collect_market_data(test_symbol)
    if snapshot:
        print(f"\nMarket snapshot for {test_symbol}:")
        print(f"Price: {snapshot['price']}")
        print(f"EMA20: {snapshot['ema20']}")
        print(f"RSI14: {snapshot['rsi']}")
        print(f"MACD: {snapshot['macd']}")
        print(f"VWAP: {snapshot['vwap']}")
    else:
        print(f"Failed to collect data for {test_symbol}")
