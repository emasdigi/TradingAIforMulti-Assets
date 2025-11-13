#!/usr/bin/env python3
"""
Configuration file for the Multi-LLM Trading Bot.
"""
import os
from pathlib import Path
from utils.utils import yaml_parser, json_parser


# --- DATA DIRECTORY ---
DATA_DIR = Path(os.getenv("TRADEBOT_DATA_DIR", "data"))
DATA_DIR.mkdir(exist_ok=True)

# Two LLM Models for Focused Testing (Claude Sonnet and Gemini Pro)
LLM_MODELS = {
    "deepseek_v3.1": {
        "model_id": "deepseek/deepseek-chat-v3.1",
        "name": "DeepSeek V3.1",
        "provider": "DeepSeek",
        "max_tokens": 4000,
        "temperature": 0.7,
    },
    "qwen3_max": {
        "model_id": "qwen/qwen3-max",
        "name": "Qwen3 Max",
        "provider": "Qwen",
        "max_tokens": 4000,
        "temperature": 0.7,
    },
}

INTERVAL = "3m"  # 3-minute candles
CHECK_INTERVAL = 3 * 60  # Check every 3 minutes (when candle closes)
START_CAPITAL = 100000000 # 100000000 idr is 6k USD
CAPITAL_PER_LLM = START_CAPITAL  # $10,000 per LLM (2 models total)

# --- INDICATOR SETTINGS ---
EMA_LEN = 20
RSI_LEN = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
ATR_LEN = 14

# --- RISK MANAGEMENT ---
RISK_FREE_RATE = float(os.getenv("RISK_FREE_RATE", "0.05"))  # 5% annual risk-free rate
MAX_POSITION_SIZE = 0.1  # Max 10% of capital per position
STOP_LOSS_PCT = 0.02  # 2% stop loss
PROFIT_TARGET_PCT = 0.04  # 4% profit target

# --- TRADING RULES ---
MIN_CONFIDENCE = 0.6  # Minimum confidence for trade execution
MAX_LEVERAGE = 10  # Maximum leverage allowed
MAX_DAILY_TRADES = 20  # Maximum trades per day per LLM

globalConfigFilePath = "config/config.global.yaml"
globalCONFIG = yaml_parser(globalConfigFilePath)
filePathEnv = globalCONFIG["filePathEnv"]
if os.path.isfile(filePathEnv):
    env = json_parser(filePathEnv)["env"]
else:
    env = os.environ.get("env", globalCONFIG["VM_env"])


envConfigFilePath = f"config/config.{env}.yaml"
envCONFIG = yaml_parser(envConfigFilePath)

# AWS Utils
from utils.awsUtils import AWS
aws = AWS(env=env, configFilePath=f"config/config.{env}.yaml", secret_name=envCONFIG.get('awsSecretName', ''))

# MONGO DB
MONGO_DB_CONFIG = aws.get_aws_secret_manager_value(
    key=envCONFIG['mongo']['secretName']
)
MONGO_DB_HOST = MONGO_DB_CONFIG.get("DATABASES_PLUANG_MONGO_HOST")
MONGO_DB_USERNAME = MONGO_DB_CONFIG.get("DATABASES_PLUANG_MONGO_USER")
MONGO_DB_PASSWORD = MONGO_DB_CONFIG.get("DATABASES_PLUANG_MONGO_PASSWORD")
MONGO_APP_NAME = envCONFIG['mongo']["appName"]


ASSET_MODE = os.getenv("ASSET_MODE", "idss")

PROJECT_S3_PATH = envCONFIG['projectS3Path']
AI_TRADING_BOT_CONFIG = aws.get_aws_secret_manager_value(key=envCONFIG['tradingBotSecretName'])

# --- BINANCE API CONFIGURATION ---
BN_API_KEY = AI_TRADING_BOT_CONFIG.get("BN_API_KEY", "")
BN_SECRET = AI_TRADING_BOT_CONFIG.get("BN_SECRET", "")

# --- OPENROUTER MULTI-LLM CONFIGURATION ---
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY = AI_TRADING_BOT_CONFIG.get("OPENROUTER_API_KEY", "")


# --- ALPACA API CONFIGURATION ---
ALPACA_API_KEY = AI_TRADING_BOT_CONFIG.get("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = AI_TRADING_BOT_CONFIG.get("ALPACA_SECRET_KEY", "")

TELEGRAM_BOT_TOKEN = AI_TRADING_BOT_CONFIG.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = AI_TRADING_BOT_CONFIG.get("TELEGRAM_CHAT_ID", "")

if ASSET_MODE.lower() == "crypto":
    # --- TRADING PARAMETERS ---
    SYMBOLS = ["ETHUSDT", "SOLUSDT", "XRPUSDT", "BTCUSDT", "DOGEUSDT", "BNBUSDT"]
    SYMBOL_TO_COIN = {
        "ETHUSDT": "ETH",
        "SOLUSDT": "SOL",
        "XRPUSDT": "XRP",
        "BTCUSDT": "BTC",
        "DOGEUSDT": "DOGE",
        "BNBUSDT": "BNB",
    }
elif ASSET_MODE.lower() == "idss":
    SYMBOLS = ["BBCA", "GOTO", "BYAN", "BMRI", "BBRI", "TLKM", "ASII", "TPIA", "BBNI", "UNVR", "HMSP"]
    SYMBOL_TO_COIN = {
        "BBCA": "BBCA",
        "GOTO": "GOTO",
        "BYAN": "BYAN",
        "BMRI": "BMRI",
        "BBRI": "BBRI",
        "TLKM": "TLKM",
        "ASII": "ASII",
        "TPIA": "TPIA",
        "BBNI": "BBNI",
        "UNVR": "UNVR",
        "HMSP": "HMSP",
    }
elif ASSET_MODE.lower() == "us_stock":
    SYMBOLS = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA", "NVDA", "IBIT", "TQQQ", "SQQQ", "UVIX"]
    SYMBOL_TO_COIN = {
        "AAPL": "AAPL",
        "MSFT": "MSFT",
        "GOOG": "GOOG",
        "AMZN": "AMZN",
        "TSLA": "TSLA",
        "NVDA": "NVDA",
        "IBIT": "IBIT",
        "TQQQ": "TQQQ",
        "SQQQ": "SQQQ",
        "UVIX": "UVIX",
    }


# --- LOGGING ---
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"
