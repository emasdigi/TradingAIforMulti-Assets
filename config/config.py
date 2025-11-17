#!/usr/bin/env python3
"""
Configuration file for the Multi-LLM Trading Bot.
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from utils.utils import json_parser, yaml_parser

# Load environment variables from .env file
load_dotenv()

# --- DATA DIRECTORY ---
DATA_DIR = Path(os.getenv("TRADEBOT_DATA_DIR", "data"))
DATA_DIR.mkdir(exist_ok=True)

# --- BINANCE API CONFIGURATION ---
BN_API_KEY = os.getenv("BN_API_KEY", "")
BN_SECRET = os.getenv("BN_SECRET", "")

# --- OPENROUTER MULTI-LLM CONFIGURATION ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# --- OPENAI DIRECT ACCESS CONFIGURATION ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

# --- LEGACY / DIRECT LLM FALLBACK ---
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-4o")

# Two LLM Models for Focused Testing (Claude Sonnet and Gemini Pro)
LLM_MODELS = {
    "deepseek_v3.1": {
        "model_id": "deepseek/deepseek-chat-v3.1",
        "name": "DeepSeek V3.1",
        "provider": "DeepSeek",
        "max_tokens": 10000,
        "temperature": 0.7,
    },
    "qwen3_max": {
        "model_id": "qwen/qwen3-max",
        "name": "Qwen3 Max",
        "provider": "Qwen",
        "max_tokens": 10000,
        "temperature": 0.7,
    },
    "gemini_pro": {
        "model_id": "google/gemini-2.5-pro",
        "name": "Gemini 2.5 Pro",
        "provider": "Google",
        "max_tokens": 10000,
        "temperature": 0.7,
    },
    "grok4": {
        "model_id": "x-ai/grok-4",
        "name": "Grok 4",
        "provider": "Grok",
        "max_tokens": 10000,
        "temperature": 0.7,
        "reasoning": {"effort": "low"},
    },
    "gpt5": {
        "model_id": "openai/gpt-5",
        "name": "GPT-5",
        "provider": "OpenAI",
        "max_tokens": 10000,
        "temperature": 1,
        "reasoning": {"enabled": False},
        "response_format": {"type": "json_object"},
    },
    "claude_sonnet_4.5": {
        "model_id": "anthropic/claude-sonnet-4.5",
        "name": "Claude Sonnet 4.5",
        "provider": "Anthropic",
        "max_tokens": 10000,
        "temperature": 0.7,
    },
    "kimi_k2": {
        "model_id": "moonshotai/kimi-k2-thinking",
        "name": "Kimik2",
        "provider": "MoonshotAI",
        "max_tokens": 10000,
        "temperature": 0.7,
    },
    # "doubao_seed": {
    #     "model_id": "bytedance/seed-oss-36b-instruct",
    #     "name": "Doubao Seed",
    #     "provider": "Bytedance",
    #     "max_tokens": 10000,
    #     "temperature": 0.7,
    # },
}
# --- TELEGRAM NOTIFICATIONS ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

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

aws = AWS(
    env=env,
    configFilePath=f"config/config.{env}.yaml",
    secret_name=envCONFIG.get("awsSecretName", ""),
)

# MONGO DB
MONGO_DB_CONFIG = aws.get_aws_secret_manager_value(key=envCONFIG["mongo"]["secretName"])
MONGO_DB_HOST = MONGO_DB_CONFIG.get("DATABASES_PLUANG_MONGO_HOST")
MONGO_DB_USERNAME = MONGO_DB_CONFIG.get("DATABASES_PLUANG_MONGO_USER")
MONGO_DB_PASSWORD = MONGO_DB_CONFIG.get("DATABASES_PLUANG_MONGO_PASSWORD")
MONGO_APP_NAME = envCONFIG["mongo"]["appName"]


ASSET_MODE = os.getenv("ASSET_MODE", "idss")

PROJECT_S3_PATH = envCONFIG["projectS3Path"]
AI_TRADING_BOT_CONFIG = aws.get_aws_secret_manager_value(
    key=envCONFIG["tradingBotSecretName"]
)


# --- ALPACA API CONFIGURATION ---
ALPACA_API_KEY = AI_TRADING_BOT_CONFIG.get("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = AI_TRADING_BOT_CONFIG.get("ALPACA_SECRET_KEY", "")

# --- NEWS & FUNDAMENTALS ---
NEWS_REFRESH_INTERVAL = int(
    os.getenv("NEWS_REFRESH_INTERVAL", str(3 * 60 * 60))
)  # seconds

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
    SYMBOLS = [
        "BBCA",
        "GOTO",
        "BYAN",
        "BMRI",
        "BBRI",
        "TLKM",
        "ASII",
        "TPIA",
        "BBNI",
        "UNVR",
        "HMSP",
    ]
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
    SYMBOLS = [
        "AAPL",
        "MSFT",
        "GOOG",
        "AMZN",
        "TSLA",
        "NVDA",
        "IBIT",
        "TQQQ",
        "SQQQ",
        "UVIX",
    ]
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

INTERVAL = "5m"  # 5-minute candles
CHECK_INTERVAL = 5 * 60  # Check every 5 minutes (when candle closes)
START_CAPITAL = 100000000  # Rp 100,000,000 (~$6,500 USD) for IDSS mode
CAPITAL_PER_LLM = START_CAPITAL  # Each LLM model gets full starting capital

# --- FEE & EXECUTION COSTS ---
TRADING_FEE_RATE = float(
    os.getenv("TRADING_FEE_RATE", "0.0003")
)  # 0.03% per side by default

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

# --- SUMMARY GENERATION SETTINGS ---
PROFESSIONAL_SUMMARY_MAX_TOKENS = 500  # Max tokens for professional portfolio summary
PROFESSIONAL_SUMMARY_TEMPERATURE = 0.7  # Temperature for professional summary
SHORT_SUMMARY_MAX_TOKENS = 100  # Max tokens for short Gen-Z summary
SHORT_SUMMARY_TEMPERATURE = 0.8  # Higher temperature for more creative/casual tone

# --- TRADING RULES ---
MIN_CONFIDENCE = 0.6  # Minimum confidence for trade execution
MAX_LEVERAGE = 10  # Maximum leverage allowed
MAX_DAILY_TRADES = 20  # Maximum trades per day per LLM

# --- LOGGING ---
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"
