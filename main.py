#!/usr/bin/env python3
"""
Main entry point for the multi-LLM trading bot.
"""

import asyncio
from dotenv import load_dotenv
from bot.trading_workflow import run_trading_loop
# Load environment variables from .env file
if __name__ == "__main__":
    run_trading_loop("deepseek_v3.1")
