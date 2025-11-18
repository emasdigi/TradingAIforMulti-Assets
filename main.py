#!/usr/bin/env python3
"""Main entry point for the multi-LLM trading bot."""

import os
import threading
import time
from typing import List

from dotenv import load_dotenv

from config import config
from bot.trading_workflow import run_trading_loop


def _resolve_models_to_run() -> List[str]:
    env_value = os.getenv("ACTIVE_LLM_MODELS", "")
    if env_value:
        models = [model.strip() for model in env_value.split(",") if model.strip()]
    else:
        models = list(config.LLM_MODELS.keys())
    return models


def _start_model_threads(models: List[str]) -> None:
    threads = []
    for model_name in models:
        if model_name not in config.LLM_MODELS:
            print(f"⚠️  Model '{model_name}' is not configured; skipping.")
            continue

        thread = threading.Thread(
            target=run_trading_loop,
            args=(model_name,),
            name=f"TradingLoop-{model_name}",
            daemon=False,
        )
        thread.start()
        threads.append(thread)
        print(f"✅ Started trading loop for model '{model_name}'")

    try:
        for thread in threads:
            thread.join()
    except KeyboardInterrupt:
        print("🛑 Shutdown signal received; waiting for worker threads to exit...")


if __name__ == "__main__":
    load_dotenv()
    models_to_run = _resolve_models_to_run()
    if not models_to_run:
        raise SystemExit("No LLM models specified to run. Set ACTIVE_LLM_MODELS or configure LLM_MODELS in config.")

    print("🚀 Launching trading bot with models:", ", ".join(models_to_run))
    time.sleep(1)
    _start_model_threads(models_to_run)