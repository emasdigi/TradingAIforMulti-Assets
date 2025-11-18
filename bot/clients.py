#!/usr/bin/env python3
"""
API client initialization for Binance and LLM providers.
"""
import logging
from typing import Optional

from binance.client import Client
from openai import OpenAI
from requests.exceptions import RequestException, Timeout

from config import config

# --- Client Singletons ---
_binance_client: Optional[Client] = None
_llm_client: Optional[OpenAI] = None


def get_binance_client() -> Optional[Client]:
    """Return a connected Binance client singleton or None if initialization failed."""
    global _binance_client

    _binance_client = Client(config.BN_API_KEY, config.BN_SECRET, testnet=False)

    return _binance_client


def get_llm_client() -> Optional[OpenAI]:
    """Return an initialized OpenAI-compatible client singleton."""
    global _llm_client

    _llm_client = OpenAI(
        api_key=config.OPENROUTER_API_KEY,
        base_url=config.OPENROUTER_BASE_URL,
        max_retries=0,
    )
    return _llm_client


if __name__ == "__main__":
    binance_client = get_binance_client()
    llm_client = get_llm_client()
    # print(binance_client.get_account())

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "who are you?"},
    ]
    response = llm_client.chat.completions.create(
        model=config.LLM_MODEL_NAME,
        messages=messages,
        temperature=0.7,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    print(response.choices[0].message.content)
