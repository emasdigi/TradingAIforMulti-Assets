#!/usr/bin/env python3
"""
Main trading workflow, state management, and execution logic.
"""
import json
import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Iterable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from colorama import Fore, Style
from openai import OpenAI

from config import config
from . import clients
from utils import utils

if config.ASSET_MODE.lower() == "crypto":
    from . import data_processing as data_processing
    from . import prompts_v1 as prompts
elif config.ASSET_MODE.lower() == "idss":
    from . import data_processing_idss as data_processing
    from . import prompts_idss as prompts
else:
    from . import data_processing as data_processing
    from . import prompts as prompts


def _to_float(value: Any) -> Optional[float]:
    """Convert value to float when possible, otherwise return None."""
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(result):
        return None
    return result


def _to_serializable(value: Any) -> Any:
    """Recursively convert numpy types so they can be JSON-serialized."""
    if isinstance(value, dict):
        return {key: _to_serializable(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_to_serializable(item) for item in value]
    if isinstance(value, tuple):
        return [_to_serializable(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return float(value)
    return value


def is_idss_break_time() -> bool:
    """
    Check if it's currently break time for Indonesian Stock Market (IDSS).
    Break time: 12:00 - 13:30 WIB
    """
    wib_tz = ZoneInfo("Asia/Jakarta")
    now = datetime.now(wib_tz)

    # Check if weekend (no break on weekends, market is just closed)
    if now.weekday() >= 5:
        return False

    # Break time: 12:00 - 13:30 WIB
    break_start = now.replace(hour=12, minute=0, second=0, microsecond=0)
    break_end = now.replace(hour=13, minute=30, second=0, microsecond=0)

    return break_start <= now < break_end


def is_market_open() -> bool:
    """
    Check if the market is currently open based on ASSET_MODE.

    US Stock Market: 9:30 AM - 4:00 PM ET, Monday-Friday
    Indonesian Stock Market (IDSS):
        - Session 1: 09:00 - 12:00 WIB
        - Break: 12:00 - 13:30 WIB
        - Session 2: 13:30 - 15:00 WIB (includes pre-closing auction)
        - Monday-Friday, WIB (GMT+7)
    """
    if config.ASSET_MODE.lower() == "us_stock":
        # US stock market hours
        et_tz = ZoneInfo("America/New_York")
        now = datetime.now(et_tz)

        # Check if weekend
        if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False

        # Check if within market hours (9:30 AM - 4:00 PM ET)
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)

        return market_open <= now <= market_close

    elif config.ASSET_MODE.lower() == "idss":
        # Indonesian stock market hours (IDX - Indonesia Stock Exchange)
        wib_tz = ZoneInfo("Asia/Jakarta")  # WIB = GMT+7
        now = datetime.now(wib_tz)

        # Check if weekend
        if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False

        # Session 1: 09:00 - 12:00 WIB
        session1_open = now.replace(hour=9, minute=0, second=0, microsecond=0)
        session1_close = now.replace(hour=12, minute=0, second=0, microsecond=0)

        # Session 2: 13:30 - 15:00 WIB (includes pre-closing auction)
        session2_open = now.replace(hour=13, minute=30, second=0, microsecond=0)
        session2_close = now.replace(hour=15, minute=0, second=0, microsecond=0)

        # Check if within either trading session
        in_session1 = session1_open <= now < session1_close
        in_session2 = session2_open <= now <= session2_close

        return in_session1 or in_session2

    else:
        # For crypto or other asset modes, market is always open
        return True


class MarketDataCoordinator:
    """Coordinates shared market data between concurrent trading loops."""

    def __init__(self):
        self._lock = threading.Lock()
        self._update_event = threading.Event()
        self._market_snapshots: Dict[str, Dict[str, Any]] = {}
        self._last_updated: Optional[datetime] = None
        self._fetch_in_progress = False

    def get_market_snapshots(
        self, wait_for_update: bool = True
    ) -> Dict[str, Dict[str, Any]]:
        if wait_for_update:
            self._update_event.wait(timeout=config.CHECK_INTERVAL * 2)

        with self._lock:
            return {
                coin: snapshot.copy()
                for coin, snapshot in self._market_snapshots.items()
            }

    def fetch_and_update(self) -> Dict[str, Dict[str, Any]]:
        freshness_threshold = max(1, int(config.CHECK_INTERVAL * 0.1))
        now = datetime.now(timezone.utc)

        with self._lock:
            if (
                self._last_updated
                and (now - self._last_updated).total_seconds() < freshness_threshold
                and self._market_snapshots
            ):
                return {
                    coin: snapshot.copy()
                    for coin, snapshot in self._market_snapshots.items()
                }

            if self._fetch_in_progress:
                logging.debug(
                    "[MarketDataCoordinator] Fetch in progress; waiting for data..."
                )
                wait_event = self._update_event
            else:
                self._fetch_in_progress = True
                self._update_event.clear()
                wait_event = None

        if wait_event is not None:
            wait_event.wait(timeout=config.CHECK_INTERVAL * 2)
            with self._lock:
                return {
                    coin: snapshot.copy()
                    for coin, snapshot in self._market_snapshots.items()
                }

        try:
            logging.info(
                "[MarketDataCoordinator] Fetching market data for all symbols..."
            )
            market_snapshots: Dict[str, Dict[str, Any]] = {}
            for symbol in config.SYMBOLS:
                snapshot = data_processing.collect_market_data(symbol)
                if snapshot:
                    market_snapshots[snapshot["coin"]] = snapshot

            if len(market_snapshots) != len(config.SYMBOLS):
                logging.warning(
                    "[MarketDataCoordinator] Incomplete market data snapshot fetched."
                )

            with self._lock:
                self._market_snapshots = market_snapshots
                self._last_updated = datetime.now(timezone.utc)
                self._update_event.set()

            logging.info(
                "[MarketDataCoordinator] Market data updated (%s coins) at %s",
                len(market_snapshots),
                self._last_updated,
            )

            return {
                coin: snapshot.copy() for coin, snapshot in market_snapshots.items()
            }

        finally:
            with self._lock:
                self._fetch_in_progress = False


_market_coordinator = MarketDataCoordinator()

if config.ASSET_MODE.lower() == "crypto":
    from . import data_processing_crypto as data_processing
    from . import prompts_v1 as prompts
elif config.ASSET_MODE.lower() == "us_stock":
    from . import data_processing_stock as data_processing
    from . import prompts_stock as prompts
elif config.ASSET_MODE.lower() == "idss":
    from . import data_processing_idss as data_processing
    from . import prompts_idss as prompts


class TradingState:
    """Manages the full state of the trading bot."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.initial_capital: float = getattr(
            config, "CAPITAL_PER_LLM", config.START_CAPITAL
        )
        self.balance: float = self.initial_capital
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.equity_history: list[float] = []
        self.start_time: datetime = datetime.now(timezone.utc)
        self.invocation_count: int = 0
        self.current_iteration_messages: list[str] = []
        self.current_iteration_trades: list[Dict[str, Any]] = []
        self.total_fees_paid: float = 0.0
        self._state_dir: Path = config.DATA_DIR / self.model_name
        self._state_dir.mkdir(parents=True, exist_ok=True)
        self._state_file: Path = self._state_dir / "portfolio_state.json"
        self._state_csv: Path = self._state_dir / "portfolio_state.csv"

    def load_state(self):
        """Load persisted balance and positions if available."""
        data: Dict[str, Any] = {}

        # Try to download from S3 first
        download_success = config.aws.download_directory_from_s3(
            s3_base_path=config.PROJECT_S3_PATH, local_directory=config.DATA_DIR
        )

        if download_success["success"] > 0:
            logging.info(
                "Successfully downloaded state from S3: %s", config.PROJECT_S3_PATH
            )
        else:
            logging.info("Could not download from S3, will check for local state file")

        if self._state_file.exists():
            try:
                with open(self._state_file, "r") as f:
                    data = json.load(f)
            except Exception as e:
                logging.error(
                    "[%s] Failed to load state JSON (%s); attempting CSV fallback.",
                    self.model_name,
                    e,
                    exc_info=True,
                )

        if data:
            balance_candidate = data.get("last_total_balance", data.get("balance"))
            balance_value = _to_float(balance_candidate) or self.initial_capital
            self.balance = balance_value

            positions = data.get("positions")
            if isinstance(positions, dict):
                self.positions = positions
                for pos in self.positions.values():
                    if "fees_paid" not in pos:
                        pos["fees_paid"] = (
                            abs(pos["quantity"])
                            * pos["entry_price"]
                            * config.TRADING_FEE_RATE
                        )
                    # Backward compatibility: rename risk_usd to risk_idr for Indonesian stocks
                    if "risk_usd" in pos and "risk_idr" not in pos:
                        pos["risk_idr"] = pos.pop("risk_usd")

            fees_candidate = data.get(
                "last_total_fees_paid", data.get("total_fees_paid")
            )
            fees_value = _to_float(fees_candidate)
            if fees_value is not None:
                self.total_fees_paid = fees_value

            last_equity = _to_float(data.get("last_total_equity"))

            history_loaded = self._hydrate_equity_history()
            if not history_loaded and last_equity is not None:
                self.equity_history.append(last_equity)

            logging.info(
                "[%s] Loaded state from %s (balance: %.2f, positions: %d)",
                self.model_name,
                self._state_file,
                self.balance,
                len(self.positions),
            )

            if self.positions:
                for coin, pos in self.positions.items():
                    logging.info(
                        "[%s]   Restored position: %s %s @ $%.4f (qty: %.4f, leverage: %dx)",
                        self.model_name,
                        coin,
                        pos.get("side", "").upper(),
                        pos.get("entry_price", 0),
                        pos.get("quantity", 0),
                        pos.get("leverage", 1),
                    )
            return

        self._load_state_from_csv()

    def _load_state_from_csv(self):
        """Fallback loader that restores state from the latest CSV snapshot."""
        if not self._state_csv.exists():
            logging.info(
                "[%s] No existing state file found; starting fresh.", self.model_name
            )
            return

        try:
            df = pd.read_csv(self._state_csv)
        except Exception as e:
            logging.error(
                "[%s] Failed to read state CSV %s: %s",
                self.model_name,
                self._state_csv,
                e,
                exc_info=True,
            )
            return

        if df.empty:
            logging.info(
                "[%s] State CSV %s is empty; starting fresh.",
                self.model_name,
                self._state_csv,
            )
            return

        latest = df.iloc[-1]

        balance_value = _to_float(latest.get("total_balance"))
        if balance_value is not None:
            self.balance = balance_value

        fees_value = _to_float(latest.get("total_fees_paid"))
        if fees_value is not None:
            self.total_fees_paid = fees_value

        equity_value = _to_float(latest.get("total_equity"))

        positions_raw = latest.get("position_details")
        positions_loaded = 0
        if isinstance(positions_raw, str) and positions_raw.strip():
            try:
                self.positions = json.loads(positions_raw)
                for pos in self.positions.values():
                    if "fees_paid" not in pos:
                        pos["fees_paid"] = (
                            abs(pos["quantity"])
                            * pos["entry_price"]
                            * config.TRADING_FEE_RATE
                        )
                positions_loaded = len(self.positions)
            except json.JSONDecodeError as e:
                logging.error(
                    "[%s] Failed to decode position details from CSV: %s",
                    self.model_name,
                    e,
                    exc_info=True,
                )

        history_loaded = self._hydrate_equity_history(df)
        if not history_loaded and equity_value is not None:
            self.equity_history.append(equity_value)

        logging.info(
            "[%s] Restored state from CSV %s (balance: %.2f, positions: %d)",
            self.model_name,
            self._state_csv,
            self.balance,
            positions_loaded,
        )

        if self.positions:
            for coin, pos in self.positions.items():
                logging.info(
                    "[%s]   Restored position: %s %s @ $%.4f (qty: %.4f, leverage: %dx)",
                    self.model_name,
                    coin,
                    pos.get("side", "").upper(),
                    pos.get("entry_price", 0),
                    pos.get("quantity", 0),
                    pos.get("leverage", 1),
                )

    def _hydrate_equity_history(self, df: Optional[pd.DataFrame] = None) -> bool:
        """Populate equity history from CSV data for Sharpe calculations."""
        try:
            if df is None:
                if not self._state_csv.exists():
                    return False
                df = pd.read_csv(self._state_csv, usecols=["total_equity"])
        except Exception as e:
            logging.error(
                "[%s] Failed to load equity history from %s: %s",
                self.model_name,
                self._state_csv,
                e,
                exc_info=True,
            )
            return False

        if "total_equity" not in df.columns:
            return False

        series = pd.to_numeric(df["total_equity"], errors="coerce").dropna()
        if series.empty:
            return False

        self.equity_history = series.tolist()
        return True

    def save_state(self, latest_summary: Optional[Dict[str, Any]] = None):
        """Persist current balance, equity, and open positions."""
        payload: Dict[str, Any] = {
            "balance": _to_float(self.balance) or self.balance,
            "positions": _to_serializable(self.positions),
            "initial_capital": _to_float(self.initial_capital) or self.initial_capital,
            "start_time": self.start_time.isoformat(),
            "total_fees_paid": _to_float(self.total_fees_paid) or self.total_fees_paid,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        if latest_summary:
            total_balance = _to_float(latest_summary.get("total_balance"))
            total_equity = _to_float(latest_summary.get("total_equity"))
            total_margin = _to_float(latest_summary.get("total_margin"))
            total_return_pct = _to_float(latest_summary.get("total_return_pct"))
            total_fees_paid = _to_float(latest_summary.get("total_fees_paid"))

            if total_balance is not None:
                payload["last_total_balance"] = total_balance
            if total_equity is not None:
                payload["last_total_equity"] = total_equity
            if total_margin is not None:
                payload["last_total_margin"] = total_margin
            if total_return_pct is not None:
                payload["last_total_return_pct"] = total_return_pct
            if total_fees_paid is not None:
                payload["last_total_fees_paid"] = total_fees_paid

        try:
            self._state_dir.mkdir(parents=True, exist_ok=True)
            with open(self._state_file, "w") as f:
                json.dump(payload, f, indent=2)

            # Upload to S3
            upload_success = config.aws.upload_directory_to_s3(
                local_directory=str(config.DATA_DIR) + "/" + self.model_name,
                s3_base_path=str(config.PROJECT_S3_PATH) + self.model_name,
            )
            if upload_success["success"] > 0:
                logging.info(
                    "State saved locally and uploaded to S3: %s",
                    str(config.PROJECT_S3_PATH) + self.model_name,
                )
            else:
                logging.warning("State saved locally but S3 upload failed")
        except Exception as e:
            logging.error(
                "[%s] Failed to save state: %s", self.model_name, e, exc_info=True
            )

    def calculate_unrealized_pnl(self, coin: str, current_price: float) -> float:
        pos = self.positions[coin]
        if pos["side"] == "long":
            return (current_price - pos["entry_price"]) * pos["quantity"]
        return (pos["entry_price"] - current_price) * pos["quantity"]

    def get_summary(self, market_snapshots: Dict[str, Any]) -> Dict[str, Any]:
        """Calculates a full summary of the current portfolio state."""
        total_margin = sum(float(p.get("margin", 0.0)) for p in self.positions.values())
        total_equity = self.balance + total_margin

        # Update each position with current unrealized PnL
        for coin, pos in self.positions.items():
            price = market_snapshots.get(coin, {}).get("price", pos["entry_price"])
            unrealized_pnl = self.calculate_unrealized_pnl(coin, price)
            pos["unrealized_pnl"] = unrealized_pnl
            pos["current_price"] = price
            total_equity += unrealized_pnl

        total_return_pct = (
            (total_equity - self.initial_capital) / self.initial_capital
        ) * 100
        net_unrealized_pnl = total_equity - self.balance - total_margin

        # Add equity to history for ratio calculations
        self.equity_history.append(total_equity)

        # Calculate Sharpe and Sortino ratios
        sharpe_ratio = calculate_sharpe_ratio(
            self.equity_history, config.CHECK_INTERVAL, config.RISK_FREE_RATE
        )

        # Format position details as JSON string for CSV storage
        position_details = json.dumps(self.positions) if self.positions else ""

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_balance": self.balance,
            "positions": self.positions,
            "start_time": self.start_time,
            "invocation_count": self.invocation_count,
            "total_equity": total_equity,
            "total_margin": total_margin,
            "total_return_pct": total_return_pct,
            "num_positions": len(self.positions),
            "position_details": position_details,
            "net_unrealized_pnl": net_unrealized_pnl,
            "sharpe_ratio": sharpe_ratio,
            "total_fees_paid": self.total_fees_paid,
        }


def get_llm_decisions(
    state: TradingState, market_snapshots: Dict[str, Any], model_name: str
) -> Optional[Dict[str, Any]]:
    """Calls the LLM API and returns trading decisions."""
    model_config = config.LLM_MODELS[model_name]

    api_key = config.OPENROUTER_API_KEY
    base_url = config.OPENROUTER_BASE_URL

    llm_client = OpenAI(api_key=api_key, base_url=base_url)
    if not llm_client:
        logging.error("LLM client is not available.")
        return None

    prompt = prompts.create_trading_prompt(
        state.get_summary(market_snapshots), market_snapshots
    )

    try:
        utils.log_ai_message(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "direction": "sent",
                "role": "system",
                "content": prompts.TRADING_RULES_PROMPT,
                "metadata": {"model": model_name},
            },
            model_name=model_name,
        )
        utils.log_ai_message(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "direction": "sent",
                "role": "user",
                "content": prompt,
                "metadata": {"model": model_name},
            },
            model_name=model_name,
        )
        model_config = config.LLM_MODELS[model_name]

        response = llm_client.chat.completions.create(
            model=model_config["model_id"],
            messages=[
                {"role": "system", "content": prompts.TRADING_RULES_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=model_config["temperature"],
            max_tokens=model_config["max_tokens"],
            response_format={"type": "json_object"},
        )
        message = response.choices[0].message

        raw_content = message.content
        parsed_payload = getattr(message, "parsed", None)

        normalized_content: Optional[str] = None

        if raw_content:
            if isinstance(raw_content, list):
                normalized_content = "".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in raw_content
                ).strip()
            else:
                normalized_content = str(raw_content).strip()

        model_name_lower = model_name.lower()
        if normalized_content and "claude" in model_name_lower:
            normalized_content = (
                normalized_content.replace("```json", "").replace("```", "").strip()
            )

        decision_payload: Any
        if parsed_payload is not None:
            if isinstance(parsed_payload, (dict, list)):
                decision_payload = parsed_payload
            else:
                try:
                    decision_payload = json.loads(json.dumps(parsed_payload))
                except TypeError:
                    logging.error(
                        "[%s] Parsed payload type %s is not JSON serializable.",
                        model_name,
                        type(parsed_payload),
                    )
                    return None
            log_content = normalized_content or json.dumps(decision_payload)
        else:
            if not normalized_content:
                logging.error("LLM API returned empty content.")
                return None
            try:
                decision_payload = json.loads(normalized_content)
            except json.JSONDecodeError as exc:
                logging.error(
                    "Failed to parse LLM content as JSON: %s", exc, exc_info=True
                )
                logging.debug("Raw LLM content: %s", normalized_content)
                return None
            log_content = normalized_content

        utils.log_ai_message(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "direction": "received",
                "role": "assistant",
                "content": log_content,
                "metadata": {"id": response.id},
            },
            model_name=model_name,
        )

        if not normalized_content:
            logging.error("LLM API returned empty content.")
            return None

        return decision_payload

    except Exception as e:
        logging.error(f"Error calling LLM API: {e}", exc_info=True)
        return None


def check_stop_loss_take_profit(state: TradingState, market_snapshots: Dict[str, Any]):
    """Check and execute stop loss / take profit for all positions."""
    for coin in list(state.positions.keys()):
        if coin not in market_snapshots:
            continue

        current_price = market_snapshots[coin]["price"]
        pos = state.positions[coin]

        # Check stop loss
        if pos["side"] == "long" and current_price <= pos["stop_loss"]:
            execute_trade(
                state,
                coin,
                {"signal": "close", "justification": "Stop loss hit"},
                current_price,
            )
        elif pos["side"] == "short" and current_price >= pos["stop_loss"]:
            execute_trade(
                state,
                coin,
                {"signal": "close", "justification": "Stop loss hit"},
                current_price,
            )

        # Check take profit
        elif pos["side"] == "long" and current_price >= pos["profit_target"]:
            execute_trade(
                state,
                coin,
                {"signal": "close", "justification": "Take profit hit"},
                current_price,
            )
        elif pos["side"] == "short" and current_price <= pos["profit_target"]:
            execute_trade(
                state,
                coin,
                {"signal": "close", "justification": "Take profit hit"},
                current_price,
            )


def calculate_unrealized_pnl(
    coin: str, current_price: float, pos: Dict[str, Any]
) -> float:
    """Calculate unrealized PnL for a position."""
    if pos["side"] == "long":
        return (current_price - pos["entry_price"]) * pos["quantity"]
    else:  # short
        return (pos["entry_price"] - current_price) * pos["quantity"]


def generate_portfolio_summary(
    state: TradingState, market_snapshots: Dict[str, Any], summary_data: Dict[str, Any]
) -> tuple[Optional[str], Optional[str]]:
    """
    Generate portfolio summaries using an LLM.
    Returns tuple: (professional_summary, short_gen_z_summary)
    """
    api_key = config.OPENROUTER_API_KEY
    if not api_key:
        logging.error(
            "OPENROUTER_API_KEY not available for portfolio summary generation."
        )
        return None, None

    llm_client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")

    # Use cheap but decent models for summaries
    # Options:
    # - "google/gemini-flash-1.5-8b" - Very fast and cheap
    # - "qwen/qwen-2.5-7b-instruct" - Extremely cheap (~$0.05/$0.50 per 1M tokens)
    # - "meta-llama/llama-3.1-8b-instruct:free" - Free tier on OpenRouter
    SUMMARY_MODEL = "openai/gpt-4o-mini"

    # Create a structured data summary for the LLM
    portfolio_context = {
        "total_equity": summary_data["total_equity"],
        "total_balance": summary_data["total_balance"],
        "total_return_pct": summary_data["total_return_pct"],
        "sharpe_ratio": summary_data["sharpe_ratio"],
        "num_positions": summary_data["num_positions"],
        "net_unrealized_pnl": summary_data["net_unrealized_pnl"],
        "total_fees_paid": summary_data.get("total_fees_paid", 0.0),
        "positions": [],
    }

    # Add position details
    for coin, pos in state.positions.items():
        current_price = market_snapshots.get(coin, {}).get("price", pos["entry_price"])
        unrealized_pnl = state.calculate_unrealized_pnl(coin, current_price)
        pnl_pct = (
            (unrealized_pnl / (pos["margin"])) * 100 if pos.get("margin", 0) > 0 else 0
        )

        portfolio_context["positions"].append(
            {
                "coin": coin,
                "side": pos["side"],
                "entry_price": pos["entry_price"],
                "current_price": current_price,
                "quantity": pos["quantity"],
                "leverage": pos.get("leverage", 1),
                "unrealized_pnl": unrealized_pnl,
                "pnl_pct": pnl_pct,
                "profit_target": pos["profit_target"],
                "stop_loss": pos["stop_loss"],
                "justification": pos.get("justification", ""),
                "invalidation_condition": pos.get("invalidation_condition", ""),
                "confidence": pos.get("confidence", 0.5),
                "fees_paid": pos.get("fees_paid", 0.0),
            }
        )

    # Professional summary user prompt
    professional_user_prompt = f"""Please provide a concise portfolio update based on the following data:

Portfolio Metrics:
- Total Equity: ${portfolio_context['total_equity']:.2f}
- Available Cash: ${portfolio_context['total_balance']:.2f}
- Total Return: {portfolio_context['total_return_pct']:.2f}%
- Sharpe Ratio: {f"{portfolio_context['sharpe_ratio']:.2f}" if portfolio_context['sharpe_ratio'] is not None else 'N/A (insufficient data)'}
- Unrealized P&L: ${portfolio_context['net_unrealized_pnl']:.2f}
- Fees Paid (lifetime): ${portfolio_context['total_fees_paid']:.2f}

Current Positions ({portfolio_context['num_positions']}):
{json.dumps(portfolio_context['positions'], indent=2) if portfolio_context['positions'] else "No open positions"}

Please provide a professional, concise summary for the client."""

    # Short summary user prompt
    short_user_prompt = f"""Create a super short Gen-Z style portfolio update:

- Total Return: {portfolio_context['total_return_pct']:.2f}%
- Positions: {portfolio_context['num_positions']}
- P&L: ${portfolio_context['net_unrealized_pnl']:.2f}
- Fees Paid: ${portfolio_context['total_fees_paid']:.2f}

{json.dumps(portfolio_context['positions'], indent=2) if portfolio_context['positions'] else "No positions right now"}

Generate ONE punchy sentence summarizing your portfolio stance."""

    try:
        # Generate professional summary
        response1 = llm_client.chat.completions.create(
            model=SUMMARY_MODEL,
            messages=[
                {"role": "system", "content": prompts.PROFESSIONAL_SUMMARY_PROMPT},
                {"role": "user", "content": professional_user_prompt},
            ],
            temperature=config.PROFESSIONAL_SUMMARY_TEMPERATURE,
            max_tokens=config.PROFESSIONAL_SUMMARY_MAX_TOKENS,
        )
        professional_summary = response1.choices[0].message.content
        professional_summary = (
            professional_summary.strip() if professional_summary else None
        )

        # Generate short Gen-Z summary
        response2 = llm_client.chat.completions.create(
            model=SUMMARY_MODEL,
            messages=[
                {"role": "system", "content": prompts.SHORT_SUMMARY_PROMPT},
                {"role": "user", "content": short_user_prompt},
            ],
            temperature=config.SHORT_SUMMARY_TEMPERATURE,
            max_tokens=config.SHORT_SUMMARY_MAX_TOKENS,
        )
        short_summary = response2.choices[0].message.content
        short_summary = short_summary.strip() if short_summary else None

        return professional_summary, short_summary

    except Exception as e:
        logging.error(f"Error generating portfolio summary: {e}", exc_info=True)
        return None, None


MIN_DATA_POINTS_FOR_SHARPE = int(60 / (config.CHECK_INTERVAL / 60))


def calculate_sharpe_ratio(
    equity_values: Iterable[float],
    period_seconds: float,
    risk_free_rate: float = config.RISK_FREE_RATE,
) -> Optional[float]:
    """
    Compute the annualized Sharpe ratio from equity snapshots.

    Args:
        equity_values: Sequence of equity values in chronological order.
        period_seconds: Average period between snapshots (used to annualize).
        risk_free_rate: Annualized risk-free rate (decimal form).

    Returns:
        Annualized Sharpe ratio, or None if insufficient data or calculation fails.
    """
    values = [
        float(v)
        for v in equity_values
        if isinstance(v, (int, float, np.floating)) and np.isfinite(v)
    ]

    # Require minimum number of data points for meaningful Sharpe calculation
    # With 5-minute intervals, 12 points = 1 hour, 18 points = 1.5 hours
    MIN_DATA_POINTS = int(60 / (config.CHECK_INTERVAL / 60))
    if len(values) < MIN_DATA_POINTS:
        return None

    # Use log returns for better numerical stability with small changes
    # log(1 + r) ≈ r for small r, but handles large changes better
    log_values = np.log(np.array(values, dtype=float))
    log_returns = np.diff(log_values)

    # Filter out invalid returns
    log_returns = log_returns[np.isfinite(log_returns)]
    if log_returns.size < max(2, MIN_DATA_POINTS_FOR_SHARPE - 1):
        return None

    period_seconds = (
        float(period_seconds)
        if period_seconds and period_seconds > 0
        else config.CHECK_INTERVAL
    )
    periods_per_year = (365 * 24 * 60 * 60) / period_seconds
    if not np.isfinite(periods_per_year) or periods_per_year <= 0:
        return None

    # Calculate per-period risk-free rate
    # For log returns, we need log(1 + rf_period), but for small rates, ln(1+r) ≈ r
    # Using the exact log form for precision
    per_period_rf_simple = risk_free_rate / periods_per_year
    per_period_rf_log = np.log(1.0 + per_period_rf_simple)

    # Mean excess return per period
    mean_return = log_returns.mean()
    excess_return = mean_return - per_period_rf_log

    if not np.isfinite(excess_return):
        return None

    return_std = np.std(log_returns, ddof=1)
    if return_std <= 0 or not np.isfinite(return_std):
        return None

    # Calculate annualized Sharpe ratio
    # For log returns: Sharpe = (mean_return - rf) / std * sqrt(periods_per_year)
    sharpe = (excess_return / return_std) * np.sqrt(periods_per_year)

    if not np.isfinite(sharpe):
        return None

    return float(sharpe)


def execute_trade(
    state: TradingState, coin: str, decision: Dict[str, Any], price: float
):
    """Executes a single trade based on an LLM decision."""
    # ... (Implementation of execute_entry and execute_close from original bot.py)
    # This function would be much larger in a real implementation.
    logging.info(
        f"Executing trade for {coin} based on signal: {decision.get('signal')}"
    )
    signal = decision.get("signal")
    fee_rate = getattr(config, "TRADING_FEE_RATE", 0.0)
    if signal == "entry":
        # valid no existing position
        if coin in state.positions:
            logging.warning(
                "[%s] Already have position for %s, skipping entry",
                state.model_name,
                coin,
            )
            return

        # extract decision parameters
        side = decision.get("side", "hold")
        leverage = decision.get("leverage", 10)
        quantity = decision.get("quantity", 0.0)
        profit_target = decision.get("profit_target", 0.0)
        stop_loss = decision.get("stop_loss", 0.0)
        risk_idr = decision.get("risk_idr", 0.0)
        # Calculate position size based on risk
        stop_distance = abs(price - stop_loss)
        if stop_distance == 0:
            logging.warning(
                "[%s] %s: Invalid stop loss, skipping", state.model_name, coin
            )
            return

        if price <= 0:
            logging.warning(
                "[%s] %s: Invalid price %.6f, skipping", state.model_name, coin, price
            )
            return

        if leverage <= 0:
            logging.warning(
                "[%s] %s: Invalid leverage %.4f, skipping",
                state.model_name,
                coin,
                leverage,
            )
            return

        if quantity <= 0:
            logging.warning(
                "[%s] %s: Non-positive quantity %.6f, skipping",
                state.model_name,
                coin,
                quantity,
            )
            return

        original_quantity = quantity
        position_value = abs(quantity * price)
        margin_required = position_value / leverage
        entry_fee = position_value * fee_rate
        total_cost = margin_required + entry_fee

        # Check sufficient balance
        if total_cost > state.balance:
            affordability_denominator = price * ((1.0 / leverage) + fee_rate)
            if affordability_denominator <= 0:
                logging.warning(
                    "[%s] %s: Invalid affordability denominator %.6f, skipping",
                    state.model_name,
                    coin,
                    affordability_denominator,
                )
                return

            max_affordable_quantity = state.balance / affordability_denominator
            adjusted_quantity = min(original_quantity, max_affordable_quantity)

            if adjusted_quantity <= 0:
                logging.warning(
                    "[%s] %s: Insufficient balance for minimum position after fees (need %.2f, have %.2f)",
                    state.model_name,
                    coin,
                    total_cost,
                    state.balance,
                )
                return

            if adjusted_quantity < original_quantity:
                logging.info(
                    "[%s] %s: Scaling quantity from %.6f to %.6f to cover entry fees",
                    state.model_name,
                    coin,
                    original_quantity,
                    adjusted_quantity,
                )

            quantity = adjusted_quantity
            position_value = abs(quantity * price)
            margin_required = position_value / leverage
            entry_fee = position_value * fee_rate
            total_cost = margin_required + entry_fee
            risk_idr = stop_distance * quantity

        if total_cost > state.balance:
            logging.warning(
                "[%s] %s: Insufficient balance for margin + fee after adjustment (need %.2f, have %.2f)",
                state.model_name,
                coin,
                total_cost,
                state.balance,
            )
            return

        # Create position
        state.positions[coin] = {
            "side": side,
            "quantity": quantity,
            "entry_price": price,
            "current_price": price,
            "profit_target": profit_target,
            "stop_loss": stop_loss,
            "leverage": leverage,
            "confidence": decision.get("confidence", 0.5),
            "margin": margin_required,
            "risk_idr": risk_idr,
            "unrealized_pnl": 0.0,  # Initialize at 0 on entry
            "invalidation_condition": decision.get("invalidation_condition", ""),
            "justification": decision.get("justification", ""),
            "fees_paid": entry_fee,
        }

        # Update balance
        state.balance -= total_cost
        state.total_fees_paid += entry_fee
        logging.info(
            "[%s] ENTRY: %s %s %.4f @ $%.4f | Fee: $%.4f",
            state.model_name,
            coin,
            side.upper(),
            quantity,
            price,
            entry_fee,
        )

        trade_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "coin": coin,
            "action": "ENTRY",
            "side": side,
            "quantity": quantity,
            "price": price,
            "profit_target": profit_target,
            "stop_loss": stop_loss,
            "leverage": leverage,
            "confidence": decision.get("confidence", 0.5),
            "pnl": 0.0,  # No PnL on entry
            "balance_after": state.balance,
            "reason": decision.get("justification", "AI entry signal"),
            "fee": entry_fee,
        }
        trade_data["net_pnl"] = trade_data["pnl"] - entry_fee

        utils.log_trade(trade_data, model_name=state.model_name)
        state.current_iteration_trades.append(trade_data)

    elif signal == "close":
        if coin not in state.positions:
            logging.warning("[%s] %s: No position to close", state.model_name, coin)
            return

        pos = state.positions[coin]
        pnl = calculate_unrealized_pnl(coin, price, pos)
        close_fee = abs(pos["quantity"]) * price * fee_rate
        total_position_fees = pos.get("fees_paid", 0.0) + close_fee
        net_trade_pnl = pnl - close_fee
        net_position_pnl = pnl - total_position_fees

        # Return margin + PnL
        state.balance += pos["margin"] + pnl - close_fee
        state.total_fees_paid += close_fee

        logging.info(
            "[%s] CLOSE: %s %s @ $%.4f | PnL: $%.2f | Fee: $%.4f | Net: $%.2f",
            state.model_name,
            coin,
            pos["side"].upper(),
            price,
            pnl,
            close_fee,
            net_position_pnl,
        )

        trade_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "coin": coin,
            "action": "CLOSE",
            "side": pos["side"],
            "quantity": pos["quantity"],
            "price": price,
            "profit_target": pos["profit_target"],
            "stop_loss": pos["stop_loss"],
            "leverage": pos["leverage"],
            "confidence": pos["confidence"],
            "pnl": pnl,
            "balance_after": state.balance,
            "reason": decision.get("justification", "AI close signal"),
            "fee": close_fee,
        }
        trade_data["net_pnl"] = net_trade_pnl
        trade_data["position_fee_total"] = total_position_fees
        trade_data["position_net_pnl"] = net_position_pnl

        utils.log_trade(trade_data, model_name=state.model_name)
        state.current_iteration_trades.append(trade_data)

        # Remove position
        del state.positions[coin]


def is_idss_break_time() -> bool:
    """
    Check if it's currently break time for Indonesian Stock Market (IDSS).
    Break time: 12:00 - 13:30 WIB
    """
    wib_tz = ZoneInfo("Asia/Jakarta")
    now = datetime.now(wib_tz)

    # Check if weekend (no break on weekends, market is just closed)
    if now.weekday() >= 5:
        return False

    # Break time: 12:00 - 13:30 WIB
    break_start = now.replace(hour=12, minute=0, second=0, microsecond=0)
    break_end = now.replace(hour=13, minute=30, second=0, microsecond=0)

    return break_start <= now < break_end


def is_market_open() -> bool:
    """
    Check if the market is currently open based on ASSET_MODE.

    US Stock Market: 9:30 AM - 4:00 PM ET, Monday-Friday
    Indonesian Stock Market (IDSS):
        - Session 1: 09:00 - 12:00 WIB
        - Break: 12:00 - 13:30 WIB
        - Session 2: 13:30 - 15:00 WIB (includes pre-closing auction)
        - Monday-Friday, WIB (GMT+7)
    """
    if config.ASSET_MODE.lower() == "us_stock":
        # US stock market hours
        et_tz = ZoneInfo("America/New_York")
        now = datetime.now(et_tz)

        # Check if weekend
        if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False

        # Check if within market hours (9:30 AM - 4:00 PM ET)
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)

        return market_open <= now <= market_close

    elif config.ASSET_MODE.lower() == "idss":
        # Indonesian stock market hours (IDX - Indonesia Stock Exchange)
        wib_tz = ZoneInfo("Asia/Jakarta")  # WIB = GMT+7
        now = datetime.now(wib_tz)

        # Check if weekend
        if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False

        # Session 1: 09:00 - 12:00 WIB
        session1_open = now.replace(hour=9, minute=0, second=0, microsecond=0)
        session1_close = now.replace(hour=12, minute=0, second=0, microsecond=0)

        # Session 2: 13:30 - 15:00 WIB (includes pre-closing auction)
        session2_open = now.replace(hour=13, minute=30, second=0, microsecond=0)
        session2_close = now.replace(hour=15, minute=0, second=0, microsecond=0)

        # Check if within either trading session
        in_session1 = session1_open <= now < session1_close
        in_session2 = session2_open <= now <= session2_close

        return in_session1 or in_session2

    else:
        # For crypto or other asset modes, market is always open
        return True


def run_trading_loop(model_name: str):
    """The main event loop for the trading bot."""
    utils.setup_logging()
    utils.init_csv_files(model_name=model_name)

    state = TradingState(model_name)
    state.load_state()

    logging.info("[%s] Initializing clients...", model_name)

    # Only check Binance client for crypto mode
    if config.ASSET_MODE.lower() == "crypto":
        if not clients.get_binance_client():
            logging.critical(
                "[%s] Failed to initialize Binance client. Exiting.", model_name
            )
            return

    # LLM client is always required
    if not clients.get_llm_client():
        logging.critical("[%s] Failed to initialize LLM client. Exiting.", model_name)
        return

    currency_symbol = utils.get_currency_symbol()
    logging.info(
        "[%s] Starting capital: %s%.2f",
        model_name,
        currency_symbol,
        state.initial_capital,
    )
    logging.info(
        "[%s] Monitoring: %s", model_name, list(config.SYMBOL_TO_COIN.values())
    )

    while True:
        try:
            # Check if market is still open (for US stocks)
            if config.ASSET_MODE.lower() == "us_stock":
                if not is_market_open():
                    et_tz = ZoneInfo("America/New_York")
                    current_time = datetime.now(et_tz).strftime("%Y-%m-%d %H:%M:%S %Z")
                    logging.info(
                        f"Market is now closed (current time: {current_time}). Waiting for market to open..."
                    )
                    state.save_state()
                    state.invocation_count = 0  # Reset iteration count
                    logging.info(
                        "Sleeping for 1 minute before checking market status again..."
                    )
                    time.sleep(60)  # Sleep for 1 minute
                    continue  # Skip this iteration and retry

            if config.ASSET_MODE.lower() == "idss":
                # Check if it's break time (12:00 - 13:30 WIB)
                if is_idss_break_time():
                    wib_tz = ZoneInfo("Asia/Jakarta")
                    current_time = datetime.now(wib_tz).strftime("%H:%M:%S %Z")
                    logging.info(
                        f"IDX market is on break (current time: {current_time}). Sleeping for 1 minute..."
                    )
                    time.sleep(60)  # Sleep for 1 minute
                    continue  # Skip this iteration and retry

                # Check if market is closed for the day
                if not is_market_open():
                    wib_tz = ZoneInfo("A sia/Jakarta")
                    current_time = datetime.now(wib_tz).strftime("%Y-%m-%d %H:%M:%S %Z")
                    logging.info(
                        f"IDX market is now closed (current time: {current_time}). Waiting for market to open..."
                    )
                    state.save_state()
                    state.invocation_count = 0  # Reset iteration count
                    logging.info(
                        "Sleeping for 1 minute before checking market status again..."
                    )
                    time.sleep(60)  # Sleep for 1 minute
                    continue  # Skip this iteration and retry

            state.invocation_count += 1
            state.current_iteration_messages = []
            state.current_iteration_trades = []  # Reset trades for this iteration

            now_utc = datetime.now(timezone.utc)
            header = f"\n{Fore.CYAN}[{model_name}] {'='*20} Iteration {state.invocation_count} - {now_utc.strftime('%Y-%m-%d %H:%M:%S UTC')} {'='*20}{Style.RESET_ALL}"
            print(header)
            state.current_iteration_messages.append(utils.strip_ansi_codes(header))

            # 1. Fetch shared market data snapshot
            market_snapshots = _market_coordinator.fetch_and_update()
            if not market_snapshots:
                logging.warning(
                    "[%s] No market data available; skipping iteration.", model_name
                )
                time.sleep(config.CHECK_INTERVAL)
                continue

            logging.info("[%s] Checking stop loss / take profit...", model_name)
            check_stop_loss_take_profit(state, market_snapshots)

            # 2. Get LLM decisions
            logging.info("[%s] Requesting trading decisions from LLM...", model_name)
            decisions = get_llm_decisions(state, market_snapshots, model_name)

            # 3. Execute trades
            if decisions:
                for coin, decision in decisions.items():
                    if coin not in config.SYMBOL_TO_COIN.values():
                        continue

                    # Validate that decision is a dictionary
                    if not isinstance(decision, dict):
                        logging.warning(
                            "[%s] Invalid decision format for %s: expected dict, got %s. Skipping.",
                            model_name,
                            coin,
                            type(decision).__name__,
                        )
                        continue

                    utils.log_ai_decision(
                        {
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "model": model_name,
                            "coin": coin,
                            **decision,
                        },
                        model_name=model_name,
                    )

                    current_price = market_snapshots.get(coin, {}).get("price")
                    if not current_price:
                        continue

                    signal = decision.get("signal")
                    if signal == "entry":
                        execute_trade(state, coin, decision, current_price)
                    elif signal == "close":
                        execute_trade(state, coin, decision, current_price)
                    elif signal == "hold" and coin in state.positions:
                        # Update position metadata with latest decision rationale
                        # This keeps justification and other fields current
                        pos = state.positions[coin]
                        if "justification" in decision:
                            pos["justification"] = decision["justification"]
                        if "invalidation_condition" in decision:
                            pos["invalidation_condition"] = decision[
                                "invalidation_condition"
                            ]
                        if "confidence" in decision:
                            pos["confidence"] = decision.get(
                                "confidence", pos.get("confidence", 0.5)
                            )
            else:
                logging.warning("[%s] No decisions received from LLM.", model_name)

            # 4. Log and display summary
            summary = state.get_summary(market_snapshots)

            # 4.5. Generate and display professional portfolio summary
            professional_summary, short_summary = generate_portfolio_summary(
                state, market_snapshots, summary
            )

            if professional_summary:
                summary_header = f"\n{Fore.GREEN}{'='*20} Portfolio Manager Summary {'='*20}{Style.RESET_ALL}"
                print(summary_header)
                print(f"{Fore.WHITE}{professional_summary}{Style.RESET_ALL}")
                print(f"{Fore.GREEN}{'='*63}{Style.RESET_ALL}\n")
                logging.info(
                    "[%s] Portfolio Summary: %s", model_name, professional_summary
                )

                # Add summaries to the state dict before logging
                summary["portfolio_summary"] = professional_summary
            else:
                summary["portfolio_summary"] = ""

            if short_summary:
                short_header = f"{Fore.CYAN}{'='*20} Quick Take (Gen-Z Style) {'='*20}{Style.RESET_ALL}"
                print(short_header)
                print(f"{Fore.YELLOW}💰 {short_summary}{Style.RESET_ALL}")
                print(f"{Fore.CYAN}{'='*63}{Style.RESET_ALL}\n")
                logging.info("[%s] Short Summary: %s", model_name, short_summary)

                summary["short_summary"] = short_summary
            else:
                summary["short_summary"] = ""

            utils.log_portfolio_state(summary, model_name=model_name)
            state.save_state(latest_summary=summary)

            # 5. Send Telegram notification
            if config.TELEGRAM_BOT_TOKEN and config.TELEGRAM_CHAT_ID:
                try:
                    telegram_message = utils.format_trading_signal_message(
                        new_trades=state.current_iteration_trades,
                        positions=state.positions,
                        market_snapshots=market_snapshots,
                        short_summary=summary.get("short_summary", ""),
                        total_equity=summary["total_equity"],
                        total_return_pct=summary["total_return_pct"],
                        net_unrealized_pnl=summary["net_unrealized_pnl"],
                        total_fees_paid=summary.get("total_fees_paid", 0.0),
                        model_name=model_name,
                    )
                    utils.send_telegram_message(telegram_message, parse_mode="HTML")
                    logging.info(
                        "[%s] Telegram notification sent successfully", model_name
                    )
                except Exception as e:
                    logging.error(
                        "[%s] Failed to send Telegram notification: %s",
                        model_name,
                        e,
                        exc_info=True,
                    )

            # 6. Wait for next interval
            logging.info(
                "[%s] Waiting %d seconds...", model_name, config.CHECK_INTERVAL
            )
            time.sleep(config.CHECK_INTERVAL)

        except KeyboardInterrupt:
            logging.info("[%s] Shutdown signal received.", model_name)
            state.save_state()
            break
        except Exception as e:
            logging.error("[%s] Error in main loop: %s", model_name, e, exc_info=True)
            state.save_state()
            time.sleep(60)
