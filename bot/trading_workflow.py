#!/usr/bin/env python3
"""
Main trading workflow, state management, and execution logic.
"""
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Iterable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from colorama import Fore, Style

from . import clients, utils
from config import config

from openai import OpenAI

if config.ASSET_MODE.lower() == "crypto":
    from . import data_processing_crypto as data_processing
    from . import prompts_v1 as prompts
elif config.ASSET_MODE.lower() == "us_stock":
    from . import data_processing_stock as data_processing
    from . import prompts_idss as prompts
elif config.ASSET_MODE.lower() == "idss":
    from . import data_processing_idss as data_processing
    from . import prompts_stock as prompts

class TradingState:
    """Manages the full state of the trading bot."""

    def __init__(self):
        self.balance: float = config.START_CAPITAL
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.equity_history: list[float] = []
        self.start_time: datetime = datetime.now(timezone.utc)
        self.invocation_count: int = 0
        self.current_iteration_messages: list[str] = []
        self.current_iteration_trades: list[Dict[str, Any]] = []

    def load_state(self):
        """Load persisted balance and positions if available."""
        # Try to download from S3 first
        download_success = config.aws.download_directory_from_s3(s3_base_path=config.PROJECT_S3_PATH, local_directory=config.DATA_DIR)
        
        if download_success["success"] > 0:
            logging.info("Successfully downloaded state from S3: %s", config.PROJECT_S3_PATH)
        else:
            logging.info("Could not download from S3, will check for local state file")
        
        # Check if local file exists (either downloaded or already present)
        if not utils.STATE_JSON.exists():
            logging.info("No existing state file found; starting fresh.")
            return
            
        try:
            with open(utils.STATE_JSON, "r") as f:
                data = json.load(f)
            self.balance = float(data.get("balance", config.START_CAPITAL))
            
            # Load positions if they exist
            if "positions" in data and isinstance(data["positions"], dict):
                self.positions = data["positions"]
                source = "S3" if download_success else "local file"
                logging.info(
                    "Loaded state from %s (balance: %.2f, positions: %d)",
                    source,
                    self.balance,
                    len(self.positions)
                )
                # Log each position for visibility
                for coin, pos in self.positions.items():
                    logging.info(
                        "  Restored position: %s %s @ $%.4f (qty: %.4f, leverage: %dx)",
                        coin,
                        pos.get("side", "").upper(),
                        pos.get("entry_price", 0),
                        pos.get("quantity", 0),
                        pos.get("leverage", 1)
                    )
            else:
                source = "S3" if download_success else "local file"
                logging.info(
                    "Loaded state from %s (balance: %.2f, no positions)", 
                    source, 
                    self.balance
                )
        except Exception as e:
            logging.error("Failed to load state: %s", e, exc_info=True)

    def save_state(self):
        """Persist current balance and open positions."""
        try:
            # Save to local file
            with open(utils.STATE_JSON, "w") as f:
                json.dump(
                    {"balance": self.balance, "positions": self.positions}, f, indent=2
                )
            
            # Upload to S3
            upload_success = config.aws.upload_directory_to_s3(local_directory=config.DATA_DIR, s3_base_path=config.PROJECT_S3_PATH)
            if upload_success["success"] > 0:
                logging.info("State saved locally and uploaded to S3: %s", config.PROJECT_S3_PATH)
            else:
                logging.warning("State saved locally but S3 upload failed")
                
        except Exception as e:
            logging.error("Failed to save state: %s", e, exc_info=True)

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
            (total_equity - config.START_CAPITAL) / config.START_CAPITAL
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
        }


def get_llm_decisions(
    state: TradingState, market_snapshots: Dict[str, Any], model_name: str
) -> Optional[Dict[str, Any]]:
    """Calls the LLM API and returns trading decisions."""
    api_key = config.OPENROUTER_API_KEY
    if not api_key:
        logging.error("❌ OPENROUTER_API_KEY is not loaded!")
    else:
        logging.info(f"✅ OPENROUTER_API_KEY loaded (length: {len(api_key)})")

    llm_client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
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
            }
        )
        utils.log_ai_message(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "direction": "sent",
                "role": "user",
                "content": prompt,
                "metadata": {"model": model_name},
            }
        )
        model_config = config.LLM_MODELS[model_name]

        response = llm_client.chat.completions.create(
            model=model_config["model_id"],
            messages=[
                {"role": "system", "content": prompts.TRADING_RULES_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=4000,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        utils.log_ai_message(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "direction": "received",
                "role": "assistant",
                "content": content,
                "metadata": {"id": response.id},
            }
        )
        
        if not content:
            logging.error("LLM API returned empty content.")
            return None

        return json.loads(content)

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

Current Positions ({portfolio_context['num_positions']}):
{json.dumps(portfolio_context['positions'], indent=2) if portfolio_context['positions'] else "No open positions"}

Please provide a professional, concise summary for the client."""

    # Short summary user prompt
    short_user_prompt = f"""Create a super short Gen-Z style portfolio update:

- Total Return: {portfolio_context['total_return_pct']:.2f}%
- Positions: {portfolio_context['num_positions']}
- P&L: ${portfolio_context['net_unrealized_pnl']:.2f}

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
            temperature=0.7,
            max_tokens=500,
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
            temperature=0.8,  # Slightly higher for more creative/casual tone
            max_tokens=100,
        )
        short_summary = response2.choices[0].message.content
        short_summary = short_summary.strip() if short_summary else None

        return professional_summary, short_summary

    except Exception as e:
        logging.error(f"Error generating portfolio summary: {e}", exc_info=True)
        return None, None


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
    MIN_DATA_POINTS = 12
    if len(values) < MIN_DATA_POINTS:
        return None

    # Use log returns for better numerical stability with small changes
    # log(1 + r) ≈ r for small r, but handles large changes better
    log_values = np.log(np.array(values, dtype=float))
    log_returns = np.diff(log_values)

    # Filter out invalid returns
    log_returns = log_returns[np.isfinite(log_returns)]
    if log_returns.size == 0:
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

    # Standard deviation of returns
    return_std = np.std(
        log_returns, ddof=1
    )  # Use sample std (ddof=1) for better estimate

    # Require minimum std threshold to avoid division by extremely small numbers
    # This prevents unrealistic Sharpe ratios from tiny variations
    MIN_STD_THRESHOLD = 1e-6  # 0.0001% minimum std
    if return_std < MIN_STD_THRESHOLD:
        return None

    # Calculate annualized Sharpe ratio
    # For log returns: Sharpe = (mean_return - rf) / std * sqrt(periods_per_year)
    sharpe = (excess_return / return_std) * np.sqrt(periods_per_year)

    if not np.isfinite(sharpe):
        return None

    # Cap Sharpe ratio at reasonable values (±10) to avoid extreme outliers
    # Real-world Sharpe ratios rarely exceed ±5
    sharpe = np.clip(sharpe, -10.0, 10.0)

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
    if signal == "entry":
        # valid no existing position
        if coin in state.positions:
            logging.warning(f"Already have position for {coin}, skipping entry")
            return

        # extract decision parameters
        side = decision.get("side", "hold")
        leverage = decision.get("leverage", 10)
        quantity = decision.get("quantity", 0.0)
        profit_target = decision.get("profit_target", 0.0)
        stop_loss = decision.get("stop_loss", 0.0)
        risk_usd = decision.get("risk_usd", 0.0)
        # Calculate position size based on risk
        stop_distance = abs(price - stop_loss)
        if stop_distance == 0:
            logging.warning(f"{coin}: Invalid stop loss, skipping")
            return

        position_value = quantity * price
        margin_required = position_value / leverage

        # Check sufficient balance
        if margin_required > state.balance:
            logging.warning(f"{coin}: Insufficient balance for margin")
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
            "risk_usd": risk_usd,
            "unrealized_pnl": 0.0,  # Initialize at 0 on entry
            "invalidation_condition": decision.get("invalidation_condition", ""),
            "justification": decision.get("justification", ""),
        }

        # Update balance
        state.balance -= margin_required
        logging.info(f"ENTRY: {coin} {side.upper()} {quantity:.4f} @ ${price:.4f}")
        
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
        }
        
        utils.log_trade(trade_data)
        state.current_iteration_trades.append(trade_data)

    elif signal == "close":
        if coin not in state.positions:
            logging.warning(f"{coin}: No position to close")
            return

        pos = state.positions[coin]
        pnl = calculate_unrealized_pnl(coin, price, pos)

        # Return margin + PnL
        state.balance += pos["margin"] + pnl

        logging.info(
            f"CLOSE: {coin} {pos['side'].upper()} @ ${price:.4f} | PnL: ${pnl:.2f}"
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
        }
        
        utils.log_trade(trade_data)
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
    utils.init_csv_files()

    state = TradingState()
    state.load_state()

    logging.info("Initializing clients...")
    if not clients.get_llm_client():
        logging.critical("Failed to initialize required API clients. Exiting.")
        return

    logging.info(f"Starting capital: ${config.START_CAPITAL:.2f}")
    logging.info(f"Monitoring: {list(config.SYMBOL_TO_COIN.values())}")

    while True:
        try:
            # Check if market is still open (for US stocks)
            if config.ASSET_MODE.lower() == "us_stock":
                if not is_market_open():
                    et_tz = ZoneInfo("America/New_York")
                    current_time = datetime.now(et_tz).strftime('%Y-%m-%d %H:%M:%S %Z')
                    logging.info(f"Market is now closed (current time: {current_time}). Waiting for market to open...")
                    state.save_state()
                    state.invocation_count = 0  # Reset iteration count
                    logging.info("Sleeping for 1 minute before checking market status again...")
                    time.sleep(60)  # Sleep for 1 minute
                    continue  # Skip this iteration and retry
            
            if config.ASSET_MODE.lower() == "idss":
                # Check if it's break time (12:00 - 13:30 WIB)
                if is_idss_break_time():
                    wib_tz = ZoneInfo("Asia/Jakarta")
                    current_time = datetime.now(wib_tz).strftime('%H:%M:%S %Z')
                    logging.info(f"IDX market is on break (current time: {current_time}). Sleeping for 1 minute...")
                    time.sleep(60)  # Sleep for 1 minute
                    continue  # Skip this iteration and retry
                
                # Check if market is closed for the day
                if not is_market_open():
                    wib_tz = ZoneInfo("Asia/Jakarta")
                    current_time = datetime.now(wib_tz).strftime('%Y-%m-%d %H:%M:%S %Z')
                    logging.info(f"IDX market is now closed (current time: {current_time}). Waiting for market to open...")
                    state.save_state()
                    state.invocation_count = 0  # Reset iteration count
                    logging.info("Sleeping for 1 minute before checking market status again...")
                    time.sleep(60)  # Sleep for 1 minute
                    continue  # Skip this iteration and retry


            state.invocation_count += 1
            state.current_iteration_messages = []
            state.current_iteration_trades = []  # Reset trades for this iteration

            header = f"\n{Fore.CYAN}{'='*20} Iteration {state.invocation_count} - {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')} {'='*20}{Style.RESET_ALL}"
            print(header)
            state.current_iteration_messages.append(utils.strip_ansi_codes(header))

            # 1. Fetch market data
            logging.info("Fetching market data for all symbols...")
            market_snapshots = {}
            for symbol in config.SYMBOLS:
                snapshot = data_processing.collect_market_data(symbol)
                if snapshot:
                    market_snapshots[snapshot["coin"]] = snapshot

            if len(market_snapshots) != len(config.SYMBOLS):
                logging.warning("Failed to fetch market data for one or more symbols.")

            logging.info("Checking stop loss / take profit...")
            check_stop_loss_take_profit(state, market_snapshots)

            # 2. Get LLM decisions
            logging.info("Requesting trading decisions from LLM...")
            decisions = get_llm_decisions(state, market_snapshots, model_name)

            # 3. Execute trades
            if decisions:
                for coin, decision in decisions.items():
                    if coin not in config.SYMBOL_TO_COIN.values():
                        continue

                    utils.log_ai_decision(
                        {
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "coin": coin,
                            **decision,
                        }
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
                logging.warning("No decisions received from LLM.")

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
                logging.info(f"Portfolio Summary: {professional_summary}")

                # Add summaries to the state dict before logging
                summary["portfolio_summary"] = professional_summary
            else:
                summary["portfolio_summary"] = ""

            if short_summary:
                short_header = f"{Fore.CYAN}{'='*20} Quick Take (Gen-Z Style) {'='*20}{Style.RESET_ALL}"
                print(short_header)
                print(f"{Fore.YELLOW}💰 {short_summary}{Style.RESET_ALL}")
                print(f"{Fore.CYAN}{'='*63}{Style.RESET_ALL}\n")
                logging.info(f"Short Summary: {short_summary}")

                summary["short_summary"] = short_summary
            else:
                summary["short_summary"] = ""

            utils.log_portfolio_state(summary)

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
                    )
                    utils.send_telegram_message(telegram_message, parse_mode="HTML")
                    logging.info("Telegram notification sent successfully")
                except Exception as e:
                    logging.error(f"Failed to send Telegram notification: {e}", exc_info=True)

            # 6. Wait for next interval
            logging.info(f"Waiting {config.CHECK_INTERVAL} seconds...")
            time.sleep(config.CHECK_INTERVAL)

        except KeyboardInterrupt:
            logging.info("Shutdown signal received.")
            state.save_state()
            break
        except Exception as e:
            logging.error(f"Error in main loop: {e}", exc_info=True)
            state.save_state()
            time.sleep(60)
