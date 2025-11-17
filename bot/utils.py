#!/usr/bin/env python3
"""
Utility functions for the trading bot.
"""
import logging
import sys
import csv
import json
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests

from config import config


# --- LOGGING ---
def setup_logging():
    """Initializes logging configuration with console output."""
    # Create root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Remove any existing handlers
    logger.handlers.clear()
    
    # Create console handler with formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(console_handler)
    
    # Optionally add file handler for persistent logs
    try:
        log_file = "trading_bot.log"
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logging.info(f"Logging configured. Console + File output enabled ({log_file})")
    except Exception as e:
        logging.info(f"Logging configured. Console output enabled (File logging failed: {e})")


# --- CSV & STATE MANAGEMENT ---
STATE_CSV = config.DATA_DIR / "portfolio_state.csv"
STATE_JSON = config.DATA_DIR / "portfolio_state.json"
TRADES_CSV = config.DATA_DIR / "trade_history.csv"
DECISIONS_CSV = config.DATA_DIR / "ai_decisions.csv"
MESSAGES_CSV = config.DATA_DIR / "ai_messages.csv"

STATE_COLUMNS = [
    "timestamp",
    "total_balance",
    "total_equity",
    "total_return_pct",
    "num_positions",
    "position_details",
    "total_margin",
    "net_unrealized_pnl",
    "sharpe_ratio",
    "portfolio_summary",
    "short_summary",
]


def init_csv_files() -> None:
    """Initialize CSV files with headers if they don't exist."""
    files_to_init = {
        STATE_CSV: STATE_COLUMNS,
        TRADES_CSV: [
            "timestamp",
            "coin",
            "action",
            "side",
            "quantity",
            "price",
            "profit_target",
            "stop_loss",
            "leverage",
            "confidence",
            "pnl",
            "balance_after",
            "reason",
        ],
        DECISIONS_CSV: [
            "timestamp",
            "model",
            "coin",
            "signal",
            "reasoning",
            "confidence",
        ],
        MESSAGES_CSV: ["timestamp", "direction", "role", "content", "metadata"],
    }
    for path, header in files_to_init.items():
        if not path.exists():
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(header)


def log_portfolio_state(state: Dict[str, Any]) -> None:
    """Log current portfolio state to CSV."""
    with open(STATE_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([state.get(col, "") for col in STATE_COLUMNS])


def log_trade(trade_data: Dict[str, Any]) -> None:
    """Log trade execution to CSV."""
    header = [
        "timestamp",
        "coin",
        "action",
        "side",
        "quantity",
        "price",
        "profit_target",
        "stop_loss",
        "leverage",
        "confidence",
        "pnl",
        "balance_after",
        "reason",
    ]
    with open(TRADES_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([trade_data.get(col, "") for col in header])


def log_ai_decision(decision_data: Dict[str, Any]) -> None:
    """Log AI decision to CSV."""
    header = ["timestamp", "model", "coin", "signal", "reasoning", "confidence"]
    with open(DECISIONS_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([decision_data.get(col, "") for col in header])


def log_ai_message(message_data: Dict[str, Any]) -> None:
    """Log raw messages exchanged with the AI provider to CSV."""
    header = ["timestamp", "direction", "role", "content", "metadata"]
    with open(MESSAGES_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([message_data.get(col, "") for col in header])


# --- TELEGRAM ---
ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")


def strip_ansi_codes(text: str) -> str:
    """Remove ANSI color codes so Telegram receives plain text."""
    return ANSI_ESCAPE_RE.sub("", text)


def send_telegram_message(text: str, parse_mode: str = None) -> None:
    """Send a notification message to Telegram if credentials are configured."""
    if not config.TELEGRAM_BOT_TOKEN or not config.TELEGRAM_CHAT_ID:
        return

    try:
        payload = {
            "chat_id": config.TELEGRAM_CHAT_ID,
            "text": text,
        }
        if parse_mode:
            payload["parse_mode"] = parse_mode
            
        response = requests.post(
            f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendMessage",
            json=payload,
            timeout=10,
        )
        if response.status_code != 200:
            logging.warning(
                "Telegram notification failed (%s): %s",
                response.status_code,
                response.text,
            )
    except Exception as exc:
        logging.error("Error sending Telegram message: %s", exc)


def format_trading_signal_message(
    new_trades: List[Dict[str, Any]],
    positions: Dict[str, Dict[str, Any]],
    market_snapshots: Dict[str, Any],
    short_summary: str = None,
    total_equity: float = 0,
    total_return_pct: float = 0,
    net_unrealized_pnl: float = 0,
) -> str:
    """
    Format a trading signal message for Telegram with trade alerts, position updates, and summary.
    
    Args:
        new_trades: List of trade dictionaries from the current iteration
        positions: Current open positions
        market_snapshots: Current market prices
        short_summary: Optional AI-generated short summary
        total_equity: Total portfolio equity
        total_return_pct: Total return percentage
        net_unrealized_pnl: Net unrealized PnL
        
    Returns:
        Formatted message string ready to send to Telegram
    """
    lines = []
    
    # Header
    lines.append("🤖 <b>Trading Bot Update</b> 🤖")
    lines.append("=" * 30)
    lines.append("")
    
    # === SECTION 1: NEW TRADES (TOP) ===
    lines.append("📊 <b>NEW TRADES</b>")
    lines.append("─" * 30)
    
    if not new_trades:
        lines.append("✅ No new trades")
    else:
        for trade in new_trades:
            action = trade.get("action", "").upper()
            coin = trade.get("coin", "")
            side = trade.get("side", "").upper()
            price = trade.get("price", 0)
            quantity = trade.get("quantity", 0)
            pnl = trade.get("pnl", 0)
            reason = trade.get("reason", "")
            
            if action == "ENTRY":
                emoji = "🟢" if side == "LONG" else "🔴"
                lines.append(f"{emoji} <b>{action} {coin} {side}</b>")
                lines.append(f"   💰 Price: ${price:.4f}")
                lines.append(f"   📦 Quantity: {quantity:.4f}")
                lines.append(f"   🎯 Target: ${trade.get('profit_target', 0):.4f}")
                lines.append(f"   🛡️ Stop Loss: ${trade.get('stop_loss', 0):.4f}")
                lines.append(f"   ⚡ Leverage: {trade.get('leverage', 1)}x")
                lines.append(f"   💭 {reason}")
            elif action == "CLOSE":
                emoji = "✅" if pnl > 0 else "❌"
                pnl_emoji = "💚" if pnl > 0 else "🔻"
                lines.append(f"{emoji} <b>{action} {coin} {side}</b>")
                lines.append(f"   💰 Price: ${price:.4f}")
                lines.append(f"   {pnl_emoji} P&L: ${pnl:.2f}")
                lines.append(f"   💭 {reason}")
            
            lines.append("")
    
    lines.append("")
    
    # === SECTION 2: POSITION UPDATES ===
    lines.append("📈 <b>CURRENT POSITIONS</b>")
    lines.append("─" * 30)
    
    if not positions:
        lines.append("📭 No open positions")
    else:
        for coin, pos in positions.items():
            side = pos.get("side", "").upper()
            entry_price = pos.get("entry_price", 0)
            current_price = market_snapshots.get(coin, {}).get("price", entry_price)
            quantity = pos.get("quantity", 0)
            leverage = pos.get("leverage", 1)
            profit_target = pos.get("profit_target", 0)
            stop_loss = pos.get("stop_loss", 0)
            
            # Calculate unrealized PnL
            if side == "LONG":
                unrealized_pnl = (current_price - entry_price) * quantity
            else:
                unrealized_pnl = (entry_price - current_price) * quantity
            
            margin = pos.get("margin", 0)
            pnl_pct = (unrealized_pnl / margin * 100) if margin > 0 else 0
            
            emoji = "🟢" if side == "LONG" else "🔴"
            pnl_emoji = "💚" if unrealized_pnl > 0 else "🔻"
            
            lines.append(f"{emoji} <b>{coin} {side}</b> ({leverage}x)")
            lines.append(f"   💰 Entry: ${entry_price:.4f} → Current: ${current_price:.4f}")
            lines.append(f"   📦 Quantity: {quantity:.4f}")
            lines.append(f"   {pnl_emoji} Unrealized P&L: ${unrealized_pnl:.2f} ({pnl_pct:+.2f}%)")
            lines.append(f"   🎯 Target: ${profit_target:.4f} | 🛡️ Stop: ${stop_loss:.4f}")
            lines.append("")
    
    lines.append("")
    
    # === SECTION 3: PORTFOLIO SUMMARY ===
    lines.append("💼 <b>PORTFOLIO SUMMARY</b>")
    lines.append("─" * 30)
    lines.append(f"💵 Total Equity: ${total_equity:.2f}")
    lines.append(f"📊 Total Return: {total_return_pct:+.2f}%")
    lines.append(f"💹 Unrealized P&L: ${net_unrealized_pnl:+.2f}")
    lines.append(f"📍 Open Positions: {len(positions)}")
    
    if short_summary:
        lines.append("")
        lines.append(f"💭 <i>{short_summary}</i>")
    
    lines.append("")
    lines.append("=" * 30)
    lines.append(f"⏰ {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    
    return "\n".join(lines)
