#!/usr/bin/env python3
"""
Utility functions for the trading bot.
"""
import logging
import csv
import json
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import sys
import requests
import yaml
from decimal import Decimal
import pytz
from datetime import timedelta

# Avoid circular import by importing config lazily
_config = None


def _get_config():
    global _config
    if _config is None:
        from config import config

        _config = config
    return _config


# --- CSV & STATE MANAGEMENT ---
# These will be initialized lazily to avoid circular import
STATE_CSV = None
STATE_JSON = None
TRADES_CSV = None
DECISIONS_CSV = None
MESSAGES_CSV = None


def _init_csv_paths():
    """Initialize CSV path constants (called on first use to avoid circular import)."""
    global STATE_CSV, STATE_JSON, TRADES_CSV, DECISIONS_CSV, MESSAGES_CSV
    if STATE_CSV is None:
        config = _get_config()
        STATE_CSV = config.DATA_DIR / "portfolio_state.csv"
        STATE_JSON = config.DATA_DIR / "portfolio_state.json"
        TRADES_CSV = config.DATA_DIR / "trade_history.csv"
        DECISIONS_CSV = config.DATA_DIR / "ai_decisions.csv"
        MESSAGES_CSV = config.DATA_DIR / "ai_messages.csv"


# --- LOGGING ---
def setup_logging():
    """Initializes logging configuration with console output."""
    config = _get_config()

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
        fmt="%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(console_handler)

    # Optionally add file handler for persistent logs
    try:
        log_file = config.DATA_DIR / "trading_bot.log"
        file_handler = logging.FileHandler(log_file, mode="a")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logging.info(f"Logging configured. Console + File output enabled ({log_file})")
    except Exception as e:
        logging.info(
            f"Logging configured. Console output enabled (File logging failed: {e})"
        )


_CSV_LOCKS = {
    "state": threading.Lock(),
    "trades": threading.Lock(),
    "decisions": threading.Lock(),
    "messages": threading.Lock(),
}

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
    "total_fees_paid",
    "portfolio_summary",
    "short_summary",
]


def _ensure_model_directory(model_name: Optional[str]) -> Path:
    """Ensure the base directory for the given model exists and return it."""
    config = _get_config()
    if not model_name:
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        return config.DATA_DIR

    model_dir = config.DATA_DIR / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


def _resolve_csv_path(base_path: Path, model_name: Optional[str]) -> Path:
    if not model_name:
        base_path.parent.mkdir(parents=True, exist_ok=True)
        return base_path

    model_dir = _ensure_model_directory(model_name)
    return model_dir / base_path.name


def init_csv_files(model_name: Optional[str] = None) -> None:
    """Initialize CSV files with headers if they don't exist."""
    _init_csv_paths()  # Ensure CSV paths are initialized
    files_to_init = {
        STATE_CSV: ("state", STATE_COLUMNS),
        TRADES_CSV: (
            "trades",
            [
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
                "net_pnl",
                "fee",
                "balance_after",
                "reason",
                "position_fee_total",
                "position_net_pnl",
            ],
        ),
        DECISIONS_CSV: (
            "decisions",
            [
                "timestamp",
                "model",
                "coin",
                "signal",
                "reasoning",
                "confidence",
            ],
        ),
        MESSAGES_CSV: (
            "messages",
            ["timestamp", "direction", "role", "content", "metadata"],
        ),
    }
    for base_path, (lock_name, header) in files_to_init.items():
        target_path = _resolve_csv_path(base_path, model_name)
        lock = _CSV_LOCKS[lock_name]
        with lock:
            if not target_path.exists():
                target_path.parent.mkdir(parents=True, exist_ok=True)
                with open(target_path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(header)
                continue

            # Ensure existing files have the expected header (handle upgrades)
            try:
                with open(target_path, "r", newline="") as f:
                    reader = csv.reader(f)
                    existing_header = next(reader, [])
                    remaining_rows = list(reader)
            except Exception as exc:
                logging.warning(
                    "Failed to read CSV header for %s (%s); rewriting with new header.",
                    target_path,
                    exc,
                )
                existing_header = []
                remaining_rows = []

            if existing_header != header:
                target_path.parent.mkdir(parents=True, exist_ok=True)
                with open(target_path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(header)
                    for row in remaining_rows:
                        row_dict = {
                            existing_header[idx]: row[idx]
                            for idx in range(min(len(row), len(existing_header)))
                        }
                        writer.writerow([row_dict.get(col, "") for col in header])


def log_portfolio_state(
    state: Dict[str, Any], model_name: Optional[str] = None
) -> None:
    """Log current portfolio state to CSV."""
    _init_csv_paths()  # Ensure CSV paths are initialized
    target_path = _resolve_csv_path(STATE_CSV, model_name)
    with _CSV_LOCKS["state"]:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([state.get(col, "") for col in STATE_COLUMNS])


def log_trade(trade_data: Dict[str, Any], model_name: Optional[str] = None) -> None:
    """Log trade execution to CSV."""
    _init_csv_paths()  # Ensure CSV paths are initialized
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
        "net_pnl",
        "fee",
        "balance_after",
        "reason",
        "position_fee_total",
        "position_net_pnl",
    ]
    target_path = _resolve_csv_path(TRADES_CSV, model_name)
    with _CSV_LOCKS["trades"]:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([trade_data.get(col, "") for col in header])


def log_ai_decision(
    decision_data: Dict[str, Any], model_name: Optional[str] = None
) -> None:
    """Log AI decision to CSV."""
    _init_csv_paths()  # Ensure CSV paths are initialized
    header = ["timestamp", "model", "coin", "signal", "reasoning", "confidence"]
    # Support newer decision payloads that provide `justification` instead of `reasoning`
    if "reasoning" not in decision_data or not decision_data.get("reasoning"):
        if "justification" in decision_data:
            reasoning_value = decision_data.get("justification")
            decision_data = {
                **decision_data,
                "reasoning": reasoning_value if reasoning_value is not None else "",
            }
    target_path = _resolve_csv_path(DECISIONS_CSV, model_name)
    with _CSV_LOCKS["decisions"]:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([decision_data.get(col, "") for col in header])


def log_ai_message(
    message_data: Dict[str, Any], model_name: Optional[str] = None
) -> None:
    """Log raw messages exchanged with the AI provider to CSV."""
    _init_csv_paths()  # Ensure CSV paths are initialized
    header = ["timestamp", "direction", "role", "content", "metadata"]
    target_path = _resolve_csv_path(MESSAGES_CSV, model_name)
    with _CSV_LOCKS["messages"]:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([message_data.get(col, "") for col in header])


# --- TELEGRAM ---
ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")


def strip_ansi_codes(text: str) -> str:
    """Remove ANSI color codes so Telegram receives plain text."""
    return ANSI_ESCAPE_RE.sub("", text)


def send_telegram_message(text: str, parse_mode: str = None) -> None:
    """Send a notification message to Telegram if credentials are configured."""
    config = _get_config()
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


def get_currency_symbol() -> str:
    """Get the appropriate currency symbol based on ASSET_MODE."""
    config = _get_config()
    if config.ASSET_MODE.lower() == "idss":
        return "Rp"
    return "$"


def format_trading_signal_message(
    new_trades: List[Dict[str, Any]],
    positions: Dict[str, Dict[str, Any]],
    market_snapshots: Dict[str, Any],
    short_summary: str = None,
    total_equity: float = 0,
    total_return_pct: float = 0,
    net_unrealized_pnl: float = 0,
    total_fees_paid: float = 0,
    model_name: Optional[str] = None,
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
    currency = get_currency_symbol()
    lines = []

    # Header
    header_title = "🤖 <b>Trading Bot Update</b> 🤖"
    if model_name:
        header_title += f"\nModel: <b>{model_name}</b>"
    lines.append(header_title)
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
                lines.append(f"   💰 Price: {currency}{price:.4f}")
                lines.append(f"   📦 Quantity: {quantity:.4f}")
                lines.append(
                    f"   🎯 Target: {currency}{trade.get('profit_target', 0):.4f}"
                )
                lines.append(
                    f"   🛡️ Stop Loss: {currency}{trade.get('stop_loss', 0):.4f}"
                )
                lines.append(f"   ⚡ Leverage: {trade.get('leverage', 1)}x")
                entry_fee = trade.get("fee", 0) or 0
                if entry_fee:
                    lines.append(f"   🧾 Fee (entry): {currency}{entry_fee:.2f}")
                net_entry = trade.get("net_pnl")
                if net_entry not in (None, ""):
                    lines.append(f"   📉 Net Impact: {currency}{net_entry:.2f}")
                lines.append(f"   💭 {reason}")
            elif action == "CLOSE":
                emoji = "✅" if pnl > 0 else "❌"
                pnl_emoji = "💚" if pnl > 0 else "🔻"
                lines.append(f"{emoji} <b>{action} {coin} {side}</b>")
                lines.append(f"   💰 Price: {currency}{price:.4f}")
                lines.append(f"   {pnl_emoji} P&L: {currency}{pnl:.2f}")
                close_fee = trade.get("fee", 0) or 0
                net_trade = trade.get("net_pnl")
                if close_fee or net_trade not in (None, ""):
                    fee_line = f"   🧾 Fee (exit): {currency}{close_fee:.2f}"
                    if net_trade not in (None, ""):
                        fee_line += f" | Net This Trade: {currency}{net_trade:.2f}"
                    lines.append(fee_line)
                position_net = trade.get("position_net_pnl")
                if position_net not in (None, ""):
                    lines.append(
                        f"   📊 Position Net (incl. entry fees): {currency}{position_net:.2f}"
                    )
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
            lines.append(
                f"   💰 Entry: {currency}{entry_price:.4f} → Current: {currency}{current_price:.4f}"
            )
            lines.append(f"   📦 Quantity: {quantity:.4f}")
            lines.append(
                f"   {pnl_emoji} Unrealized P&L: {currency}{unrealized_pnl:.2f} ({pnl_pct:+.2f}%)"
            )
            lines.append(
                f"   🎯 Target: {currency}{profit_target:.4f} | 🛡️ Stop: {currency}{stop_loss:.4f}"
            )
            lines.append("")

    lines.append("")

    # === SECTION 3: PORTFOLIO SUMMARY ===
    lines.append("💼 <b>PORTFOLIO SUMMARY</b>")
    lines.append("─" * 30)
    lines.append(f"💵 Total Equity: {currency}{total_equity:.2f}")
    lines.append(f"📊 Total Return: {total_return_pct:+.2f}%")
    lines.append(f"💹 Unrealized P&L: {currency}{net_unrealized_pnl:+.2f}")
    lines.append(f"🧾 Fees Paid (lifetime): {currency}{total_fees_paid:.2f}")
    lines.append(f"📍 Open Positions: {len(positions)}")

    if short_summary:
        lines.append("")
        lines.append(f"💭 <i>{short_summary}</i>")

    lines.append("")
    lines.append("=" * 30)
    lines.append(f"⏰ {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")

    return "\n".join(lines)


def is_today(timestamp: int) -> bool:
    tz = pytz.FixedOffset(420)  # GMT-4 is UTC-4 hours, i.e., -240 minutes
    dt = datetime.fromtimestamp(timestamp, tz)
    return dt.date() == datetime.now().date()

def is_next_day(prev_timestamp: int, new_timestamp: int) -> bool:
    tz = pytz.FixedOffset(420)  # GMT-4 is UTC-4 hours, i.e., -240 minutes

    new_dt = datetime.fromtimestamp(new_timestamp, tz)
    new_date = new_dt.date()

    previous_dt = datetime.fromtimestamp(prev_timestamp, tz)
    previous_date = previous_dt.date()

    # Check if the new date is exactly one day after the previous date
    date_difference = new_date - previous_date
    return date_difference >= timedelta(days=1)


def current_epoch() -> int:
    """return current timestamp

    Returns:
        int: timestamp in seconds
    """
    now = datetime.now()
    return int(now.timestamp())


def yaml_parser(filePath: str) -> dict:
    """
    Parses a YAML file and returns its content as a dictionary.
    The function opens the file located at `filePath` and uses the `yaml` library to parse it.

    Args:
        filePath (str): The path to the YAML file to be parsed.

    Returns:
        dict: The parsed YAML data.
    """
    with open(filePath) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def json_parser(filePath: str) -> dict:
    """
    Parses a JSON file and returns its content as a dictionary.
    The function opens the file located at `filePath` and uses the `json` library to parse it.

    Args:
        filePath (str): The path to the JSON file to be parsed.

    Returns:
        dict: The parsed JSON data.
    """
    with open(filePath) as f:
        value = json.load(f)
    return value


def json_exporter(d, filePath):
    """
    Exports a dictionary to a JSON file at the specified path with pretty formatting.
    This function writes the dictionary `d` to a file specified by `filePath`, formatting the
    JSON output with an indentation of 4 spaces.

    Args:
        d (dict): The dictionary to export.
        filePath (str): The path where the JSON file will be saved.
    """
    with open(filePath, "w") as fp:
        json.dump(d, fp, indent=4)


def format_float_by_given_granularity_into_str(
    rawNumber: float, referenceStr: str
) -> str:
    """format the raw number into a str as per referenceStr granularity

    Args:
        rawNumber (float): raw number
        referenceStr (str): refence number is string

    Returns:
        str: fotmatted number
    """
    return str(
        (
            Decimal(int(Decimal(rawNumber) / Decimal(referenceStr)))
            * Decimal(referenceStr)
        ).quantize(Decimal(referenceStr))
    )
