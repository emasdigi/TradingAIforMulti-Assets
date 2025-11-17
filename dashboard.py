#!/usr/bin/env python3
"""Streamlit dashboard for monitoring the DeepSeek trading bot."""
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Any, Iterable
from html import escape

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from binance.client import Client
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = BASE_DIR / "data"
DATA_DIR = Path(os.getenv("TRADEBOT_DATA_DIR", str(DEFAULT_DATA_DIR))).expanduser()
DATA_DIR.mkdir(parents=True, exist_ok=True)

STATE_CSV = DATA_DIR / "portfolio_state.csv"
TRADES_CSV = DATA_DIR / "trade_history.csv"
DECISIONS_CSV = DATA_DIR / "ai_decisions.csv"
MESSAGES_CSV = DATA_DIR / "ai_messages.csv"
ENV_PATH = BASE_DIR / ".env"
DEFAULT_RISK_FREE_RATE = 0.05
DEFAULT_SNAPSHOT_SECONDS = 300.0
DEFAULT_START_CAPITAL = 10_000.0
DEFAULT_TRADING_FEE_RATE = 0.0003

COIN_TO_SYMBOL: Dict[str, str] = {
    "ETH": "ETHUSDT",
    "SOL": "SOLUSDT",
    "XRP": "XRPUSDT",
    "BTC": "BTCUSDT",
    "DOGE": "DOGEUSDT",
    "BNB": "BNBUSDT",
}

if ENV_PATH.exists():
    load_dotenv(ENV_PATH)
else:
    load_dotenv()

BN_API_KEY = os.getenv("BN_API_KEY", "")
BN_SECRET = os.getenv("BN_SECRET", "")


def resolve_starting_capital(default: float = DEFAULT_START_CAPITAL) -> float:
    env_value = os.getenv("START_CAPITAL")
    if not env_value:
        return default
    try:
        capital = float(env_value)
    except (TypeError, ValueError):
        logging.warning(
            "Invalid START_CAPITAL value '%s'; using default %.2f", env_value, default
        )
        return default
    if not np.isfinite(capital) or capital <= 0:
        logging.warning(
            "Non-positive START_CAPITAL value '%s'; using default %.2f",
            env_value,
            default,
        )
        return default
    return capital


def discover_model_directories() -> Dict[str, Path]:
    """Return mapping of model name to data directory."""
    model_dirs: Dict[str, Path] = {}

    # Legacy data (pre multi-model) stored directly under DATA_DIR
    if STATE_CSV.exists():
        model_dirs["combined"] = DATA_DIR

    for path in sorted(DATA_DIR.iterdir()):
        if not path.is_dir():
            continue
        portfolio_csv = path / "portfolio_state.csv"
        if portfolio_csv.exists():
            model_dirs[path.name] = path

    return model_dirs


def get_model_csv_path(model_name: str, filename: str) -> Path:
    model_dirs = discover_model_directories()
    base_path = model_dirs.get(model_name)
    if base_path is None:
        return Path()
    return base_path / filename


def format_model_label(model_name: str) -> str:
    if model_name == "combined":
        return "All Models (legacy)"
    return model_name.replace("_", " ").title()


def resolve_risk_free_rate() -> float:
    """Return annualized risk-free rate configured for Sortino ratio."""
    env_value = os.getenv("SORTINO_RISK_FREE_RATE") or os.getenv("RISK_FREE_RATE")
    if env_value is None:
        return DEFAULT_RISK_FREE_RATE
    try:
        return float(env_value)
    except (TypeError, ValueError):
        logging.warning(
            "Invalid SORTINO_RISK_FREE_RATE/RISK_FREE_RATE value '%s'; using default %.4f",
            env_value,
            DEFAULT_RISK_FREE_RATE,
        )
        return DEFAULT_RISK_FREE_RATE


RISK_FREE_RATE = resolve_risk_free_rate()
STARTING_CAPITAL = resolve_starting_capital()


def resolve_trading_fee_rate(default: float = DEFAULT_TRADING_FEE_RATE) -> float:
    env_value = os.getenv("TRADING_FEE_RATE")
    if not env_value:
        return default
    try:
        fee_rate = float(env_value)
    except (TypeError, ValueError):
        logging.warning(
            "Invalid TRADING_FEE_RATE value '%s'; using default %.6f",
            env_value,
            default,
        )
        return default
    if fee_rate < 0:
        logging.warning(
            "Negative TRADING_FEE_RATE value '%s'; using default %.6f",
            env_value,
            default,
        )
        return default
    return fee_rate


TRADING_FEE_RATE = resolve_trading_fee_rate()

BINANCE_CLIENT: Client | None = None
if BN_API_KEY and BN_SECRET:
    try:
        BINANCE_CLIENT = Client(BN_API_KEY, BN_SECRET, testnet=False)
    except Exception as exc:
        logging.warning("Unable to initialize Binance client: %s", exc)
else:
    logging.info("Binance credentials not provided; live prices disabled.")


def load_csv(path: Path, parse_dates: List[str] | None = None) -> pd.DataFrame:
    """Load a CSV into a DataFrame, returning empty frame when missing."""
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=parse_dates)


@st.cache_data(ttl=15)
def get_portfolio_state(csv_path: str) -> pd.DataFrame:
    path = Path(csv_path)
    df = load_csv(path, parse_dates=["timestamp"])
    if df.empty:
        return df

    numeric_cols = [
        "total_balance",
        "total_equity",
        "total_return_pct",
        "num_positions",
        "total_margin",
        "net_unrealized_pnl",
        "sharpe_ratio",
        "total_fees_paid",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df.sort_values("timestamp", inplace=True)
    df.set_index("timestamp", inplace=True)
    return df


@st.cache_data(ttl=15)
def get_trades(csv_path: str) -> pd.DataFrame:
    path = Path(csv_path)
    df = load_csv(path, parse_dates=["timestamp"])
    if df.empty:
        return df
    if {"net_pnl", "balance_after"}.issubset(df.columns):
        net_numeric = pd.to_numeric(df["net_pnl"], errors="coerce")
        balance_numeric = pd.to_numeric(df["balance_after"], errors="coerce")
        misaligned_net = (
            balance_numeric.isna() & net_numeric.notna() & df["balance_after"].isna()
        )
        if misaligned_net.any():
            df.loc[misaligned_net, "balance_after"] = df.loc[misaligned_net, "net_pnl"]
            df.loc[misaligned_net, "net_pnl"] = "0"
    if {"fee", "reason"}.issubset(df.columns):
        reason_missing = df["reason"].isna() | (
            df["reason"].astype(str).str.strip() == ""
        )
        fee_as_text = df["fee"].astype(str)
        fee_numeric = pd.to_numeric(df["fee"], errors="coerce")
        misaligned_fee = (
            reason_missing & fee_numeric.isna() & (fee_as_text.str.strip() != "")
        )
        if misaligned_fee.any():
            df.loc[misaligned_fee, "reason"] = df.loc[misaligned_fee, "fee"]
            df.loc[misaligned_fee, "fee"] = ""
    df.sort_values("timestamp", inplace=True, ascending=False)
    numeric_cols = [
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
        "position_fee_total",
        "position_net_pnl",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


@st.cache_data(ttl=15)
def get_ai_decisions(csv_path: str) -> pd.DataFrame:
    path = Path(csv_path)
    df = load_csv(path, parse_dates=["timestamp"])
    if df.empty:
        return df
    df.sort_values("timestamp", inplace=True, ascending=False)
    return df


@st.cache_data(ttl=15)
def get_ai_messages(csv_path: str) -> pd.DataFrame:
    path = Path(csv_path)
    df = load_csv(path, parse_dates=["timestamp"])
    if df.empty:
        return df
    df.sort_values("timestamp", inplace=True, ascending=False)
    return df


@st.cache_data(ttl=60)
def get_local_btc_price_series(csv_path: str) -> pd.DataFrame:
    """Extract BTC prices from logged AI messages (no external calls)."""
    path = Path(csv_path)
    messages_df = load_csv(path, parse_dates=["timestamp"])
    if messages_df.empty or "content" not in messages_df.columns:
        return pd.DataFrame()

    pattern = re.compile(r"BTC MARKET SNAPSHOT.*?- Price:\s*([0-9.,]+)", re.DOTALL)

    def _extract_price(text: str) -> float | None:
        matches = pattern.findall(str(text))
        if not matches:
            return None
        raw_value = matches[-1].replace(",", "")
        try:
            return float(raw_value)
        except ValueError:
            return None

    messages_df["btc_price"] = messages_df["content"].apply(_extract_price)
    price_df = (
        messages_df.dropna(subset=["btc_price"])[["timestamp", "btc_price"]]
        .drop_duplicates(subset=["timestamp"])
        .sort_values("timestamp")
    )
    price_df["btc_price"] = pd.to_numeric(price_df["btc_price"], errors="coerce")
    price_df.dropna(subset=["btc_price"], inplace=True)
    return price_df


def _coerce_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(result):
        return None
    return result


def _parse_json_positions(data: Any) -> List[Dict[str, Any]] | None:
    if isinstance(data, str):
        try:
            parsed = json.loads(data)
        except json.JSONDecodeError:
            return None
    else:
        parsed = data

    if not isinstance(parsed, dict):
        return None

    rows: List[Dict[str, Any]] = []
    for coin, payload in parsed.items():
        if not isinstance(payload, dict):
            continue

        row: Dict[str, Any] = {
            "coin": coin,
            "side": str(payload.get("side", "")).upper(),
            "quantity": _coerce_float(payload.get("quantity")),
            "entry_price": _coerce_float(payload.get("entry_price")),
            "current_price": _coerce_float(payload.get("current_price")),
            "profit_target": _coerce_float(payload.get("profit_target")),
            "stop_loss": _coerce_float(payload.get("stop_loss")),
            "leverage": _coerce_float(payload.get("leverage")),
            "margin": _coerce_float(payload.get("margin")),
            "unrealized_pnl": _coerce_float(payload.get("unrealized_pnl")),
            "risk_idr": _coerce_float(payload.get("risk_idr")),
            "fees_paid": _coerce_float(payload.get("fees_paid")),
            "justification": payload.get("justification", ""),
            "invalidation_condition": payload.get("invalidation_condition", ""),
        }
        rows.append(row)

    return rows


def _parse_legacy_positions(text: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for chunk in text.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            symbol, side, rest = chunk.split(":")
            quantity, entry_price = rest.split("@")
        except ValueError:
            continue

        rows.append(
            {
                "coin": symbol,
                "side": side,
                "quantity": _coerce_float(quantity),
                "entry_price": _coerce_float(entry_price),
            }
        )
    return rows


def parse_positions(position_payload: Any) -> pd.DataFrame:
    """Normalize position payloads (JSON, dict, or legacy string) into a DataFrame."""
    if position_payload is None or (
        isinstance(position_payload, float) and np.isnan(position_payload)
    ):
        return pd.DataFrame()

    if isinstance(position_payload, dict):
        rows = _parse_json_positions(position_payload)
    elif isinstance(position_payload, str):
        text = position_payload.strip()
        if not text or text.lower() == "no positions":
            return pd.DataFrame()
        rows = _parse_json_positions(text)
        if rows is None:
            rows = _parse_legacy_positions(text)
    else:
        return pd.DataFrame()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    numeric_cols: Iterable[str] = (
        "quantity",
        "entry_price",
        "current_price",
        "profit_target",
        "stop_loss",
        "leverage",
        "margin",
        "unrealized_pnl",
        "risk_idr",
        "fees_paid",
    )
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def prepare_summary_html(text: Any) -> str | None:
    """Convert plain-text summary content into safe HTML paragraphs."""
    if isinstance(text, float) and pd.isna(text):
        return None
    if text is None:
        return None

    normalized = str(text).replace("\r\n", "\n").strip()
    if not normalized:
        return None

    paragraphs = [para.strip() for para in normalized.split("\n\n") if para.strip()]
    if not paragraphs:
        paragraphs = [normalized]

    html_parts: List[str] = []
    for para in paragraphs:
        lines = [line.strip() for line in para.split("\n")]
        escaped_lines = [escape(line, quote=False) for line in lines if line]
        if not escaped_lines:
            continue
        html_parts.append("<p>" + "<br>".join(escaped_lines) + "</p>")

    if not html_parts:
        return None

    return "".join(html_parts)


def fetch_current_prices(coins: List[str]) -> Dict[str, float | None]:
    """Fetch latest market prices for the provided coin tickers."""
    prices: Dict[str, float | None] = {coin: None for coin in coins}
    if not BINANCE_CLIENT:
        return prices

    for coin in coins:
        symbol = COIN_TO_SYMBOL.get(coin.upper(), f"{coin.upper()}USDT")
        try:
            ticker = BINANCE_CLIENT.get_symbol_ticker(symbol=symbol)
            prices[coin] = float(ticker["price"])
        except Exception as exc:
            logging.warning("Failed to fetch price for %s: %s", symbol, exc)
            prices[coin] = None
    return prices


def estimate_period_seconds(
    index: pd.Index, default: float = DEFAULT_SNAPSHOT_SECONDS
) -> float:
    """Infer measurement cadence from a datetime-like index."""
    if index.size < 2:
        return default
    try:
        diffs = index.to_series().diff().dropna()
    except Exception:
        return default
    if diffs.empty:
        return default
    try:
        period_seconds = diffs.dt.total_seconds().median()
    except AttributeError:
        period_seconds = default
    if not period_seconds or not np.isfinite(period_seconds) or period_seconds <= 0:
        return default
    return float(period_seconds)


MIN_RETURNS_FOR_SHARPE = int(60 / (DEFAULT_SNAPSHOT_SECONDS / 60))


def compute_sharpe_ratio(state_df: pd.DataFrame, risk_free_rate: float) -> float | None:
    """Compute annualized Sharpe ratio from equity snapshots."""
    if state_df.empty or "total_equity" not in state_df.columns:
        return None

    equity = pd.to_numeric(state_df["total_equity"], errors="coerce").dropna()
    if equity.size < 2:
        return None

    try:
        equity = equity.sort_index()
    except Exception:
        pass

    log_equity = np.log(equity)
    log_returns = log_equity.diff().dropna()
    if log_returns.empty:
        return None

    if log_returns.size < MIN_RETURNS_FOR_SHARPE:
        return None

    period_seconds = estimate_period_seconds(equity.index)
    if not np.isfinite(period_seconds) or period_seconds <= 0:
        return None

    periods_per_year = (365 * 24 * 60 * 60) / period_seconds
    if not np.isfinite(periods_per_year) or periods_per_year <= 0:
        return None

    per_period_rf = np.log(1.0 + risk_free_rate / periods_per_year)
    mean_excess = log_returns.mean() - per_period_rf
    if not np.isfinite(mean_excess):
        return None

    std = log_returns.std(ddof=1)
    if std is None or std <= 0 or not np.isfinite(std):
        return None

    sharpe = mean_excess / std * np.sqrt(periods_per_year)
    if not np.isfinite(sharpe):
        return None

    return float(sharpe)


def compute_sortino_ratio(
    state_df: pd.DataFrame, risk_free_rate: float
) -> float | None:
    """Compute annualized Sortino ratio from total equity snapshots."""
    if state_df.empty or "total_equity" not in state_df.columns:
        return None

    equity = pd.to_numeric(state_df["total_equity"], errors="coerce").dropna()
    if equity.size < 2:
        return None

    returns = equity.pct_change().dropna()
    if returns.empty:
        return None

    period_seconds = estimate_period_seconds(equity.index)
    seconds_per_year = 365 * 24 * 60 * 60
    periods_per_year = seconds_per_year / period_seconds
    if not np.isfinite(periods_per_year) or periods_per_year <= 0:
        return None

    per_period_rf = risk_free_rate / periods_per_year
    excess_return = returns.mean() - per_period_rf
    if not np.isfinite(excess_return):
        return None

    downside = np.minimum(returns - per_period_rf, 0.0)
    downside_deviation = np.sqrt(np.mean(np.square(downside)))
    if downside_deviation <= 0 or not np.isfinite(downside_deviation):
        return None

    sortino = (excess_return / downside_deviation) * np.sqrt(periods_per_year)
    return float(sortino) if np.isfinite(sortino) else None


def render_combined_equity_chart(
    state_map: Dict[str, pd.DataFrame],
    btc_price_map: Dict[str, pd.DataFrame] | None = None,
) -> None:
    frames: List[pd.DataFrame] = []
    timeline_frames: List[pd.DataFrame] = []
    series_order: List[str] = []

    for model_name, df in state_map.items():
        if df.empty or "total_equity" not in df.columns:
            continue

        reset = df.reset_index().copy()
        reset["timestamp"] = pd.to_datetime(reset["timestamp"], errors="coerce")
        reset["Value"] = pd.to_numeric(reset["total_equity"], errors="coerce")
        label = format_model_label(model_name)
        reset["Series"] = label
        frame = reset[["timestamp", "Series", "Value"]]
        frames.append(frame)
        timeline_frames.append(frame[["timestamp"]])
        series_order.append(label)

    if not frames:
        st.info("No equity data available across models yet.")
        return

    btc_caption = None
    if btc_price_map:
        longest_btc: pd.DataFrame | None = None
        for series in btc_price_map.values():
            if series is None or series.empty or "btc_price" not in series.columns:
                continue
            if longest_btc is None or len(series) > len(longest_btc):
                longest_btc = series

        if longest_btc is not None and timeline_frames:
            timeline = pd.concat(timeline_frames, ignore_index=True)
            timeline["timestamp"] = pd.to_datetime(
                timeline["timestamp"], utc=True, errors="coerce"
            )
            timeline = (
                timeline.dropna(subset=["timestamp"])
                .drop_duplicates()
                .sort_values("timestamp")
            )

            btc_df = longest_btc.copy()
            btc_df["timestamp"] = pd.to_datetime(
                btc_df["timestamp"], utc=True, errors="coerce"
            )
            btc_df = btc_df.dropna(subset=["timestamp"]).sort_values("timestamp")

            if not timeline.empty and not btc_df.empty:
                benchmark = pd.merge_asof(
                    timeline,
                    btc_df,
                    on="timestamp",
                    direction="backward",
                )
                benchmark["btc_price"] = benchmark["btc_price"].ffill().bfill()
                valid_prices = benchmark["btc_price"].dropna()
                if not valid_prices.empty:
                    base_price = float(valid_prices.iloc[0])
                    if base_price > 0:
                        base_investment = 10_000.0
                        btc_values = base_investment * (
                            benchmark["btc_price"] / base_price
                        )
                        btc_frame = pd.DataFrame(
                            {
                                "timestamp": benchmark["timestamp"],
                                "Series": "BTC buy & hold",
                                "Value": btc_values,
                            }
                        )
                        frames.append(btc_frame)
                        series_order.append("BTC buy & hold")

    combined_df = pd.concat(frames, ignore_index=True).dropna(
        subset=["timestamp", "Value"]
    )
    combined_df.sort_values("timestamp", inplace=True)

    lower = float(combined_df["Value"].min())
    upper = float(combined_df["Value"].max())
    if not np.isfinite(lower) or not np.isfinite(upper):
        st.info("No valid equity data available for charting.")
        return

    span = upper - lower
    if span <= 0:
        span = max(upper * 0.02, 1.0)
    padding = span * 0.1
    lower_bound = max(0.0, lower - padding)
    upper_bound = upper + padding

    series_scale = alt.Scale(domain=list(dict.fromkeys(series_order)))

    chart = (
        alt.Chart(combined_df)
        .mark_line(interpolate="monotone")
        .encode(
            x=alt.X("timestamp:T", title="Time"),
            y=alt.Y(
                "Value:Q",
                title="Portfolio Equity ($)",
                scale=alt.Scale(domain=[lower_bound, upper_bound]),
            ),
            color=alt.Color("Series:N", title="Series", scale=series_scale),
            tooltip=[
                alt.Tooltip("timestamp:T", title="Timestamp"),
                alt.Tooltip("Series:N", title="Series"),
                alt.Tooltip("Value:Q", title="Equity", format="$.2f"),
            ],
        )
        .properties(height=320)
        .interactive()
    )

    baseline = (
        alt.Chart(pd.DataFrame({"Value": [STARTING_CAPITAL]}))
        .mark_rule(color="#888888", strokeDash=[6, 3])
        .encode(y="Value:Q")
    )

    combined_chart = (chart + baseline).resolve_scale(color="independent")

    st.altair_chart(combined_chart, use_container_width=True)  # type: ignore[arg-type]
    if btc_caption:
        st.caption(btc_caption)


def render_portfolio_tab(
    state_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    btc_series: pd.DataFrame | None = None,
) -> None:
    if state_df.empty:
        st.info("No portfolio data logged yet.")
        return

    latest = state_df.iloc[-1]
    positions_df = parse_positions(latest.get("position_details", ""))
    exit_fee_estimate = 0.0
    price_map: Dict[str, float | None] = {}

    if not positions_df.empty:
        coins = positions_df.get("coin")
        if coins is None:
            coins = pd.Series([], dtype=object)
        unique_coins = [coin for coin in coins.dropna().unique().tolist() if coin]
        price_map = fetch_current_prices(unique_coins)
        live_prices = coins.map(price_map)
        if "current_price" in positions_df.columns:
            positions_df["current_price"] = live_prices.combine_first(
                positions_df["current_price"]
            )
        else:
            positions_df["current_price"] = live_prices
        positions_df["current_price"] = pd.to_numeric(
            positions_df["current_price"], errors="coerce"
        )
        positions_df["current_price"] = positions_df["current_price"].fillna(
            positions_df["entry_price"]
        )

        def _row_unrealized(row: pd.Series) -> float | None:
            price = row.get("current_price")
            entry_price = row.get("entry_price")
            quantity = row.get("quantity")
            if (
                quantity is None
                or pd.isna(quantity)
                or entry_price is None
                or pd.isna(entry_price)
            ):
                return None
            if price is None or pd.isna(price):
                price = entry_price
            diff = price - entry_price
            if str(row.get("side", "")).lower() == "short":
                diff = entry_price - price
            return diff * quantity

        positions_df["unrealized_pnl"] = positions_df.apply(_row_unrealized, axis=1)  # type: ignore[arg-type]
        positions_df["fees_paid"] = pd.to_numeric(
            positions_df.get("fees_paid"), errors="coerce"
        ).fillna(0.0)

        fee_rate = TRADING_FEE_RATE or 0.0
        if fee_rate > 0:
            price_for_exit = positions_df["current_price"].fillna(
                positions_df["entry_price"]
            )
            qty_abs = positions_df["quantity"].abs()
            positions_df["estimated_exit_fee"] = qty_abs * price_for_exit * fee_rate
            exit_fee_estimate = float(
                positions_df["estimated_exit_fee"].sum(skipna=True)
            )
            if not np.isfinite(exit_fee_estimate):
                exit_fee_estimate = 0.0
        else:
            positions_df["estimated_exit_fee"] = 0.0

    total_fees_paid = latest.get("total_fees_paid", 0.0)
    if pd.isna(total_fees_paid):
        total_fees_paid = 0.0
    total_fees_paid = float(total_fees_paid)
    margin_allocated = latest.get("total_margin", 0.0)
    if pd.isna(margin_allocated):
        margin_allocated = 0.0
    margin_allocated = float(margin_allocated)
    unrealized_pnl = latest.get("net_unrealized_pnl", np.nan)
    if pd.isna(unrealized_pnl):
        unrealized_pnl = (
            latest["total_equity"] - latest["total_balance"] - margin_allocated
        )

    prev_unrealized = 0.0
    if len(state_df) > 1:
        prior = state_df.iloc[-2]
        prev_margin = prior.get("total_margin", 0.0)
        if pd.isna(prev_margin):
            prev_margin = 0.0
        prev_margin = float(prev_margin)
        prev_unrealized = prior.get("net_unrealized_pnl", np.nan)
        if pd.isna(prev_unrealized):
            prev_unrealized = (
                prior["total_equity"] - prior["total_balance"] - prev_margin
            )

    realized_pnl = 0.0
    if (
        not trades_df.empty
        and "action" in trades_df.columns
        and "pnl" in trades_df.columns
    ):
        actions = trades_df["action"].fillna("").str.upper()
        realized_pnl = trades_df.loc[actions == "CLOSE", "pnl"].sum(skipna=True)
        if not np.isfinite(realized_pnl):
            realized_pnl = 0.0

    sharpe_ratio = None
    if "sharpe_ratio" in state_df.columns:
        sharpe_series = pd.to_numeric(state_df["sharpe_ratio"], errors="coerce")
        sharpe_series = sharpe_series.dropna()
        if not sharpe_series.empty:
            sharpe_ratio = float(sharpe_series.iloc[-1])

    sortino_ratio = compute_sortino_ratio(state_df, RISK_FREE_RATE)

    col_a, col_b, col_c, col_d, col_e, col_f, col_g, col_h = st.columns(8)
    col_a.metric("Total Equity", f"${latest['total_equity']:,.2f}")
    col_b.metric("Total Return %", f"{latest['total_return_pct']:,.2f}%")
    col_c.metric("Margin Allocated", f"${margin_allocated:,.2f}")
    col_d.metric("Available Balance", f"${latest['total_balance']:,.2f}")
    col_e.metric(
        "Unrealized PnL",
        f"${unrealized_pnl:,.2f}",
        # delta=f"${unrealized_pnl - prev_unrealized:.2f}",
    )
    col_f.metric("Realized PnL", f"${realized_pnl:,.2f}")
    col_g.metric(
        "Sharpe Ratio",
        f"{sharpe_ratio:,.2f}" if sharpe_ratio is not None else "N/A",
    )
    col_h.metric(
        "Sortino Ratio",
        f"{sortino_ratio:,.2f}" if sortino_ratio is not None else "N/A",
    )

    fee_col_a, fee_col_b = st.columns(2)
    fee_col_a.metric("Fees Paid (lifetime)", f"${total_fees_paid:,.2f}")
    fee_col_b.metric("Est. Exit Fees (open positions)", f"${exit_fee_estimate:,.2f}")
    st.caption(
        f"Fee calculations assume {TRADING_FEE_RATE * 100:.3f}% per side; adjust the TRADING_FEE_RATE env var if needed."
    )

    portfolio_summary = latest.get("portfolio_summary")
    short_summary_text = latest.get("short_summary")

    portfolio_summary_html = prepare_summary_html(portfolio_summary)
    short_summary_html = prepare_summary_html(short_summary_text)

    if portfolio_summary_html or short_summary_html:
        col_summary_a, col_summary_b = st.columns(2)

        if portfolio_summary_html:
            with col_summary_a:
                st.markdown("**Portfolio Summary**")
                st.markdown(
                    f"<div style='white-space:normal; line-height:1.55'>{portfolio_summary_html}</div>",
                    unsafe_allow_html=True,
                )

        if short_summary_html:
            with col_summary_b:
                st.markdown("**Quick Take**")
                st.markdown(
                    f"<div style='white-space:normal; line-height:1.55'>{short_summary_html}</div>",
                    unsafe_allow_html=True,
                )

    st.subheader("Equity Over Time (with BTC benchmark)")
    base_investment = STARTING_CAPITAL

    chart_frames = [
        pd.DataFrame(
            {
                "timestamp": state_df.index,
                "Series": "Portfolio equity",
                "Value": pd.to_numeric(
                    state_df["total_equity"], errors="coerce"
                ).values,
            }
        )
    ]

    btc_caption = None
    if btc_series is not None and not btc_series.empty and len(state_df.index) > 0:
        timeline = state_df.reset_index()[["timestamp"]].copy()
        timeline["timestamp"] = pd.to_datetime(
            timeline["timestamp"], utc=True, errors="coerce"
        )
        timeline = timeline.dropna(subset=["timestamp"]).sort_values("timestamp")

        btc_df = btc_series.copy()
        btc_df["timestamp"] = pd.to_datetime(
            btc_df["timestamp"], utc=True, errors="coerce"
        )
        btc_df = btc_df.dropna(subset=["timestamp"]).sort_values("timestamp")

        if not timeline.empty and not btc_df.empty:
            benchmark = pd.merge_asof(
                timeline,
                btc_df,
                on="timestamp",
                direction="backward",
            )
            benchmark["btc_price"] = benchmark["btc_price"].ffill().bfill()
            valid_prices = benchmark["btc_price"].dropna()
            if not valid_prices.empty:
                base_price = float(valid_prices.iloc[0])
                if base_price > 0:
                    btc_values = base_investment * (benchmark["btc_price"] / base_price)
                    chart_frames.append(
                        pd.DataFrame(
                            {
                                "timestamp": benchmark["timestamp"],
                                "Series": "BTC buy & hold",
                                "Value": btc_values,
                            }
                        )
                    )

    equity_chart_df = pd.concat(chart_frames).dropna(subset=["timestamp", "Value"])
    equity_chart_df.sort_values("timestamp", inplace=True)

    lower = float(equity_chart_df["Value"].min())
    upper = float(equity_chart_df["Value"].max())
    span = upper - lower
    if span <= 0:
        span = max(upper * 0.02, 1.0)
    padding = span * 0.1
    lower_bound = max(0.0, lower - padding)
    upper_bound = upper + padding

    equity_chart = (
        alt.Chart(equity_chart_df)
        .mark_line(interpolate="monotone")
        .encode(
            x=alt.X("timestamp:T", title="Time"),
            y=alt.Y(
                "Value:Q",
                title="Value ($)",
                scale=alt.Scale(domain=[lower_bound, upper_bound]),
            ),
            color=alt.Color(
                "Series:N",
                title="Series",
                scale=alt.Scale(
                    domain=["Portfolio equity", "BTC buy & hold"],
                    range=["#f58518", "#4c78a8"],
                ),
            ),
            tooltip=[
                alt.Tooltip("timestamp:T", title="Timestamp"),
                alt.Tooltip("Series:N", title="Series"),
                alt.Tooltip("Value:Q", title="Value", format="$.2f"),
            ],
        )
        .properties(height=280)
        .interactive()
    )
    baseline = (
        alt.Chart(pd.DataFrame({"Value": [base_investment]}))
        .mark_rule(color="#888888", strokeDash=[6, 3])
        .encode(y="Value:Q")
    )
    combined_chart = (equity_chart + baseline).resolve_scale(color="independent")
    st.altair_chart(combined_chart, use_container_width=True)  # type: ignore[arg-type]
    if btc_caption:
        st.caption(btc_caption)

    st.subheader("Open Positions")
    if positions_df.empty:
        st.write("No open positions.")
    else:
        if positions_df["current_price"].isna().all():
            st.caption("Live price lookup unavailable; showing entry data only.")

        st.dataframe(
            positions_df,
            column_config={
                "quantity": st.column_config.NumberColumn(format="%.4f"),
                "entry_price": st.column_config.NumberColumn(format="Rp%.2f"),
                "current_price": st.column_config.NumberColumn(format="Rp%.2f"),
                "profit_target": st.column_config.NumberColumn(format="Rp%.2f"),
                "stop_loss": st.column_config.NumberColumn(format="Rp%.2f"),
                "margin": st.column_config.NumberColumn(format="Rp%.2f"),
                "risk_idr": st.column_config.NumberColumn(format="Rp%.2f"),
                "fees_paid": st.column_config.NumberColumn(format="Rp%.2f"),
                "unrealized_pnl": st.column_config.NumberColumn(format="Rp%.2f"),
                "estimated_exit_fee": st.column_config.NumberColumn(format="Rp%.2f"),
            },
            use_container_width=True,
        )


def _extract_decision_justifications(messages_df: pd.DataFrame) -> pd.DataFrame:
    """Parse assistant responses to recover per-coin justification text."""
    if messages_df.empty or "content" not in messages_df.columns:
        return pd.DataFrame(columns=["timestamp", "coin", "reasoning"])

    assistant_df = messages_df[messages_df.get("role", "") == "assistant"].copy()
    if assistant_df.empty:
        return pd.DataFrame(columns=["timestamp", "coin", "reasoning"])

    records: List[Dict[str, Any]] = []
    for _, row in assistant_df.iterrows():
        content = row.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        try:
            payload = json.loads(content)
        except (TypeError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        for coin, coin_payload in payload.items():
            if not isinstance(coin_payload, dict):
                continue
            justification = coin_payload.get("justification")
            if not justification:
                continue
            records.append(
                {
                    "timestamp": row.get("timestamp"),
                    "coin": coin,
                    "reasoning": justification,
                }
            )

    if not records:
        return pd.DataFrame(columns=["timestamp", "coin", "reasoning"])

    df = pd.DataFrame(records)
    df = df.dropna(subset=["timestamp", "coin", "reasoning"])
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "coin", "reasoning"])
    df.sort_values(["coin", "timestamp"], inplace=True)
    return df


def _merge_decisions_with_justifications(
    decisions_df: pd.DataFrame, justifications_df: pd.DataFrame
) -> pd.DataFrame:
    """Merge decision rows with parsed justifications per coin."""
    if decisions_df.empty or justifications_df.empty:
        return decisions_df

    merged_frames: List[pd.DataFrame] = []
    for coin in sorted(decisions_df["coin"].dropna().unique()):
        decisions_slice = decisions_df[decisions_df["coin"] == coin].sort_values(
            "timestamp"
        )
        just_slice = justifications_df[justifications_df["coin"] == coin].sort_values(
            "timestamp"
        )
        if just_slice.empty:
            merged_frames.append(decisions_slice)
            continue

        merged = pd.merge_asof(
            decisions_slice,
            just_slice,
            on="timestamp",
            direction="backward",
            tolerance=pd.Timedelta("5min"),
        )

        if "reasoning_y" in merged.columns:
            merged["reasoning"] = merged["reasoning_x"].combine_first(
                merged["reasoning_y"]
            )
            merged.drop(columns=["reasoning_x", "reasoning_y"], inplace=True)
        merged_frames.append(merged)

    if not merged_frames:
        return decisions_df

    merged_decisions = pd.concat(merged_frames, ignore_index=True)
    return merged_decisions


def render_trades_tab(trades_df: pd.DataFrame) -> None:
    if trades_df.empty:
        st.info("No trades recorded yet.")
        return

    st.dataframe(
        trades_df,
        column_config={
            "timestamp": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm:ss"),
            "quantity": st.column_config.NumberColumn(format="%.4f"),
            "price": st.column_config.NumberColumn(format="Rp%.2f"),
            "profit_target": st.column_config.NumberColumn(format="Rp%.2f"),
            "stop_loss": st.column_config.NumberColumn(format="Rp%.2f"),
            "pnl": st.column_config.NumberColumn(format="Rp%.2f"),
            "net_pnl": st.column_config.NumberColumn(format="Rp%.2f"),
            "fee": st.column_config.NumberColumn(format="Rp%.2f"),
            "balance_after": st.column_config.NumberColumn(format="Rp%.2f"),
            "position_fee_total": st.column_config.NumberColumn(format="Rp%.2f"),
            "position_net_pnl": st.column_config.NumberColumn(format="Rp%.2f"),
        },
        use_container_width=True,
        height=420,
    )


def render_ai_tab(decisions_df: pd.DataFrame, messages_df: pd.DataFrame) -> None:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Recent AI Decisions")
        if decisions_df.empty:
            st.write("No decisions yet.")
        else:
            decisions_display = decisions_df.copy()
            justifications = _extract_decision_justifications(messages_df)
            if not justifications.empty:
                merged = _merge_decisions_with_justifications(
                    decisions_display, justifications
                )
                decisions_display = merged.sort_values(
                    "timestamp", ascending=False
                ).copy()
            else:
                decisions_display.sort_values(
                    "timestamp", ascending=False, inplace=True
                )
            st.dataframe(
                decisions_display.head(50),
                column_config={
                    "timestamp": st.column_config.DatetimeColumn(
                        format="YYYY-MM-DD HH:mm:ss"
                    ),
                    "confidence": st.column_config.NumberColumn(format="%.2f"),
                },
                use_container_width=True,
            )

    with col2:
        st.subheader("Recent AI Messages")
        if messages_df.empty:
            st.write("No messages logged yet.")
        else:
            st.dataframe(
                messages_df.head(50),
                column_config={
                    "timestamp": st.column_config.DatetimeColumn(
                        format="YYYY-MM-DD HH:mm:ss"
                    ),
                },
                use_container_width=True,
            )


def main() -> None:
    st.set_page_config(page_title="Trading Bot Monitor", layout="wide")
    st.title("Trading Bot Monitor")
    st.caption("Trading bot dashboard for the MultiLLM trading bot.")

    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

    model_dirs = discover_model_directories()
    if not model_dirs:
        st.warning("No model data found yet. Run the trading bot to generate logs.")
        return

    model_names = list(model_dirs.keys())
    model_labels = {name: format_model_label(name) for name in model_names}

    state_map: Dict[str, pd.DataFrame] = {}
    trades_map: Dict[str, pd.DataFrame] = {}
    decisions_map: Dict[str, pd.DataFrame] = {}
    messages_map: Dict[str, pd.DataFrame] = {}
    btc_price_map: Dict[str, pd.DataFrame] = {}

    for model_name, path in model_dirs.items():
        state_map[model_name] = get_portfolio_state(str(path / "portfolio_state.csv"))
        trades_map[model_name] = get_trades(str(path / "trade_history.csv"))
        decisions_map[model_name] = get_ai_decisions(str(path / "ai_decisions.csv"))
        messages_map[model_name] = get_ai_messages(str(path / "ai_messages.csv"))
        btc_price_map[model_name] = get_local_btc_price_series(
            str(path / "ai_messages.csv")
        )

    st.subheader("Combined Equity Across Models")
    render_combined_equity_chart(state_map, btc_price_map)

    tabs = st.tabs([model_labels[name] for name in model_names])

    for tab, model_name in zip(tabs, model_names):
        with tab:
            label = model_labels[model_name]
            st.markdown(f"### Portfolio Overview – {label}")
            render_portfolio_tab(
                state_map.get(model_name, pd.DataFrame()),
                trades_map.get(model_name, pd.DataFrame()),
                btc_price_map.get(model_name),
            )

            st.markdown("---")
            st.subheader("Trade Log")
            render_trades_tab(trades_map.get(model_name, pd.DataFrame()))

            st.markdown("---")
            st.subheader("AI Activity")
            render_ai_tab(
                decisions_map.get(model_name, pd.DataFrame()),
                messages_map.get(model_name, pd.DataFrame()),
            )


if __name__ == "__main__":
    main()
