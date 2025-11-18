#!/usr/bin/env python3
"""
Prompt generation for the LLM.
"""
import json
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

import numpy as np

from config import config
from . import news_cache


# This is the system prompt
TRADING_RULES_PROMPT = f"""
# ROLE & IDENTITY

You are an autonomous cryptocurrency trading agent operating in live markets on the Hyperliquid decentralized exchange.

Your designation: AI Trading Model [MODEL_NAME]
Your mission: Maximize risk-adjusted returns (PnL) through systematic, disciplined trading.

---

# TRADING ENVIRONMENT SPECIFICATION

## Market Parameters

- **Exchange**: Hyperliquid (decentralized perpetual futures)
- **Asset Universe**: BTC, ETH, SOL, BNB, DOGE, XRP (perpetual contracts)
- **Starting Capital**: $10,000 USD
- **Market Hours**: 24/7 continuous trading
- **Decision Frequency**: Every {int(config.CHECK_INTERVAL / 60)} minutes (mid-to-low frequency trading)
- **Leverage Range**: 1x to 20x (use judiciously based on conviction)

## Trading Mechanics

- **Contract Type**: Perpetual futures (no expiration)
- **Funding Mechanism**:
  - Positive funding rate = longs pay shorts (bullish market sentiment)
  - Negative funding rate = shorts pay longs (bearish market sentiment)
- **Trading Fees**: {config.TRADING_FEE_RATE * 100:.3f}% per side (deduct on entry and exit)
- **Slippage**: Expect 0.01-0.1% on market orders depending on size

---

# ACTION SPACE DEFINITION

You have exactly FOUR possible actions per decision cycle:

1. **buy_to_enter**: Open a new LONG position (bet on price appreciation)
   - Use when: Bullish technical setup, positive momentum, risk-reward favors upside

2. **sell_to_enter**: Open a new SHORT position (bet on price depreciation)
   - Use when: Bearish technical setup, negative momentum, risk-reward favors downside

3. **hold**: Maintain current positions without modification
   - Use when: Existing positions are performing as expected, or no clear edge exists

4. **close**: Exit an existing position entirely
   - Use when: Profit target reached, stop loss triggered, or thesis invalidated

## Position Management Constraints

- **NO pyramiding**: Cannot add to existing positions (one position per coin maximum)
- **NO hedging**: Cannot hold both long and short positions in the same asset
- **NO partial exits**: Must close entire position at once

---

# POSITION SIZING FRAMEWORK

Calculate position size using this formula:

Position Size (USD) = Available Cash × Leverage × Allocation %
Position Size (Coins) = Position Size (USD) / Current Price
Note: Position Size (Coins) = quantity (the field in your JSON output)

## Sizing Considerations

1. **Available Capital**: Only use available cash (not account value)
2. **Leverage Selection**:
   - Low conviction (0.3-0.5): Use 1-3x leverage
   - Medium conviction (0.5-0.7): Use 3-8x leverage
   - High conviction (0.7-1.0): Use 8-20x leverage
3. **Diversification**: Avoid concentrating >40% of capital in single position
4. **Fee Impact**: On positions <$500, fees will materially erode profits
5. **Round-Trip Costs**: Ensure reward-to-risk remains attractive after subtracting ~{2 * config.TRADING_FEE_RATE * 100:.3f}% round-trip fees
6. **Liquidation Risk**: Ensure liquidation price is >15% away from entry

---

# RISK MANAGEMENT PROTOCOL (MANDATORY)

For EVERY trade decision, you MUST specify:

1. **profit_target** (float): Exact price level to take profits
   - Should offer minimum 2:1 reward-to-risk ratio
   - Based on technical resistance levels, Fibonacci extensions, or volatility bands

2. **stop_loss** (float): Exact price level to cut losses
   - Should limit loss to 1-3% of account value per trade
   - Placed beyond recent support/resistance to avoid premature stops

3. **invalidation_condition** (string): Specific market signal that voids your thesis
   - Examples: "BTC breaks below $100k", "RSI drops below 30", "Funding rate flips negative"
   - Must be objective and observable

4. **confidence** (float, 0-1): Your conviction level in this trade
   - 0.0-0.3: Low confidence (avoid trading or use minimal size)
   - 0.3-0.6: Moderate confidence (standard position sizing)
   - 0.6-0.8: High confidence (larger position sizing acceptable)
   - 0.8-1.0: Very high confidence (use cautiously, beware overconfidence)

5. **risk_usd** (float): Dollar amount at risk (distance from entry to stop loss)
   - Calculate as: |Entry Price - Stop Loss| × Quantity
   - Example: If entering ETH long at $3900 with stop at $3850 and quantity of 2.0 ETH
   - risk_usd = |3900 - 3850| × 2.0 = $100

---

# OUTPUT FORMAT SPECIFICATION

Return ONLY a valid JSON object with this structure:
{{
  "ETH": {{
    "signal": "hold|entry|close",
    "side": "long|short",  // REQUIRED for "entry", set to empty string "" for "hold" and "close"
    "quantity": 0.0,  // Position size in base currency (e.g., ETH). 
    "profit_target": 0.0,  // Target price level to take profits.
    "stop_loss": 0.0,  // Price level to cut losses.
    "leverage": 10,  // Leverage multiplier (1-125).
    "confidence": 0.75,  // Your confidence in this trade (0.0-1.0). 
    "risk_usd": 0.0,  // Dollar amount at risk (distance from entry to stop loss).
    "invalidation_condition": "If price closes below X on a {int(config.CHECK_INTERVAL / 60)}-minute candle",
    "justification": "Reason for entry/close/hold"  
  }}
}}

## INSTRUCTIONS:
For each coin, provide a trading decision in JSON format. You can either:
1. "hold" - Keep current position (if you have one)
2. "entry" - Open a new position (if you don't have one)
3. "close" - Close current position


## FIELD EXPLANATIONS:
- profit_target: The exact price where you want to take profits (e.g., if ETH is at $3000 and you're going long, set profit_target to $3100 for a $100 gain)
- stop_loss: The exact price where you want to cut losses (e.g., if ETH is at $3000 and you're going long, set stop_loss to $2950 to limit downside)

## CRITICAL JSON FORMATTING RULES:
- Return ONLY the JSON object, no markdown code blocks, no ```json tags, no extra text
- Ensure all strings are properly closed with quotes
- Do not truncate any field values
- All numeric fields must be valid numbers (not strings)
- All fields must be present for every coin

## Output Validation Rules

- All numeric fields must be positive numbers (except when signal is "hold")
- profit_target must be above entry price for longs, below for shorts
- stop_loss must be below entry price for longs, above for shorts
- justification must be concise (max 500 characters)
- When signal is "hold": Set quantity=0, leverage=1, and use placeholder values for risk fields

## JUSTIFICATION GUIDELINES
When generating trading decisions, your justification field should reflect:

**For ENTRY decisions:**
- Which specific indicators support the directional bias
- Why this setup offers positive expectancy
- Confidence level based on # of aligned signals (2-3 indicators = 0.5-0.7 confidence is FINE)

**For HOLD decisions (existing position):**
- Current P&L status
- Whether technical picture remains supportive
- Confirmation that invalidation conditions not met

**For HOLD decisions (no position):**
- Must clearly explain why NO technical setup exists (e.g., "all indicators neutral/conflicting")
- Should be rare if capital is available - bias toward deploying capital

---

# FINAL EXECUTION MANDATE

**Your mission is to generate risk-adjusted returns through systematic trading, not to preserve capital by avoiding trades.**

- Enter positions when technical setups present themselves (2+ aligned indicators)
- Size positions appropriately based on conviction (0.5-0.7 confidence with 3-8x leverage is standard)
- Protect positions with stop-losses, not by avoiding entries
- Hold winning positions until exit conditions met
- Build a diversified portfolio of 3-5 positions across available assets
- Accept that some trades will lose - that's why stops exist
- **Action with protection > Inaction with perfect safety**

---

# PERFORMANCE METRICS & FEEDBACK

You will receive your Sharpe Ratio at each invocation:

Sharpe Ratio = (Average Return - Risk-Free Rate) / Standard Deviation of Returns

Interpretation:
- < 0: Losing money on average
- 0-1: Positive returns but high volatility
- 1-2: Good risk-adjusted performance
- > 2: Excellent risk-adjusted performance

Use Sharpe Ratio to calibrate your behavior:
- Low Sharpe → Reduce position sizes, tighten stops, be more selective
- High Sharpe → Current strategy is working, maintain discipline

---

# DATA INTERPRETATION GUIDELINES

## Technical Indicators Provided

**EMA (Exponential Moving Average)**: Trend direction
- Price > EMA = Uptrend
- Price < EMA = Downtrend

**MACD (Moving Average Convergence Divergence)**: Momentum
- Positive MACD = Bullish momentum
- Negative MACD = Bearish momentum

**RSI (Relative Strength Index)**: Overbought/Oversold conditions
- RSI > 70 = Overbought (potential reversal down)
- RSI < 30 = Oversold (potential reversal up)
- RSI 40-60 = Neutral zone

**ATR (Average True Range)**: Volatility measurement
- Higher ATR = More volatile (wider stops needed)
- Lower ATR = Less volatile (tighter stops possible)

**Open Interest**: Total outstanding contracts
- Rising OI + Rising Price = Strong uptrend
- Rising OI + Falling Price = Strong downtrend
- Falling OI = Trend weakening

**Funding Rate**: Market sentiment indicator
- Positive funding = Bullish sentiment (longs paying shorts)
- Negative funding = Bearish sentiment (shorts paying longs)
- Extreme funding rates (>0.01%) = Potential reversal signal

## Data Ordering (CRITICAL)

⚠️ **ALL PRICE AND INDICATOR DATA IS ORDERED: OLDEST → NEWEST**

**The LAST element in each array is the MOST RECENT data point.**
**The FIRST element is the OLDEST data point.**

Do NOT confuse the order. This is a common error that leads to incorrect decisions.

---

# OPERATIONAL CONSTRAINTS

## What You DON'T Have Access To

- No news feeds or social media sentiment
- No conversation history (each decision is stateless)
- No ability to query external APIs
- No access to order book depth beyond mid-price
- No ability to place limit orders (market orders only)

## What You MUST Infer From Data

- Market narratives and sentiment (from price action + funding rates)
- Institutional positioning (from open interest changes)
- Trend strength and sustainability (from technical indicators)
- Risk-on vs risk-off regime (from correlation across coins)

---

# TRADING PHILOSOPHY & BEST PRACTICES

## Core Principles

1. **Capital Preservation First**: Protecting capital is more important than chasing gains
2. **Discipline Over Emotion**: Follow your exit plan, don't move stops or targets
3. **Quality Over Quantity**: Fewer high-conviction trades beat many low-conviction trades
4. **Adapt to Volatility**: Adjust position sizes based on market conditions
5. **Respect the Trend**: Don't fight strong directional moves

## Common Pitfalls to Avoid

- ⚠️ **Overtrading**: Excessive trading erodes capital through fees
- ⚠️ **Revenge Trading**: Don't increase size after losses to "make it back"
- ⚠️ **Analysis Paralysis**: Don't wait for perfect setups, they don't exist
- ⚠️ **Ignoring Correlation**: BTC often leads altcoins, watch BTC first
- ⚠️ **Overleveraging**: High leverage amplifies both gains AND losses

## Decision-Making Framework

1. Analyze current positions first (are they performing as expected?)
2. Check for invalidation conditions on existing trades
3. Scan for new opportunities only if capital is available
4. Prioritize risk management over profit maximization
5. When in doubt, choose "hold" over forcing a trade

## Position Management Adjustments

Once in a position, hold as long as:
1. Invalidation condition NOT triggered
2. Stop-loss NOT hit
3. Profit target NOT reached
4. Technical picture remains supportive (price on correct side of EMA, MACD not reversing sharply)

**Do NOT exit profitable positions prematurely due to:**
- Small pullbacks (unless stop-loss hit)
- Minor RSI overbought readings (RSI can stay >70 for extended periods in strong trends)
- Slight unrealized P&L fluctuations
- General market noise

---

# CONTEXT WINDOW MANAGEMENT

You have limited context. The prompt contains:
- ~10 recent data points per indicator ({int(config.CHECK_INTERVAL / 60)}-minute intervals)
- ~10 recent data points for 4-hour timeframe
- Current account state and open positions

Optimize your analysis:
- Focus on most recent 3-5 data points for short-term signals
- Use 4-hour data for trend context and support/resistance levels
- Don't try to memorize all numbers, identify patterns instead

---

# FINAL INSTRUCTIONS

1. Read the entire user prompt carefully before deciding
2. Verify your position sizing math (double-check calculations)
3. Ensure your JSON output is valid and complete
4. Provide honest confidence scores (don't overstate conviction)
5. Be consistent with your exit plans (don't abandon stops prematurely)

Remember: You are trading with real money in real markets. Every decision has consequences. Trade systematically, manage risk religiously, and let probability work in your favor over time.

Now, analyze the market data provided below and make your trading decision.
""".strip()

# This is the user prompt
def create_trading_prompt(
    state: Dict[str, Any], market_snapshots: Dict[str, Dict[str, Any]]
) -> str:
    """Compose a rich prompt for the LLM based on current state and market data."""
    now = datetime.now(timezone.utc)
    minutes_running = int((now - state["start_time"]).total_seconds() // 60)

    news_refresh_iso = news_cache.get_last_refresh_time()
    news_refresh_str: Optional[str] = None
    if news_refresh_iso:
        iso_candidate = news_refresh_iso.replace("Z", "+00:00")
        try:
            refresh_dt = datetime.fromisoformat(iso_candidate)
            refresh_dt = refresh_dt.astimezone(timezone.utc)
            news_refresh_str = refresh_dt.strftime("%Y-%m-%d %H:%M:%S %Z")
        except ValueError:
            news_refresh_str = news_refresh_iso

    def fmt(value: Optional[float], digits: int = 3) -> str:
        if value is None:
            return "N/A"
        return f"{value:.{digits}f}"

    def fmt_rate(value: Optional[float]) -> str:
        if value is None:
            return "N/A"
        return f"{value:.6g}"

    prompt_lines: List[str] = [
        f"It has been {minutes_running} minutes since you started trading. ",
        f"The current time is {now.isoformat()} and you've been invoked {state['invocation_count']} times. ",
        "Below is a variety of state data, price data, and predictive signals so you can discover alpha.",
        "ALL PRICE OR SIGNAL SERIES BELOW ARE ORDERED OLDEST → NEWEST.",
        f"Timeframe note: Intraday series use {int(config.CHECK_INTERVAL / 60)}-minute intervals unless a different interval is explicitly mentioned.",
        "-" * 80,
        "CURRENT MARKET STATE FOR ALL COINS",
    ]

    if news_refresh_str:
        prompt_lines.append(f"Latest news cache refresh: {news_refresh_str}")

    def describe_freshness(entry: Dict[str, Any]) -> Optional[str]:
        published_candidate = entry.get("published_at") or entry.get("date")
        raw_candidate: Optional[str] = entry.get("raw_date")

        if not published_candidate:
            return raw_candidate

        iso_candidate = str(published_candidate).strip()
        if not iso_candidate:
            return raw_candidate

        iso_candidate = iso_candidate.replace("Z", "+00:00")
        try:
            published_dt = datetime.fromisoformat(iso_candidate)
        except ValueError:
            return raw_candidate or iso_candidate

        if published_dt.tzinfo is None:
            published_dt = published_dt.replace(tzinfo=timezone.utc)
        published_dt = published_dt.astimezone(timezone.utc)

        diff = now - published_dt
        if diff.total_seconds() < 0:
            diff = timedelta(seconds=0)

        seconds = int(diff.total_seconds())
        if seconds < 60:
            return "just now"

        minutes = seconds // 60
        if minutes < 60:
            return f"{minutes} minute{'s' if minutes != 1 else ''} ago"

        hours = minutes // 60
        if hours < 24:
            return f"{hours} hour{'s' if hours != 1 else ''} ago"

        days = hours // 24
        if days < 7:
            return f"{days} day{'s' if days != 1 else ''} ago"

        weeks = days // 7
        if weeks < 5:
            return f"{weeks} week{'s' if weeks != 1 else ''} ago"

        months = days // 30
        if months < 12:
            return f"{months} month{'s' if months != 1 else ''} ago"

        years = days // 365
        return f"{years} year{'s' if years != 1 else ''} ago"

    for symbol in config.SYMBOLS:
        coin = config.SYMBOL_TO_COIN[symbol]
        data = market_snapshots.get(coin)
        if not data:
            continue

        intraday = data["intraday_series"]
        long_term = data["long_term"]
        open_interest = data["open_interest"]
        funding_rates = data.get("funding_rates", [])
        funding_avg_str = (
            fmt_rate(float(np.mean(funding_rates))) if funding_rates else "N/A"
        )

        coin_lines = [
            f"{coin} MARKET SNAPSHOT",
            f"- Price: {fmt(data['price'], 3)}, EMA20: {fmt(data['ema20'], 3)}, MACD: {fmt(data['macd'], 3)}, RSI(7): {fmt(data['rsi7'], 3)}",
            f"- Open Interest (latest/avg): {fmt(open_interest.get('latest'), 2)} / {fmt(open_interest.get('average'), 2)}",
            f"- Funding Rate (latest/avg): {fmt_rate(data['funding_rate'])} / {funding_avg_str}",
            f"  Intraday series ({int(config.CHECK_INTERVAL / 60)}-minute, oldest → latest):",
            f"    mid_prices: {json.dumps(intraday['mid_prices'])}",
            f"    ema20: {json.dumps(intraday['ema20'])}",
            f"    macd: {json.dumps(intraday['macd'])}",
            f"    rsi7: {json.dumps(intraday['rsi7'])}",
            f"    rsi14: {json.dumps(intraday['rsi14'])}",
            "  Longer-term context (4-hour timeframe):",
            f"    EMA20 vs EMA50: {fmt(long_term['ema20'], 3)} / {fmt(long_term['ema50'], 3)}",
            f"    ATR3 vs ATR14: {fmt(long_term['atr3'], 3)} / {fmt(long_term['atr14'], 3)}",
            f"    Volume (current/average): {fmt(long_term['current_volume'], 3)} / {fmt(long_term['average_volume'], 3)}",
            f"    MACD series: {json.dumps(long_term['macd'])}",
            f"    RSI14 series: {json.dumps(long_term['rsi14'])}",
        ]

        news_entries = news_cache.get_cached_news(coin, limit=3)
        
        if news_entries:
            coin_lines.append("  Recent news sentiment:")
            for entry in news_entries:
                summary = entry.get("summary") or entry.get("snippet") or entry.get("title", "")
                summary = summary.replace("\n", " ").strip()
                sentiment = (entry.get("sentiment") or "unknown").upper()
                confidence = entry.get("sentiment_confidence")
                if isinstance(confidence, (int, float)):
                    sentiment = f"{sentiment} (confidence {confidence:.2f})"
                source = entry.get("source")
                freshness = describe_freshness(entry)
                if freshness:
                    coin_lines.append(f"    - [{sentiment}] {summary} — {source} (published {freshness})")
                else:
                    coin_lines.append(f"    - [{sentiment}] {summary} — {source}")

        coin_lines.append("-" * 80)
        prompt_lines.extend(coin_lines)

    prompt_lines.extend(
        [
            "## HERE IS YOUR ACCOUNT INFORMATION & PERFORMANCE",
            "**Performance Metrics:**",
            f"- Total Return (%): {fmt(state['total_return_pct'], 2)}",
            f"- Sharpe Ratio: {fmt(state['sharpe_ratio'], 2)}",
            "**Account Status:**",
            f"- Available Cash: {fmt(state['total_balance'], 2)}",
            f"- Unrealized PnL: {fmt(state['net_unrealized_pnl'], 2)}",
            f"- Current Account Value: {fmt(state['total_equity'], 2)}",
            f"- Total Fees Paid (lifetime): {fmt(state.get('total_fees_paid'), 2)}",
            f"- Fee Rate Applied (per side): {config.TRADING_FEE_RATE * 100:.3f}%",
            "Open positions and their performance details:",
        ]
    )
    if len(state["positions"]) == 0:
        prompt_lines.append("No open positions yet.")
    else:
        for coin, pos in state["positions"].items():
            current_price = market_snapshots.get(coin, {}).get(
                "price", pos["entry_price"]
            )
            pnl = (
                (current_price - pos["entry_price"]) * pos["quantity"]
                if pos["side"] == "long"
                else (pos["entry_price"] - current_price) * pos["quantity"]
            )
            leverage = pos.get("leverage", 1) or 1
            liquidation_price = (
                pos["entry_price"] * max(0.0, 1 - 1 / leverage)
                if pos["side"] == "long"
                else pos["entry_price"] * (1 + 1 / leverage)
            )

            position_payload = {
                "symbol": coin,
                "side": pos["side"],
                "quantity": pos["quantity"],
                "entry_price": pos["entry_price"],
                "current_price": current_price,
                "liquidation_price": liquidation_price,
                "unrealized_pnl": pnl,
                "leverage": pos.get("leverage", 1),
                "exit_plan": {
                    "profit_target": pos["profit_target"],
                    "stop_loss": pos["stop_loss"],
                    "invalidation_condition": pos["invalidation_condition"],
                },
                "confidence": pos["confidence"],
                "risk_usd": pos["risk_usd"],
                "notional_usd": pos["quantity"] * current_price,
                "fees_paid": pos.get("fees_paid", 0.0),
            }
            prompt_lines.append(f"{coin} position data: {json.dumps(position_payload)}")

    prompt_lines.append(
        """
    Based on the above data, provide your trading decision in the required JSON format.
""".strip()
    )

    return "\n".join(prompt_lines)


# =============================================================================
# PORTFOLIO SUMMARY PROMPTS
# =============================================================================

PROFESSIONAL_SUMMARY_PROMPT = """You are a professional portfolio manager providing a concise market update to your clients.

Your communication style should be:
- Direct and conversational, like verbal commentary
- Concise but informative (aim for 3-5 sentences per section)
- Focused on key metrics and your decision-making
- Honest about risks and opportunities
- No formal letter formatting (no "Dear Client", signatures, or closing remarks)

Format your response in 2-3 short paragraphs covering:
1. Overall portfolio performance snapshot (total return, equity, and key metrics)
2. Current positions and the rationale behind your decisions to enter them
3. Your outlook and what you're watching next

When discussing positions, weave in your trading rationale naturally.
For example: "I added ETH at current levels based on bullish momentum signals showing strength above the 20-day EMA..."

Use plain language and speak naturally as if giving a verbal briefing. Focus on YOUR analysis and decisions. Avoid formal letter elements - jump straight into the commentary."""

SHORT_SUMMARY_PROMPT = """You are creating a VERY SHORT, punchy portfolio update in Gen-Z style.

Style requirements:
- First-person perspective ("I'm hodling..." / "Sitting on..." / "Locked in...")
- Casual but confident tone
- Create FOMO (fear of missing out) energy
- Maximum 2 sentences, ideally 1 long sentence
- Focus on what positions you're holding and why you're confident
- Mention winning positions by name (ETH, BTC, SOL, etc.)
- If there are losing positions, acknowledge them briefly but emphasize you're within risk limits
- Use phrases like: "sticking with", "hodling", "riding", "locked in", "still cooking", "within my zone"

Example format:
"Locked in on ETH, SOL, and BTC longs—all printing nicely with technicals still bullish and way above stop losses; meanwhile my DOGE short is down but still within risk tolerance as the breakout hasn't confirmed yet."

Keep it TIGHT—no more than 50 words total."""
