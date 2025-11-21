import config

test = f"""
INSTRUCTIONS:
For each coin, provide a trading decision in JSON format. You can either:
1. "hold" - Keep current position (if you have one)
2. "entry" - Open a new position (if you don't have one)
3. "close" - Close current position

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
    "invalidation_condition": "If price closes below X on a {int(config.CHECK_INTERVAL / 60)}-minute candle",
    "justification": "Reason for entry/close/hold"  
  }}
}}

FIELD EXPLANATIONS:
- profit_target: The exact price where you want to take profits (e.g., if ETH is at $3000 and you're going long, set profit_target to $3100 for a $100 gain)
- stop_loss: The exact price where you want to cut losses (e.g., if ETH is at $3000 and you're going long, set stop_loss to $2950 to limit downside)

CRITICAL JSON FORMATTING RULES:
- Return ONLY the JSON object, no markdown code blocks, no ```json tags, no extra text
- Ensure all strings are properly closed with quotes
- Do not truncate any field values
- All numeric fields must be valid numbers (not strings)
- All fields must be present for every coin

IMPORTANT TRADING RULES:
- Only suggest entries if you see strong opportunities
- Use proper risk management with appropriate stop losses
- Provide clear invalidation conditions for entries
""".strip()

print(test)