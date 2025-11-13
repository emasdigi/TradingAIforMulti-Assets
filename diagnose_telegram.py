#!/usr/bin/env python3
"""
Helper script to diagnose Telegram connection issues
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

print("\n" + "="*60)
print("TELEGRAM CONNECTION DIAGNOSTIC")
print("="*60 + "\n")

# Check if credentials are set
print("1. Checking credentials...")
if not TELEGRAM_BOT_TOKEN:
    print("   ❌ TELEGRAM_BOT_TOKEN is not set in .env file")
    exit(1)
else:
    print(f"   ✅ Bot Token: {TELEGRAM_BOT_TOKEN[:10]}...")

if not TELEGRAM_CHAT_ID:
    print("   ❌ TELEGRAM_CHAT_ID is not set in .env file")
    exit(1)
else:
    print(f"   ✅ Chat ID: {TELEGRAM_CHAT_ID}")

# Test bot token validity
print("\n2. Testing bot token...")
try:
    response = requests.get(
        f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getMe",
        timeout=10
    )
    if response.status_code == 200:
        bot_info = response.json()
        if bot_info.get("ok"):
            print(f"   ✅ Bot token is valid!")
            print(f"   Bot name: @{bot_info['result']['username']}")
            print(f"   Bot ID: {bot_info['result']['id']}")
        else:
            print(f"   ❌ Bot token error: {bot_info}")
    else:
        print(f"   ❌ Failed to verify bot token (status {response.status_code})")
except Exception as e:
    print(f"   ❌ Error checking bot: {e}")

# Get recent updates
print("\n3. Checking for messages to the bot...")
try:
    response = requests.get(
        f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates",
        timeout=10
    )
    if response.status_code == 200:
        data = response.json()
        if data.get("ok") and data.get("result"):
            updates = data["result"]
            if updates:
                print(f"   ✅ Found {len(updates)} message(s)")
                print("\n   Recent chat IDs found:")
                seen_chat_ids = set()
                for update in updates:
                    if "message" in update:
                        chat_id = update["message"]["chat"]["id"]
                        chat_type = update["message"]["chat"]["type"]
                        first_name = update["message"]["chat"].get("first_name", "")
                        username = update["message"]["chat"].get("username", "")
                        
                        if chat_id not in seen_chat_ids:
                            seen_chat_ids.add(chat_id)
                            print(f"      • Chat ID: {chat_id}")
                            print(f"        Name: {first_name}")
                            if username:
                                print(f"        Username: @{username}")
                            print(f"        Type: {chat_type}")
                            
                            # Check if this matches the configured chat ID
                            if str(chat_id) == str(TELEGRAM_CHAT_ID):
                                print(f"        ✅ This matches your configured TELEGRAM_CHAT_ID!")
                            print()
                
                if str(TELEGRAM_CHAT_ID) not in [str(chat_id) for chat_id in seen_chat_ids]:
                    print(f"\n   ⚠️  Your configured chat ID ({TELEGRAM_CHAT_ID}) was NOT found in recent messages!")
                    print(f"   💡 Try using one of the chat IDs listed above instead.")
            else:
                print("   ⚠️  No messages found. Have you sent a message to your bot yet?")
                print("   💡 Please:")
                print("      1. Open Telegram")
                print("      2. Search for your bot")
                print("      3. Send any message (like 'hello' or '/start')")
                print("      4. Run this script again")
        else:
            print(f"   ❌ Error: {data}")
except Exception as e:
    print(f"   ❌ Error checking updates: {e}")

# Try to send a test message
print("\n4. Testing message sending to configured chat ID...")
try:
    response = requests.post(
        f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
        json={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": "🧪 Test message from diagnostic script!"
        },
        timeout=10
    )
    if response.status_code == 200:
        data = response.json()
        if data.get("ok"):
            print("   ✅ Message sent successfully!")
            print("   Check your Telegram - you should have received a test message!")
        else:
            print(f"   ❌ Telegram API error: {data}")
            if "chat not found" in str(data):
                print("\n   💡 SOLUTION: The chat ID is incorrect or you haven't messaged the bot yet.")
                print("      1. Make sure you've sent a message to your bot first")
                print("      2. Use one of the chat IDs found in step 3 above")
    else:
        print(f"   ❌ HTTP error {response.status_code}: {response.text}")
except Exception as e:
    print(f"   ❌ Error sending message: {e}")

print("\n" + "="*60)
print("DIAGNOSTIC COMPLETE")
print("="*60 + "\n")

