# 🌐 Quick Guide: Share Your Dashboard

## For People on Same WiFi/Network

**Easiest option - No installation needed!**

```bash
cd front_end
./run_dashboard_network.sh
```

The script will show you a URL like:
```
Network: http://192.168.1.100:8501
```

**Share this URL with anyone on the same WiFi!** They can open it in their browser.

---

## For People Anywhere on Internet

### Option 1: ngrok (Quick & Temporary)

**Setup (one-time):**
```bash
# Install ngrok
brew install ngrok

# Sign up at https://ngrok.com and get your token
ngrok config add-authtoken YOUR_TOKEN_HERE
```

**Run:**
```bash
cd front_end
./run_dashboard_ngrok.sh
```

You'll get a public URL like:
```
https://abc123.ngrok.io
```

**Share this URL with anyone!** Works from anywhere in the world.

⚠️ **Note**: URL changes each time you restart (free plan)

---

### Option 2: Streamlit Cloud (Permanent & Free)

**Best for long-term sharing**

1. **Push code to GitHub:**
```bash
cd /Users/rz/Documents/tradingagent/TradingAIforMulti-Assets
git add front_end/
git add data/  # If you want to include data
git commit -m "Add dashboard"
git push origin main
```

2. **Deploy on Streamlit Cloud:**
   - Go to https://share.streamlit.io/
   - Sign in with GitHub
   - Click "New app"
   - Repository: `YOUR_USERNAME/TradingAIforMulti-Assets`
   - Branch: `main` (or `multi_llm`)
   - Main file path: `front_end/dashboard.py`
   - Click "Deploy"

3. **Your permanent URL:**
```
https://yourapp.streamlit.app
```

**Advantages:**
- ✅ Always online (24/7)
- ✅ Free forever
- ✅ Auto-updates from GitHub
- ✅ HTTPS included
- ✅ Custom domain possible

---

## Quick Comparison

| Method | Best For | Setup Time | Cost | Accessible From |
|--------|----------|------------|------|-----------------|
| **Network Mode** | Same WiFi users | 30 sec | Free | Local network only |
| **ngrok** | Quick demo | 5 min | Free* | Internet (anywhere) |
| **Streamlit Cloud** | Permanent hosting | 10 min | Free | Internet (anywhere) |

*ngrok free tier has rate limits

---

## 🔒 Security Tips

### Add Password Protection

Edit `dashboard.py` and add this at the start of `main()`:

```python
def main():
    # Add password protection
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    
    if not st.session_state.authenticated:
        password = st.text_input("Enter Password", type="password")
        if st.button("Login"):
            if password == "your_secure_password_here":
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Incorrect password")
        st.stop()
    
    # Rest of your dashboard code...
```

---

## 🆘 Troubleshooting

### "Connection refused" error
- Check if firewall is blocking port 8501
- On Mac: System Settings → Network → Firewall → Allow Streamlit
- On Linux: `sudo ufw allow 8501`

### Can't access from external network
- Make sure you're using `--server.address 0.0.0.0`
- Check router firewall settings
- Consider using ngrok instead

### Dashboard is slow over network
- Reduce auto-refresh frequency in dashboard.py
- Optimize data loading (cache data)
- Use compression in config.toml

---

## Need Help?

See full documentation in `EXTERNAL_ACCESS.md`
