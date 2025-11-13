import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import os
from pathlib import Path
import pytz

# Page Configuration
st.set_page_config(
    page_title="AI Trading Bot Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS for minimalist design
st.markdown(
    """
<style>
    /* Main theme colors */
    :root {
        --bg-color: #FAFAFA;
        --text-primary: #2C3E50;
        --text-secondary: #7F8C8D;
        --accent-blue: #3498DB;
        --accent-green: #27AE60;
        --accent-red: #E74C3C;
        --border-color: #E0E0E0;
    }
    
    /* Global styles */
    .main {
        background-color: var(--bg-color);
    }
    
    /* Header styles */
    .dashboard-header {
        text-align: center;
        padding: 20px 0;
        border-bottom: 2px solid var(--border-color);
        margin-bottom: 30px;
    }
    
    .dashboard-title {
        font-size: 2.5rem;
        font-weight: 500;
        color: var(--text-primary);
        margin-bottom: 10px;
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 8px;
        border: 1px solid var(--border-color);
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: var(--text-secondary);
        margin-bottom: 8px;
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 600;
        color: var(--text-primary);
        font-family: 'Monaco', monospace;
    }
    
    .metric-change {
        font-size: 0.85rem;
        margin-top: 5px;
    }
    
    .positive {
        color: var(--accent-green);
    }
    
    .negative {
        color: var(--accent-red);
    }
    
    .neutral {
        color: var(--accent-blue);
    }
    
    /* Badge styles */
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
    }
    
    .badge-long {
        background: #D5F5E3;
        color: #27AE60;
    }
    
    .badge-short {
        background: #FADBD8;
        color: #E74C3C;
    }
    
    .badge-entry {
        background: #D6EAF8;
        color: #3498DB;
    }
    
    .badge-exit {
        background: #F5CBA7;
        color: #E67E22;
    }
    
    .badge-hold {
        background: #E8E8E8;
        color: #7F8C8D;
    }
    
    /* AI Decision Card */
    .decision-card {
        background: white;
        padding: 16px;
        border-radius: 8px;
        border: 1px solid var(--border-color);
        margin-bottom: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    
    .decision-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 12px;
    }
    
    .coin-symbol {
        font-size: 1.2rem;
        font-weight: 600;
        color: var(--text-primary);
    }
    
    .timestamp-small {
        font-size: 0.75rem;
        color: var(--text-secondary);
    }
    
    /* Progress bar for confidence */
    .confidence-bar {
        height: 8px;
        border-radius: 4px;
        overflow: hidden;
        background: #E0E0E0;
        margin: 8px 0;
    }
    
    .confidence-fill {
        height: 100%;
        transition: width 0.3s ease;
    }
    
    /* Table styles */
    .dataframe {
        font-size: 0.9rem !important;
    }
    
    /* Refresh indicator */
    .refresh-info {
        text-align: center;
        color: var(--text-secondary);
        font-size: 0.85rem;
        margin-top: 10px;
    }
    
    /* Footer */
    .dashboard-footer {
        border-top: 2px solid var(--border-color);
        padding: 20px 0;
        margin-top: 40px;
        text-align: center;
        color: var(--text-secondary);
        font-size: 0.85rem;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Expander customization */
    .streamlit-expanderHeader {
        font-size: 0.9rem;
        color: var(--text-secondary);
    }
</style>
""",
    unsafe_allow_html=True,
)

# Data paths (relative to project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


# Helper functions
def load_data():
    """Load all CSV data files"""
    try:
        # Note: CSV timestamps are in UTC, we need to convert to UTC+7 (WIB)
        utc = pytz.UTC

        def process_timestamp(df):
            """Process timestamp column, handling both tz-aware and naive timestamps"""
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            # Check if already timezone-aware
            if df["timestamp"].dt.tz is None:
                # Naive timestamps - localize to UTC
                df["timestamp"] = df["timestamp"].dt.tz_localize(utc)
            else:
                # Already timezone-aware - convert to UTC
                df["timestamp"] = df["timestamp"].dt.tz_convert(utc)
            return df

        ai_decisions = pd.read_csv(DATA_DIR / "ai_decisions.csv")
        ai_decisions = process_timestamp(ai_decisions)

        portfolio_state = pd.read_csv(DATA_DIR / "portfolio_state.csv")
        portfolio_state = process_timestamp(portfolio_state)

        trade_history = pd.read_csv(DATA_DIR / "trade_history.csv")
        trade_history = process_timestamp(trade_history)

        # AI messages - handle large file
        ai_messages = pd.read_csv(DATA_DIR / "ai_messages.csv")
        ai_messages = process_timestamp(ai_messages)

        return ai_decisions, portfolio_state, trade_history, ai_messages
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None, None


def parse_position_details(position_json):
    """Parse position details from JSON string"""
    import json

    try:
        return json.loads(position_json.replace("'", '"'))
    except:
        return {}


def format_currency(value):
    """Format currency values"""
    return f"${value:,.2f}"


def format_percent(value):
    """Format percentage values"""
    return f"{value:+.2f}%"


def get_color_for_value(value):
    """Get color based on positive/negative value"""
    if value > 0:
        return "positive"
    elif value < 0:
        return "negative"
    else:
        return "neutral"


def get_confidence_color(confidence):
    """Get color gradient for confidence score"""
    if confidence >= 0.7:
        return "#27AE60"  # Green
    elif confidence >= 0.4:
        return "#F39C12"  # Orange
    else:
        return "#E74C3C"  # Red


def to_jakarta_time(timestamp):
    """Convert timestamp to Jakarta timezone (WIB/UTC+7)
    Note: CSV timestamps are in UTC, so this converts UTC to UTC+7"""
    jkt_tz = pytz.timezone("Asia/Jakarta")  # UTC+7 (WIB)

    # If timestamp is timezone-aware, convert it directly
    if timestamp.tzinfo is not None:
        return timestamp.astimezone(jkt_tz)

    # If timestamp is naive, assume it's UTC (as per CSV source)
    utc = pytz.UTC
    timestamp_utc = utc.localize(timestamp)

    # Convert to Jakarta time (UTC+7)
    return timestamp_utc.astimezone(jkt_tz)


def time_ago(timestamp):
    """Convert timestamp to relative time"""
    now = datetime.now()
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=None)
    if now.tzinfo:
        now = now.replace(tzinfo=None)

    diff = now - timestamp

    if diff.days > 0:
        return f"{diff.days}d ago"
    elif diff.seconds >= 3600:
        hours = diff.seconds // 3600
        return f"{hours}h ago"
    elif diff.seconds >= 60:
        minutes = diff.seconds // 60
        return f"{minutes}m ago"
    else:
        return "just now"


def render_header(portfolio_state, last_update):
    """Render dashboard header with key metrics"""
    st.markdown('<div class="dashboard-header">', unsafe_allow_html=True)
    st.markdown(
        '<h1 class="dashboard-title">AI Trading Bot Dashboard</h1>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if portfolio_state is not None and not portfolio_state.empty:
        latest = portfolio_state.iloc[-1]

        # Key metrics row - now with 5 columns
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.markdown(
                f"""
            <div class="metric-card">
                <div class="metric-label">Total Account Value</div>
                <div class="metric-value">{format_currency(latest['total_equity'])}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                f"""
            <div class="metric-card">
                <div class="metric-label">Available Cash</div>
                <div class="metric-value">{format_currency(latest['total_balance'])}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col3:
            unrealized_pnl = latest["net_unrealized_pnl"]
            pnl_color = get_color_for_value(unrealized_pnl)
            st.markdown(
                f"""
            <div class="metric-card">
                <div class="metric-label">Unrealized PnL</div>
                <div class="metric-value {pnl_color}">{format_currency(unrealized_pnl)}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col4:
            # Total Return % as a separate metric
            total_return_pct = latest.get("total_return_pct", 0.0)
            return_color = get_color_for_value(total_return_pct)
            st.markdown(
                f"""
            <div class="metric-card">
                <div class="metric-label">Total Return</div>
                <div class="metric-value {return_color}">{format_percent(total_return_pct)}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col5:
            st.markdown(
                f"""
            <div class="metric-card">
                <div class="metric-label">Open Positions</div>
                <div class="metric-value">{int(latest['num_positions'])}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    # Last updated info (in Jakarta time)
    jkt_time = to_jakarta_time(last_update)
    st.markdown(
        f'<div class="refresh-info">Last Updated: {jkt_time.strftime("%Y-%m-%d %H:%M:%S")} WIB</div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")


def render_positions_table(portfolio_state):
    """Render current positions table"""
    st.subheader("📊 Current Positions")

    if portfolio_state is not None and not portfolio_state.empty:
        latest = portfolio_state.iloc[-1]
        positions = parse_position_details(latest["position_details"])

        if positions:
            for coin, details in positions.items():
                with st.container():
                    col1, col2, col3, col4 = st.columns([1, 2, 2, 1])

                    with col1:
                        st.markdown(f"**{coin}**")
                        side_class = (
                            "badge-long" if details["side"] == "long" else "badge-short"
                        )
                        st.markdown(
                            f'<span class="badge {side_class}">{details["side"].upper()}</span>',
                            unsafe_allow_html=True,
                        )

                    with col2:
                        st.markdown(
                            f"**Entry:** {format_currency(details['entry_price'])}"
                        )
                        st.markdown(
                            f"**Quantity:** {details['quantity']} @ {details['leverage']}x"
                        )

                        # Show unrealized PnL if available
                        unrealized_pnl = details.get("unrealized_pnl", 0)
                        pnl_color = get_color_for_value(unrealized_pnl)
                        st.markdown(
                            f'<span class="{pnl_color}">**Unrealized PnL:** {format_currency(unrealized_pnl)}</span>',
                            unsafe_allow_html=True,
                        )

                    with col3:
                        st.markdown(
                            f"🎯 Target: {format_currency(details['profit_target'])}"
                        )
                        st.markdown(f"🛑 Stop: {format_currency(details['stop_loss'])}")

                    with col4:
                        margin = details["margin"]
                        st.markdown(f"**Margin:** {format_currency(margin)}")
                        confidence = details.get("confidence", 0)
                        st.progress(confidence)
                        st.caption(f"Confidence: {confidence:.0%}")

                    # Progress bar showing price position
                    entry = details["entry_price"]
                    target = details["profit_target"]
                    stop = details["stop_loss"]

                    # Position on scale from stop to target
                    if target > stop:
                        position_pct = (entry - stop) / (target - stop)
                        st.progress(max(0, min(1, position_pct)))

                    # Justification expander
                    with st.expander("💡 View Trade Justification"):
                        st.write(
                            details.get("justification", "No justification available")
                        )

                    st.markdown("---")
        else:
            st.info("No open positions")
    else:
        st.warning("No portfolio data available")


def render_recent_trades(trade_history):
    """Render recent trades timeline"""
    st.subheader("📈 Recent Trades")

    if trade_history is not None and not trade_history.empty:
        # Get last 10 trades
        recent_trades = trade_history.sort_values("timestamp", ascending=False).head(10)

        for _, trade in recent_trades.iterrows():
            with st.container():
                col1, col2, col3, col4 = st.columns([1, 2, 2, 1])

                with col1:
                    # Convert to Jakarta time for display
                    jkt_time = to_jakarta_time(trade["timestamp"])
                    # Display with WIB indicator
                    time_str = jkt_time.strftime("%m/%d %H:%M")
                    st.caption(f"{time_str} WIB")
                    action_class = (
                        "badge-entry" if trade["action"] == "ENTRY" else "badge-exit"
                    )
                    st.markdown(
                        f'<span class="badge {action_class}">{trade["action"]}</span>',
                        unsafe_allow_html=True,
                    )

                with col2:
                    side_class = (
                        "badge-long" if trade["side"] == "long" else "badge-short"
                    )
                    st.markdown(
                        f"**{trade['coin']}** <span class='badge {side_class}'>{trade['side'].upper()}</span>",
                        unsafe_allow_html=True,
                    )
                    st.caption(f"Qty: {trade['quantity']}")

                with col3:
                    if trade["action"] == "ENTRY":
                        st.markdown(f"**Entry:** {format_currency(trade['price'])}")
                        st.caption(f"Leverage: {trade['leverage']}x")
                    else:
                        st.markdown(f"**Exit:** {format_currency(trade['price'])}")
                        pnl = trade.get("pnl", 0)
                        pnl_color = get_color_for_value(pnl)
                        st.markdown(
                            f'<span class="{pnl_color}">**PnL:** {format_currency(pnl)}</span>',
                            unsafe_allow_html=True,
                        )

                with col4:
                    confidence = trade.get("confidence", 0)
                    st.progress(confidence)
                    st.caption(f"{confidence:.0%}")

                # Justification expander
                with st.expander("📝 AI Reasoning"):
                    st.write(trade.get("reason", "No reason provided"))

                st.markdown("---")
    else:
        st.info("No trade history available")


def render_portfolio_summaries(portfolio_state):
    """Render portfolio summary messages in chat-like format"""
    st.subheader("🤖 AI Decision")

    if portfolio_state is not None and not portfolio_state.empty:
        # Get the last 10 summaries (newest to oldest - no reversal)
        recent_summaries = portfolio_state.sort_values(
            "timestamp", ascending=False
        ).head(10)

        # Display newest to oldest (top to bottom)
        for _, snapshot in recent_summaries.iterrows():
            short_summary = snapshot.get("short_summary", "")
            portfolio_summary = snapshot.get("portfolio_summary", "")
            timestamp = snapshot["timestamp"]

            # Skip if no summary available
            if pd.isna(short_summary) or short_summary == "":
                continue

            # Convert to Jakarta time and format
            jkt_time = to_jakarta_time(timestamp)
            timestamp_str = jkt_time.strftime("%m/%d %H:%M:%S")

            # Create chat-like message card
            st.markdown(
                f"""
            <div class="decision-card" style="background: #f8f9fa; border-left: 4px solid #3498DB;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                    <span style="font-weight: 600; color: #3498DB; font-size: 0.9rem;">DEEPSEEK CHAT V3.1</span>
                    <span style="font-size: 0.75rem; color: #7F8C8D;">{timestamp_str}</span>
                </div>
                <div style="font-size: 0.9rem; color: #2C3E50; line-height: 1.6;">
                    {short_summary}
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

            # Expandable section for full portfolio summary
            if not pd.isna(portfolio_summary) and portfolio_summary != "":
                with st.expander("💬 View Full Portfolio Summary"):
                    # Clean up the portfolio summary text
                    # Remove markdown formatting artifacts and display as clean text
                    import re

                    # Replace common markdown patterns with plain text
                    cleaned_summary = portfolio_summary

                    # Remove inline code/italics artifacts (text between asterisks or underscores)
                    cleaned_summary = re.sub(
                        r"\*\*([^*]+)\*\*", r"\1", cleaned_summary
                    )  # Bold
                    cleaned_summary = re.sub(
                        r"\*([^*]+)\*", r"\1", cleaned_summary
                    )  # Italic
                    cleaned_summary = re.sub(
                        r"_([^_]+)_", r"\1", cleaned_summary
                    )  # Italic underscore

                    # Display with proper line breaks and formatting
                    st.text_area(
                        "Portfolio Summary",
                        cleaned_summary,
                        height=300,
                        disabled=True,
                        label_visibility="collapsed",
                    )

            st.markdown(
                "<div style='margin-bottom: 8px;'></div>", unsafe_allow_html=True
            )
    else:
        st.info("No portfolio summaries available")


def render_portfolio_chart(portfolio_state):
    """Render portfolio performance chart"""
    st.subheader("📊 Portfolio Performance")

    if portfolio_state is not None and not portfolio_state.empty:
        fig = go.Figure()

        # Calculate dynamic y-axis range for better visibility
        min_equity = portfolio_state["total_equity"].min()
        max_equity = portfolio_state["total_equity"].max()
        equity_range = max_equity - min_equity

        # Add padding (10% on each side, or minimum $50 if range is very small)
        padding = max(equity_range * 0.1, 50)
        y_min = min_equity - padding
        y_max = max_equity + padding

        # Convert timestamps to Jakarta time for display
        portfolio_state_copy = portfolio_state.copy()
        portfolio_state_copy["timestamp_wib"] = portfolio_state_copy["timestamp"].apply(
            lambda x: to_jakarta_time(x)
        )

        # Account value line
        fig.add_trace(
            go.Scatter(
                x=portfolio_state_copy["timestamp_wib"],
                y=portfolio_state_copy["total_equity"],
                mode="lines+markers",
                name="Account Value",
                line=dict(color="#3498DB", width=2),
                marker=dict(size=6, color="#3498DB"),
                fill="tonexty",
                fillcolor="rgba(52, 152, 219, 0.1)",
            )
        )

        # Add markers for significant changes
        portfolio_state_copy["equity_change"] = portfolio_state_copy[
            "total_equity"
        ].pct_change()

        fig.update_layout(
            xaxis_title="Time (WIB)",
            yaxis_title="Account Value ($)",
            hovermode="x unified",
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(color="#2C3E50"),
            height=400,
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis=dict(showgrid=True, gridcolor="#E0E0E0", zeroline=False),
            yaxis=dict(
                showgrid=True,
                gridcolor="#E0E0E0",
                zeroline=False,
                range=[y_min, y_max],  # Dynamic range to show variations
            ),
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No portfolio performance data available")


def render_footer(ai_decisions, portfolio_state):
    """Render dashboard footer with stats"""
    st.markdown('<div class="dashboard-footer">', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Trading Model:** DeepSeek v3.1")

    with col2:
        if ai_decisions is not None and not ai_decisions.empty:
            total_invocations = len(ai_decisions)
            st.markdown(f"**Total Invocations:** {total_invocations}")
        else:
            st.markdown("**Total Invocations:** 0")

    with col3:
        if portfolio_state is not None and not portfolio_state.empty:
            start_time = portfolio_state["timestamp"].min()
            elapsed = datetime.now() - start_time.replace(tzinfo=None)
            hours = int(elapsed.total_seconds() // 3600)
            minutes = int((elapsed.total_seconds() % 3600) // 60)
            st.markdown(f"**Runtime:** {hours}h {minutes}m")
        else:
            st.markdown("**Runtime:** N/A")

    st.markdown("</div>", unsafe_allow_html=True)


# Main dashboard
def main():
    # Initialize session state for auto-refresh
    if "last_refresh" not in st.session_state:
        st.session_state.last_refresh = datetime.now(pytz.UTC)
    else:
        # Ensure last_refresh is timezone-aware (for backwards compatibility)
        if st.session_state.last_refresh.tzinfo is None:
            st.session_state.last_refresh = pytz.UTC.localize(
                st.session_state.last_refresh
            )

    if "refresh_interval" not in st.session_state:
        st.session_state.refresh_interval = 300  # 5 minutes in seconds

    # Manual refresh button
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        if st.button("🔄 Refresh Now"):
            st.session_state.last_refresh = datetime.now(pytz.UTC)
            st.rerun()

    with col2:
        # Countdown timer
        time_since_refresh = (
            datetime.now(pytz.UTC) - st.session_state.last_refresh
        ).seconds
        time_until_refresh = st.session_state.refresh_interval - time_since_refresh
        minutes = time_until_refresh // 60
        seconds = time_until_refresh % 60
        st.caption(f"Next refresh in: {minutes}:{seconds:02d}")

    # Load data
    ai_decisions, portfolio_state, trade_history, ai_messages = load_data()

    # Render header
    render_header(portfolio_state, st.session_state.last_refresh)

    # Main content area
    col_left, col_right = st.columns([3, 2])

    with col_left:
        # Current positions
        render_positions_table(portfolio_state)
        st.markdown("<br>", unsafe_allow_html=True)

        # Recent trades
        render_recent_trades(trade_history)

    with col_right:
        # Portfolio summaries (chat-like messages)
        render_portfolio_summaries(portfolio_state)
        st.markdown("<br>", unsafe_allow_html=True)

        # Portfolio chart
        render_portfolio_chart(portfolio_state)

    # Footer
    render_footer(ai_decisions, portfolio_state)

    # Auto-refresh logic
    time_since_refresh = (
        datetime.now(pytz.UTC) - st.session_state.last_refresh
    ).seconds
    if time_since_refresh >= st.session_state.refresh_interval:
        st.session_state.last_refresh = datetime.now(pytz.UTC)
        st.rerun()

    # Sleep briefly to allow UI to update countdown
    time.sleep(1)
    st.rerun()


if __name__ == "__main__":
    main()
