# Dashboard Layout Preview

## Visual Structure

```
┌─────────────────────────────────────────────────────────────────────┐
│                   AI TRADING BOT DASHBOARD                          │
├─────────────────────────────────────────────────────────────────────┤
│  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐       │
│  │  $10,022  │  │  $7,002   │  │   +$22    │  │     3     │       │
│  │ Total Val │  │   Cash    │  │ Unrlzd PnL│  │ Positions │       │
│  │  +0.23%   │  │           │  │           │  │           │       │
│  └───────────┘  └───────────┘  └───────────┘  └───────────┘       │
│                                                                      │
│          Last Updated: 2025-11-02 15:37:28                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────────────────────┬────────────────────────────┐  │
│  │ LEFT COLUMN (60%)               │ RIGHT COLUMN (40%)         │  │
│  ├─────────────────────────────────┼────────────────────────────┤  │
│  │ 📊 CURRENT POSITIONS            │ 🤖 AI DECISION SUMMARY     │  │
│  │                                 │                            │  │
│  │ ┌───────────────────────────┐   │ ┌────────────────────────┐ │  │
│  │ │ ETH              LONG     │   │ │ ETH        HOLD        │ │  │
│  │ │ Entry: $3,893.13         │   │ │ Confidence: ████░░ 40% │ │  │
│  │ │ Qty: 1.282 @ 5x          │   │ │ 2 mins ago             │ │  │
│  │ │ 🎯 Target: $3,940        │   │ └────────────────────────┘ │  │
│  │ │ 🛑 Stop: $3,870          │   │                            │  │
│  │ │ Progress: ████████░░░     │   │ ┌────────────────────────┐ │  │
│  │ │ [View Justification ▼]   │   │ │ SOL        HOLD        │ │  │
│  │ └───────────────────────────┘   │ │ Confidence: █████░ 50% │ │  │
│  │                                 │ │ 2 mins ago             │ │  │
│  │ ┌───────────────────────────┐   │ └────────────────────────┘ │  │
│  │ │ SOL              LONG     │   │                            │  │
│  │ │ Entry: $186.70           │   │ ┌────────────────────────┐ │  │
│  │ │ Qty: 26.8 @ 5x           │   │ │ BNB        HOLD        │ │  │
│  │ │ 🎯 Target: $190.00       │   │ │ Confidence: █████░ 55% │ │  │
│  │ │ 🛑 Stop: $184.00         │   │ │ 2 mins ago             │ │  │
│  │ │ Progress: ████████░░░     │   │ └────────────────────────┘ │  │
│  │ │ [View Justification ▼]   │   │                            │  │
│  │ └───────────────────────────┘   │                            │  │
│  │                                 │                            │  │
│  │ ┌───────────────────────────┐   │ ──────────────────────────  │  │
│  │ │ BNB              LONG     │   │                            │  │
│  │ │ Entry: $1,091.37         │   │ 📊 PORTFOLIO PERFORMANCE   │  │
│  │ │ Qty: 7.32 @ 8x           │   │                            │  │
│  │ │ 🎯 Target: $1,110.00     │   │  $10,040 ┐                 │  │
│  │ │ 🛑 Stop: $1,080.00       │   │  $10,020 │  ●─────●        │  │
│  │ │ Progress: ████████░░░     │   │  $10,000 │ ╱       ╲       │  │
│  │ │ [View Justification ▼]   │   │   $9,980 │●         ●──●  │  │
│  │ └───────────────────────────┘   │   $9,960 └─────────────────│  │
│  │                                 │          7:22  7:30  7:37  │  │
│  ├─────────────────────────────────┤                            │  │
│  │ 📈 RECENT TRADES                │  (Optimized Y-scale!)      │  │
│  │                                 │                            │  │
│  │ ┌───────────────────────────┐   │                            │  │
│  │ │ 11m ago      ENTRY        │   │                            │  │
│  │ │ BNB LONG                  │   │                            │  │
│  │ │ Entry: $1,091.37 @ 8x     │   │                            │  │
│  │ │ Confidence: 70%           │   │                            │  │
│  │ │ [AI Reasoning ▼]          │   │                            │  │
│  │ └───────────────────────────┘   │                            │  │
│  └─────────────────────────────────┴────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────────┤
│                            FOOTER                                    │
│  Trading Model: DeepSeek v3.1  |  Total Invocations: 32  |  0h 15m  │
└─────────────────────────────────────────────────────────────────────┘
```

## Color Coding

### Status Badges
- **LONG**: Green background (#D5F5E3), Green text (#27AE60)
- **SHORT**: Red background (#FADBD8), Red text (#E74C3C)
- **ENTRY**: Blue background (#D6EAF8), Blue text (#3498DB)
- **EXIT**: Orange background (#F5CBA7), Orange text (#E67E22)
- **HOLD**: Gray background (#E8E8E8), Gray text (#7F8C8D)

### Values
- **Positive PnL**: Green (#27AE60)
- **Negative PnL**: Red (#E74C3C)
- **Neutral**: Blue (#3498DB)

### Confidence Bars
- **High (≥70%)**: Green (#27AE60)
- **Medium (40-69%)**: Orange (#F39C12)
- **Low (<40%)**: Red (#E74C3C)

## Interactive Elements

1. **🔄 Refresh Now** button (top-left)
2. **Countdown Timer** showing time until next auto-refresh
3. **Expandable Trade Justifications** - Click to see full AI reasoning
4. **Expandable AI Reasoning** - Click to see detailed trade explanations
5. **Hoverable Chart** - Hover over portfolio chart for exact values
6. **Progress Bars** - Visual representation of confidence scores

## Chart Optimization

### Before:
- Y-axis: $0 to $10,000
- Small changes invisible (0.2% of height)
- Looked like a flat line

### After (Optimized):
- Y-axis: Dynamic range (e.g., $9,960 to $10,040)
- Same changes now 50% of height
- Clear visualization of ups and downs
- Automatic padding (10% or minimum $50)
- Markers on data points for clarity

## Responsive Behavior

### Desktop (>1200px)
- Two-column layout (60/40 split)
- All metrics in single row
- Full width charts

### Tablet (768-1200px)
- Two-column layout with adjusted spacing
- Metrics may wrap to multiple rows

### Mobile (<768px)
- Single column layout
- Stacked sections
- Scrollable tables
- Condensed metrics

## Auto-Refresh

- **Interval**: 5 minutes (300 seconds)
- **Indicator**: Countdown timer in header
- **Manual Override**: Refresh button always available
- **Smooth Updates**: No page flash, data updates seamlessly

## Design Highlights

- Clean white background (#FAFAFA)
- Subtle borders and shadows
- Generous whitespace (16px minimum)
- Monaco monospace font for numbers
- Sans-serif for text
- Color-coded badges and values
- Progressive disclosure (expandable sections)

