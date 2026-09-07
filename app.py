"""
Investment Portfolio Analyzer
=============================
A Streamlit web app that lets a user build a portfolio of stocks, pull live
market data, and see returns, risk metrics (volatility, Sharpe ratio, max
drawdown), asset allocation, and performance vs. a benchmark (e.g. S&P 500).

Styled after a trading-terminal aesthetic: dark background, monospaced
data, amber accent, green/red for positive/negative figures.

Run locally with:
    pip install -r requirements.txt
    streamlit run app.py

For the full look, keep the accompanying .streamlit/config.toml file in a
.streamlit folder next to this script — it sets the native dark theme that
Streamlit's own widgets (buttons, tables, inputs) read from.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import yfinance as yf

# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Investment Portfolio Analyzer",
    page_icon=None,
    layout="wide",
)

TRADING_DAYS_PER_YEAR = 252
MIN_RELIABLE_TRADING_DAYS = 60  # roughly 3 months — below this, annualizing
                                 # a short stretch of performance into a
                                 # full-year figure gets statistically shaky

# --------------------------------------------------------------------------
# Design system — dark terminal palette
# --------------------------------------------------------------------------

BG = "#0A0E14"            # near-black background
SURFACE = "#10151C"       # panel / card background
INK = "#D7DCE1"            # primary text, light grey
MUTED = "#6E7A87"          # secondary text
ACCENT = "#F5A623"         # amber — headers, active elements
POSITIVE = "#00C805"       # terminal green
NEGATIVE = "#FF3B30"       # terminal red
BENCHMARK_LINE = "#5B6B7C"
HAIRLINE = "#232B36"

CUSTOM_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&display=swap');

html, body, [class^="css"], [class*=" css"] {{
    font-family: 'IBM Plex Mono', 'Courier New', monospace !important;
}}

.stApp {{
    background-color: {BG};
}}

h1, h2, h3 {{
    font-family: 'IBM Plex Mono', monospace !important;
    color: {ACCENT} !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-size: 1.05rem !important;
}}

h1 {{
    font-size: 1.4rem !important;
}}

/* Remove Streamlit's auto anchor-link icon on headers (set via anchor=False
   in code too — this is a fallback in case any header omits it) */
[data-testid="stHeaderActionElements"] {{
    display: none !important;
}}

p, li, span, label {{
    color: {INK};
}}

[data-testid="stCaptionContainer"] {{
    color: {MUTED} !important;
    font-size: 0.78rem !important;
}}

/* Metric cards */
[data-testid="stMetric"] {{
    background-color: {SURFACE};
    border: 1px solid {HAIRLINE};
    border-radius: 0px;
    padding: 0.85rem 1rem 0.75rem 1rem;
}}
[data-testid="stMetricValue"] {{
    font-family: 'IBM Plex Mono', monospace !important;
    color: {INK} !important;
    font-weight: 600 !important;
}}
[data-testid="stMetricLabel"] {{
    color: {MUTED} !important;
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}}
[data-testid="stMetricDelta"] {{
    font-family: 'IBM Plex Mono', monospace !important;
}}

/* Buttons */
.stButton > button {{
    background-color: {ACCENT};
    color: #0A0E14;
    border: none;
    border-radius: 0px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-size: 0.8rem;
}}
.stButton > button:hover {{
    background-color: #D18E1B;
    color: #0A0E14;
}}
.stButton > button:focus {{
    box-shadow: 0 0 0 2px {ACCENT}55;
}}

/* Dataframes */
[data-testid="stDataFrame"] {{
    font-family: 'IBM Plex Mono', monospace;
    border: 1px solid {HAIRLINE};
}}

hr {{
    border-color: {HAIRLINE};
}}

section[data-testid="stSidebar"] {{
    border-right: 1px solid {HAIRLINE};
}}

/* Divider rule under the title */
.header-rule {{
    border: none;
    border-top: 1px solid {HAIRLINE};
    margin-top: 0.5rem;
    margin-bottom: 1.4rem;
}}

/* Sidebar labels uppercase, muted, small — terminal control-panel feel */
section[data-testid="stSidebar"] label {{
    color: {MUTED} !important;
    font-size: 0.72rem !important;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

PLOTLY_TEMPLATE = "plotly_dark"
CHART_FONT = dict(family="IBM Plex Mono, monospace", color=INK, size=12)


def style_fig(fig: go.Figure, height: int = 360) -> go.Figure:
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        font=CHART_FONT,
        height=height,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor=SURFACE,
        paper_bgcolor=SURFACE,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    fig.update_xaxes(showgrid=False, linecolor=HAIRLINE)
    fig.update_yaxes(showgrid=True, gridcolor=HAIRLINE, zerolinecolor=HAIRLINE)
    return fig


# --------------------------------------------------------------------------
# Data helpers (cached so we don't re-hit the API on every rerun)
# --------------------------------------------------------------------------


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_price_history(tickers: tuple[str, ...], period: str) -> pd.DataFrame:
    """Download adjusted close prices for a tuple of tickers."""
    if not tickers:
        return pd.DataFrame()
    data = yf.download(
        list(tickers), period=period, auto_adjust=True, progress=False
    )
    if data.empty:
        return pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"]
    else:
        close = data[["Close"]]
        close.columns = tickers
    return close.dropna(how="all")


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_current_prices(tickers: tuple[str, ...]) -> dict:
    prices = {}
    for t in tickers:
        try:
            hist = yf.Ticker(t).history(period="1d")
            if not hist.empty:
                prices[t] = float(hist["Close"].iloc[-1])
        except Exception:
            pass
    return prices


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_company_names(tickers: tuple[str, ...]) -> dict:
    names = {}
    for t in tickers:
        try:
            info = yf.Ticker(t).info
            names[t] = info.get("shortName", t)
        except Exception:
            names[t] = t
    return names


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_ticker_currencies(tickers: tuple[str, ...]) -> dict:
    """Each ticker's native trading currency, e.g. 'USD', 'TRY', 'INR'.
    Defaults to USD if it can't be determined, since that's correct for
    the overwhelming majority of US-listed tickers."""
    currencies = {}
    for t in tickers:
        currency = None
        try:
            fast = yf.Ticker(t).fast_info
            currency = fast.get("currency") if hasattr(fast, "get") else getattr(fast, "currency", None)
        except Exception:
            pass
        if not currency:
            try:
                currency = yf.Ticker(t).info.get("currency")
            except Exception:
                pass
        currencies[t] = (currency or "USD").upper()
    return currencies


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fx_rate_series(currency: str, period: str) -> pd.Series:
    """Units of `currency` per 1 USD, as a daily series — e.g. for TRY,
    how many Turkish Lira equal one US dollar on each date. Used to
    convert a foreign-currency price into USD by dividing by this rate."""
    if currency == "USD":
        return pd.Series(dtype=float)
    pair = f"USD{currency}=X"
    try:
        data = yf.download(pair, period=period, auto_adjust=True, progress=False)
    except Exception:
        return pd.Series(dtype=float)
    if data.empty:
        return pd.Series(dtype=float)
    if isinstance(data.columns, pd.MultiIndex):
        return data["Close"][pair].dropna()
    return data["Close"].dropna()


# --------------------------------------------------------------------------
# Metric calculations
# --------------------------------------------------------------------------


def compute_daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna(how="all")


def annualized_return(cumulative_return: float, num_days: int) -> float:
    if num_days <= 0:
        return 0.0
    years = num_days / TRADING_DAYS_PER_YEAR
    if years <= 0:
        return 0.0
    return (1 + cumulative_return) ** (1 / years) - 1


def annualized_volatility(daily_returns: pd.Series) -> float:
    return daily_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)


def sharpe_ratio(daily_returns: pd.Series, risk_free_rate: float) -> float:
    excess_daily = daily_returns - (risk_free_rate / TRADING_DAYS_PER_YEAR)
    std = excess_daily.std()
    if std == 0 or np.isnan(std):
        return 0.0
    return (excess_daily.mean() / std) * np.sqrt(TRADING_DAYS_PER_YEAR)


def max_drawdown(cumulative_series: pd.Series) -> float:
    running_max = cumulative_series.cummax()
    drawdown = (cumulative_series - running_max) / running_max
    return drawdown.min()


def fmt_pct(x: float) -> str:
    return f"{x * 100:,.2f}%"


def fmt_signed_pct(x: float) -> str:
    sign = "+" if x >= 0 else ""
    return f"{sign}{x * 100:,.2f}%"


def colorize_by_sign(row: pd.Series, skip_cols=("Metric", "Ticker"), skip_rows=("Volatility, annualized",)) -> list:
    """Row-wise style function for a pandas Styler: colors numeric cells
    green if >= 0, red if < 0. Skips label columns and rows where sign
    doesn't carry a good/bad meaning (e.g. volatility)."""
    styles = [""] * len(row)
    if row.name in skip_rows:
        return styles
    for i, col in enumerate(row.index):
        if col in skip_cols:
            continue
        val = row[col]
        if isinstance(val, (int, float, np.floating)) and not pd.isna(val):
            color = POSITIVE if val >= 0 else NEGATIVE
            styles[i] = f"color: {color}; font-weight: 600"
    return styles


# --------------------------------------------------------------------------
# Sidebar — portfolio input
# --------------------------------------------------------------------------

st.sidebar.subheader("Portfolio setup", anchor=False)
st.sidebar.caption(
    "Enter the tickers and share counts you actually hold. Prices are "
    "pulled live from Yahoo Finance."
)

if "holdings" not in st.session_state:
    st.session_state.holdings = pd.DataFrame(
        {"Ticker": pd.Series(dtype="str"), "Shares": pd.Series(dtype="float")}
    )

holdings_editor_output = st.sidebar.data_editor(
    st.session_state.holdings,
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "Ticker": st.column_config.TextColumn("Ticker", help="e.g. AAPL"),
        "Shares": st.column_config.NumberColumn(
            "Shares", min_value=0.0, step=1.0, format="%.2f"
        ),
    },
    key="holdings_editor",
)

# Deliberately NOT writing holdings_editor_output back into
# st.session_state.holdings here. This widget has a `key`, so Streamlit
# already tracks its state internally and automatically across reruns.
# Manually mirroring that state into a second variable and feeding it
# back in as this same widget's starting value creates two competing
# sources of truth — which is what was causing rows to vanish or the
# whole table to reset. st.session_state.holdings is only ever used to
# seed the very first render, before the widget has any state of its own.

period_label_to_code = {
    "1 month": "1mo",
    "3 months": "3mo",
    "6 months": "6mo",
    "1 year": "1y",
    "2 years": "2y",
    "5 years": "5y",
    "10 years": "10y",
}
period_label = st.sidebar.selectbox(
    "History window", list(period_label_to_code.keys()), index=3
)
period_code = period_label_to_code[period_label]

benchmark_ticker = st.sidebar.text_input("Benchmark ticker", value="SPY")

risk_free_rate_pct = st.sidebar.number_input(
    "Risk-free rate, annual (%)",
    min_value=0.0,
    max_value=15.0,
    value=4.5,
    step=0.1,
    help="Used in the Sharpe ratio. Roughly the current 3-month T-bill yield.",
)
risk_free_rate = risk_free_rate_pct / 100

run = st.sidebar.button("Run analysis", type="primary", use_container_width=True)

st.sidebar.divider()
st.sidebar.caption(
    "Data provided by Yahoo Finance and may be delayed. For informational "
    "purposes only; not investment advice."
)

# --------------------------------------------------------------------------
# Main body — header
# --------------------------------------------------------------------------

st.title("Investment Portfolio Analyzer", anchor=False)
st.caption(
    "Enter holdings in the sidebar, then run the analysis to see value, "
    "returns, risk, allocation, and performance against a benchmark."
)
st.markdown("<hr class='header-rule'/>", unsafe_allow_html=True)

# Clean holdings input into a separate copy — never write this back into
# st.session_state.holdings, since that would reshape the editor's own
# backing data and cause the "first entry disappears" glitch above.
holdings_df = holdings_editor_output.dropna(subset=["Ticker"]).copy()
holdings_df["Ticker"] = holdings_df["Ticker"].str.strip().str.upper()
holdings_df = holdings_df[holdings_df["Ticker"] != ""]
holdings_df = holdings_df[holdings_df["Shares"] > 0]

# Combine duplicate tickers by summing shares — entering TSLA twice (3 + 3)
# should behave like one TSLA position of 6 shares, not two separate 3-share
# rows that silently overwrite or double-count each other downstream.
duplicate_tickers = holdings_df["Ticker"][holdings_df["Ticker"].duplicated()].unique().tolist()
if duplicate_tickers:
    st.info(
        f"Combined duplicate entries for: {', '.join(duplicate_tickers)}. "
        f"Share counts for the same ticker are summed automatically."
    )
holdings_df = holdings_df.groupby("Ticker", as_index=False)["Shares"].sum()

if not run and "last_results" not in st.session_state:
    st.info("Set up holdings in the sidebar, then select Run analysis.")
    st.stop()

if holdings_df.empty:
    st.warning("Add at least one ticker with a positive share count.")
    st.stop()

tickers = tuple(holdings_df["Ticker"].tolist())
benchmark_ticker = benchmark_ticker.strip().upper() or "SPY"
all_tickers = tuple(sorted(set(tickers) | {benchmark_ticker}))

with st.spinner("Fetching market data..."):
    prices = fetch_price_history(all_tickers, period_code)
    current_prices = fetch_current_prices(tickers)
    company_names = fetch_company_names(tickers)

def has_usable_data(df: pd.DataFrame, ticker: str, min_points: int = 2) -> bool:
    """A ticker only counts as valid if it actually returned real price
    points — not just a column that exists but is empty or all-NaN
    (which happens for fake tickers, delisted symbols, or ones with no
    trading activity in the selected window)."""
    return ticker in df.columns and df[ticker].notna().sum() >= min_points


invalid_tickers = [t for t in tickers if not has_usable_data(prices, t)]
if invalid_tickers:
    st.warning(
        f"Please enter a valid ticker — no usable price data found for: "
        f"{', '.join(invalid_tickers)}. These holdings were excluded from "
        f"the analysis below."
    )

held_tickers = [t for t in tickers if has_usable_data(prices, t)]
if not held_tickers:
    st.error("Please enter a valid ticker. None of the tickers entered returned usable price data.")
    st.stop()

benchmark_available = has_usable_data(prices, benchmark_ticker)
if not benchmark_available:
    st.warning(
        f"Please enter a valid ticker — no usable price data found for "
        f"benchmark '{benchmark_ticker}'. Benchmark comparison will be skipped."
    )

# --------------------------------------------------------------------------
# Portfolio value over time
# --------------------------------------------------------------------------

shares_map = dict(zip(holdings_df["Ticker"], holdings_df["Shares"]))

portfolio_prices = prices[held_tickers].dropna(how="all")
portfolio_prices = portfolio_prices.ffill().dropna(how="any")

if portfolio_prices.empty:
    st.error(
        "Please enter a valid ticker. The selected holdings don't share "
        "any overlapping trading days in this history window — try a "
        "longer window or different tickers."
    )
    st.stop()

# --------------------------------------------------------------------------
# Currency conversion — every holding is converted to USD so the portfolio
# total, returns, and allocation are all measured in a single, consistent
# currency. Without this, a ticker priced in e.g. Turkish Lira would be
# added to a USD total as if 1 TRY = 1 USD, producing a meaningless number.
# --------------------------------------------------------------------------

currencies = fetch_ticker_currencies(tuple(held_tickers))
foreign_currencies = sorted({c for c in currencies.values() if c != "USD"})

converted_tickers = []
fx_failed_tickers = []

for currency in foreign_currencies:
    fx_series = fetch_fx_rate_series(currency, period_code)
    tickers_in_currency = [t for t in held_tickers if currencies.get(t) == currency]
    if fx_series.empty:
        fx_failed_tickers.extend(tickers_in_currency)
        continue
    fx_aligned = fx_series.reindex(portfolio_prices.index).ffill().bfill()
    if fx_aligned.isna().all():
        fx_failed_tickers.extend(tickers_in_currency)
        continue
    for t in tickers_in_currency:
        portfolio_prices[t] = portfolio_prices[t] / fx_aligned
        latest_rate = fx_aligned.iloc[-1]
        if t in current_prices and latest_rate:
            current_prices[t] = current_prices[t] / latest_rate
        converted_tickers.append(t)

if fx_failed_tickers:
    st.warning(
        f"Couldn't fetch a live exchange rate for: {', '.join(fx_failed_tickers)}. "
        f"These holdings were excluded from the analysis."
    )
    held_tickers = [t for t in held_tickers if t not in fx_failed_tickers]
    if not held_tickers:
        st.error("Please enter a valid ticker. No holdings remained after currency conversion.")
        st.stop()
    portfolio_prices = portfolio_prices[held_tickers]

if converted_tickers:
    converted_labels = [f"{t} ({currencies[t]})" for t in converted_tickers]
    st.info(
        f"Converted to USD using live exchange rates: {', '.join(converted_labels)}."
    )

shares_vector = np.array([shares_map[t] for t in held_tickers])
portfolio_value_series = portfolio_prices.mul(shares_vector, axis=1).sum(axis=1)
portfolio_value_series.name = "Portfolio"

current_value = sum(
    current_prices.get(t, portfolio_prices[t].iloc[-1]) * shares_map[t]
    for t in held_tickers
)
start_value = portfolio_value_series.iloc[0]
total_return = (current_value - start_value) / start_value if start_value else 0.0

daily_returns_assets = compute_daily_returns(portfolio_prices)
portfolio_daily_returns = compute_daily_returns(portfolio_value_series.to_frame())["Portfolio"]

vol = annualized_volatility(portfolio_daily_returns)
sharpe = sharpe_ratio(portfolio_daily_returns, risk_free_rate)
ann_return = annualized_return(total_return, len(portfolio_value_series))
mdd = max_drawdown(portfolio_value_series)

# --------------------------------------------------------------------------
# Benchmark comparison
# --------------------------------------------------------------------------
# (benchmark_available was already determined above, right after fetching
# prices, so a bad benchmark ticker can't crash this section either)

if benchmark_available:
    benchmark_prices = prices[benchmark_ticker].dropna()
    benchmark_cum = benchmark_prices / benchmark_prices.iloc[0] - 1
    benchmark_daily_returns = compute_daily_returns(benchmark_prices.to_frame())[benchmark_ticker]
    benchmark_total_return = benchmark_prices.iloc[-1] / benchmark_prices.iloc[0] - 1
    benchmark_ann_return = annualized_return(benchmark_total_return, len(benchmark_prices))
    benchmark_vol = annualized_volatility(benchmark_daily_returns)
    benchmark_sharpe = sharpe_ratio(benchmark_daily_returns, risk_free_rate)
    benchmark_mdd = max_drawdown(benchmark_prices)
else:
    benchmark_cum = None

portfolio_cum = portfolio_value_series / portfolio_value_series.iloc[0] - 1

# --------------------------------------------------------------------------
# Short-window caution
# --------------------------------------------------------------------------
# Annualizing is just math — (1 + return) scaled up to a full year — but
# over a short window it can turn an ordinary week or month into a wildly
# overstated (or understated) "yearly" figure. Flag it before showing the
# numbers, not after, so it's read as a caveat rather than an excuse.

num_trading_days = len(portfolio_value_series)
if num_trading_days < MIN_RELIABLE_TRADING_DAYS:
    st.warning(
        f"This history window only covers {num_trading_days} trading days. "
        f"Annualized return, volatility, and Sharpe ratio extrapolate that "
        f"short stretch out to a full year and can be misleading — a good "
        f"or bad few weeks doesn't necessarily predict a full year. Consider "
        f"a longer history window (6 months or more) for more reliable "
        f"annualized figures."
    )

# --------------------------------------------------------------------------
# Top metrics row
# --------------------------------------------------------------------------

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Portfolio value", f"${current_value:,.2f}", fmt_signed_pct(total_return))
m2.metric(
    "Total return",
    fmt_pct(total_return),
    fmt_signed_pct(total_return - benchmark_total_return) + " vs. benchmark"
    if benchmark_available
    else None,
)
m3.metric("Annualized return", fmt_pct(ann_return))
m4.metric("Volatility, annualized", fmt_pct(vol))
m5.metric("Sharpe ratio", f"{sharpe:,.2f}")

st.caption(f"Maximum drawdown over the period: {fmt_pct(mdd)}")

st.divider()

# --------------------------------------------------------------------------
# Portfolio vs. benchmark table
# --------------------------------------------------------------------------

if benchmark_available:
    st.subheader("Portfolio vs. benchmark", anchor=False)
    comparison_df = pd.DataFrame(
        {
            "Metric": [
                "Total return",
                "Annualized return",
                "Volatility, annualized",
                "Sharpe ratio",
                "Maximum drawdown",
            ],
            "Portfolio": [total_return, ann_return, vol, sharpe, mdd],
            benchmark_ticker: [
                benchmark_total_return,
                benchmark_ann_return,
                benchmark_vol,
                benchmark_sharpe,
                benchmark_mdd,
            ],
        }
    ).set_index("Metric")

    pct_rows = ["Total return", "Annualized return", "Volatility, annualized", "Maximum drawdown"]
    ratio_rows = ["Sharpe ratio"]
    comparison_styler = (
        comparison_df.style.format("{:.2%}", subset=pd.IndexSlice[pct_rows, :])
        .format("{:.2f}", subset=pd.IndexSlice[ratio_rows, :])
        .apply(colorize_by_sign, axis=1)
    )
    st.dataframe(comparison_styler, use_container_width=True)
    st.divider()

# --------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------

col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("Portfolio value over time", anchor=False)
    fig_value = go.Figure()
    fig_value.add_trace(
        go.Scatter(
            x=portfolio_value_series.index,
            y=portfolio_value_series.values,
            mode="lines",
            line=dict(color=ACCENT, width=1.6),
            name="Portfolio",
        )
    )
    fig_value.update_yaxes(title_text="Value ($)")
    st.plotly_chart(style_fig(fig_value), use_container_width=True)

    st.subheader(f"Cumulative return vs. {benchmark_ticker}", anchor=False)
    compare_fig = go.Figure()
    compare_fig.add_trace(
        go.Scatter(
            x=portfolio_cum.index,
            y=portfolio_cum.values * 100,
            name="Portfolio",
            mode="lines",
            line=dict(color=ACCENT, width=1.6),
        )
    )
    if benchmark_available:
        compare_fig.add_trace(
            go.Scatter(
                x=benchmark_cum.index,
                y=benchmark_cum.values * 100,
                name=benchmark_ticker,
                mode="lines",
                line=dict(color=BENCHMARK_LINE, width=1.4, dash="dash"),
            )
        )
    else:
        st.caption(f"Benchmark ticker '{benchmark_ticker}' unavailable.")
    compare_fig.update_yaxes(title_text="Cumulative return (%)")
    st.plotly_chart(style_fig(compare_fig), use_container_width=True)

with col_right:
    st.subheader("Asset allocation", anchor=False)
    alloc_values = [current_prices.get(t, portfolio_prices[t].iloc[-1]) * shares_map[t] for t in held_tickers]
    alloc_df = pd.DataFrame({"Ticker": held_tickers, "Value": alloc_values})
    palette = [ACCENT, "#00C805", "#5B6B7C", "#C97A1A", "#3E8E41", "#8A99A8"]
    fig_pie = px.pie(
        alloc_df,
        names="Ticker",
        values="Value",
        hole=0.55,
        color_discrete_sequence=palette,
    )
    fig_pie.update_traces(textfont=CHART_FONT, marker=dict(line=dict(color=SURFACE, width=2)))
    st.plotly_chart(style_fig(fig_pie, height=320), use_container_width=True)

    st.subheader("Holdings", anchor=False)
    holdings_display = alloc_df.copy()
    holdings_display["Company"] = holdings_display["Ticker"].map(company_names)
    holdings_display["Shares"] = holdings_display["Ticker"].map(shares_map)
    holdings_display["Price"] = holdings_display["Ticker"].apply(
        lambda t: current_prices.get(t, portfolio_prices[t].iloc[-1])
    )
    holdings_display["Weight"] = holdings_display["Value"] / holdings_display["Value"].sum()
    holdings_display = holdings_display[["Ticker", "Company", "Shares", "Price", "Value", "Weight"]]
    st.dataframe(
        holdings_display.style.format(
            {"Price": "${:,.2f}", "Value": "${:,.2f}", "Weight": "{:.1%}", "Shares": "{:,.2f}"}
        ),
        use_container_width=True,
        hide_index=True,
    )

st.divider()

# --------------------------------------------------------------------------
# Per-asset risk / return table
# --------------------------------------------------------------------------

st.subheader("Per-asset risk and return", anchor=False)

rows = []
for t in held_tickers:
    r = daily_returns_assets[t].dropna()
    if r.empty:
        continue
    asset_cum_return = portfolio_prices[t].iloc[-1] / portfolio_prices[t].iloc[0] - 1
    rows.append(
        {
            "Ticker": t,
            "Total return": asset_cum_return,
            "Annualized return": annualized_return(asset_cum_return, len(r)),
            "Volatility, annualized": annualized_volatility(r),
            "Sharpe": sharpe_ratio(r, risk_free_rate),
        }
    )
per_asset_df = pd.DataFrame(rows)
per_asset_styler = per_asset_df.style.format(
    {
        "Total return": "{:.2%}",
        "Annualized return": "{:.2%}",
        "Volatility, annualized": "{:.2%}",
        "Sharpe": "{:.2f}",
    }
).apply(colorize_by_sign, axis=1)
st.dataframe(per_asset_styler, use_container_width=True, hide_index=True)

# --------------------------------------------------------------------------
# Correlation heatmap
# --------------------------------------------------------------------------

if len(held_tickers) > 1:
    st.subheader("Correlation matrix", anchor=False)
    corr = daily_returns_assets[held_tickers].corr()
    fig_corr = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale=[NEGATIVE, SURFACE, POSITIVE],
        zmin=-1,
        zmax=1,
        aspect="auto",
    )
    fig_corr.update_traces(textfont=CHART_FONT)
    st.plotly_chart(style_fig(fig_corr, height=380), use_container_width=True)

st.session_state.last_results = True

st.divider()
st.caption(
    "Sharpe ratio, volatility, and returns are computed from historical daily "
    "price data and annualized assuming 252 trading days per year. This tool "
    "is for informational and educational purposes only and does not "
    "constitute investment advice."
)