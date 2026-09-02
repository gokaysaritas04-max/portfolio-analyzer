# 📈 Investment Portfolio Analyzer

A Streamlit web app for tracking and analyzing a stock portfolio using live
market data. Enter your holdings, and instantly see portfolio value,
returns, risk metrics, asset allocation, and performance vs. a benchmark
(e.g. the S&P 500).

## Features

- **Live prices** — pulls current and historical data from Yahoo Finance via `yfinance`
- **Editable holdings table** — add/remove tickers and share counts on the fly
- **Portfolio value over time** — line chart of total portfolio value
- **Return metrics** — total return, annualized return
- **Risk metrics** — annualized volatility, Sharpe ratio, max drawdown
- **Benchmark comparison** — cumulative return of your portfolio vs. any benchmark ticker (default: SPY)
- **Asset allocation** — pie chart + table of current weights
- **Per-asset breakdown** — return/risk stats for each individual holding
- **Correlation matrix** — heatmap of how your holdings move relative to each other

## Tech Stack

- [Streamlit](https://streamlit.io/) — web app framework
- [yfinance](https://pypi.org/project/yfinance/) — market data API
- [Plotly](https://plotly.com/python/) — interactive charts
- [pandas](https://pandas.pydata.org/) / [NumPy](https://numpy.org/) — data manipulation & calculations

## Setup

```bash
# 1. Clone/download this folder, then from inside it:
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501`.

## How the metrics are calculated

- **Annualized Return**: `(1 + total_return)^(252/num_trading_days) - 1`
- **Annualized Volatility**: standard deviation of daily returns × √252
- **Sharpe Ratio**: `(mean daily excess return / std of daily excess return) × √252`,
  where excess return subtracts the daily risk-free rate (user-adjustable in the sidebar)
- **Max Drawdown**: largest peak-to-trough decline in portfolio value over the selected period

## Possible Extensions (good for a "v2" resume bullet)

- Monte Carlo simulation for future portfolio value projections
- Efficient frontier / mean-variance optimization to suggest optimal weights
- Support for CSV import of trade history to compute real cost-basis and realized/unrealized gains
- Sector/industry breakdown using ticker fundamentals
- Deploy to Streamlit Community Cloud for a live demo link on your resume

## Disclaimer

This tool is for educational purposes only and does not constitute
financial advice.
