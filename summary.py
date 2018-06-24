#!/usr/bin/env python
"""
This script generates a summary of the Stock-Trak performance.

The standard deviation is computed with daily returns and the Sharpe Ratio uses the geometric average daily returns.
The date range used in the above calculations is May 28, 2018 until the current data.

Input files required in current working directory:
    FLTR.csv - Yahoo Finance formatted historical performance of VanEck Vectors Investment Grade Floating Rate ETF.
    URTH.csv - Yahoo Finance formatted historical performance of iShares MSCI World ETF.
    OpenPosition.csv - Contains the current Stock-Trak portfolio.
    portfolio_history.csv - Contains the historical performance of the Stock-Trak account.
"""

from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from scipy.stats.mstats import gmean

# ***************************************************************************************
# Set version number
# ***************************************************************************************
__version__ = '1.0'


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Main body of Code
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

start_date = datetime(2018, 5, 28)

# Read Risk-Free performance
# -----------------------------------------------------------------------------------
risk_free = pd.read_csv("FLTR.csv", parse_dates=["Date"], usecols=["Date", "Adj Close"])
risk_free.set_index("Date", inplace=True)
risk_free = risk_free.resample('D').interpolate()
start_price = risk_free.loc[start_date, "Adj Close"]
risk_free['Return'] = 100.0 * (risk_free['Adj Close'] - start_price) / start_price
risk_free['Daily_Return'] = risk_free['Adj Close'].pct_change()
r_f = 100.0 * (gmean(risk_free.loc[risk_free.index > start_date, 'Daily_Return'] + 1.0) - 1.0)
risk_free['Daily_Return'] = risk_free['Daily_Return'] * 100.0
std_f = risk_free['Daily_Return'].std()

# Read benchmark performance
# -----------------------------------------------------------------------------------
benchmark = pd.read_csv("URTH.csv", parse_dates=["Date"], usecols=["Date", "Adj Close"])
benchmark.set_index("Date", inplace=True)
benchmark = benchmark.resample('D').interpolate()
benchmark = benchmark.loc[benchmark.index >= start_date]
start_price = benchmark.loc[start_date, "Adj Close"]
benchmark['Return'] = 100.0 * (benchmark['Adj Close'] - start_price) / start_price - risk_free['Return']
benchmark['Daily_Return'] = benchmark['Adj Close'].pct_change()
r_b = 100.0 * (gmean(benchmark.loc[benchmark.index > start_date, 'Daily_Return'] + 1.0) - 1.0)
benchmark['Daily_Return'] = 100.0 * benchmark['Daily_Return']
std_b = benchmark['Daily_Return'].std()
sharpe_b = (r_b - r_f) / (std_b - std_f)

# Read portfolio performance
# -----------------------------------------------------------------------------------
portfolio = pd.read_csv("portfolio_history.csv", parse_dates=["Date"], usecols=["Date", "Value"])
portfolio.set_index("Date", inplace=True)
portfolio = portfolio.resample('D').interpolate()
portfolio = portfolio.loc[portfolio.index >= start_date]
start_price = portfolio.loc[start_date, "Value"]
portfolio['Return'] = 100.0 * (portfolio['Value'] - start_price) / start_price - risk_free['Return']
portfolio['Daily_Return'] = portfolio['Value'].pct_change()
r_p = 100.0 * (gmean(portfolio.loc[portfolio.index > start_date, 'Daily_Return'] + 1.0) - 1.0)
portfolio['Daily_Return'] = 100.0 * portfolio['Daily_Return']
std_p = portfolio['Daily_Return'].std()
sharpe_p = (r_p - r_f) / (std_p - std_f)

# Plot the returns
# -----------------------------------------------------------------------------------
fig, ax = plt.subplots()
label = "Benchmark (Daily Geo. Mean r={:.2f}%, std={:.2f}%, sharpe={:.2f})".format(r_b, std_b, sharpe_b)
ax.plot(benchmark.Return, label=label, marker=".")
label = "Portfolio (Daily Geo. Mean r={:.2f}%, std={:.2f}%, sharpe={:.2f})".format(r_p, std_p, sharpe_p)
ax.plot(portfolio.Return, label=label, marker=".")
ax.set_title("Benchmark (URTH) and Portfolio Performance")
plt.ylabel("% Cumulative Returns above Risk-Free (FLTR)")
ax.set_ylim(top=8.0)
lgd = ax.legend(loc='upper left', fancybox=True, shadow=True, ncol=1, fontsize=9)
plt.grid(True)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%d"))
fig.autofmt_xdate()
fig.savefig('summary.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.close()

# Read current portfolio
# -----------------------------------------------------------------------------------
current_portfolio = pd.read_csv("OpenPosition.csv", usecols=["Currency", "MarketValue"])
currency = current_portfolio.groupby("Currency").sum()
currency["Percent"] = 100.0 * currency["MarketValue"] / currency["MarketValue"].sum()

# Plot currency distribution
# -----------------------------------------------------------------------------------
ax = currency.plot.pie(y="MarketValue", title="Currency Exposures", figsize=(5, 5))
ax.legend(labels=["{}, {:.1f}%".format(n, currency.loc[n, "Percent"]) for n in currency.index])
ax.set_ylabel("")
fig = ax.get_figure()
fig.savefig('currency.png')
plt.close()
