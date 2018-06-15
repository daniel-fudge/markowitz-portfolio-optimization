#!/usr/bin/env python
"""
This script generates a summary of the Stock-Trak performance.

Input files required in current working directory:
    FLTR.csv - Yahoo Finance formatted historical performance of VanEck Vectors Investment Grade Floating Rate ETF.
    URTH.csv - Yahoo Finance formatted historical performance of iShares MSCI World ETF.
    portfolio_history.csv - Contains the Stock-Trak formatted historical performance of the Stock-Trak account.
"""

from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ***************************************************************************************
# Set version number
# ***************************************************************************************
__version__ = '1.0'


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Main body of Code
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

start_date = datetime(2018, 5, 29)

# Read Risk-Free performance
# -----------------------------------------------------------------------------------
risk_free = pd.read_csv("FLTR.csv", parse_dates=["Date"], usecols=["Date", "Adj Close"])
risk_free.set_index("Date", inplace=True)
risk_free = risk_free.resample('D').interpolate()
start_price = risk_free.loc[start_date, "Adj Close"]
risk_free['Return'] = 100.0 * (risk_free['Adj Close'] - start_price) / start_price

# Read benchmark performance
# -----------------------------------------------------------------------------------
benchmark = pd.read_csv("URTH.csv", parse_dates=["Date"], usecols=["Date", "Adj Close"])
benchmark.set_index("Date", inplace=True)
benchmark = benchmark.resample('D').interpolate()
benchmark = benchmark.loc[benchmark.index >= start_date]
start_price = benchmark.loc[start_date, "Adj Close"]
benchmark['Return'] = 100.0 * (benchmark['Adj Close'] - start_price) / start_price - risk_free['Return']
std_b = benchmark['Return'].std()
sharpe_b = benchmark['Return'].iloc[-1] / std_b

# Read portfolio performance
# -----------------------------------------------------------------------------------
portfolio = pd.read_csv("portfolio_history.csv", parse_dates=["Date"], usecols=["Date", "Value"])
portfolio.set_index("Date", inplace=True)
portfolio = portfolio.resample('D').interpolate()
portfolio = portfolio.loc[portfolio.index >= start_date]
start_price = portfolio.loc[start_date, "Value"]
portfolio['Return'] = 100.0 * (portfolio['Value'] - start_price) / start_price - risk_free['Return']
std_p = portfolio['Return'].std()
sharpe_p = portfolio['Return'].iloc[-1] / std_p

# Plot the returns
# -----------------------------------------------------------------------------------
fig, ax = plt.subplots()
label = "Benchmark (r={:.2f}%, std={:.2f}%, sharpe={:.2f})".format(benchmark['Return'].iloc[-1], std_b, sharpe_b)
ax.plot(benchmark.Return, label=label, marker=".")
label = "Portfolio (r={:.2f}%, std={:.2f}%, sharpe={:.2f})".format(portfolio['Return'].iloc[-1], std_p, sharpe_p)
ax.plot(portfolio.Return, label=label, marker=".")
ax.set_title("Benchmark (URTH) and Portfolio Performance")
plt.ylabel("% Cumulative Returns above Risk-Free (FLTR)")
ax.set_ylim(0.0, 4.0)
lgd = ax.legend(loc='upper left', fancybox=True, shadow=True, ncol=1, fontsize=9)
plt.grid(True)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%d"))
fig.autofmt_xdate()
fig.savefig('summary.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.close()
