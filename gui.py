#!/usr/bin/env python
"""
ECCN: EAR99

This script generates an interactive mean-variance plot with optimal portfolio selection.

Input files required in current working directory:
    override.csv - Contains hard coded expected monthly return values.
    stocks/[name].csv - The [name] refers to the stock ticker.  Each csv file needs to contains the "Date" and
        "Adj Close" columns.  Daily frequency is required.
"""

from datetime import date, timedelta
from glob import glob
import numpy as np
from os import chdir, getcwd, remove
from os.path import isfile
from scipy.optimize import minimize
from shutil import move

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

# ***************************************************************************************
# Set version number
# ***************************************************************************************
__version__ = '1.0'


# ***************************************************************************************
def make_portfolio():
    """Create the optimal portfolio.

    Input files required in current working directory:
        settings.csv - Contains the following settings under the "name" and "value" columns.
            risk_free_rate (float):  The risk free rate in percentage, i.e. 10% as 10.0.
            utility_a (float):  The "A" constant in the quadratic utility function; u = e - 1/2 * A * std^2
"""

    r_free = settings["risk_free_rate"]
    std_free = settings["risk_free_std"]
    a = settings["utility_a"]

    # Set the weight bounds and constraints
    # -----------------------------------------------------------------------------------
    weights = np.ones(book.shape[0]) / book.shape[0]

    weight_con = ({'type': 'eq', 'fun': lambda x: x.sum() - 1.0},)
    weight_bnd = weights.shape[0] * [(0.0, 1.0)]
    shorting = "n"
    if "shorting" in settings.index:
        if settings['shorting'].lower() in ['t', 'true', 'y', 'yes']:
            weight_bnd = weights.shape[0] * [(-1.0, 2.0)]
            shorting = "y"
        elif settings['shorting'].lower() == 'limited':
            weight_con += ({'type': 'ineq', 'fun': lambda x: 2.0 - np.abs(x).sum()},)
            weight_bnd = weights.shape[0] * [(-1.0, 1.0)]
            shorting = "L"

    # Determine the optimal portfolio
    # -----------------------------------------------------------------------------------
    res = minimize(negative_slope, x0=weights, bounds=weight_bnd, constraints=weight_con)
    opt_weights = res.x
    opt_r = (opt_weights * book.loc[:, 'return']).sum()
    opt_std = get_std(opt_weights)

    # Determine the minimum variance portfolio
    # -----------------------------------------------------------------------------------
    res = minimize(get_std, x0=opt_weights, bounds=weight_bnd, constraints=weight_con)
    min_var_r = (res.x * book.loc[:, 'return']).sum()
    min_var_std = get_std(res.x)

    # Determine complete portfolio point
    # -----------------------------------------------------------------------------------
    y = (opt_r - r_free) / (a * opt_std ** 2)
    complete_r = y * opt_r + (1.0 - y) * r_free
    complete_std = y * opt_std
    u = complete_r - 0.5 * a * complete_std ** 2

    # Create minimum variance frontier
    # -----------------------------------------------------------------------------------
    frontier_r = np.linspace(min(min_var_r, book.loc[:, 'return'].min()), max(opt_r, book.loc[:, 'return'].max()))
    frontier_std = np.empty_like(frontier_r)
    weights = opt_weights.copy()
    for i in range(frontier_r.shape[0]):

        r_con = weight_con + ({'type': 'eq', 'fun': lambda x: (x * book.loc[:, 'return']).sum() - frontier_r[i]},)
        res = minimize(get_std, x0=weights, bounds=weight_bnd, constraints=r_con)
        weights = res.x
        frontier_std[i] = get_std(res.x)

    # Create the utility function for plotting
    # -----------------------------------------------------------------------------------
    utility_std = np.linspace(0, max(opt_std, complete_std, book.loc[:, 'std'].max()) * 1.1, num=100)
    utility_r = u + 0.5 * a * utility_std ** 2

    # Create the capital allocation line (CAL)
    # -----------------------------------------------------------------------------------
    cal_std = [std_free, utility_std[-1]]
    cal_r = [r_free, r_free + (utility_std[-1] - r_free) * (opt_r - r_free) / (opt_std - std_free)]

    # Make a plot
    # -----------------------------------------------------------------------------------
    fig, ax1 = plt.subplots(nrows=1, ncols=1)
    ax1 = plt.subplot(111)
    ax1.set_xlabel("Standard Deviation of Monthly Returns")
    ax1.set_ylabel("Expected Monthly Returns")
    ax1.grid()

    ax1.plot(utility_std, utility_r, color='orange', label='Utility (A={}, U={:.1f})'.format(a, u))
    if shorting == 'y':
        ax1.plot(frontier_std, frontier_r, color='b', label='Min Var. Frontier (Shorting)')
    elif shorting == 'n':
        ax1.plot(frontier_std, frontier_r, color='b', label='Min Var. Frontier (No Shorting)')
    else:
        ax1.plot(frontier_std, frontier_r, color='b', label='M.V. Frontier (Short with sum(|w|) < 2.0')
    ax1.plot(cal_std, cal_r, color='g', label='Capital Allocation Line (CAL)')
    ax1.scatter(opt_std, opt_r, color='g', marker='d', label='Optimum Portfolio', zorder=8)
    ax1.scatter(min_var_std, min_var_r, color='b', marker='s', label='Global Min Var. Portfolio')
    ax1.scatter(complete_std, complete_r, color='m', marker='*',
                label='Portfolio (A={}, U={:.1f}, y={:.1f}%)'.format(a, u, 100.0*y), zorder=9, s=80)
    ax1.scatter(std_free, r_free, color='grey', marker='o',
                label='"Risk-Free" ({})'.format(settings["risk_free_name"]))
    ax1.scatter(book.loc[:, 'std'], book.loc[:, 'return'], color='k', marker='.', label='Stocks', zorder=10)
    ax1.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0., fontsize=8)
    ax1.set_ylim(0, 20)
    ax1.set_xlim(left=0)

    fig.savefig("portfolio.png", orientation='landscape', bbox_inches="tight")
    plt.close(fig)

    # Save portfolio weights
    # -----------------------------------------------------------------------------------
    weights = pd.Series(data=opt_weights.round(2), index=book.index)
    weights = 100.0 * y * weights.loc[weights != 0.00] / weights.sum()
    weights[settings["risk_free_name"]] = 100.0 * (1.0 - y)
    weights.to_csv("weights.csv", index_label="name", float_format='%.0f')


# ***************************************************************************************
def negative_slope(weights):
    """Calculates the negative slope from the given weighted portfolio to the risk-free asset."""
    r = (weights * book.loc[:, 'return']).sum()
    std = get_std(weights)

    return (settings["risk_free_rate"] - r) / (std - settings["risk_free_std"])


# ***************************************************************************************
def get_std(weights):
    """Calculates the portfolio standard deviation from the given weights."""

    std = 0.0
    for i in range(book.shape[0]):
        std += (weights * book_cov.iloc[:, i]).sum() * weights[i]
    std = np.sqrt(std)

    return std


# ***************************************************************************************
def override():
    """Override trended return values with manually supplied values.

    Input files required in current working directory:
        override.csv - Monthly return values to override trend values.
"""

    if isfile("override.csv"):
        manual_returns = pd.read_csv("override.csv")
        manual_returns.set_index("name", inplace=True)
        tmp_str = "{} has predicted value of {:.1f}% that is overridden by {:.1f}%."
        for name in manual_returns.index.intersection(book.index).tolist():
            print(tmp_str.format(name, book.loc[name, "return"], manual_returns.loc[name, "return"]))
            book.loc[name, "return"] = manual_returns.loc[name, "return"]


# ***************************************************************************************
def trend():
    """Analyze daily stock prices to determine standard deviation and expected monthly return.

    Input files required in current working directory:
        stocks/[name].csv - The [name] refers to the stock ticker.  Each csv file needs to contains the "Date" and
            "Adj Close" columns.  Daily frequency is required.

    Returns:
        pd.DataFrame:  [name, [std, return]]  Each row is a stock by the name and columns are the standard deviation and
            expected monthly return.
        pd.DataFrame:  [name, name]  The covariance matrix of the monthly stock returns.
"""

    # Create DataFrame with index starting 36 months ago
    # ---------------------------------------------------------------------------------------
    now = date.today()
    start = now - timedelta(days=3*365.25)
    prices = pd.DataFrame(index=pd.date_range(start, now))

    # Read stock prices
    # ---------------------------------------------------------------------------------------
    chdir("stocks")
    for file_name in glob("*.csv"):
        csv = pd.read_csv(file_name, usecols=["Date", "Adj Close"])
        name = file_name[:-4]
        csv.rename(columns={'Adj Close': name}, inplace=True)
        csv[name] = pd.to_numeric(csv[name], errors='coerce', downcast='float')
        csv['Date'] = pd.to_datetime(csv['Date'])
        csv.set_index('Date', inplace=True)
        prices = pd.concat([prices, csv], axis=1)
    prices = prices.resample('D').interpolate().bfill().ffill()
    prices = prices.loc[prices.index >= start.strftime("%Y-%m-%d"), :]
    returns = 100.0 * (prices.iloc[28:, :] - prices.iloc[:-28, :].values) / prices.iloc[:-28, :].values
    stocks = pd.DataFrame(index=prices.columns.tolist(), columns=["std", "return"])
    chdir(cwd)

    # Fit trend to rates, predict one month and calculate standard deviation
    # ---------------------------------------------------------------------------------------
    prediction_duration = 12*28
    x = np.arange(prediction_duration)
    fits = {}
    for name in returns.columns:
        fits[name] = np.poly1d(np.polyfit(x, returns[name].values[-prediction_duration:], 1))
        stocks.loc[name, "return"] = fits[name](prediction_duration + 28)
        stocks.loc[name, "std"] = returns.loc[:, name].std()

    # Create pdf report
    # ---------------------------------------------------------------------------------------
    pdf_name = 'stocks.pdf'
    tmp_pdf_name = '.tmp.pdf'
    for name in [n for n in [pdf_name, tmp_pdf_name] if isfile(n)]:
        remove(name)
    pdf = PdfPages(tmp_pdf_name)

    t = [now - timedelta(days=prediction_duration), now, now + timedelta(days=28)]
    title_text = "{}, Monthly STD = {:.1f}%, Predicted Monthly Return = {:.1f}%"
    for name in prices.columns:
        fig = plt.figure(figsize=(8.5, 11))
        ax1 = plt.subplot(211)
        plt.title(title_text.format(name, stocks.loc[name, "std"], stocks.loc[name, "return"]))
        ax1.plot(prices.index, prices[name], color='k', marker=',')
        ax1.set_ylabel("Price")
        ax1.grid()

        ax2 = plt.subplot(212, sharex=ax1)
        ax2.plot(returns.index, returns[name], color='k', marker=',')
        ax2.set_ylabel("Monthly Return in Percent")
        ax2.grid()

        r = [fits[name](0), fits[name](prediction_duration), stocks.loc[name, "return"]]
        ax2.plot(t[:2], r[:2], color='b')
        ax2.plot(t[1:], r[1:], color='g')

        pdf.savefig(fig, papertype='letter', orientation='landscape', pad_inches=0.25)

    pdf.close()
    move(tmp_pdf_name, pdf_name)

    # Remove "risk-free" asset
    # -----------------------------------------------------------------------------------
    if "risk_free_name" in settings.index:
        if settings["risk_free_name"] in stocks.index:
            stocks.drop(settings["risk_free_name"], inplace=True, axis=0)
            returns.drop(settings["risk_free_name"], inplace=True, axis=1)

    # Calculate the covariance
    # -----------------------------------------------------------------------------------
    cov = returns.cov()

    return stocks, cov


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Main body of Code
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

cwd = getcwd()

# Read the settings
# -----------------------------------------------------------------------------------
settings = pd.read_csv("settings.csv", index_col="name", squeeze=True)
settings = settings.str.strip()
settings = pd.to_numeric(settings, errors='coerce').fillna(settings)

# Initialize the book with historical trend and calculate the monthly return covariance
# ---------------------------------------------------------------------------------------
book, book_cov = trend()

# Updated book with manually entered values
# ---------------------------------------------------------------------------------------
override()

# Create portfolio
# ---------------------------------------------------------------------------------------
make_portfolio()
