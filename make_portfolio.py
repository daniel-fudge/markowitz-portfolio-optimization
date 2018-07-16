#!/usr/bin/env python
"""
This script performs an optimal portfolio selection based on the Sharpe ratio and utility maximization.

Notes:
    [CC] is the Currency Code such as "gbp" for Great Britain Pound.
    [SE] is the Stock Extension used in the Yahoo Finance name on the stock price history files, such as "to" for the
        Toronto Stock Exchange.  The [SE] is used to make to the associated [CC].

Input files required in current working directory:

    OpenPosition.csv - Current StockTrak portfolio positions.
    override.csv - Contains hard coded expected monthly return values.
    settings.csv - Contains the following settings.
        risk_free_name (str):  The risk free asset name.
        utility_a (float):  The utility factor use to make asset allocation.
        shorting (str):  The shorting method, which can be:
            n = No shorting.
            y = Full shorting.
            limited = The absolute value of the weights can't be greater than 1.5.
        portfolio_value (float):  The desired portfolio value used to determine number of stock to purchase / sell.
        [CC]_cad (float):  The CC to Canadian dollar conversion rate.
        [SE] (str):  The [CC] associated with the given [SE].
    stocks/[name].csv - The [name] refers to the stock ticker.  Each csv file needs to contains the "Date" and
        "Adj Close" columns.  Daily frequency is required.
"""

from datetime import date, timedelta
from glob import glob
import numpy as np
from os import chdir, getcwd, remove
from os.path import isfile
import pandas as pd
from scipy.optimize import minimize
from shutil import move

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

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

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
            weight_con += ({'type': 'ineq', 'fun': lambda x: 1.5 - np.abs(x).sum()},)
            weight_bnd = weights.shape[0] * [(-1.0, 1.0)]
            shorting = "L"

    # Determine the optimal portfolio
    # -----------------------------------------------------------------------------------
    print("Generating optimal portfolio.")
    res = minimize(negative_slope, x0=weights, bounds=weight_bnd, constraints=weight_con)
    opt_weights = res.x
    opt_r = (opt_weights * book.loc[:, 'return']).sum()
    opt_std = get_std(opt_weights)

    # Determine the minimum variance portfolio
    # -----------------------------------------------------------------------------------
    print("Generating minimum variance portfolio.")
    res = minimize(get_std, x0=opt_weights, bounds=weight_bnd, constraints=weight_con)
    min_var_r = (res.x * book.loc[:, 'return']).sum()
    min_var_std = get_std(res.x)

    # Determine complete portfolio point
    # -----------------------------------------------------------------------------------
    y = (opt_r - risk_free["return"]) / (a * opt_std ** 2)
    if y > 1.0:
        print("Optimal y is {:.1f}%, but limiting to 100.0%".format(100.0 * y))
        y = 1.0
    complete_r = y * opt_r + (1.0 - y) * risk_free["return"]
    complete_std = y * opt_std
    u = complete_r - 0.5 * a * complete_std ** 2

    # Compile desired portfolio
    # -----------------------------------------------------------------------------------
    print("Saving portfolio.csv")
    portfolio = pd.DataFrame(columns=["percentage"], index=book.index)
    portfolio.loc[:, 'percentage'] = opt_weights
    portfolio = portfolio.loc[portfolio.percentage.round(2) != 0.00, :]
    portfolio['percentage'] = y * portfolio['percentage'] / portfolio['percentage'].sum()
    portfolio.loc[settings["risk_free_name"], 'percentage'] = 1.0 - y
    portfolio['value_cad'] = portfolio['percentage'] * settings['portfolio_value']

    for name in portfolio.index:
        tmp_a = name.lower().split(".")
        if len(tmp_a) == 1:
            ex = settings['usd_cad']
        else:
            ex = settings['{}_cad'.format(settings[tmp_a[1]])]

        if name == settings['risk_free_name']:
            portfolio.loc[settings['risk_free_name'], 'price_cad'] = risk_free["close"] * ex
        else:
            portfolio.loc[name, 'price_cad'] = book.loc[name, 'close'] * ex

    portfolio['desired_number'] = portfolio['value_cad'] / portfolio['price_cad']

    # Read current portfolio and calculate the necessary trades
    # -----------------------------------------------------------------------------------
    current_portfolio = pd.read_csv("OpenPosition.csv")
    current_portfolio.rename(columns={"Quantity": "current_number"}, inplace=True)
    for i in current_portfolio.index:
        currency = current_portfolio.loc[i, "Currency"].lower()
        if currency != "usd":
            market = settings[settings == currency].index[0].upper()
            current_portfolio.loc[i, "Symbol"] = "{}.{}".format(current_portfolio.loc[i, "Symbol"], market)
    current_portfolio.set_index("Symbol", inplace=True)

    portfolio = pd.concat([portfolio, current_portfolio["current_number"]], axis=1)
    for name in ["current_number", "percentage", "value_cad", "desired_number"]:
        portfolio[name] = portfolio[name].fillna(0.0)
    missing = portfolio.index[portfolio.price_cad.isnull()].tolist()

    for name in missing:
        tmp_a = name.lower().split(".")
        if len(tmp_a) == 1:
            ex = settings['usd_cad']
        else:
            ex = settings['{}_cad'.format(settings[tmp_a[1]])]
        portfolio.loc[name, 'price_cad'] = current_portfolio.loc[name, "LastPrice"] * ex

    portfolio["buy"] = portfolio['desired_number'] - portfolio['current_number']

    # Save portfolio
    # -----------------------------------------------------------------------------------
    portfolio = portfolio.astype({n: int for n in ["desired_number", "current_number", "buy"]})
    portfolio.to_csv("portfolio.csv", index_label="ticker", float_format='%.2f')

    # Create the utility function for plotting
    # -----------------------------------------------------------------------------------
    utility_std = np.linspace(0, max(opt_std, complete_std, book.loc[:, 'std'].max()) * 1.1, num=100)
    utility_r = u + 0.5 * a * utility_std ** 2

    # Create the capital allocation line (CAL)
    # -----------------------------------------------------------------------------------
    cal_std = [risk_free["std"], utility_std[-1]]
    d_r = (utility_std[-1] - risk_free["return"]) * (opt_r - risk_free["return"]) / (opt_std - risk_free["std"])
    cal_r = [risk_free["return"], risk_free["return"] + d_r]

    # Create minimum variance frontier
    # -----------------------------------------------------------------------------------
    print("Generating minimum variance frontier.")
    frontier_r = np.linspace(min(min_var_r, book.loc[:, 'return'].min()),
                             max(opt_r, book.loc[:, 'return'].max()), num=10)
    frontier_std = np.empty_like(frontier_r)
    weights = opt_weights.copy()
    for i in range(frontier_r.shape[0]):

        r_con = weight_con + ({'type': 'eq', 'fun': lambda x: (x * book.loc[:, 'return']).sum() - frontier_r[i]},)
        res = minimize(get_std, x0=weights, bounds=weight_bnd, constraints=r_con)
        weights = res.x
        frontier_std[i] = get_std(res.x)

    # Make a plot
    # -----------------------------------------------------------------------------------
    print("Making portfolio.png.")
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
        ax1.plot(frontier_std, frontier_r, color='b', label='M.V. Frontier (Short with sum(|w|) < 1.5)')
    ax1.plot(cal_std, cal_r, color='g', label='Capital Allocation Line (CAL)')
    ax1.scatter(opt_std, opt_r, color='g', marker='d', label='Optimum Portfolio', zorder=8)
    ax1.scatter(min_var_std, min_var_r, color='b', marker='s', label='Global Min Var. Portfolio')
    ax1.scatter(complete_std, complete_r, color='m', marker='*',
                label='Portfolio (A={}, U={:.1f}, y={:.1f}%)'.format(a, u, 100.0*y), zorder=9, s=80)
    ax1.scatter(risk_free["std"], risk_free["return"], color='grey', marker='o',
                label='"Risk-Free" ({})'.format(settings["risk_free_name"]))
    ax1.scatter(book.loc[:, 'std'], book.loc[:, 'return'], color='k', marker='.', label='Stocks', zorder=10)
    ax1.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0., fontsize=8)
    max_axis = 1.05 * max(opt_r, complete_r, book.loc[:, 'std'].max())
    ax1.set_ylim(top=max_axis)
    ax1.set_xlim(left=0, right=max_axis)

    fig.savefig("portfolio.png", orientation='landscape', bbox_inches="tight")
    plt.close(fig)


# ***************************************************************************************
def negative_slope(weights):
    """Calculates the negative slope from the given weighted portfolio to the risk-free asset."""
    r = (weights * book.loc[:, 'return']).sum()
    std = get_std(weights)

    return (risk_free["return"] - r) / (std - risk_free["std"])


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
        pd.Series:  [std, return, close]  The price information for the given risk-free asset.
        pd.DataFrame:  [name, [std, return, close]]  Each row is a stock by the name and columns are the standard
            deviation, expected monthly return and today's adjusted close value.
        pd.DataFrame:  [name, name]  The covariance matrix of the monthly stock returns.
"""

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

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
    stocks = pd.DataFrame(index=prices.columns.tolist(), columns=["std", "return", "close"])
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
        stocks.loc[name, "close"] = prices.loc[now, name]

    # Create pdf report
    # ---------------------------------------------------------------------------------------
    pdf_name = 'stocks.pdf'
    print("Making {}.".format(pdf_name))
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
        plt.close(fig)

    pdf.close()
    move(tmp_pdf_name, pdf_name)

    # Remove "risk-free" asset
    # -----------------------------------------------------------------------------------
    rf = stocks.loc[settings["risk_free_name"], :]
    stocks.drop(settings["risk_free_name"], inplace=True, axis=0)
    returns.drop(settings["risk_free_name"], inplace=True, axis=1)

    # Calculate the covariance
    # -----------------------------------------------------------------------------------
    cov = returns.cov()

    return rf, stocks, cov


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Main body of Code
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

cwd = getcwd()

# Read the settings
# -----------------------------------------------------------------------------------
settings = pd.read_csv("settings.csv", index_col="name", squeeze=True)
settings = settings.str.strip()
settings = pd.to_numeric(settings, errors='coerce').fillna(settings)
settings['cad_cad'] = 1.0

# Initialize the book with historical trend and calculate the monthly return covariance
# ---------------------------------------------------------------------------------------
risk_free, book, book_cov = trend()

# Updated book with manually entered values
# ---------------------------------------------------------------------------------------
override()

# Create portfolio
# ---------------------------------------------------------------------------------------
make_portfolio()
