import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import yfinance as yf


def get_data(stocks, start, end):
    stockData = yf.download(stocks, start=start, end=end)
    stockData = stockData['Close']
    returns = stockData.pct_change()
    meanReturns = returns.mean()
    covMatrix = returns.cov()
    return meanReturns, covMatrix

stockList = ['APLD', 'GOOG', 'LRCX']
endDate = dt.datetime.now()
startDate = endDate - dt.timedelta(days=300)

meanReturns, covMatrix = get_data(stockList, startDate, endDate)
print(meanReturns)

weights = np.random.random(len(meanReturns))
weights /= np.sum(weights)

print(weights)

#Monte Carlo Method
mc_sims = 10
timeframe = 100 # in days

meanM = np.full(shape=(timeframe, len(meanReturns)), fill_value=meanReturns)
meanM = meanM.T

portfolio_sims = np.full(shape=(timeframe, mc_sims), fill_value=0.0)

initialPortfolio = 3500

for m in range(mc_sims):
    Z = np.random.normal(size=(timeframe, len(weights)))
    L = np.linalg.cholesky(covMatrix)

    dailyReturns = meanM + np.inner(L, Z)
    portfolio_sims[:, m] = np.cumprod(np.inner(weights, dailyReturns.T) + 1) * initialPortfolio

plt.plot(portfolio_sims)
plt.ylabel('Portfolio Value ($)')
plt.xlabel('Days')
plt.title('Monte Carlo Simulation: Portfolio Value over Time')
plt.show()