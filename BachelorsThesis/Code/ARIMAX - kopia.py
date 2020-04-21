#https://mc.ai/11-classical-time-series-forecasting-methods-in-python-cheat-sheet/

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font='IPAGothic')
import numpy as np
import statsmodels.api as sm

from sklearn.metrics import mean_squared_error

stock_prices = pd.read_csv('MSFT_close_csv.csv', header=0, parse_dates=[0], index_col=0, squeeze=True)
twitter_data = pd.read_csv('time_series_MSFT_2020-02-03_-_2020-04-03.csv', header=0, parse_dates=[0], index_col=0, squeeze=True)

# Grid Search ARIMAX Model Hyperparameters
'''
def evaluate_arimax_model(X, Y, arima_order):
    warnings.filterwarnings("ignore")
    # prepare training dataset:
    training_start, training_end = 0, int(len(stock_prices) * 0.66)
    testing_start, testing_end = int(len(stock_prices) * 0.66), len(stock_prices)

    stock_training, stock_test = stock_prices[training_start:training_end], stock_prices[testing_start:testing_end]
    twitter_training, twitter_test = twitter_data[training_start:training_end], twitter_data[testing_start:testing_end]
    stock_history = [x for x in stock_training]
    twitter_history = [x for x in twitter_training]

    # make predictions
    predictions = list()
    for t in range(len(stock_test)):
        model = sm.tsa.statespace.SARIMAX(stock_history,
                                          order = arima_order,
                                          seasonal_order=(0,0,0,0),
                                          exog = twitter_history,
                                          freq=None,
                                       enforce_stationarity=False,
                                       enforce_invertibility=False)
        model_fit = model.fit()
        yhat = model_fit.predict()[0]
        predictions.append(yhat)
        stock_history.append(stock_test[t])
        twitter_history.append(twitter_test[t])
 
    # mse = mean_squared_error(stock_test, predictions)
    rmse = np.sqrt(mean_squared_error(stock_test, predictions))
    return rmse

# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(stock, twitter, p_values, d_values, q_values):
    stock = stock.astype('float32')
    twitter = twitter.astype('float32')
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p,d,q)
                try:
                    rmse = evaluate_arimax_model(stock, twitter, order)
                    if rmse < best_score:
                        best_score, best_cfg = rmse, order
                    print('ARIMAX%s RMSE=%.3f' % (order,rmse))
                except:
                    continue
    
    print('Best ARIMAX%s RMSE=%.3f' % (best_cfg, best_score))

# evaluate parameters
p_values = [0, 1, 2, 3, 4, 5]
d_values = [0, 1, 2, 3, 4, 5]
q_values = [0, 1, 2, 3, 4, 5]

evaluate_models(stock_prices.values, twitter_data.values, p_values, d_values, q_values)
'''

# Rolling Forecast with Selected ARIMAX Model: Walk Forward Validation
'''
# split into train and test sets
training_start, training_end = 0, int(len(stock_prices) * 0.66)
testing_start, testing_end = int(len(stock_prices) * 0.66), len(stock_prices)

stock_training, stock_test = stock_prices[training_start:training_end], stock_prices[testing_start:testing_end]
twitter_training, twitter_test = twitter_data[training_start:training_end], twitter_data[testing_start:testing_end]
stock_history = [x for x in stock_training]
twitter_history = [x for x in twitter_training]
predictions = []

# walk-forward validation
for t in range(len(stock_test)):
    model = sm.tsa.statespace.SARIMAX(stock_history,
                                          order = (0,0,1),
                                          seasonal_order=(0,0,0,0),
                                          exog = twitter_history,
                                          freq=None,
                                          enforce_stationarity=False,
                                          enforce_invertibility=False)
    model_fit = model.fit()
    output = model_fit.predict()[0]
    #print (output)
    #yhat = output[0]
    #predictions.append(yhat)
    predictions.append(output)
    obs = stock_test[t]
    stock_history.append(obs)
    twitter_history.append(twitter_test[t])

# evaluate forecasts
rmse = np.sqrt(mean_squared_error(stock_test, predictions))
print('Test RMSE: %.3f' % rmse)
print(model_fit.summary())
'''

# Plot forecasts against actual outcomes
'''
plt.rc('figure', figsize=(18, 3))
plt.plot(stock_test.values, label='Actual stock price')
plt.plot(predictions, color='red', label='Predicted stock price')
plt.title("ARIMAX(0,0,1) - Predicted Stock Prices Against Actual Stock Prices - Microsoft")
plt.ylabel("Stock Price")
plt.xlabel("Day")
plt.legend()
plt.show()
'''
