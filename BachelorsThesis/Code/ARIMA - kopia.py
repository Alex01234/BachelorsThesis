# https://ucilnica.fri.uni-lj.si/mod/resource/view.php?id=28089

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font='IPAGothic')
import warnings
warnings.filterwarnings("ignore")
from statsmodels.graphics.tsaplots import plot_acf
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.graphics.gofplots import qqplot
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.arima_model import ARIMA

s = pd.read_csv('MSFT_close_csv.csv', header=0, parse_dates=[0], index_col=0, squeeze=True)


# Grid Search ARIMA Model Hyperparameters
'''
# evaluate an ARIMA model for a given order (p,d,q)
def evaluate_arima_model(X, arima_order):
    # prepare training dataset
    train_size = int(len(X) * 0.66)
    train, test = X[0:train_size], X[train_size:]
    history = [x for x in train]

    # make predictions
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, order=arima_order)
        model_fit = model.fit(disp=0)
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test[t])
    # calculate out of sample error
    rmse = np.sqrt(mean_squared_error(test, predictions))
    return rmse

# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(dataset, p_values, d_values, q_values):
    dataset = dataset.astype('float32')
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p,d,q)
                try:
                    rmse = evaluate_arima_model(dataset, order)
                    if rmse < best_score:
                        best_score, best_cfg = rmse, order
                    print('ARIMA%s RMSE=%.3f' % (order,rmse))
                except:
                    continue
    
    print('Best ARIMA%s RMSE=%.3f' % (best_cfg, best_score))

# evaluate parameters
p_values = [0, 1, 2, 3, 4, 5]
d_values = [0, 1, 2, 3, 4, 5]
q_values = [0, 1, 2, 3, 4, 5]

evaluate_models(s.values, p_values, d_values, q_values)
'''

# Rolling Forecast with Selected ARIMA Model: Walk Forward Validation
'''
# split into train and test sets
X = s.values
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = []

# walk-forward validation
for t in range(len(test)):
    model = ARIMA(history, order=(1,1,0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    #print('predicted=%f, expected=%f' % (yhat, obs))

# evaluate forecasts
rmse = np.sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)
print(model_fit.summary())
# https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima_model.ARIMAResults.html#statsmodels.tsa.arima_model.ARIMAResults
print('conf_int')
print(model_fit.conf_int())
# https://www.machinelearningplus.com/time-series/arima-model-time-series-forecasting-python/
# https://www.digitalocean.com/community/tutorials/a-guide-to-time-series-forecasting-with-arima-in-python-3
print('train')
print(train)
print('history')
print(history)
print('test')
print(test)
print('predictions')
print(predictions)
'''


# Plot forecasts against actual outcomes
'''
# plot forecasts against actual outcomes
plt.rc('figure', figsize=(18, 3))
plt.plot(test, label='Actual stock price')
plt.plot(predictions, color='red', label='Predicted stock price')
plt.title("ARIMA(1,1,0) - Predicted Stock Prices Against Actual Stock Prices - Microsoft")
plt.ylabel("Stock Price")
plt.xlabel("Day")
plt.legend()
plt.show()
'''
