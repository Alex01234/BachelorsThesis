from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import seaborn as sns
sns.set(font='IPAGothic')

#LSTM Multivariate code inspired by Brownlee at machinelearningmastery.com
'''
Sources:
https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/?fbclid=IwAR232BDqAFLnD52J6lfBUeNIU0rcLMHqN0hJsQcf0XUIgIfcnQfB5otYrKA
https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/?fbclid=IwAR0j_lhsDOJFRxAbh0w7vmPIK5Mux1ExFd23NvKiRxe-0UfJsGejf6CISEA
'''


# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg
	
# load dataset, The code is adjusted for a dataset with two variables excluding date.
dataset = read_csv('LSTM_MSFT_data_merged.csv', header=0, index_col=0)
values = dataset.values
valuesUnchanged = read_csv('MSFT_close_csv.csv', header=0, parse_dates=[0], index_col=0, squeeze=True)
# transform the data to be stationary
raw_values = valuesUnchanged.values
# integer encode direction
encoder = LabelEncoder()
values[:,1] = encoder.fit_transform(values[:,1])
# ensure all data is float
values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# frame as supervised learning
reframed = series_to_supervised(scaled, 1, 1)
# drop columns we don't want to predict
reframed.drop(reframed.columns[[3]], axis=1, inplace=True)

# split into train and test sets
values = reframed.values
train_size = int((len(values) * 0.66))
train = values[:train_size, :]
test = values[train_size:, :]
# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

# design network
history_list = list()
model = Sequential()
model.add(LSTM(100, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam',metrics=['accuracy'])
# fit network
history = model.fit(train_X, train_y, epochs=680, batch_size=4, validation_data=(test_X, test_y), verbose=2, shuffle=False)
			
#plot history
pyplot.rc('figure', figsize=(18, 3))
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.title("LSTM Multivariate - Train Loss Versus Test Loss - Microsoft")
pyplot.ylabel('loss')
pyplot.xlabel('epoch')
pyplot.legend(['train', 'test'], loc='upper right')
pyplot.show()


# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]
#invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]
# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)


# line plot of observed vs predicted
start = train_size + 1
pyplot.rc('figure', figsize=(18, 3))
pyplot.plot(raw_values[start:], label='Actual stock price')
pyplot.plot(inv_yhat[-train_size:],color='red', label='Predicted stock price')
pyplot.title("LSTM Multivariate - Predicted Stock Prices Against Actual Stock Prices - Microsoft")
pyplot.ylabel("Stock Price")
pyplot.xlabel("Day")
pyplot.legend()
pyplot.show()
