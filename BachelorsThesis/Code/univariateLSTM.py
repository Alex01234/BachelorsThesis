import numpy
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import seaborn as sns
sns.set(font='IPAGothic')

#LSTM Univariate code inspired by Brownlee at machinelearningmastery.com
'''
Sources:
https://machinelearningmastery.com/time-series-forecasting-long-short-term-memory-network-python/?fbclid=IwAR3zVjgn82ZRo4Zi-VTOpSiT1xyinUkiVrvK-V2JxpY1qZMNFLS6GQV6g_Y
https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/?fbclid=IwAR0j_lhsDOJFRxAbh0w7vmPIK5Mux1ExFd23NvKiRxe-0UfJsGejf6CISEA
'''

# load the dataset
dataframe = pandas.read_csv('AAPL_close_csv.csv', usecols=[1], engine='python')
dataset = dataframe.values
dataset = dataset.astype('float32')
raw_values = dataset

# normalize dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# split into train and test sets
train_size = int(len(dataset) * 0.63)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
raw_values_test = raw_values[train_size:len(raw_values),:]

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)
	
	# reshape into X=t and Y=t+1
look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))


# create and fit the LSTM network
model = Sequential()
model.add(LSTM(8, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
model.fit(trainX, trainY, epochs=500, batch_size=4,validation_data=(testX, testY), verbose=2)
#plot history
plt.rc('figure', figsize=(18, 3))
plt.plot(model.history.history['loss'], label='train')
plt.plot(model.history.history['val_loss'], label='test')
plt.title("LSTM Univariate - Train Loss Versus Test Loss - Apple")
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
# calculate root mean squared error
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.3f RMSE' % (testScore))


# plot baseline and predictions
plt.rc('figure', figsize=(18, 3))
plt.plot(raw_values_test[:15], label='Actual stock price')
plt.plot(testPredict[:,0],color='red', label='Predicted stock price')
plt.title("LSTM Univariate - Predicted Stock Prices Against Actual Stock Prices - Apple")
plt.ylabel("Stock Price")
plt.xlabel("Day")
plt.legend()
plt.show()
