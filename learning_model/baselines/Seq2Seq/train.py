from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from pandas import concat
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, RepeatVector, Dropout, TimeDistributed
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras.optimizers import Adam, RMSprop
from keras.layers.advanced_activations import LeakyReLU
from math import sqrt
from matplotlib import pyplot
from numpy import array
import numpy as np

dataset_filename = './dataset.csv'

n_lag = 6
n_seq = 3
n_jump=1
n_test = 8000
path_model = './model/short.h5'
np.random.seed(seed=1)

n_batch=8
n_epochs = 300
n_neurons = 500

# date-time parsing function for loading the dataset
def parser(x):
	return datetime.strptime('190'+x, '%Y-%m')

# shuffle dataset
def shuffle_along_axis(a, axis):
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a,idx,axis=axis)

# convert time series into supervised learning problem
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True, n_jump=1, n_test=1):
	agg = np.zeros((data.shape[0], data.shape[1], n_in+n_out+1)) # for base value
	# input sequence (t-n, ... t-1)
	print('historical')
	for i in range(n_in+1, data.shape[0]-n_jump-n_out, n_jump):
		historical_data = data[(i-n_in-1):i, :]
		agg[i, :, 0:n_in+1] = np.swapaxes(historical_data, 1, 0)
		
	# forecast sequence (t, t+1, ... t+n)
	print('forecast')
	for i in range(n_in+1, data.shape[0]-n_jump-n_out, n_jump):
		forecast_data = data[i:(i+n_out), :]
		agg[i, :, n_in+1:] = np.swapaxes(forecast_data, 1, 0)
	
	#new for training
	agg = agg[:(agg.shape[0]-n_test)]
	
	agg = np.take(agg,np.random.permutation(agg.shape[0]),axis=0,out=agg)
	return agg

# create a differenced series
def difference(dataset, interval=1):
	for i in range(interval, len(dataset)):
		value = dataset[i, :] - dataset[i - interval, :]
		value = np.reshape(value, (1, -1))
		dataset[i - interval, :] = value
	return dataset

# transform series into train and test sets for supervised learning
def prepare_data(series, n_test, n_lag, n_seq,n_jump):
	# drop all-zero columns and refill nan
	series = series.replace(0,np.nan).dropna(axis=1,how="all")
	series = series.fillna(0)
	# extract raw values
	raw_values = series.values
	n_vars = raw_values.shape[1]
	# transform data to be stationary
	#diff_values = difference(raw_values, 1)
	diff_values = raw_values
	# rescale values to 0, 1
	scaler = MinMaxScaler(feature_range=(0, 1))
	#scaled_values = scaler.fit_transform(diff_values)
	scaled_values =diff_values
	# transform into supervised learning problem X, y
	supervised = series_to_supervised(scaled_values, n_lag, n_seq, n_jump, n_test=n_test)
	supervised = np.swapaxes(supervised, 2, 1)
	# split into train and test sets
	train, test = supervised, None
	return scaler, train, test, n_vars

# fit an LSTM network to training data
def fit_lstm(train, n_lag, n_seq, n_vars, n_batch, nb_epoch, n_neurons, path_model):
	#train = np.take(train,np.random.permutation(train.shape[0]),axis=0,out=train)
	#train = train[:int(train.shape[0])]
	print('Maxmin:', np.max(train), np.min(train))
	# reshape training into [samples, timesteps, features]
	y = train[:, n_lag+1:, :]	
	#y = y.reshape(y.shape[0], n_vars*n_seq)
	train = train[:, 1:n_lag+1, :]
	print('Shape:', train.shape, y.shape)
	activation = 'relu'
	# design network
	model = Sequential()
	
	# encoder
	model.add(LSTM(n_neurons, return_sequences=True, activation=activation, input_shape=(n_lag, n_vars)))
	model.add(LSTM(n_neurons, return_sequences=True, activation=activation))
	model.add(LSTM(n_neurons, return_sequences=True, activation=activation))
	model.add(LSTM(n_neurons, return_sequences=True, activation=activation))
	model.add(LSTM(n_neurons, activation=activation))
	model.add(RepeatVector(n_seq))
	
	# decoder
	model.add(LSTM(n_neurons, return_sequences=True, activation=activation))
	model.add(LSTM(n_neurons, return_sequences=True, activation=activation))
	model.add(LSTM(n_neurons, return_sequences=True, activation=activation))
	model.add(LSTM(n_neurons, return_sequences=True, activation=activation))
	model.add(LSTM(n_neurons, return_sequences=True, activation=activation))
 
	# prediction
	#model.add(Dense(n_neurons, activation=activation))
	#model.add(LeakyReLU(alpha=0.3))
	model.add(Dense(n_vars, activation='relu'))
	model.compile(loss='mse', optimizer=Adam(lr=1e-3, decay=5e-4), metrics=['mse'])
	model.summary()
	
	es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=500)
	mc = ModelCheckpoint(path_model, monitor='val_loss', mode='min', verbose=1, save_best_only=True)
	model.fit(train, y, epochs=nb_epoch, batch_size=n_batch, validation_split=.05, verbose=2,shuffle=True, callbacks=[es, mc])
	return model

# load dataset
series = read_csv(dataset_filename, header=0, parse_dates=[0], index_col=0)

# prepare data
scaler, train, test, n_vars = prepare_data(series, n_test, n_lag, n_seq, n_jump)

# training
model = fit_lstm(train, n_lag, n_seq, n_vars, n_batch, n_epochs, n_neurons, path_model)
