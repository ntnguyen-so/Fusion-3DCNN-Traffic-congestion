from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from pandas import concat
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import Adam
from keras.models import load_model
from math import sqrt
from matplotlib import pyplot
from numpy import array
import numpy as np

dataset_filename = './timeseries_avg_reduced.csv'
model_filename = './model/long.h5'
result_filename = './evaluation/s6_4h_s6_4h.csv'
n_jump=8
n_lag = 6
n_seq = 6
n_test=8000
n_batch=1

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
	
	agg = agg[(agg.shape[0]-n_test):]
	
	return agg

# transform series into train and test sets for supervised learning
def prepare_data(series, n_test, n_lag, n_seq,n_jump):
	# drop all-zero columns and refill nan
	series = series.replace(0,np.nan).dropna(axis=1,how="all")
	series = series.fillna(0)
	# extract raw values
	raw_values = series.values
	n_vars = raw_values.shape[1]
	diff_values = raw_values
	# rescale values to 0, 1
	scaler = MinMaxScaler(feature_range=(0, 1))
	scaled_values =diff_values
	# transform into supervised learning problem X, y
	supervised = series_to_supervised(scaled_values, n_lag, n_seq, n_jump, n_test=n_test)
	supervised = np.swapaxes(supervised, 2, 1)
	# split into train and test sets
	train, test = None, supervised
	return scaler, train, test, n_vars

# make one forecast with an LSTM,
def forecast_lstm(model, X, n_batch):
	# make forecast
	forecast = model.predict(X, batch_size=n_batch)
	# convert to array
	return forecast

# evaluate the persistence model
def make_forecasts(model, n_batch, train, test, n_lag, n_seq):
	# reshape training into [samples, timesteps, features]
	X_test = test[:, 1:n_lag+1, :]
	y_test = test[:, n_lag+1:, :]	

	forecasts = None
	for i in range(len(test)):
		#print('make_forecasts', i, len(test))
		X = X_test[i, :, :]
		X = np.expand_dims(X, axis=0)
		y = y_test[i, :]
		# make forecast
		forecast = forecast_lstm(model, X, n_batch)
		#print(np.unique(forecast, return_counts=True))
		forecast = np.append(X, forecast, axis=1)
		# store the forecast
		if forecasts is None:
			forecasts = forecast
		else:
			forecasts = np.vstack((forecasts, forecast))
	return forecasts

# invert differenced forecast
def inverse_difference(last_ob, forecast):
	# invert first forecast
	inverted = forecast[0]
	inverted = np.reshape(inverted, (1, -1))
	# propagate difference forecast using inverted first value
	for i in range(1, len(forecast)):
		inverted_row = forecast[i]
		inverted_row = np.reshape(inverted_row, (1, -1))

		inverted = np.vstack((inverted, inverted_row))
	return inverted

# inverse data transform on forecasts
def inverse_transform(series, forecasts, scaler, n_test):
	inverted = None
	for i in range(len(forecasts)):
		# create array from forecast
		forecast = array(forecasts[i])
		# invert scaling
		inv_scale = forecast
		# invert differencing
		last_ob = inv_scale[0]
		inv_diff = inverse_difference(last_ob, inv_scale)
		inv_diff = np.expand_dims(inv_diff, axis=0)
		# store
		if inverted is None:			
			inverted = inv_diff
		else:
			inverted = np.vstack((inverted, inv_diff))
	return inverted

# evaluate the RMSE for each forecast time step
def evaluate_forecasts(test, forecasts, n_lag, n_seq, result_file, timestart, n_vars):
	for i in range(len(test)):
		actual = test[i, n_lag+1:, :]
		predicted = forecasts[i, n_lag:, :]

		mse = mean_squared_error(actual, predicted)
		rmse = sqrt(mean_squared_error(actual, predicted))
		mae = mean_absolute_error(actual, predicted)
		
		result_line = '{0},{1},{2},{3},{4},{5}'.format(timestart + i, np.sum(actual), np.sum(predicted), mse, mae, rmse)
		result_file.write(result_line + '\n')
		print(result_line)

# reset the results
# f = open (result_filename, 'w')
# f.write('id,err_MSE,err_RMSE,err_MAE,GT,PD\n')
# f.close()

# load dataset
series = DataFrame()
i = 0

timestart = 0
model = None
chunksize = 50
series = read_csv(dataset_filename, header=0, parse_dates=[0], index_col=0)

# prepare data
scaler, train, TEST, n_vars = prepare_data(series, n_test, n_lag, n_seq, n_jump)

model = None
for i in range(TEST.shape[0]//chunksize):
	result_file = open(result_filename, 'a')
	test = TEST[(i*chunksize): (min((i+1)*chunksize, TEST.shape[0]))]
	print(test.shape)
	print('load_model')
	if model is None:
		model = load_model(model_filename)
		model.summary()
	# make forecasts
	print('make_forecasts')
	forecasts = make_forecasts(model, n_batch, train, test, n_lag, n_seq)
	# inverse transform forecasts and test
	print('inverse_transform forecasts')
	forecasts = inverse_transform(series, forecasts, scaler, n_test+2)
	print('inverse_transform actual')
	#test = np.swapaxes(test, 2, 1)
	actual = inverse_transform(series, test, scaler, n_test+2)
	# evaluate forecasts
	print('evaluate_forecasts')
	evaluate_forecasts(actual, forecasts, n_lag, n_seq, result_file, timestart, n_vars)
	result_file.close()

	timestart += chunksize