import sys
sys.path.append('../..')

import pandas as pd
import numpy as np
from sklearn import metrics as M
from statsmodels.tsa.vector_ar.var_model import VAR
from utils.metrics import *
import warnings
warnings.filterwarnings("ignore")


#######################
## Configure dataset ##
#######################
step = 0
result_files = ['short.csv', 'medium.csv', 'long.csv']
lookbacks    = [6, 12, 48]
future_steps = [3, 3, 6]
jumps        = [1, 2, 8]
data_file = 'timeseries_avg.csv'

# logging
def logging(resultFile, mode, contentLine):
    f = open(resultFile, mode)
    f.write(contentLine)
    f.close()
    
################
## Evaluation ##
################
# Load and define evaluation point
dataset = pd.read_csv(data_file, engine='python', header=None)
print(dataset.head())
dataset = dataset.to_numpy()
dataset = dataset.astype(float)
print(dataset.shape)
hours = dataset[:, 0]
start_idx = np.argwhere(hours == 201505100000)# 201505100000)
start_idx = start_idx[0,0]
print(start_idx)

while step < len(result_files):
    result_file_id = step
    for hour in range(start_idx, len(hours)):    
        # get past congestion statistics
        history_congestion = dataset[hour-lookbacks[result_file_id] : hour : jumps[result_file_id], 1:]

        # form ARIMA model, and perform prediction
        model = VAR(history_congestion)
        model_fit = model.fit(trend='nc') # nc no constant and trend
        pd_congestion = model_fit.forecast(model_fit.y, steps=max(future_steps))

        
        gt_congestion = dataset[hour : hour+future_steps[result_file_id]*jumps[result_file_id] : jumps[result_file_id], 1:]
        pd_congestion_local = pd_congestion[:future_steps[result_file_id]]            

        # evalute
        error_MSE           = mse(gt_congestion, pd_congestion_local)
        error_MAE           = mae(gt_congestion, pd_congestion_local)
        error_RMSE          = rmse(gt_congestion, pd_congestion_local)

        result_line = '{0},\
                {1},{2},{3}'.format(str(int(dataset[hour, 0])) ,
                                        error_MSE, error_MAE, error_RMSE)
        print(result_line)
        logging(result_files[result_file_id], 'a',  result_line + '\n')
    
    step += 1