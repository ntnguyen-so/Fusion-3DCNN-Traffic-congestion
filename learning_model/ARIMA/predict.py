import sys
sys.path.append('../')

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
result_files = ['short.csv', 'medium_.csv', 'long.csv']
future_steps = [3, 6, 48]
data_file = 'timeseries_short.csv'

area_size = 4000
num_areas = 3

# logging
def logging(resultFile, mode, contentLine):
    f = open(resultFile, mode)
    f.write(contentLine)
    f.close()
    
################
## Evaluation ##
################
# Write header
for result_file in result_files:
    logging(result_file, 'w', 'datetime,err_MSE,err_MAE,err_RMSE\n')

# Load and define evaluation point
dataset = pd.read_csv(data_file, engine='python', header=None)
dataset = dataset.to_numpy()
dataset = dataset.astype(float)
hours = dataset[:, 0]
start_idx = np.argwhere(hours == 201505010030)
start_idx = start_idx[0,0]

for hour in range(start_idx, len(hours)):    
    # get past congestion statistics of 10 days
    history_congestion = dataset[hour-960:hour, 1:]

    for area_id in range(num_areas):
        history_congestion_area = history_congestion[:, area_id*area_size : (area_id+1)*area_size]

        # form ARIMA model, and perform prediction
        model = VAR(history_congestion_area)
        model_fit = model.fit(trend='ctt') #"ctt" - constant, linear and quadratic trend
        pd_congestion = model_fit.forecast(model_fit.y, steps=max(future_steps))

        for result_file_id in range(len(result_files)):
                gt_congestion = dataset[hour : hour+future_steps[result_file_id], 
                                        1 + area_id*area_size: 1 + (area_id+1)*area_size
                                        ]
                pd_congestion_local = pd_congestion[:future_steps[result_file_id]]                

                # evalute
                error_MSE           = mse(gt_congestion, pd_congestion_local)
                error_MAE           = mae(gt_congestion, pd_congestion_local)
                error_RMSE          = rmse(gt_congestion, pd_congestion_local)

                result_line = '{0},\
                        {1},{2},{3}'.format(str(int(dataset[hour, 0])) + '_' + str(area_id+1), 
                                                error_MSE, error_MAE, error_RMSE)
                print(result_line)
                logging(result_files[result_file_id], 'a',  result_line + '\n')
