import numpy as np
import math
from sklearn import metrics as M

# mean square error
def mse(gt, pd):
    return M.mean_squared_error(gt, pd)

# mean absolute error
def mae(gt, pd):
    return M.mean_absolute_error(gt, pd)

# root mean square error
def rmse(gt, pd):
    return math.sqrt(mse(gt, pd))
