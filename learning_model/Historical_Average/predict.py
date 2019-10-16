import numpy as np
import os, fnmatch
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
from utils.metrics import *

#######################
## Configure dataset ##
#######################
dataset_path = '/mnt/7E3B52AF2CE273C0/Thesis/Final-Thesis-Output/raster_imgs/CRSA/dataset/acm_short'
WD = {
    'input': {
        'factors'    : dataset_path + '/in_seq/',
        'predicted'  : dataset_path + '/out_seq/'    
    },    
    'output': {
        'results'         : './medium.csv'
    }
}

FACTOR = {
    # factor channel index
    'Input_congestion'        : 0,
    'Input_rainfall'          : 1,
    'Input_sns'               : 2,
    'Input_accident'          : 3,   
    'default'                 : 0
}

MAIN_FACTOR = {
    # factor channel index
    'Input_congestion'        : 0,
    'Input_rainfall'          : 1,
    'Input_accident'          : 3
}

MAX_FACTOR = {
    'Input_congestion'        : 2736,
    'Input_rainfall'          : 288,
    'Input_sns'               : 1,
    'Input_accident'          : 1,
    'default'                 : 288,
}

LINK_FACTOR = {
    'Input_congestion'      : (4,5,6,7),
    'Input_rainfall'        : (2,3,6,7),
    'Input_accident'        : (1,3,5,7)
}

BOUNDARY_AREA = {
    0 : [ 20, 80,   50,  100],
    1 : [ 40, 100,  100, 180],
    2 : [ 20, 80,   180, 250]
}

PADDING = {
    0 : [ 0,  60, 30, 80],
    1 : [ 0,  60,  0, 80],
    2 : [ 0,  60,  0, 70]
}

GLOBAL_SIZE_X = [6, 60, 80, 4]
GLOBAL_SIZE_Y = [3, 60, 80, 1]

print('Loading testing data...')
testDataFiles = fnmatch.filter(os.listdir(WD['input']['factors']), '201*30.npz')
testDataFiles.sort()
numSamples = len(testDataFiles)
print('Nunber of testing data = {0}'.format(numSamples))

###########################################
## Load data for training and evaluating ##
###########################################
# Load training data
def loadDataFile(path, areaId, mode):
    try:
        data = np.load(path)
        data = data['arr_0']
    except Exception:
        data = None
    if mode == 'X'  :
        mask = np.zeros(GLOBAL_SIZE_X)
    else:
        mask = np.zeros(GLOBAL_SIZE_Y)
    data = data[:, BOUNDARY_AREA[areaId][0]:BOUNDARY_AREA[areaId][1], BOUNDARY_AREA[areaId][2]:BOUNDARY_AREA[areaId][3], :]
    mask[:, PADDING[areaId][0]:PADDING[areaId][1], PADDING[areaId][2]:PADDING[areaId][3], :] = data

    return mask

def appendFactorData(factorName, factorData, X):
    # Load data
    data = factorData[:, :, :, FACTOR[factorName]]
    data = np.expand_dims(data, axis=3)
    data = np.expand_dims(data, axis=0)
    
    if factorName == 'Input_accident' or factorName == 'Input_sns':
        data[data > 0] = 1
    
    # Standardize data
    data = data.astype(float)
    #data /= MAX_FACTOR[factorName]

    if X[factorName] is None:
        X[factorName] = data
    else:
        X[factorName] = np.vstack((X[factorName], data))

    return X

def loadTestData(dataFiles, fileId, areaId):
    # Initalize data
    X = {}
    for key in FACTOR.keys():
        X[key] = None
    
    y = {}
    y['default'] = None    

    seqName = dataFiles[fileId]
    
    factorData = loadDataFile(WD['input']['factors'] + seqName, areaId, 'X')
    predictedData = loadDataFile(WD['input']['predicted'] + seqName, areaId, 'Y')    

    # Load factors and predicted data
    for key in FACTOR.keys():
        X = appendFactorData(key, factorData, X)
    
    y = appendFactorData('default', predictedData, y)

    del X['default']
    return X, y

# longging
def logging(mode, contentLine):
    f = open(WD['output']['results'], mode)
    f.write(contentLine)
    f.close()
    
#############################
## Build statistical model ##
#############################
def calculateHA(data, predictStep):
    data = data[0]
    predicted = np.average(data, axis=0)
    predicted0 = np.expand_dims(predicted, axis=0)
    predicted = predicted0
    for step in range(predictStep-1):
      predicted = np.concatenate((predicted, predicted0))
    predicted = np.expand_dims(predicted, axis=0)

    return predicted

################
## Evaluation ##
################
header = 'datetime,data_congestion,data_rainfall,data_accident,ground_truth,predicted,err_MSE,err_MAE,err_RMSE'
logging('w', header  + '\n')

max_congestion = 0
max_rainfall = 0
start = 0 
numSamples = numSamples
for fileId in range(start, numSamples):
    for areaId in range(len(BOUNDARY_AREA)):
        Xtest, ytest = loadTestData(testDataFiles, fileId, areaId)

        ypredicted = calculateHA(Xtest['Input_congestion'], 3)

        datetime = str(testDataFiles[fileId].split('.')[0]) + '_' + str(areaId+1)

        data_congestion     = np.sum(Xtest['Input_congestion'] * MAX_FACTOR['Input_congestion'])
        data_rainfall       = np.sum(Xtest['Input_rainfall']   * MAX_FACTOR['Input_rainfall'])
        data_accident       = np.sum(Xtest['Input_accident']   * MAX_FACTOR['Input_accident'])    
        data_sns            = np.sum(Xtest['Input_sns']        * MAX_FACTOR['Input_sns'])    
        
        gt_congestion       = ytest['default']      #* MAX_FACTOR['Input_congestion']
        pd_congestion       = ypredicted            #* MAX_FACTOR['Input_congestion']
        gt_congestion       = np.reshape(gt_congestion, (1, -1))
        pd_congestion       = np.reshape(pd_congestion, (1, -1))

        error_MSE           = mse(gt_congestion, pd_congestion)
        error_MAE           = mae(gt_congestion, pd_congestion)
        error_RMSE          = rmse(gt_congestion, pd_congestion)
        
        results = '{0},{1},{2},{3},{4},\
                {5},{6},\
                {7},{8},{9}'.format(
                    datetime, data_congestion, data_rainfall, data_accident, data_sns,
                    np.sum(gt_congestion), np.sum(pd_congestion),
                    error_MSE, error_MAE, error_RMSE
                )

        #print(results)
        #logging('a', results + '\n')
        if np.max(np.max(Xtest['Input_congestion'])) > max_congestion:
            max_congestion = np.max(np.max(Xtest['Input_congestion']))
            print('max_congestion', max_congestion)

        if np.max(np.max(Xtest['Input_rainfall'])) > max_rainfall:
            max_rainfall = np.max(np.max(Xtest['Input_rainfall']))
            print('max_rainfall', max_rainfall)