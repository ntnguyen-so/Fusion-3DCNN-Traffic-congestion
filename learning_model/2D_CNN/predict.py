import numpy as np
import os, fnmatch
import matplotlib.pyplot as plt
from keras import *
import sys
sys.path.append('../')
from utils.metrics import *

#######################
## Configure dataset ##
#######################
dataset_path = '/media/tnnguyen/7E3B52AF2CE273C0/Thesis/Final-Thesis-Output/raster_imgs/CRSA/dataset/medium'
WD = {
    'input': {
        'test' : {
          'factors'    : dataset_path + '/in_seq/',
          'predicted'  : dataset_path + '/out_seq/'
        },
    'model_weights' : './training_output/weights_medium.h5'
    },    
    'loss': './evaluation/medium.csv'
}

FACTOR = {
    # factor channel index
    'Input_congestion'        : 0,
    'Input_rainfall'          : 1,
    'Input_sns'               : 2,
    'Input_accident'          : 3,
    'default'                 : 0
}

MAX_FACTOR = {
    'Input_congestion'        : 5000,
    'Input_rainfall'          : 150,
    'Input_sns'               : 1,
    'Input_accident'          : 1,
    'default'                 : 5000,
}

BOUNDARY_AREA = {
    0 : [ 20, 80,   50,  100],
    1 : [ 40, 100,  100, 180],
    2 : [ 20, 80,   180, 250]
}
#imgShape = (6,60,80,1)
PADDING = {
    0 : [ 0,  60, 30, 80],
    1 : [ 0,  60,  0, 80],
    2 : [ 0,  60,  0, 70]
}

GLOBAL_SIZE_X = [6, 60, 80, 4]
GLOBAL_SIZE_Y = [3, 60, 80, 1]

print('Loading testing data...')
testDataFiles = fnmatch.filter(os.listdir(WD['input']['test']['factors']), '2015*30.npz')
testDataFiles.sort()
numSamples = len(testDataFiles)
print('Nunber of testing data = {0}'.format(numSamples))

###########################################
## Load data for training and evaluating ##
###########################################
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
    data = np.swapaxes(data, 0, 3)
    
    if factorName == 'Input_accident' or factorName == 'Input_sns':
        data[data > 0] = 1
    
    # Standardize data
    data = data.astype(float)
    data /= MAX_FACTOR[factorName]

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
    
    factorData = loadDataFile(WD['input']['test']['factors'] + seqName, areaId, 'X')
    predictedData = loadDataFile(WD['input']['test']['predicted'] + seqName, areaId, 'Y')    

    # Load factors and predicted data
    for key in FACTOR.keys():
        X = appendFactorData(key, factorData, X)
    
    y = appendFactorData('default', predictedData, y)

    del X['default']
    return X, y

def logging(mode, contentLine):
    f = open(WD['loss'], mode)
    f.write(contentLine)
    f.close()

##########################
## Build learning model ##
##########################
def buildCNN(cnnInputs, imgShape, filters, kernelSize, factorName, isFusion=False, cnnOutputs=None):
    if isFusion == True:
        cnnInput = layers.add(cnnOutputs, name='Fusion_{0}'.format(factorName))
    else:
        cnnInput = layers.Input(shape=imgShape, name='Input_{0}'.format(factorName))

    for i in range(len(filters)):
        counter = i+1
        if i == 0:
            cnnOutput = cnnInput

        cnnOutput = layers.Conv2D(filters=filters[i], kernel_size=kernelSize, strides=1, padding='same', activation='tanh',
                                  name='Conv2D_{0}{1}'.format(factorName, counter))(cnnOutput)
        cnnOutput = layers.BatchNormalization(name='BN_{0}{1}'.format(factorName, counter))(cnnOutput)
    
    if cnnInputs is not None:
        cnnModel = Model(inputs=cnnInputs, outputs=cnnOutput)
    else:
        cnnModel = Model(inputs=cnnInput, outputs=cnnOutput)
    return cnnModel
    
def buildPrediction(orgInputs, filters, kernelSize, lastOutputs=None):
    predictionOutput = None
    for i in range(len(filters)):
        counter = i + 1
        if i == 0:
            if lastOutputs is not None:
                predictionOutput = lastOutputs
            else:
                predictionOutput = orgInputs
                    
        predictionOutput = layers.Conv2D(filters=filters[i], kernel_size=kernelSize, strides=1, padding='same', activation='sigmoid', 
                                         name='Conv2D_prediction{0}1'.format(counter))(predictionOutput)        
        predictionOutput = layers.Conv2D(filters=filters[i], kernel_size=kernelSize, strides=1, padding='same', activation='relu', 
                                         name='Conv2D_prediction{0}2'.format(counter))(predictionOutput)
        
    predictionOutput = Model(inputs=orgInputs, outputs=predictionOutput)
    return predictionOutput

def buildCompleteModel(imgShape, filtersDict, kernelSizeDict):
    # Define architecture for learning individual factors
    filters = filtersDict['factors']
    kernelSize= kernelSizeDict['factors']

    filtersCongestion = list()
    for filter in range(len(filters)-1):
        filtersCongestion.append(int(filters[filter]*1.0))
    filtersCongestion.append(filters[-1])
    
    print(filtersCongestion)
    congestionCNNModel   = buildCNN(cnnInputs=None, imgShape=imgShape, filters=filtersCongestion, kernelSize=kernelSize, factorName='congestion')

    # Define architecture for prediction layer
    filters = filtersDict['prediction']
    kernelSize= kernelSizeDict['prediction']
    predictionModel     = buildPrediction(orgInputs=[congestionCNNModel.input],
                                          filters=filters,
                                          kernelSize=kernelSize,
                                          lastOutputs=congestionCNNModel.output
                                         )            

    return predictionModel

###############################
## Define model architecture ##
###############################
imgShape = (60,80,6)
targetImgShape=(60,80,3)
filtersDict = {}; filtersDict['factors'] = [128, 128, 256, 256, 256, 256, 128]; filtersDict['prediction'] = [64, targetImgShape[2]]
kernelSizeDict = {}; kernelSizeDict['factors'] = (3,3); kernelSizeDict['prediction'] = (3,3)

predictionModel = buildCompleteModel(imgShape, filtersDict, kernelSizeDict)
predictionModel.summary()
utils.plot_model(predictionModel,to_file='architecture.png',show_shapes=True)

#################################
## Load weights for prediction ##
#################################
predictionModel = buildCompleteModel(imgShape, filtersDict, kernelSizeDict)
predictionModel.load_weights(WD['input']['model_weights'])

################
## Evaluation ##
################
header = 'datetime,data_congestion,data_rainfall,data_accident,data_sns,ground_truth,predicted,err_MSE,err_MAE,err_RMSE'
logging('w', header  + '\n')

start = 0 
numSamples = numSamples
for fileId in range(start, numSamples):
    for areaId in range(len(BOUNDARY_AREA)):
        Xtest, ytest = loadTestData(testDataFiles, fileId, areaId)
        ypredicted = predictionModel.predict(Xtest)

        datetime = testDataFiles[fileId].split('.')[0] + '_' + str(areaId+1)

        data_congestion     = np.sum(Xtest['Input_congestion'] * MAX_FACTOR['Input_congestion'])
        data_rainfall       = np.sum(Xtest['Input_rainfall']   * MAX_FACTOR['Input_rainfall'])
        data_accident       = np.sum(Xtest['Input_accident']   * MAX_FACTOR['Input_accident'])    
        data_sns            = np.sum(Xtest['Input_sns']        * MAX_FACTOR['Input_sns'])    
        
        gt_congestion       = ytest['default']      * MAX_FACTOR['Input_congestion']
        pd_congestion       = ypredicted            * MAX_FACTOR['Input_congestion']
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

        print(results)
        logging('a', results + '\n')
