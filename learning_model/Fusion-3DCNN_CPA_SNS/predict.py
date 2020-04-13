import numpy as np
import os, fnmatch
import matplotlib.pyplot as plt
from keras import *
import sys
sys.path.append('../')
from utils.metrics import *
from data_config import *

#######################
## Configure dataset ##
#######################
dataset_path = '/mnt/7E3B52AF2CE273C0/Thesis/dataset/dataset/s6_4h_s6_4h'
WD = {
    'input': {
        'test' : {
          'factors'    : dataset_path + '/in_seq/',
          'predicted'  : dataset_path + '/out_seq/'
        },
    'model_weights' : '/mnt/7E3B52AF2CE273C0/Thesis/dataset/source_code/Fusion-3DCNN-Traffic-congestion/learning_model/Fusion-3DCNN_CPA_SNS/training_output/model/epoch_15000.h5'
    },    
    'loss': './evaluation/s6_4h_s6_4h_075.csv'
}


REDUCED_WEIGHT = 0.75

print('Loading testing data...')
testDataFiles = fnmatch.filter(os.listdir(WD['input']['test']['factors']), '2015*30.npz')
testDataFiles.sort()
numTestDataFiles = len(testDataFiles)
print('Nunber of testing data = {0}'.format(numTestDataFiles))

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
    
def fuseFactors(factorName, factorData):
    main_factor = factorData[:, :, :, FACTOR[factorName]]
    main_factor = np.expand_dims(main_factor, axis=3)
    main_factor = np.expand_dims(main_factor, axis=0)
    main_factor = main_factor.astype(float)
    main_factor /= MAX_FACTOR[factorName]

    if factorName == 'Input_accident':
        main_factor[main_factor > 0] = 1

    if factorName != 'default' and factorName != 'Input_sns':
        secondary_factor = factorData[:, :, :, FACTOR['Input_sns']]
        secondary_factor = np.expand_dims(secondary_factor, axis=3)
        secondary_factor = np.expand_dims(secondary_factor, axis=0)        
        secondary_factor = secondary_factor.astype(int)

        for idx in LINK_FACTOR[factorName]:
            main_factor[secondary_factor == idx] = (1-REDUCED_WEIGHT) * main_factor[secondary_factor == idx]
    
    return main_factor

def appendFactorData(factorName, factorData, X):
    data = fuseFactors(factorName, factorData)

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

        cnnOutput = layers.Conv3D(filters=filters[i], kernel_size=kernelSize, strides=1, padding='same', activation='tanh',
                                  name='Conv3D_{0}{1}'.format(factorName, counter))(cnnOutput)
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
                    
        predictionOutput = layers.Conv3D(filters=filters[i], kernel_size=kernelSize, strides=1, padding='same', activation='sigmoid', 
                                         name='Conv3D_prediction{0}1'.format(counter))(predictionOutput)        
        predictionOutput = layers.Conv3D(filters=filters[i], kernel_size=kernelSize, strides=1, padding='same', activation='relu', 
                                         name='Conv3D_prediction{0}2'.format(counter))(predictionOutput)
        
    # predictionOutput = layers.MaxPooling3D(pool_size=(2,1,1), name='output')(predictionOutput)

    predictionOutput = Model(inputs=orgInputs, outputs=predictionOutput)
    return predictionOutput

def buildCompleteModel(imgShape, filtersDict, kernelSizeDict):
    # Define architecture for learning individual factors
    filters = filtersDict['factors']
    kernelSize= kernelSizeDict['factors']
    
    congestionCNNModel   = buildCNN(cnnInputs=None, imgShape=imgShape, filters=filters, kernelSize=kernelSize, factorName='congestion')
    rainfallCNNModel     = buildCNN(cnnInputs=None, imgShape=imgShape, filters=filters, kernelSize=kernelSize, factorName='rainfall')
    accidentCNNModel     = buildCNN(cnnInputs=None, imgShape=imgShape, filters=filters, kernelSize=kernelSize, factorName='accident')

    # Define architecture for fused layers
    filters = filtersDict['factors_fusion']
    kernelSize= kernelSizeDict['factors_fusion']

    fusedCNNModel       = buildCNN(cnnInputs=[congestionCNNModel.input, rainfallCNNModel.input, accidentCNNModel.input],
                                   cnnOutputs=[congestionCNNModel.output, rainfallCNNModel.output, accidentCNNModel.output],
                                   imgShape=imgShape,
                                   filters=filters, kernelSize=kernelSize,
                                   factorName='factors', isFusion=True
                                  )

    # Define architecture for prediction layer
    filters = filtersDict['prediction']
    kernelSize= kernelSizeDict['prediction']
    predictionModel     = buildPrediction(orgInputs=[congestionCNNModel.input, rainfallCNNModel.input, accidentCNNModel.input],
                                          filters=filters,
                                          kernelSize=kernelSize,
                                          lastOutputs=fusedCNNModel.output
                                         )            

    return predictionModel

###############################
## Define model architecture ##
###############################
imgShape = (6,60,80,1)
filtersDict = {}; filtersDict['factors'] = [128, 128, 256]; filtersDict['factors_fusion'] = [256, 256, 256, 128]; filtersDict['prediction'] = [64, 1]
kernelSizeDict = {}; kernelSizeDict['factors'] = (3,3,3); kernelSizeDict['factors_fusion'] = (3,3,3); kernelSizeDict['prediction'] = (3,3,3)

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
header = 'datetime,data_congestion,data_rainfall,data_accident,data_sns_congestion,data_sns_rainfall,data_sns_accident,ground_truth,predicted,err_MSE,err_MAE,err_RMSE'
logging('w', header  + '\n')

start = 0 
numSamples = numTestDataFiles
for fileId in range(start, numSamples):
    for areaId in range(len(BOUNDARY_AREA)):
        Xtest, ytest = loadTestData(testDataFiles, fileId, areaId)
        ypredicted = predictionModel.predict(Xtest)

        datetime = testDataFiles[fileId].split('.')[0] + '_' + str(areaId+1)

        data_congestion     = np.sum(Xtest['Input_congestion'] * MAX_FACTOR['Input_congestion'])
        data_rainfall       = np.sum(Xtest['Input_rainfall']   * MAX_FACTOR['Input_rainfall'])
        data_accident       = np.sum(Xtest['Input_accident']   * MAX_FACTOR['Input_accident'])    
        data_sns            = np.sum(Xtest['Input_sns']        * MAX_FACTOR['Input_sns'])    
        
        data_sns = data_sns.astype(int)
        data_sns_congestion = np.zeros_like(data_sns)
        data_sns_rainfall = np.zeros_like(data_sns)
        data_sns_accident = np.zeros_like(data_sns)
        for msg_type in LINK_FACTOR['Input_congestion']:
            data_sns_congestion[data_sns == msg_type] = 1
        for msg_type in LINK_FACTOR['Input_rainfall']:
            data_sns_rainfall[data_sns == msg_type] = 1
        for msg_type in LINK_FACTOR['Input_accident']:
            data_sns_accident[data_sns == msg_type] = 1
        data_sns_congestion = np.sum(data_sns_congestion)
        data_sns_rainfall = np.sum(data_sns_rainfall)
        data_sns_accident = np.sum(data_sns_accident)

        
        gt_congestion       = ytest['default']      * MAX_FACTOR['Input_congestion']
        pd_congestion       = ypredicted            * MAX_FACTOR['Input_congestion']
        gt_congestion       = np.reshape(gt_congestion, (1, -1))
        pd_congestion       = np.reshape(pd_congestion, (1, -1))

        error_MSE           = mse(gt_congestion, pd_congestion)
        error_MAE           = mae(gt_congestion, pd_congestion)
        error_RMSE          = rmse(gt_congestion, pd_congestion)
        
        results = '{0},{1},{2},{3},\
                {4},{5},{6},\
                {7},{8},\
                {9},{10},{11}'.format(
                    datetime, data_congestion, data_rainfall, data_accident, 
                    data_sns_congestion, data_sns_rainfall, data_sns_accident,
                    np.sum(gt_congestion), np.sum(pd_congestion),
                    error_MSE, error_MAE, error_RMSE
                )

        print(results)
        logging('a', results + '\n')
