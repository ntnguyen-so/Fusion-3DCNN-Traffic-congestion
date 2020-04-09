import numpy as np
import os, fnmatch
import matplotlib.pyplot as plt
from keras import *
import sys
sys.path.append('../')
from utils.logger import Logger

#######################
## Configure dataset ##
#######################
dataset_path = '/mnt/7E3B52AF2CE273C0/Thesis/dataset/dataset/s6_4h_s6_4h'
WD = {
    'input': {
        'factors'    : dataset_path + '/in_seq/',
        'predicted'  : dataset_path + '/out_seq/'    
    },    
    'output': {
        'model_weights' : './training_output/model/',
        'plots'         : './training_output/monitor/'
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
    'Input_congestion'        : 2600,
    'Input_rainfall'          : 131,
    'Input_sns'               : 1,
    'Input_accident'          : 1,
    'default'                 : 2600,
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
GLOBAL_SIZE_Y = [6, 60, 80, 1]

# Get the list of factors data files
print('Loading training data...')
trainDataFiles = fnmatch.filter(os.listdir(WD['input']['factors']), '2014*.npz')
trainDataFiles.sort()
numTrainDataFiles = len(trainDataFiles)
print('Nunber of training data = {0}'.format(numTrainDataFiles))

print('Loading testing data...')
testDataFiles = fnmatch.filter(os.listdir(WD['input']['factors']), '2015*.npz')
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

def appendFactorData(factorName, factorData, X):
    # Load data
    data = factorData[:, :, :, FACTOR[factorName]]
    data = np.expand_dims(data, axis=3)
    data = np.expand_dims(data, axis=0)
    
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

def createBatch(batchSize, dataFiles):
    # Initalize data
    X = {}
    for key in FACTOR.keys():
        X[key] = None
    
    y = {}
    y['default'] = None    
    
    numDataFiles = len(dataFiles)
    i = 0
    while i < batchSize:
        fileId = np.random.randint(low=0, high=int(numDataFiles), size=1)
        fileId = fileId[0]

        areaId = np.random.randint(low=0, high=len(BOUNDARY_AREA), size=1)
        areaId = areaId[0]

        try:
            seqName = dataFiles[fileId]
            factorData = loadDataFile(WD['input']['factors'] + seqName, areaId, 'X')
            predictedData = loadDataFile(WD['input']['predicted'] + seqName, areaId, 'Y')            
                
            if not (factorData is not None and predictedData is not None):
                continue

            # Load factors and predicted data
            for key in FACTOR.keys():
                X = appendFactorData(key, factorData, X)
            
            y = appendFactorData('default', predictedData, y)

        except Exception:
            continue
        
        i += 1

    del X['default']
    return X, y

# loss function: MSE
def mean_squared_error_eval(y_true, y_pred):
    return backend.eval(backend.mean(backend.square(y_pred - y_true)))
    
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

##################################
## Configuring learning process ##
##################################
batchSize = 1
numIterations = 24000#numTrainDataFiles * len(BOUNDARY_AREA) * 2

lr = 5e-5
predictionModel.compile(optimizer=optimizers.Adam(lr=lr, decay=2e-5),
                        loss='mse',
                        metrics=['mse']
                       )

##############
## Training ##
##############
train_logger = Logger('./training_output/tensorboard/train')
test_logger = Logger('./training_output/tensorboard/test')

trainLosses = list()
testLosses = list()
start = 1

for iteration in range(start, numIterations):    
    # ============ Training progress ============#
    while True:
        X, y = createBatch(batchSize, trainDataFiles)
        trainLoss = predictionModel.train_on_batch(X, y['default'])

        if np.sum(X['Input_rainfall']) > 0 or np.sum(X['Input_accident']) > 0:
            break
        else:
            del X
            del y

    # test per epoch
    Xtest, ytest = createBatch(1, testDataFiles)      
    testLoss = predictionModel.test_on_batch(Xtest, ytest['default'])

    # ============ TensorBoard logging ============#
    # Log the scalar values
    train_info = {
        'loss': trainLoss[0],
    }
    test_info = {
        'loss': testLoss[0],
    }

    for tag, value in train_info.items():
        train_logger.scalar_summary(tag, value, step=iteration)
    for tag, value in test_info.items():
        test_logger.scalar_summary(tag, value, step=iteration)
    
    trainLosses.append(trainLoss[0])
    testLosses.append(testLoss[0])    
    print('Iteration: {:7d}; \tTrain_Loss: {:2.10f}; \tTest_Loss: {:2.10f}'.format(iteration, trainLoss[0], testLoss[0]))

    if iteration % 200 == 0:
        ypredicted = predictionModel.predict(Xtest)
        print('Iteration: {:7d}; \tTrain_Loss: {:2.10f}; \tTest_Loss: {:2.10f}; \tSum_GT: {:2.10f}; \tSum_PD: {:2.10f}'.format(
            iteration, trainLoss[0], testLoss[0], np.sum(ytest['default']), np.sum(ypredicted)))

    # save model checkpoint
    if iteration % 3000 == 0:   
        # save model weight
        predictionModel.save_weights(WD['output']['model_weights'] \
                                     + 'epoch_' + str(iteration) + '.h5')

