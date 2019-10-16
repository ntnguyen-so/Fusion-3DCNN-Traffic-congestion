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
dataset_path = '/media/tnnguyen/7E3B52AF2CE273C0/Thesis/Final-Thesis-Output/raster_imgs/CRSA/dataset/medium'
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
    'Input_accident'          : 2,
    'default'                 : 0
}

MAX_FACTOR = {
    'Input_congestion'        : 5000,
    'Input_rainfall'          : 150,
    'Input_accident'          : 1,
    'default'                 : 5000,
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

# Get the list of factors' data files
print('Loading training data...')
trainDataFiles = fnmatch.filter(os.listdir(WD['input']['factors']), '2014*30.npz')
trainDataFiles.sort()
numTrainDataFiles = len(trainDataFiles)
print('Number of training data = {0}'.format(numTrainDataFiles))

print('Loading testing data...')
testDataFiles = fnmatch.filter(os.listdir(WD['input']['factors']), '2015*30.npz')
testDataFiles.sort()
numTestDataFiles = len(testDataFiles)
print('Number of testing data = {0}'.format(numTestDataFiles))

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

##################################
## Configuring learning process ##
##################################
batchSize = 1
numIterations = numTrainDataFiles * len(BOUNDARY_AREA) * 1

lr = 5e-5
predictionModel.compile(optimizer=optimizers.Adam(lr=lr, decay=1e-5),
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
    X, y = createBatch(batchSize, trainDataFiles)
    trainLoss = predictionModel.train_on_batch(X, y['default'])

    # test per epoch
    Xtest, ytest = createBatch(1, testDataFiles)      
    ypredicted = predictionModel.predict(Xtest)
    
    testLoss = mean_squared_error_eval(ytest['default'], ypredicted)

    # ============ TensorBoard logging ============#
    # Log the scalar values
    train_info = {
        'loss': trainLoss[0],
    }
    test_info = {
        'loss': testLoss,
    }

    for tag, value in train_info.items():
        train_logger.scalar_summary(tag, value, step=iteration)
    for tag, value in test_info.items():
        test_logger.scalar_summary(tag, value, step=iteration)
    
    trainLosses.append(trainLoss[0])
    testLosses.append(testLoss)    
    print(iteration, trainLoss[0], testLoss, np.sum(ytest['default']), np.sum(ypredicted))
    
    # save model checkpoint
    if iteration % 100 == 0:   
        # save model weight
        predictionModel.save_weights(WD['output']['model_weights'] \
                                     + 'iteration_' + str(iteration) + '.h5')
