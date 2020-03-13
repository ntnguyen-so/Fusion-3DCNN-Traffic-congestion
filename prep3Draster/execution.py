import numpy as np
import os
from config import *
import sys
import fnmatch

def extractFactorMap(path):
    iCorrupted = False
    factorsMap = None

    try:
        factorsMap = np.load(path)
        factorsMap = factorsMap['arr_0']
    except Exception:
        # sometime, a npz file gets corrupted
        iCorrupted = True

    return iCorrupted, factorsMap

def extractSequence(factorsFiles, start, steps, lookback=False, delta=1):
    seq = None
    iCorrupted = False

    if lookback == True:
        loopRange = range(-steps*delta, 0, delta)
    else:
        loopRange = range(0, steps*delta, delta)

    for i in loopRange:
        factorFilename = WD['input']['sequence_prepdataset'] + factorsFiles[start + i]
        #print('--', lookback, factorFilename)
        iCorrupted, factorsMap = extractFactorMap(factorFilename)
        if iCorrupted == True:
            return None

        factorsMap = np.expand_dims(factorsMap, axis=0)

        if seq is None:
            seq = factorsMap
        else:
            seq = np.vstack((seq, factorsMap))

    return seq

def extractInOutSequence(factorsFiles, start, inSteps, outSteps, outFactor=None, crop=None):
    inputSeq = extractSequence(factorsFiles, start, inSteps, lookback=True, delta=SEQUENCE['inp_delta'])
    outputSeq = extractSequence(factorsFiles, start, outSteps, delta=SEQUENCE['out_delta'])

    iCorrupted = False
    if not(inputSeq is not None and outputSeq is not None):
        iCorrupted = True

    if crop is not None:
        if inputSeq is not None and outputSeq is not None:
            inputSeq = inputSeq[:, crop['xS'] : crop['xE'], crop['yS'] : crop['yE'], :]
            outputSeq = outputSeq[:, crop['xS'] : crop['xE'], crop['yS'] : crop['yE'], :]

    if outFactor is not None:
        outputSeq = outputSeq[:, :, :, outFactor]
        outputSeq = np.expand_dims(outputSeq, axis=3)

    return inputSeq, outputSeq, iCorrupted
    
if __name__ == '__main__':    
    inputFiles = os.listdir(WD['input']['sequence_prepdataset'])
    inputFiles.sort()  
    #print(inputFiles)

    #determineCropArea(inputFiles)    
    s = int(sys.argv[1]) + SEQUENCE['inp_len']*SEQUENCE['inp_delta']
    e = int(sys.argv[2]) + s

    for start in range(s, e):
        #if '30.npz' not in inputFiles[start]:
        #    continue
        startingTime = inputFiles[start].split('.')[0]
        print('Extracting {0}...'.format(startingTime))

        # extract in/out sequence
        inSeq, outSeq, iCorrupted = extractInOutSequence(inputFiles, start, SEQUENCE['inp_len'], SEQUENCE['out_len'], SEQUENCE['out_factor'], SEQUENCE['crop'])

        # sometime, the sequence gets corrupted
        if iCorrupted == True:
            continue

        # store data
        sequenceFilename = WD['output']['sequence_prepdataset'] + startingTime + '_'
        np.savez_compressed(WD['output']['sequence_prepdataset'] + 'in_seq/' + startingTime, inSeq)
        np.savez_compressed(WD['output']['sequence_prepdataset'] + 'out_seq/' + startingTime, outSeq)

