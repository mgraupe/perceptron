import numpy as np
import pickle

from network import  Network
from MNISTdata.readMNIST  import readMNIST

trainingEpochs = 100000


trainingData = readMNIST('training')
testData = readMNIST('testing')

imgDims = trainingData.getImgDimensions()
perceptron = Network(imgDims[0]*imgDims[1],10,0.1)

learningProgress = []
# learning 
for n in range(trainingEpochs):
    nMissclassified = 0
    nCorrectlyClassified = 0
    for i in range(trainingData.getDSsize()):
        img = trainingData.getBinImg(i)
        currentOutput = perceptron.getOutput(np.ndarray.flatten(img[1]))
        # update weights for wrong classification
        if np.argmax(currentOutput) != img[0]:
            nMissclassified+=1
            perceptron.updateWeightsBiases([1 if i==img[0] else 0 for i in range(10)],currentOutput,np.ndarray.flatten(img[1]))
        # do nothing if pattern is correctly classified
        else:
            nCorrectlyClassified +=1
    # save performance of learning epoch
    learningProgress.append([nMissclassified,nCorrectlyClassified])
    
# testing
nTestMissclassified = 0
nTestCorrectlyClassified = 0
for i in range(testData.getDSsize()):
    img = testData.getBinImg(i)
    currentOutput = perceptron.getOutput(np.ndarray.flatten(img[1]))
    if np.argmax(currentOutput) != img[0]:
        #print 'wrong evaluation'
        nTestMissclassified+=1
    else:
        nTestCorrectlyClassified +=1


pickle.dump( learningProgress, open( "learningProgress.p", "wb" ) )
pickel.dump( [nTestMissclassified,nTestCorrectlyClassified], open("testPerformance.p", "wb") )
