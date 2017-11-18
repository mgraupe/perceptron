import numpy as np
import pickle

from network import  Network
from MNISTdata.readMNIST  import readMNIST

# parameters
trainingEpochs = 50
learningRate   = 0.1
classificationNumbers = [0,1]

# read data 
trainingData = readMNIST('training')
testData = readMNIST('testing')

# create network
imgDims = trainingData.getImgDimensions()
perceptron = Network(imgDims[0]*imgDims[1],len(classificationNumbers),learningRate)

learningProgress = []
# learning 
for n in range(trainingEpochs):
    nMissclassified = 0
    nCorrectlyClassified = 0
    for i in range(trainingData.getDSsize()):
        img = trainingData.getBinImg(i)
        if img[0] in classificationNumbers:
            currentOutput = perceptron.getOutput(np.ndarray.flatten(img[1]))
            # update weights for wrong classification
            if classificationNumbers[np.argmax(currentOutput)] != img[0]:
                nMissclassified+=1
                perceptron.updateWeightsBiases([1 if i==img[0] else 0 for i in classificationNumbers],currentOutput,np.ndarray.flatten(img[1]))
            # do nothing if paimg = testData.getBinImg(i)ttern is correctly classified
            else:
                nCorrectlyClassified +=1
    # save performance of learning epoch
    learningProgress.append([nMissclassified,nCorrectlyClassified])
    print 'Learning epoch %s/%s ..... %s %% error rate' % ((n+1),trainingEpochs,nMissclassified*100./float(trainingData.getDSsize()))
    if nMissclassified == 0:
        break
    
# testing
nTestMissclassified = 0
nTestCorrectlyClassified = 0
for i in range(testData.getDSsize()):
    img = testData.getBinImg(i)
    if img[0] in classificationNumbers:
        currentOutput = perceptron.getOutput(np.ndarray.flatten(img[1]))
        if classificationNumbers[np.argmax(currentOutput)] != img[0]:
            #print 'wrong evaluation'classificationNumbers
            nTestMissclassified+=1
        else:
            nTestCorrectlyClassified +=1
print 'Test data set ..... %s %% error rate' % (nTestMissclassified*100./float(testData.getDSsize()))


pickle.dump( learningProgress, open( "learningProgress.p", "wb" ) )
pickle.dump( [nTestMissclassified,nTestCorrectlyClassified], open("testPerformance.p", "wb") )
