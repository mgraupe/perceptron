import numpy as np
import pickle

from network import  Network
from MNISTdata.readMNIST  import readMNIST

# parameters
trainingEpochs = 50
learningRate   = 0.2
classANumbers = [1]
classBNumbers = [2]

def isDesiredOutput(currentOut,number):
    desiredOut = 1 if number in classANumbers else -1
    if currentOut == desiredOut :
        return [True,desiredOut]
    else:
        return [False,desiredOut]


classificationNumbers =classANumbers + classBNumbers

# read data 
trainingData = readMNIST('training')
testData = readMNIST('testing')

# create network
imgDims = trainingData.getImgDimensions()
perceptron = Network(imgDims[0]*imgDims[1],1,learningRate)

print 'Train on %s samples, test on %s samples.' % (trainingData.getDSsize()*len(classificationNumbers)/10, testData.getDSsize()*len(classificationNumbers)/10)

learningProgress = []
# learning 
for n in range(trainingEpochs):
    nMissclassified = 0
    nCorrectlyClassified = 0
    for i in range(trainingData.getDSsize()):
        img = trainingData.getBinImg(i)
        if img[0] in classificationNumbers:
            currentOutput = perceptron.getOutput(np.ndarray.flatten(img[1]))
            evaluation = isDesiredOutput(currentOutput, img[0] )
            if evaluation[0]:
                nCorrectlyClassified += 1
            else:
                nMissclassified+=1
                perceptron.updateWeights(evaluation[1],np.ndarray.flatten(img[1]))
    # save performance of learning epoch
    learningProgress.append([nMissclassified,nCorrectlyClassified])
    print 'Learning epoch %s/%s ..... %s %% error rate in %s samples' % ((n+1),trainingEpochs,nMissclassified*100./float(nCorrectlyClassified+nMissclassified),(nCorrectlyClassified+nMissclassified))
    if nMissclassified == 0:
        break
    
# testing
nTestMissclassified = 0
nTestCorrectlyClassified = 0
for i in range(testData.getDSsize()):
    img = testData.getBinImg(i)
    if img[0] in classificationNumbers:
        currentOutput = perceptron.getOutput(np.ndarray.flatten(img[1]))
        evaluation = isDesiredOutput(currentOutput, img[0])
        if evaluation[0]:
            nTestCorrectlyClassified += 1
        else:
            nTestMissclassified+=1

print 'Test data set ..... %s %% error rate in %s samples' % (nTestMissclassified*100./float(nTestCorrectlyClassified+nTestMissclassified),(nTestCorrectlyClassified+nTestMissclassified))


pickle.dump( learningProgress, open( "learningProgress.p", "wb" ) )
pickle.dump( [nTestMissclassified,nTestCorrectlyClassified], open("testPerformance.p", "wb") )
