import numpy as np
import pickle

from network import  Network
from MNISTdata.readMNIST  import readMNIST

# parameters
trainingEpochs = 50
learningRate   = 0.2
classANumbers = [0]
classBNumbers = [7]

def isDesiredOutput(currentOut,number,inp):
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

print 'Train on %s samples, test on %s samples.' % (trainingData.getDSsize(), testData.getDSsize())

learningProgress = []
# learning 
for n in range(trainingEpochs):
    nMissclassified = 0
    nCorrectlyClassified = 0
    for i in range(trainingData.getDSsize()):
        img = trainingData.getBinImg(i)
        if img[0] in classificationNumbers:
            inp = np.concatenate((([1]),np.ndarray.flatten(img[1])))
            currentOutput = perceptron.getOutput(inp)
            evaluation = isDesiredOutput(currentOutput, img[0], inp)
            #print evaluation[0], evaluation[1]
            if evaluation[0]:
                nCorrectlyClassified += 1
            else:
                nMissclassified+=1
                perceptron.updateWeights(evaluation[1],inp)
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
        inp = np.concatenate((([1]), np.ndarray.flatten(img[1])))
        currentOutput = perceptron.getOutput(inp)
        evaluation = isDesiredOutput(currentOutput, img[0], inp)
        if evaluation[0]:
            nTestCorrectlyClassified += 1
        else:
            nTestMissclassified+=1

print 'Test data set ..... %s %% error rate' % (nTestMissclassified*100./float(testData.getDSsize()))


pickle.dump( learningProgress, open( "learningProgress.p", "wb" ) )
pickle.dump( [nTestMissclassified,nTestCorrectlyClassified], open("testPerformance.p", "wb") )
