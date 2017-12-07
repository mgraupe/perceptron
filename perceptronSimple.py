import numpy as np
import pickle

from network import  Network

# parameters
trainingEpochs = 5
learningRate   = 0.2
initalWeights  = np.array([0.,1.,0.5])
inputPoints = ([1.,-2.,1.],[1.,1.,1.],[1.,1.5,-0.5],[-1.,-2.,-1.],[-1.,-1.,-1.5],[-1.,2.,-2.])

def isDesiredOutput(currentOut,number):
    desiredOut = 1 if number in classANumbers else -1
    if currentOut == desiredOut :
        return [True,desiredOut]
    else:
        return [False,desiredOut]

# create network
perceptron = Network(2,1,learningRate,initalWeights)

learningProgress = []
# learning 
for n in range(trainingEpochs):
    nMissclassified = 0
    nCorrectlyClassified = 0
    for i in range(len(inputPoints)):
        currentOutput = perceptron.getOutput(inputPoints[i][1:])
        if currentOutput == inputPoints[i][0]:
            nCorrectlyClassified += 1
        else:
            nMissclassified+=1
            perceptron.updateWeights(inputPoints[i][0],inputPoints[i][1:])
    # save performance of learning epoch
    learningProgress.append([nMissclassified,nCorrectlyClassified])
    print 'Learning epoch %s/%s ..... %s %% error rate' % ((n+1),trainingEpochs,nMissclassified*100./float(len(inputPoints)))
    if nMissclassified == 0:
        break
    
print perceptron.weights
