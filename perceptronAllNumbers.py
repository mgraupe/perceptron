import numpy as np
import pickle
import itertools
import pdb

from network import  Network
from MNISTdata.readMNIST  import readMNIST

# parameters
trainingEpochs = 2
learningRate   = 0.2
pairs = []
voteWeights = np.zeros((10,45))
nPair = 0
for p in itertools.combinations(range(10),2):
    pairs.append(p)
    voteWeights[p[0],nPair] = 1.
    voteWeights[p[1],nPair] = -1.
    nPair+=1


def isDesiredOutput(currentOut,number,pair):
    desiredOut = 1 if number==pair[0] else -1
    if currentOut == desiredOut :
        return [True,desiredOut]
    else:
        return [False,desiredOut]

def getVote(currentOut):
    voteUnits = np.dot(voteWeights,currentOut)
    #print voteUnits, np.argmax(voteUnits)
    return np.where(voteUnits == voteUnits.max())[0]



# read data 
trainingData = readMNIST('training')
testData = readMNIST('testing')

# create network
imgDims = trainingData.getImgDimensions()
network = Network(imgDims[0]*imgDims[1],len(pairs),learningRate)

print 'Train on %s samples, test on %s samples.' % (trainingData.getDSsize(), testData.getDSsize())

learningProgress = []
pairErrorRate = 45*[1]
pairErrorRate.append(45*[1])
# learning 
for n in range(trainingEpochs):
    nPairMissclassified = np.zeros(len(pairs))
    nPairCorrectlyClassified = np.zeros(len(pairs))
    nMissclassified = 0
    nCorrectlyClassified = 0
    for i in range(trainingData.getDSsize()):
        img = trainingData.getBinImg(i)
        currentOutput = network.getOutput(np.ndarray.flatten(img[1]))
        vote = getVote(currentOutput)
        #print vote, img[0]
        if img[0] in vote:
            nCorrectlyClassified += 1
        else:
            nMissclassified += 1
        for j in range(len(pairs)):
            if (img[0] in pairs[j]) and pairErrorRate[-1][j] != 0:
                #print img[0], pairs[j], j
                # check output for single binary discriminator unit
                evaluation = isDesiredOutput(currentOutput[j], img[0], pairs[j])
                if evaluation[0]:
                    nPairCorrectlyClassified[j] += 1
                else:
                    nPairMissclassified[j]+=1
                    network.updateWeights(evaluation[1],np.ndarray.flatten(img[1]),outputUnit=j)
    # save performance of learning epoch
    learningProgress.append([nMissclassified,nCorrectlyClassified,nPairMissclassified,nPairCorrectlyClassified])
    pairErrorRate.append(nPairMissclassified*100./(nPairMissclassified+nPairCorrectlyClassified))
    print 'Learning epoch %s/%s ..... %s %% error rate in %s samples with %s %% pair error rate' % ((n+1),trainingEpochs,nMissclassified*100./float(nCorrectlyClassified+nMissclassified),(nCorrectlyClassified+nMissclassified),nPairMissclassified*100./(nPairMissclassified+nPairCorrectlyClassified))
    if nMissclassified == 0:
        break

#pdb.set_trace()
# testing
nTestMissclassified = 0
nTestCorrectlyClassified = 0
numbersMissclassified = []
for i in range(testData.getDSsize()):
    img = testData.getBinImg(i)
    currentOutput = network.getOutput(np.ndarray.flatten(img[1]))
    vote = getVote(currentOutput)
    if img[0] in vote:
        nTestCorrectlyClassified += 1
    else:
        nTestMissclassified += 1
        numbersMissclassified.append(img[0])

print 'Test data set ..... %s %% error rate in %s samples' % (nTestMissclassified*100./float(nTestCorrectlyClassified+nTestMissclassified),(nTestCorrectlyClassified+nTestMissclassified))


pickle.dump( learningProgress, open( "learningProgress.p", "wb" ) )
pickle.dump( [nTestMissclassified,nTestCorrectlyClassified], open("testPerformance.p", "wb") )
pickle.dump( numbersMissclassified, open('numbersMissclassified.p', 'wb'))
