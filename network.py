import numpy as np


class Network(object):

    def __init__(self, inputLayerSize, outputLayerSize, learningRate):
        #self.num_layers = len(sizes)
        #self.sizes = sizes
        self.eta     = learningRate
        #self.biases  = np.random.rand(outputLayerSize)
        self.weights = np.random.rand(outputLayerSize,inputLayerSize+1)
        self.inputLayerSize  = inputLayerSize
        self.outputLayerSize = outputLayerSize 
        #print np.shape(self.biases), np.shape(self.weights)
    
    def getOutput(self,a):
        aPrime = np.concatenate((([1]),a))
        return np.sign(np.dot(self.weights,aPrime))
    
    def updateWeights(self,desiredOutput,inputPattern):
        #for i in range(self.outputLayerSize):
        inputPatternPrime = np.concatenate((([1]),inputPattern))
        self.weights = self.weights + self.eta*desiredOutput*inputPatternPrime

    
    
