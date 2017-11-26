import numpy as np


class Network(object):

    def __init__(self, inputLayerSize, outputLayerSize, learningRate,initialWeights=None):
        #self.num_layers = len(sizes)
        #self.sizes = sizes
        self.eta     = learningRate
        #self.biases  = np.random.rand(outputLayerSize)
        if not (initialWeights is None):
            self.weights = initialWeights
        else:
            self.weights = (-1.+np.random.rand(outputLayerSize,inputLayerSize+1)*2.)
        self.inputLayerSize  = inputLayerSize
        self.outputLayerSize = outputLayerSize 
        #print np.shape(self.biases), np.shape(self.weights)
    
    def getOutput(self,a,outputUnit=None):
        aPrime = np.concatenate((([1]),a))
        if outputUnit: # in case only specific output unit is required
            return np.sign(np.dot(self.weights[outputUnit],aPrime))
        else:
            return np.sign(np.dot(self.weights, aPrime))
    
    def updateWeights(self,desiredOutput,inputPattern,outputUnit=None):
        #for i in range(self.outputLayerSize):
        inputPatternPrime = np.concatenate((([1]),inputPattern))
        if outputUnit:  # in case only specific output unit is required
            self.weights[outputUnit] = self.weights[outputUnit] + self.eta * desiredOutput * inputPatternPrime
        else:
            self.weights = self.weights + self.eta*desiredOutput*inputPatternPrime

    
    
