import numpy as np


class Network(object):

    def __init__(self, inputLayerSize, outputLayerSize, learningRate):
        #self.num_layers = len(sizes)
        #self.sizes = sizes
        self.eta     = learningRate
        self.biases  = np.random.rand(outputLayerSize)
        self.weights = np.random.rand(outputLayerSize,inputLayerSize)
        self.inputLayerSize  = inputLayerSize
        self.outputLayerSize = outputLayerSize 
        #print np.shape(self.biases), np.shape(self.weights)
    
    def getOutput(self,a):
        return (np.dot(self.weights,a)+self.biases)/(self.inputLayerSize+1.)
    
    def updateWeightsBiases(self,desiredOutput,actualOutput,inputPattern):
        for i in range(self.outputLayerSize):
            self.weights[i] = self.weights[i] + self.eta*(desiredOutput[i]-actualOutput[i])*inputPattern
            self.biases[i]  = self.biases[i]  + self.eta*(desiredOutput[i]-actualOutput[i])
        
    
    
