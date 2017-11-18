import os
import struct
import numpy as np
from matplotlib import pyplot
import matplotlib as mpl
        
class readMNIST:
    def __init__(self,dataset = "training", path = "."):
        """
        Python class for importing the MNIST data set.  The 'getImg' function returns a 
        tuple with the first element being the label and the second element
        being a numpy.uint8 2D array of pixel data for the given image. In the 
        'getBinImg' function, the image is return as binary array. 
        """

        if dataset is "training":
            fname_img = os.path.join(path, 'MNISTdata/train-images.idx3-ubyte')
            fname_lbl = os.path.join(path, 'MNISTdata/train-labels.idx1-ubyte')
        elif dataset is "testing":
            fname_img = os.path.join(path, 'MNISTdata/t10k-images.idx3-ubyte')
            fname_lbl = os.path.join(path, 'MNISTdata/t10k-labels.idx1-ubyte')
        else:
            raise ValueError, "dataset must be 'testing' or 'training'"

        # Load everything in some numpy arrays
        with open(fname_lbl, 'rb') as flbl:
            magic, num = struct.unpack(">II", flbl.read(8))
            self.lbl = np.fromfile(flbl, dtype=np.int8)

        with open(fname_img, 'rb') as fimg:
            magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
            self.img = np.fromfile(fimg, dtype=np.uint8).reshape(len(self.lbl), rows, cols)
        
    def getImg(self, idx):
        if idx < len(self.lbl):
            return (self.lbl[idx], self.img[idx])
        else:
            print 'index out of data-set range'
    
    def getImgDimensions(self):
        sampleImg = self.getBinImg(0)
        return np.shape(sampleImg[1])
    
    def getBinImg(self, idx):
        if idx < len(self.lbl):
            # threshold the gray-scale image
            imgb = self.img[idx]>=256./2.
            # convert the boolean array into array of zeros and ones
            imgb = np.array(imgb,dtype=int)
            return (self.lbl[idx], imgb)
        else:
            print 'index out of data-set range'
            
    def getDSsize(self):
        return len(self.lbl)
        # Create an iterator which returns each image in turn
        #for i in xrange(len(lbl)):
        #    yield get_img(i)
    def showImg(self,image):
        """
        Render a given numpy.uint8 2D array of pixel data.
        """
        fig = pyplot.figure()
        ax = fig.add_subplot(1,1,1)
        imgplot = ax.imshow(image, cmap=mpl.cm.Greys)
        imgplot.set_interpolation('nearest')
        ax.xaxis.set_ticks_position('top')
        ax.yaxis.set_ticks_position('left')
        pyplot.show()
    
