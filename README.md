# Perceptron project

The script implements a Perceptron learning to classify handwritten digits from 0 to 9. The MNIST database of handwritten digits (from http://yann.lecun.com/exdb/mnist/) is used. The dataset has a training set of 60,000 examples, and a test set of 10,000 examples. The digits have been size-normalized and centered in a fixed-size image.

Three implementations of the Perceptron exist in this repository.

### Perceptron Simple

``perceptronSimple.py`` implements a simple classification of points in the 2D plane.


### Perceptron Two Numbers

``perceptronTwoNumbers.py`` implements the classification of a pair of numbers (can be chosen in the beginning of the script) from the MNIST database. The script uses images of those to number from the training dataset to learn the classificaiton. Performance is subsequently tested using the test dataset.

### Perceptron All Numbers

``perceptronAllNumbers.py`` implements the classification of all numbers [0,...,9] present in the MNIST dataset. The script uses the pairwise classification of ``perceptronTwoNumbers.py`` and adds an additional voting layer, which counts votes from the pairwise discrimination units.


#### Plotting

``plotResults.py`` allows to display performance of the 10 number discrimination implemented in ``perceptronAllNumbers.py``.