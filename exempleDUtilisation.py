from MNISTdata.readMNIST  import readMNIST
import matplotlib.pylab as plt

# creer des instances de la classe pour les donnees de formation et de test
trainingData = readMNIST('training')
testData = readMNIST('testing')

# obtenir la taille de l'ensemble de donnees
print(trainingData.getDSsize())
print(testData.getDSsize())

# obtenir la taille d'une seule image : pixel x pixel
print(testData.getImgDimensions())

# obtenir une image : numero de l'image comme argument d'entree [0,59999]
# image est renvoyee en encodage 8 bits
img = trainingData.getImg(0)
# image est renvoyee en encodage binaire
img = trainingData.getBinImg(0)

# premier element en tuple conserve le label de l'image [0,..., 9] 
print(img[0])
# le deuxieme element du tuple contient l'image elle-meme
#plt.imshow(img[1])

