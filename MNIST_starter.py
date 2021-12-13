import numpy as np
import idx2numpy
import network
import time

# converts a 1d python list into a (1,n) row vector
def rv(vec):
    return np.array([vec])
    
# converts a 1d python list into a (n,1) column vector
def cv(vec):
    return rv(vec).T
        
# creates a (size,1) array of zeros, whose ith entry is equal to 1    
def onehot(i, size):
    vec = np.zeros(size)
    vec[i] = 1
    return cv(vec)


##################################################
# NOTE: make sure these paths are correct for your directory structure

# training data
trainingImageFile = "data/train-images.idx3-ubyte"
trainingLabelFile = "data/train-labels.idx1-ubyte"

# testing data
testingImageFile = "data/t10k-images.idx3-ubyte"
testingLabelFile = "data/t10k-labels.idx1-ubyte"


# returns the number of entries in the file, as well as a list of integers
# representing the correct label for each entry
def getLabels(labelfile):
    file = open(labelfile, 'rb')
    file.read(4)
    n = int.from_bytes(file.read(4), byteorder='big') # number of entries
    
    labelarray = bytearray(file.read())
    labelarray = [b for b in labelarray]    # convert to ints
    file.close()
    
    return n, labelarray

# returns a list containing the pixels for each image, stored as a (784, 1) numpy array
def getImgData(imagefile):
    # returns an array whose entries are each (28x28) pixel arrays with values from 0 to 255.0
    images = idx2numpy.convert_from_file(imagefile) 
    features = []
    # We want to flatten each image from a 28 x 28 to a 784 x 1 numpy array
    # CODE GOES HERE
    for img in images:
        #tempArr = np.reshape(img,784)
        temp1 = img.flatten()
        temp = cv(temp1)
        scaled = temp/255
        
        features.append(scaled)
    
    # convert to floats in [0,1] (only really necessary if you have other features, but we'll do it anyways)
    # CODE GOES HERE
   
    return features

# reads the data from the four MNIST files,
# divides the data into training and testing sets, and encodes the training vectors in onehot form
# returns a tuple (trainingData, testingData), each of which is a zipped array of features and labels
def prepData():
    ntrain, train_labels = getLabels(trainingLabelFile)
    ntest, test_labels = getLabels(testingLabelFile)
    # CODE GOES HERE

    training_features = getImgData(trainingImageFile)
    testing_features = getImgData(testingImageFile)

    trainingLabels = [onehot(label, 10) for label in train_labels]

    trainingData = zip(training_features, trainingLabels)
    testingData = zip(testing_features, test_labels)
    
       
    return (trainingData, testingData)
    

###################################################


trainingData, testingData = prepData()

net = network.Network([784,25,25,25,10])
before = time.perf_counter()
net.SGD(trainingData, 50, 5, 0.75, test_data = testingData)

after = time.perf_counter()

print(f"Finished in {after - before:0.4f} seconds")



#network.saveToFile(net,"part1.pkl")


