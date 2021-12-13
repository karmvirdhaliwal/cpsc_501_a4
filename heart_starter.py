import csv
import numpy as np
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

# given a data point, mean, and standard deviation, returns the z-score
def standardize(x, mu, sigma):
    return ((x - mu)/sigma)
    

##############################################

# reads number of data points, feature vectors and their labels from the given file
# and returns them as a tuple
def getDataFromSample(sample):

    # SBP
    sbpMean = 138.3
    sbpStdDev = 20.5
    sbp = cv([standardize(float(sample[1]),sbpMean, sbpStdDev)])

    # tobacco
    tobaccoMean = 3.64
    tobaccoStdDev = 4.59
    tobacco = cv([standardize(float(sample[2]),tobaccoMean, tobaccoStdDev)])
    
    # ldl
    ldlMean = 4.74
    ldlStdDev = 2.07
    ldl = cv([standardize(float(sample[3]),ldlMean, ldlStdDev)])
    
    # adiposity
    adiposityMean = 25.4
    adiposityStdDev = 7.77
    adiposity = cv([standardize(float(sample[4]), adiposityMean, adiposityStdDev)])
    
    # famhist
    if(sample[5] == "Present"):
        famhist = cv([1])
    elif(sample[5] == "Absent"):
        famhist = cv([0])
    #else:
        #print("Data processing error. Exiting program.")
        #quit()  

    #typea
    typeaMean = 53.1
    typeaStdDev = 9.81
    typea = cv([standardize(float(sample[6]), typeaMean, typeaStdDev)])


    #obesity
    obesityMean = 26.0
    obesityStdDev = 4.21
    obesity = cv([standardize(float(sample[7]), obesityMean, obesityStdDev)])

    #alcohol
    alcoholMean = 17.0
    alcoholStdDev = 24.5
    alcohol = cv([standardize(float(sample[8]), alcoholMean, alcoholStdDev)])

    #age
    ageFromFile = int(sample[9])
    resize = float(ageFromFile/64)
    age = cv([resize])

    # concatenate results to get feature vector
    features = np.concatenate((sbp, tobacco, ldl, adiposity, famhist, typea, obesity, alcohol, age), axis=0)

    label = int(sample[10])
    
    # return as a tuple
    return (features, label)

def readData(filename):

    with open(filename, newline='') as datafile:
        reader = csv.reader(datafile)        
        next(reader, None)  # skip the header row
        
        n = 0
        features = []
        labels = []
        
        for row in reader:
            featureVec, label = getDataFromSample(row)
            features.append(featureVec)
            labels.append(label)
            n = n + 1

    print(f"Number of data points read: {n}")
    
    #print(features)
    #print(labels)
    
    return (n, features, labels)



################################################

# reads the data from the heart.csv file,
# divides the data into training and testing sets, and encodes the training vectors in onehot form
# returns a tuple (trainingData, testingData), each of which is a zipped array of features and labels
def prepData():

    n, features, labels = readData('data/heart.csv')

    ntrain = int(n * 5/6)    
    ntest = n - ntrain

    trainingFeatures = features[:ntrain]
    trainingLabels = [onehot(label, 2) for label in labels[:ntrain]]

    testingFeatures = features[ntrain:]
    testingLabels = labels[ntrain:]

    trainingData = zip(trainingFeatures, trainingLabels)
    testingData = zip(testingFeatures, testingLabels)

    return (trainingData, testingData)


###################################################


trainingData, testingData = prepData()

net = network.Network([9,10,10,10,2])
before = time.perf_counter()
net.SGD(trainingData, 50, 2, 0.75, test_data = testingData)
after = time.perf_counter()

print(f"Finished in {after - before:0.4f} seconds")

#network.saveToFile(net,"part3.pkl")

