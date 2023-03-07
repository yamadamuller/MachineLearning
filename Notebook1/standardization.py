import numpy as np

def mySampleMean(x):
    samples = len(x) #n samples from the feature
    sum = 0 #variable that will add up the samples of the feature
    for i in range(0, samples):
        sum += x[i]

    sampleMean = sum/samples #sample mean equation as shown in the original notebook
    return sampleMean

def mySampleStd(x, mean):
    samples = len(x) #n samples from the feature
    sumSquared = 0 #variable that will add up the difference of the sample and the mean squared
    for i in range(0,samples):
        sumSquared += (x[i]-mean)**2

    sampleStd = ((1/(samples-1))*sumSquared)**0.5 #sample std equation as shown in the original notebook
    return sampleStd

def standardization(X):
    z = np.zeros_like(X) #array shaped just like the data set
    mean = np.zeros((1,np.shape(X)[1])) #array containing the means of all features
    std = np.zeros_like(mean) #array containing the std's of all the features

    for j in range(0, np.shape(z)[1]):
        mean[0,j] = mySampleMean(X[:,j]) #computes the mean for set j (ranging from 0 to number of sets)
        std[0,j] = mySampleStd(X[:,j],mean[0,j]) #computes the std for set j (ranging from 0 to number of sets)
        for i in range(0, len(z)):
            z[i,j] = (X[i,j] - mean[0,j])/std[0,j] #computes the standarized vector for position [i,j]

    return mean, std, z

n, d = 10, 3
#n, d = 1000, 3
np.random.seed(1)
#Feature 1
mu, sigma = 10, 0.1 # mean and standard deviation
x1 = np.random.normal(mu, sigma, size=(n, 1))
#Feature 2
mu, sigma = 2, 10 # mean and standard deviation
x2 = np.random.normal(mu, sigma, size=(n, 1))
#Feature 3
mu, sigma = -10, 100 # mean and standard deviation
x3 = np.random.normal(mu, sigma, size=(n, 1))

X = np.block([x1, x2, x3])
print(X)

[mean,std,setZ] = standardization(X)

standMean = np.zeros((1,np.shape(setZ)[1])) #array containing the means of all features of the standardized data
standStd = np.zeros_like(mean) #array containing the std's of all the features of the standardized data

for j in range(0, np.shape(standStd)[1]):
    standMean[0,j] = mySampleMean(setZ[:,j]) #computes the mean for set j (ranging from 0 to number of sets) standardized
    standStd[0,j] = mySampleStd(setZ[:,j],standMean[0,j]) #computes the std for set j (ranging from 0 to number of sets) standardized

print(f"The mean from the standardized data: {standMean}")
print(f"The std from the standardized data: {standStd}")