import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

def mean(input):
    return np.sum(input)/np.shape(input)[0]

def crossVariance(inputX, inputY, meanX, meanY):
    n = np.shape(inputX)[0]
    croVar = 0
    for i in range(0,n):
        croVar += (inputX[i] - meanX)*(inputY[i] - meanY)

    croVar = croVar/(n-1)
    return croVar

def variance(inputX, meanX):
    n = np.shape(inputX)[0]
    var = 0
    for i in range(0, n):
        var += (inputX[i] - meanX)**2

    var = var/(n-1)
    return var

def computeFIT(inputX, inputY):
    meanX = mean(inputX)
    meanY = mean(inputY)
    croVar = crossVariance(inputX, inputY, meanX, meanY)
    var = variance(inputX, meanX)
    m = croVar/var
    b = meanY - (meanX * m)
    return m,b

def sklCompare(inputX, inputY):
    model = LinearRegression(fit_intercept=True)
    model.fit(inputX[:, np.newaxis], inputY) #train data
    xFit = np.linspace(0, 5, 1000) #test data
    yFit = model.predict(xFit[:, np.newaxis]) #applying the model
    m = model.coef_[0]
    b = model.intercept_
    return xFit, yFit, m, b

#Random Data
rng = np.random.RandomState(2)
x = 5 * rng.rand(100)
y = 2 * x - 5 + rng.randn(100)

m, b = computeFIT(x,y)
print(f'Slope of the model: {m}')
print(f'Bias of the model: {b}')

xFit, yFit, mFit, bFit = sklCompare(x,y)
print(f'Slope of the model: {mFit}')
print(f'Bias of the model: {bFit}')

aux = np.array([0,5])
leg = list()
plt.plot(aux, m*aux + b)
leg.append('linear regression model')
plt.scatter(x, y)
leg.append('random data')
plt.legend(leg)
plt.title('simple linear regression')
plt.grid()
plt.show()
