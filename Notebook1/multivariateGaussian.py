import numpy as np
from scipy.stats import multivariate_normal

def computeP(X, mu, sigma):
    vecP = np.zeros((np.shape(X)[0],1))
    d = np.shape(X)[1]
    denPart = 1 / np.sqrt(((2 * np.pi) ** d) * (np.linalg.det(sigma)))
    for i in range(0,np.shape(vecP)[0]):
        subtract = np.expand_dims(X[i, :], axis=1) - mu
        bracketPart = np.transpose(subtract) @ (np.linalg.inv(sigma) @ subtract)
        expPart = np.exp(-0.5 * bracketPart)
        vecP[i] = denPart * expPart

    return vecP

N, d = 10, 2
mu1, sigma1 = np.array([[1], [1]]), np.diag([0.1, 0.1]) # mean and covariance of model 1
mu2, sigma2 = np.array([[-1], [-1]]), np.diag([1, 2]) # mean and covariance of model 2
X = np.random.rand(N,d)
P1 = computeP(X, mu1, sigma1)
P2 = computeP(X, mu2, sigma2)


from scipy.stats import multivariate_normal

compareMu1, compareSigma1 = np.array([1, 1]), np.diag([0.1, 0.1]) # mean and covariance of model 1
compareMu2, compareSigma2 = np.array([-1, -1]), np.diag([1, 2]) # mean and covariance of model 2

compareP1 = multivariate_normal.pdf(X,compareMu1,compareSigma1)
print(f"P computed for the model 1 using scipy.stats: \n{compareP1}")
compareP2 = multivariate_normal.pdf(X,compareMu2,compareSigma2)
print(f"P computed for the model 2 using scipy.stats: \n{compareP2}")
