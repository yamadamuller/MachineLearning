import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures

def myMSE(realY, predY):
    n = np.shape(realY)[0]
    accSum = 0
    for i in range(0,n):
        accSum += (realY[i,0] - predY[i,0])**2

    accSum = accSum/n
    return accSum

#Train data
trainData = np.array([[100, 121, 143, 159, 182], [28, 30, 25, 38, 34], [0.05100, 0.05463, 0.04559, 0.06920, 0.06192],\
                  [3401, 4004, 4561, 5308, 6032]], dtype=float)
trainData = np.transpose(trainData)

#The model proposed is g(v,t) = a0 + a1*vt + a2*vt^2
vt = trainData[:, 0] * trainData[:, 1] #element wise multiplication
X_v_t = np.expand_dims(vt, axis=1) #(n,) to (n,1) array
X_v_t_squared = np.expand_dims(vt**2, axis=1)
X = np.ones((np.shape(X_v_t)[0], 1), dtype=float)
X = np.concatenate((X, X_v_t), axis = 1) #a0 + a1*x(1) + a2*x(2)
X = np.concatenate((X, X_v_t_squared), axis = 1) #a0 + a1*x(1) + a2*x(2)
Xt = np.transpose(X)
Y = np.expand_dims(trainData[:, -1], axis=1)

#Linear regression
M = Xt @ X
half1 = np.linalg.inv(M)
half2 = Xt @ Y
theta = half1 @ half2
print(f'Coef = {np.transpose(theta[1:])}')
print(f'Intercept = {theta[0]} ')

# Precticted values
Y_predict = X @ theta

#Using scikit learn
nb_degree = 2
polynomial_features = PolynomialFeatures(degree = nb_degree)
X_TRANSF = polynomial_features.fit_transform(X_v_t)
model = linear_model.LinearRegression()
model.fit(X_TRANSF[:,[1,2]], Y)
print("Coef = ", model.coef_)
print("Intercept = ", model.intercept_)

#Singular Value Decomposition (SVD) of the matrix (X.t X)
u, s, vh = np.linalg.svd(M, full_matrices=True)
print("SVD:\n s:", s)

#Model's error
MSE = mean_squared_error(Y, Y_predict)
myMSE = myMSE(Y,Y_predict)

### Plot
plt.scatter(X_v_t, Y, color="black")
plt.plot(X_v_t, Y_predict,  color="blue", linewidth=3)
plt.grid()
title = 'MSE = {}'.format(round(MSE,2))
plt.title("Linear Regression g(v, t) = a_o + a_1.v.t + a_2.(v.t)^2 : \n " + title,
          fontsize=10)
plt.xlabel('v.t')
plt.ylabel('y')
plt.show()