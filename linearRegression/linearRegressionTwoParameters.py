import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn import linear_model

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

#The model proposed is f(v,t) = a0 + a1*vt
vt = trainData[:, 0] * trainData[:, 1] #element wise multiplication
X_v_t = np.expand_dims(vt, axis=1) #(n,) to (n,1) array
X = np.ones((np.shape(X_v_t)[0], 1), dtype=float)
X = np.concatenate((X, X_v_t), axis = 1) #a0 + a1*x(1)
Xt = np.transpose(X)
Y = np.expand_dims(trainData[:, -1], axis=1)

#Linear Regression
M = Xt @ X
half1 = np.linalg.inv(M) #[n,m]@[m,n] = [n,n]
half2 = Xt @ Y #[n,m]@[i,j] = [n,j]
theta = half1 @ half2 #[n,n]@[n,j] = [n,j]
print(f'Coef = {theta[1]}')
print(f'Intercept = {theta[0]} ')

#Singular Value Decomposition (SVD) of the matrix (X.t X)
u, s, vh = np.linalg.svd(M, full_matrices=True)
print("SVD:\n s:", s)

# Precticted values
Y_predict = X @ theta # Y = a0 + a1*x

#Model's error
MSE = mean_squared_error(Y, Y_predict)
myMSE = myMSE(Y,Y_predict)

#Using scikit learn
model = linear_model.LinearRegression()
model.fit(X_v_t, Y)
print("Coef = ", model.coef_)
print("Intercept = ", model.intercept_)

'''plt.scatter(X_v_t, Y, color="black")
plt.plot(X_v_t, Y_predict,  color="blue", linewidth=3)
plt.grid()
title = 'MSE = {}'.format(round(MSE,2))
plt.title("Linear Regression f(v, t) = ao + a1.v.t : \n " + title,
          fontsize=10)
plt.xlabel('v.t')
plt.ylabel('y')
plt.show()'''
