import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import SGDRegressor

#Train data
trainData = np.array([[100, 121, 143, 159, 182], [28, 30, 25, 38, 34], [0.05100, 0.05463, 0.04559, 0.06920, 0.06192],\
                  [3401, 4004, 4561, 5308, 6032]], dtype=float)
trainData = np.transpose(trainData)

#The model proposed is g(v,t) = a0 + a1*vt + a2*wt
vt = trainData[:, 0] * trainData[:, 1] #element wise multiplication
wt = trainData[:, 0] * trainData[:, 2] #element wise multiplication
X_v_t = np.expand_dims(vt, axis=1) #(n,) to (n,1) array
X_w_t = np.expand_dims(wt, axis=1)
X_train = np.ones((np.shape(X_v_t)[0], 1), dtype=float)
X_train = np.concatenate((X_train, X_v_t), axis = 1) #a0 + a1*x(1) + a2*x(2)
X_train = np.concatenate((X_train, X_w_t), axis = 1) #a0 + a1*x(1) + a2*x(2)
Xt_train = np.transpose(X_train)
Y_train = np.expand_dims(trainData[:, -1], axis=1)

# scaler for normalization
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
X = scaler_x.fit_transform(X_train)
Xt = np.transpose(X)
Y = scaler_y.fit_transform(Y_train)
print("Training:", X)
print("Label:", Y)

lrate = 0.005
n_epochs = 300
n_samples = X.shape[0]

#Initialization of theta
theta = np.random.rand(X.shape[1], 1)
#In stochastic gradient descent, you calculate the gradient using just a random small part of the observations instead of all of them.
for itr in range(n_epochs):
    isample = np.random.randint(0, X.shape[0])
    Y_predict = X[isample] @ theta
    Y_residuals = np.subtract(Y_predict, Y[isample])
    Loss = (Y_residuals ** 2).mean()
    grad_loss = 2*X[isample]*Y_predict - 2*X[isample]*Y[isample]
    grad_loss = np.reshape(grad_loss, (X.shape[1], 1))
    print('Iter i:', itr, '  Loss:', Loss)
    theta = theta - lrate*grad_loss

Y_predict = scaler_y.inverse_transform(X @ theta) #reverse transformation

#Using sklearn
model_sgd = SGDRegressor(learning_rate='constant', eta0 = lrate, alpha=0, early_stopping=False, max_iter=n_epochs, tol=1e-9)
X = X[:, 1:]
# Fit model based on data
model_sgd.fit(X , Y.reshape(n_samples))
# Use the model
yfit_sgd = model_sgd.predict(X)
yfit_sgd = scaler_y.inverse_transform(yfit_sgd.reshape(-1, 1)) #reverse transformation
print(yfit_sgd)

#Plot regression line
plt.scatter(X_v_t, Y_train)
plt.plot(X_v_t, yfit_sgd,  'y', label='sklearn SGDRegressor')
plt.plot(X_v_t, Y_predict,  'y', label='iterative SGD', color='blue')
plt.legend()
plt.title('Stochastic Gradient Descent')
plt.show()
