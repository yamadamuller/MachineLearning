#Based on "Implementation of Kalman Filter with Python Language"
#by Mohamed LAARAIEDH

import numpy as np
import matplotlib.pyplot as plt

#Predict: function written to be iterable
'''
xn: the state vector [m,1] of the previous iteration
A: the state matrix [m,m]
u: the input vector [p,1] of the previous iteration
B: the input effect matrix [p,n]
P: the covariance matrix of the previous step
Q: the measurement noise
'''
def kFilter_pred(xn, A, u, B, P, Q):
    x_pred = (A @ xn) + (B @ u)
    P_pred = (A @ P @ np.transpose(A)) + Q
    return x_pred, P_pred

#Update
'''
x_pred: the predicted mean
P_pred: the predicted covariance matrix
y: the new measurement (considering the measurement equation)
H: the measurement matrix
R: the measuremnet covariance
'''
def kFilter_update(x_pred, P_pred, y, H, R):
    S = (H @ P_pred @ np.transpose(H)) + R #Meas
    K = P_pred @ np.transpose(H) @ np.linalg.inv(S) #Kalman gain
    e = y - (H @ x_pred) #innovation (error)
    x_up = x_pred + (K @ e) #updated mean (state vector)
    P_up = P_pred - (K @ S @ np.transpose(K)) #updated covariance matrix
    return x_up, P_up

#Example: tracking of mobile in wireless network
np.random.seed(3)
dt = 0.1 #time step of movement

#State matrices
x = np.array([[0.], [0.], [0.1], [0.1]])
A = np.array([[1, 0, dt, 0], [0, 1, 0, dt],\
              [0, 0, 1, 0], [0, 0, 0, 1]])
P = np.diag((0.01, 0.01, 0.01, 0.01))
u = np.zeros((np.shape(x)[0],1))
Q = np.eye(np.shape(x)[0])
B = np.eye(np.shape(x)[0])

#Measurement matrices
y = np.array([[x[0,0] + np.abs(np.random.randn(1)[0])], \
               [x[1,0] + np.abs(np.random.randn(1)[0])]])
H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
R = np.eye(np.shape(y)[0])

#Applying the Kalman filter
nIter = 50 #number of iterations

#Creating arrays to save computed values
xi_pred = np.zeros(nIter)
xj_pred = np.zeros(nIter)
xi_up = np.zeros(nIter)
xj_up = np.zeros(nIter)
xi_out = np.zeros(nIter)
xj_out = np.zeros(nIter)

f = plt.figure()
for i in range(0, nIter):
    (x, P) = kFilter_pred(x, A, u, B, P, Q)
    xi_pred[i] = x[0,0]
    xj_pred[i] = x[1,0]

    (x, P) = kFilter_update(x, P, y, H, R)
    xi_up[i] = x[0,0]
    xj_up[i] = x[1,0]

    y = np.array([[xi_up[i] + np.abs(0.1 * np.random.randn(1)[0])],\
                    [xj_up[i] + np.abs(0.1 * np.random.randn(1)[0])]])
    xi_out[i] = y[0,0]
    xj_out[i] = y[1,0]

plt.plot(xi_pred, xj_pred, 'o', color='black', label="Predicted trajectory")
plt.plot(xi_up, xj_up, 'o', color='red', label="Corrected trajectory")
plt.plot(xi_out, xj_out, color='blue', label="Measured trajectory")
plt.legend()
plt.xlabel('x(m))')
plt.ylabel('y(m)')
plt.grid()
plt.show()
