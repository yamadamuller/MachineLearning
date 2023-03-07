import numpy as np
import matplotlib.pyplot as plt

# cost function
def J_func(theta1, theta2):
    return (theta1 ** 2) + (theta2 ** 2) + 3 * ((theta1 - 1) ** 2) + ((theta2 - 1) ** 2) \
           + theta1 * theta2

# gradient
def J_grad(nVars, theta1, theta2):
    grad = np.zeros((nVars,1))
    grad[0,0] = (8*theta1) + theta2 - 6
    grad[1,0] = theta1 + (4*theta2) - 2
    return grad

def gradientDescent(maxIter, points, theta1_0, theta2_0):
    points[0] = [theta1_0, theta2_0]  # initial guess
    gradJ = np.zeros_like(points)
    costFunc = np.zeros((maxIter,1))
    for i in range(0, maxIter):
        costFunc[i,0] = J_func(points[i][0], points[i][1])
        print(i, points[i], costFunc[i,0])
        computeGrad = J_grad(numVar, points[i][0], points[i][1])
        for j in range(0, np.shape(gradJ)[1]):
            gradJ[i, j] = computeGrad[j, 0]
            points[i + 1][j] = points[i][j] - gamma * computeGrad[j, 0]

    return points, costFunc, gradJ

#number of variables
numVar = 2
# step size
gamma = 0.1
# number of iterations
MAX_ITER = 10
# collect points along iterations
points = np.zeros((MAX_ITER + 1, numVar))
# initial point
theta1_0, theta2_0 = 0, 0

###########################
# gradient descent method
points, cost, gradJ = gradientDescent(MAX_ITER, points, theta1_0, theta2_0)

# draw contour lines
MIN = -0.5
MAX = 1.5
STEP = 0.1
theta1 = np.arange(MIN,MAX+STEP,STEP)
theta2 = np.arange(MIN,MAX+STEP,STEP)
Theta1, Theta2 = np.meshgrid(theta1, theta2)
J = Theta1**2 + Theta2**2 +3*(Theta1-1)**2 + (Theta2-1)**2\
    + Theta1*Theta2

fig, ax = plt.subplots(figsize=(7,7))

cs = ax.contour(Theta1,Theta2,J,10)
ax.clabel(cs, inline=True, fontsize=10)
ax.set_aspect('equal')
ax.set_title('countour plot')
plt.grid()

# draw sequence of solutions
plt.plot(points[:,0],points[:,1],'-*')
plt.show()
