import numpy as np
import matplotlib.pyplot as plt

# cost function
def J_func(x, y):
    f1 = (2*((x**2)+(y**2)))/(1 + (x**2) + (y**2))
    f2 = (((x-2)**2) + ((y-1)**2))/(1 + ((x-2)**2) + ((y-2)**2))
    return f1 + f2

# gradient
def J_grad(nVars, x, y):
    grad = np.zeros((nVars,1))
    grad[0,0] = ((4*x)/(((x**2) + (y**2) + 1)**2)) - \
                ((4*(x-2)*(y-2))/(((x**2) - (4*x) + (y**2) - (4*y) + 9)**2))
    #grad[0,0] = ((4*x)/(((x**2) + (y**2) + 1)**2)) + \
    #            ((2 * (x-2) * ((-2*y) + 4))/(((x**2) - (4*x) + (y**2) - (4*y) + 9)**2))
    grad[1,0] = ((4*y)/(((x**2) + (y**2) + 1)**2)) + \
                ((2*((x**2) - (4*x) - (y**2) + (4*y) + 1))/(((x**2) - (4*x) + (y**2) - (4*y) + 9)**2))
    return grad

def gradientDescent(maxIter, points, x_0, y_0):
    points[0] = [x_0, y_0]  # initial guess
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
MAX_ITER = 20
# collect points along iterations
points = np.zeros((MAX_ITER + 1, numVar))
# initial point
x_0, y_0 = -0.5,-0.5

###########################
# gradient descent method
points, cost, gradJ = gradientDescent(MAX_ITER, points, x_0, y_0)

# draw contour lines
MIN = -8
MAX = 8
STEP = 0.1
x = np.arange(MIN,MAX+STEP,STEP)
y = np.arange(MIN,MAX+STEP,STEP)
X, Y = np.meshgrid(x, y)
J = (2*((X**2)+(Y**2)))/(1 + (X**2) + (Y**2)) + \
    (((X-2)**2) + ((Y-1)**2))/(1 + ((X-2)**2) + ((Y-2)**2))

fig, ax = plt.subplots(figsize=(7,7))

cs = ax.contour(X,Y,J,10)
ax.clabel(cs, inline=True, fontsize=10)
ax.set_aspect('equal')
ax.set_title('countour plot')
plt.grid()

# draw sequence of solutions
plt.plot(points[:,0],points[:,1],'-*')
plt.show()