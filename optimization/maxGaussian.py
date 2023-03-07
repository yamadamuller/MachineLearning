import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def gaussian2D(X1,X2,mu,sigma):
    tensor = np.empty(np.shape(X1) + (2,)) #creates an 100x100x2 tensor for X1 and X2
    tensor[:, :, 0] = X1
    tensor[:, :, 1] = X2
    d = np.shape(mu)[0] #2 dimensional gaussian PDF
    denPart = 1 / np.sqrt(((2 * np.pi) ** d) * (np.linalg.det(sigma)))
    sigmaInv = np.linalg.inv(sigma)
    subtract = tensor - mu

    #bracketPart = np.einsum('ijk,kl->ijl', subtract, sigmaInv)
    #bracketPart = np.einsum('ijk,ijk->ij', bracketPart, subtract)

    bracketMult1 = np.zeros_like(tensor)
    for i in range(0, np.shape(X1)[0]):
        for j in range(0, np.shape(X1)[1]):
            bracketMult1[i, j, :] = subtract[i][j, :] @ sigmaInv

    bracketMult2 = np.zeros((np.shape(X1)[0], np.shape(X1)[1]))
    for i in range(0, np.shape(X1)[0]):
        for j in range(0, np.shape(X1)[1]):
            bracketMult2[i, j] = bracketMult1[i][j, :] @ subtract[i][j, :]

    expPart = np.exp(-0.5 * bracketMult2)
    vecP = denPart * expPart

    return vecP

def maxFunc(pdf1, pdf2):
    maxElems = np.zeros((np.shape(pdf1)[0],np.shape(pdf2)[1]))
    for i in range(np.shape(maxElems)[0]):
        for j in range(np.shape(maxElems)[1]):
            check1 = pdf1[i,j]
            check2 = pdf2[i,j]
            if check1 >= check2:
                maxElems[i,j] = check1
            else:
                maxElems[i,j] = check2

    return maxElems

# Our 2-dimensional distribution will be over variables X1 and X2
N = 100
x1 = np.linspace(-3, 5, N)
x2 = np.linspace(-2, 4, N)
X1, X2 = np.meshgrid(x1, x2)

# Mean vector and covariance matrix
#PDF 1
mu1 = np.array([2, 1])
Sigma1 = np.array([[1, -1], [-1, 1.5]])
#PDF 2
mu2 = np.array([-1, 0])
Sigma2 = np.array([[1, 0.5], [0.5, 1]])

Z1 = gaussian2D(X1,X2,mu1,Sigma1)
Z2 = gaussian2D(X1,X2,mu2,Sigma2)
F = maxFunc(Z1,Z2)

fig, ax = plt.subplots(figsize=(10,10))
ax = plt.axes(projection='3d')

ax.plot_surface(X1, X2, F, rstride=3, cstride=3, cmap=cm.viridis)
ax.view_init(60,-80)
ax.set_xlabel(r'$x_1$')
ax.set_ylabel(r'$x_2$')
plt.show()

fig, ax = plt.subplots(figsize=(10,10))
cs = ax.contour(X1, X2, F, 10)
ax.clabel(cs, inline=True, fontsize=10)
ax.set_aspect('equal')
ax.set_title('countour plot')
ax.set_xlabel(r'$x_1$')
ax.set_ylabel(r'$x_2$')
plt.grid()
plt.show()
