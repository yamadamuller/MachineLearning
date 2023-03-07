import numpy as np
import matplotlib.pyplot as plt

def J_func(theta1,theta2):
  return (theta1**2)+(theta2**2) + 3*((theta1-1)**2) + ((theta2-1)**2)\
          + theta1*theta2

def gridSearch1(XMIN, XMAX, DX, YMIN, YMAX, DY):
    theta1_min = 0
    theta2_min = 0
    J_min = float('inf')
    for theta1 in np.arange(XMIN, XMAX + DX, DX):
        for theta2 in np.arange(YMIN, YMAX + DY, DY):
            J = J_func(theta1, theta2)
            if J < J_min:
                J_min = J
                theta1_min = theta1
                theta2_min = theta2

    return J_min, theta1_min, theta2_min

def gridSearch2(XMIN, XMAX, DX, YMIN, YMAX, DY):
    theta1 = np.arange(XMIN, XMAX + DX, DX)
    theta2 = np.arange(YMIN, YMAX + DY, DY)
    computeJ = np.zeros((np.shape(theta1)[0],np.shape(theta2)[0]))
    for i in range(0,np.shape(computeJ)[0]):
        for j in range(0,np.shape(computeJ)[1]):
            computeJ[i,j] = np.expand_dims(J_func(theta1[i],theta2[j]), axis=0)

    idxMin = np.unravel_index(np.argmin(computeJ), np.shape(computeJ))
    J_min = computeJ[idxMin[0], idxMin[1]]
    theta1_min = theta1[idxMin[0]]
    theta2_min = theta2[idxMin[1]]

    return J_min, theta1_min, theta2_min, computeJ, idxMin

XMIN = -3
XMAX = 3
DX = 0.01
YMIN = -3
YMAX = 3
DY = DX

#J_min, theta1_min, theta2_min = gridSearch1(XMIN, XMAX, DX, YMIN, YMAX, DY)
#print(f'min: {J_min} at ({theta1_min},{theta2_min})')
J_min2, theta1_min2, theta2_min2, teste, idx = gridSearch2(XMIN, XMAX, DX, YMIN, YMAX, DY)
print(f'min: {J_min2} at ({theta1_min2},{theta2_min2})')

leg = list()
plt.figure()
plt.imshow(np.transpose(teste))
plt.scatter(idx[0],idx[1], marker='x')
leg.append(f"Minimum point = {J_min2}")
plt.legend(leg)
plt.colorbar()
plt.xlabel("Samples of theta1")
plt.ylabel("Samples of theta2")
plt.title("Cost function")
plt.show()


