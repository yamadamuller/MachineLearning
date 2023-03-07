import sympy as sym

theta1,theta2 = sym.symbols('theta1, theta2')

f1 = (theta1**2) + (theta2**2) + 3*((theta1-1)**2) + ((theta2-1)**2) + theta1*theta2
#Computing the gradient
g1 = sym.zeros(2,1)
g1[0,0] = sym.diff(f1,theta1)
g1[1,0]= sym.diff(f1,theta2)
print(f'Gradient of f(x) = {g1}')
#gradient = 0: calculate the minima
sysSol = sym.solve(g1,theta1,theta2)
print(f'Points satisfying the necessary conditions: {sysSol}')
#Computing the Hessian
H1 = sym.zeros(2,2)
H1[0,0] = sym.diff(g1[0,0],theta1)
H1[0,1] = sym.diff(g1[1,0],theta1)
H1[1,0] = sym.diff(g1[0,0],theta2)
H1[1,1] = sym.diff(g1[1,0],theta2)
print(f'Hessian of f(x) = {g1}')
print(f'Determinant of the Hessian = {sym.det(H1)}')
print(f'Trace of the Hessian = {sym.trace(H1)}')
ans = f1.subs(theta1,22/31).subs(theta2,10/31)
print(f'Minimum value of the cost function = {ans}')
