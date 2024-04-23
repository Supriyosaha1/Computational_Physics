import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp

def fun1(x,y):
    return np.vstack((y[1],-np.exp(-2*y[0])))
def fun2(x,y):
    return np.vstack((y[1],y[1]*np.cos(x)-y[0]*np.log(y[0])))
def fun3(x,y):
    return np.vstack((y[1],-(2*y[1]**3-y[0]**2*y[1])))
def fun4(x,y):
    return np.vstack((y[1], 0.5 - 0.5*y[1]**2 - 0.5*y[0]*np.sin(x)))


def bc1(ya,yb):
    return np.array([ya[0],yb[0]-np.log(2)])
def bc2(ya, yb):
    return np.array([ya[0] - 1, yb[0] - np.exp(1)])
def bc3(ya, yb):
    return np.array([ya[0] - 2**(-1/4), yb[0] - (12**0.25)/2])
def bc4(ya, yb):
    return np.array([ya[0] - 2, yb[0] - 2])

problems = [
    {'fun': fun1, 'bc': bc1, 'x_span': np.linspace(1, 2, 1000), 'label': '$y\'\'=-e^{2y}$'},
    {'fun': fun2, 'bc': bc2, 'x_span': np.linspace(0, np.pi/2, 1000), 'label': '$y\'\'= y\'\cos(x)-y \ln(y)$'},
    {'fun': fun3, 'bc': bc3, 'x_span': np.linspace(np.pi/4, np.pi/3, 1000), 'label': '$y\'\'=-(2(y\')^3 +y^2 y\')$'},
    {'fun': fun4, 'bc': bc4, 'x_span': np.linspace(0, np.pi, 1000), 'label': '$y\'\'=1/2 -(y\')^2/2 -y \sin(x)/2$'}
]

plt.figure(figsize=(10, 8))

for i, problem in enumerate(problems, start=1):
    y_guess = np.ones((2, problem['x_span'].size))
    sol = solve_bvp(problem['fun'], problem['bc'], problem['x_span'], y_guess)

    plt.subplot(2, 2, i)
    plt.plot(sol.x, sol.y[0], label='Numerical solution by solve_bvp')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Solution of ' + problem['label'])
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()
