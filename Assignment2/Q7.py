import matplotlib.pyplot as plt
import numpy as np
g=10
tf=10

t=np.linspace(0,tf,51)
h=tf/50
max_iter=10000
y=np.zeros(51)
all_y_values=[]
for i in range(max_iter):
    y_last=y.copy()
    all_y_values.append(y_last)
    y[1:-1]=0.5*(y[2:]+y[:-2])+0.5*g*h**2
    if max(abs(y-y_last))<1e-6:
        print(f"solution converged after {i+1} steps")
        break
else:
    print("Solution failed to converge")
def exact_sol(t):
    return -0.5*g*t**2 +5*g*t
for i in range(400,2001, 400):
    plt.plot(t, all_y_values[i], label=f'trial {int(i/400)}')
plt.plot(t,exact_sol(t),label='exact sol')
plt.plot(t,y,'.',label='final solution obtained by relaxation method')
plt.legend()
plt.xlabel('t')
plt.ylabel('X')
plt.title('Solution of the equation $\\frac{d^2x}{dt^2}=-g$ by relaxation method')
plt.grid()
plt.show()

