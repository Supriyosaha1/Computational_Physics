import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton

def backward_euler(f,x0,xf,y0,h=0.01):
    x_values=[x0]
    y_values=[y0]
    x=x0
    y=y0
    while x<xf:
        y=newton(lambda t:t-y-h*f(x+h,t),x0)
        x+=h
        y_values.append(y)
        x_values.append(x)

    return x_values,y_values

def f(x,y):
    return -9*y

def g(x,y):
    return -20*(y-x)**2+2*x


x_values,y_values=backward_euler(f,0,1,np.exp(1))

f_original=lambda x: np.exp(1-9*x)
plt.plot(x_values,f_original(np.array(x_values)),label="original soln")
plt.plot(x_values,y_values,label="soln obtained by backward euler ")
plt.grid()
plt.xlabel("x")
plt.ylabel("y")
plt.title(r"Solution of the differential eqn $\frac{dy}{dx}=-9y$")
plt.legend()
plt.show()

x_values_g,y_values_g=backward_euler(g,0,1,1/3)
plt.plot(x_values_g,y_values_g,label='solution obtained by backward euler')
plt.grid()
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title(r"Solution of the differential eqn $\frac{dy}{dx}=-20(y-x)^2 + 2x$")
plt.show()