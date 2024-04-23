import numpy as np
import matplotlib.pyplot as plt

def f(u, x):
    return 1 / (x**2 * (1 - u)**2 + u**2)

def rk4(h, x0, u0):
    k1 = h * f(u0, x0)
    k2 = h * f(u0 + 0.5 * h, x0 + 0.5 * k1)
    k3 = h * f(u0 + 0.5 * h, x0 + 0.5 * k2)
    k4 = h * f(u0 + h, x0 + k3)

    x_new = x0 + (1/6) * (k1 + 2*k2 + 2*k3 + k4)
    return x_new

u0 = 0
uf = 1
h = 0.01
x0 = 1
u, x = [u0], [x0]

while u[-1] < uf:
    u_new = u[-1] + h
    x_new = rk4(h, x[-1], u[-1])
    u.append(u_new)
    x.append(x_new)
    if u_new >= uf:
        break


target_t = 3.5e6 / (1 + 3.5e6)
index = np.argmin(np.abs(np.array(u) - target_t))


print("The value of x at t = 3.5e6 is:", x[index])
plt.plot(u,x)
plt.show()