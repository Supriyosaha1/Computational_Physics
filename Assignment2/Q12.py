import numpy as np
import matplotlib.pyplot as plt

def f(t, u):
    u1, u2, u3 = u
    du1_dt = u1 + 2*u2 - 2*u3 + np.exp(-t)
    du2_dt = u2 + u3 - 2*np.exp(-t)
    du3_dt = u1 + 2*u2 + np.exp(-t)
    return np.array([du1_dt, du2_dt, du3_dt])

def rk4_step(t, u, h):
    k1 = f(t, u)
    k2 = f(t + 0.5*h, u + 0.5*h*k1)
    k3 = f(t + 0.5*h, u + 0.5*h*k2)
    k4 = f(t + h, u + h*k3)
    return u + (h/6) * (k1 + 2*k2 + 2*k3 + k4)


u0 = np.array([3, -1, 1])

t0 = 0
t_max = 1
h = 0.01  

num_steps = int((t_max - t0) / h)


t_values = np.linspace(t0, t_max, num_steps+1)
u_values = np.zeros((num_steps+1, 3))
u_values[0] = u0

for i in range(num_steps):
    u_values[i+1] = rk4_step(t_values[i], u_values[i], h)


plt.plot(t_values, u_values[:,0], label='u1')
plt.plot(t_values, u_values[:,1], label='u2')
plt.plot(t_values, u_values[:,2], label='u3')
plt.xlabel('t')
plt.ylabel('u')
plt.title('Solution of the system of differential equations')
plt.legend()
plt.grid(True)
plt.show()
