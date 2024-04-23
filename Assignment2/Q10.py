import numpy as np
import matplotlib.pyplot as plt

def f(t, y):
    return (y**2 + y) / t

def rk4_step(t, y, h):
    k1 = f(t, y)
    k2 = f(t + 0.5*h, y + 0.5*h*k1)
    k3 = f(t + 0.5*h, y + 0.5*h*k2)
    k4 = f(t + h, y + h*k3)
    return y + (h/6) * (k1 + 2*k2 + 2*k3 + k4)

def adaptive_rk4(t0, y0, t_end, h0, tol):
    t_values = [t0]
    y_values = [y0]
    h = h0
    t = t0
    y = y0
    while t < t_end:
     
        y1 = rk4_step(t, y, h)
        y2 = rk4_step(t, y, h/2)
        y2 = rk4_step(t + h/2, y2, h/2)
        
  
        error = np.abs(y1 - y2)
        
        if np.max(error) < tol:
            t = t + h
            y = y1
            t_values.append(t)
            y_values.append(y)
        
        h = 0.9 * h * (tol / np.max(error))**0.2
       
        if t + h > t_end:
            h = t_end - t
    
    return t_values, y_values

# Initial conditions
t0 = 1
y0 = -2

t_values, y_values = adaptive_rk4(t0, y0, 3, 0.01, 1e-4)


plt.plot(t_values, y_values, label='Solution')
plt.scatter(t_values, y_values, color='red', label='Mesh Points')
plt.xlabel('t')
plt.ylabel('y')
plt.title('Solution of y\' = (y^2 + y)/t')
plt.legend()
plt.grid(True)
plt.show()
