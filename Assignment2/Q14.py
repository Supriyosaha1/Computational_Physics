import numpy as np
import matplotlib.pyplot as plt
def f(t,y,z):
    return (2*t*z-2*y+ t**3*np.log(t))/t**2

def second_order_ode_euler(h,t0,tf,y0,z0):
    t_new=[t0]
    y_new=[y0]
    z_new=[z0]
    y=y0
    t=t0
    z=z0
    
    while t<tf:
        y+=h*z
        z+=h*f(t,y,z)
        t+=h
        t_new.append(t)
        y_new.append(y)
        z_new.append(z)

    return t_new,y_new

# Initial conditions
t0 = 1
tf = 2
y0 = 1
z0 = 0

# Step size
h = 0.001

t_val,y_val=second_order_ode_euler(h,t0,tf,y0,z0)
def g(t):
    return 7*t/4 + (t**3*np.log(t))/2-3*t**3/4


plt.plot(t_val,y_val,label='by euler method',ls='--',color='red')
plt.scatter(t_val,g(np.array(t_val)),label="original solution",marker='o',s=5)

plt.xlabel('t')
plt.ylabel('y(t)')
plt.title('Solution of the Initial Value Problem $t^2 y\'\' -2ty\'+2y=t^3 ln(t)$')
plt.legend()
plt.grid(True)
plt.show()


