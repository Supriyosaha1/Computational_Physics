import numpy as np
import matplotlib.pyplot as plt

def euler(f,t0,y0,tf,h):
    result=[(t0,y0)]
    t=t0
    y=y0
    n=int((tf-t0)/h)
    for i in range(1,n+1):
        y+=h*f(y,t)
        t+=h
        result.append((t,y))
    return result

def f(y,t):
    return (y/t)-(y/t)**2
t0=1
tf=2
y0=1
h=0.1

solution=euler(f,t0,y0,tf,h)

y_values=[t[1] for t in solution]
t_values=[t[0] for t in solution]

plt.plot(t_values,y_values,label="solution by euler method")

def original_soln(t):
    return t/(1+np.log(t))

orig_soln=original_soln(t_values)

absolute_error=np.abs(y_values-orig_soln)
relative_error=np.abs((y_values-orig_soln)/y_values)

tt=np.linspace(1,2,1000)
plt.plot(tt,original_soln(tt),label="Original solution")
plt.grid()
plt.xlabel("t")
plt.ylabel("y(t)")
plt.legend()


print("t\t\t euler soln\t\t original soln \t\t abs error\t\t relative error ")
for t, euler_soln,orig_soln,abs_error,rel_error in zip(t_values,y_values,orig_soln,absolute_error,relative_error):
    print(f"{t:.1f}\t\t {euler_soln:.6f}\t\t{orig_soln:.6f}\t\t {abs_error:.6f}\t\t {rel_error:.6f}")


plt.show()