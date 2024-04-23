import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def dydt1(t,y):
    return t*np.exp(3*t)-2*y

def dydt2(t,y):
    return 1-(t-y)**2

def dydt3(t,y):
    return 1+y/t

def dydt4(t,y):
    return np.cos(2*t)+np.sin(3*t)

# Original solution
def original_solution1(t):
    return (1/25) * np.exp(-2 * t) * (1 - np.exp(5 * t) + 5 * np.exp(5 * t) * t)

def original_solution2(t):
    return (1 - 3 * t + t**2) / (-3 + t)

def original_solution3(t):
    return 2 * t + t * np.log(t)

def original_solution4(t):
    return (1/6) * (8 - 2 * np.cos(3 * t) + 3 * np.sin(2 * t))

problems=[
    {'dydt':dydt1,'t_span':(0,1),'y0':[0],'label':'y\'=te^(3t)-2y', 'original_solution': original_solution1},
    {'dydt':dydt2,'t_span':(2,3),'y0':[1],'label':'y\'=1-(t-y)^2', 'original_solution': original_solution2},
    {'dydt':dydt3,'t_span':(1,2),'y0':[2],'label':'y\'=1+y/t', 'original_solution': original_solution3},
    {'dydt':dydt4,'t_span':(0,1),'y0':[1],'label':'y\'=cos(2t)+sin(3t)', 'original_solution': original_solution4}
]

plt.figure(figsize=(10, 8))

for i,problem in enumerate(problems,start=1):
    sol=solve_ivp(problem['dydt'],problem['t_span'],problem['y0'],t_eval=np.linspace(*problem['t_span'], 100))

    plt.subplot(2,2,i)
    plt.plot(sol.t, sol.y[0], label='Numerical solution by solve_ivp')
    
    
    t_orig = np.linspace(*problem['t_span'], 100)
    y_orig = problem['original_solution'](t_orig)
    plt.plot(t_orig, y_orig, 'r--', label='Original solution')

    plt.xlabel('t')
    plt.ylabel('y')
    plt.title('Solution of ' + problem['label'])
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()
