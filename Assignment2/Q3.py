import numpy as np
import matplotlib.pyplot as plt
def f1(x,y,z):return z
def f2(x,y,z):return x*np.exp(x) -x +2*z-y
xf,x0=1,0
x,y,z=0,0,0
h=0.001
X,Y,Z=[x],[y],[z]
n=int((xf-x0)/h)
for i in range(n):
    a1=h*f1(x,y,z)
    b1=h*f2(x,y,z)

    a2=h*f1(x+h/2,y+a1/2,z+b1/2)
    b2=h*f2(x+h/2,y+a1/2,z+b1/2)

    a3=h*f1(x+h/2,y+a2/2,z+b2/2)
    b3=h*f2(x+h/2,y+a2/2,z+b2/2)

    a4=h*f1(x+h,y+a3,z+b3)
    b4=h*f2(x+h,y+a3,z+b3)

    y=y+(a1+2*a2+2*a3+a4)/6
    z=z+(b1+2*b2+2*b3+b4)/6
    x=x+h

    X.append(x)
    Y.append(y)
    Z.append(z)

plt.plot(X,Y,label='solution obtained by rk4')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Solution of the Initial Value Problem $y\'\' -2y\'+y=xe^x -x$')
plt.grid(True)
plt.legend()
plt.show()


