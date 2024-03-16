import numpy as np
import sys

A = np.array([[2, -1, 0], [-1, 2, -1], [0, -1, 2]])
x0 = np.array([[1], [0], [0]])
y0 = np.array([[1], [0], [0]])
lam = np.zeros(100)

for i in range(100):
    numerator=np.dot(x0.T,np.dot(np.linalg.matrix_power(A, i+1),y0))[0,0]
    denominator=np.dot(x0.T,np.dot(np.linalg.matrix_power(A, i),y0))[0,0]
    lam[i] = (numerator /denominator )
    if i > 0 and abs(lam[i] - lam[i-1]) < 0.01*lam[i]:
        eigen_vector=np.dot(np.linalg.matrix_power(A, i),y0)/np.linalg.norm(np.dot(np.linalg.matrix_power(A, i),y0))
        print(f"eigen vector found after {i}th iteration")
        print("Dominant eigenvalue:", lam[i],"eigenvector is ",eigen_vector)
        sys.exit()

# If the loop completes without finding a suitable eigenvalue
print("Dominant eigenvalue not found within 100 iterations.")


"""
OUTPUT:

eigen vector found after 8th iteration
Dominant eigenvalue: 3.3760539629005057 eigenvector is  [[ 0.51376645]
 [-0.70697036]
 [ 0.48604212]]

"""