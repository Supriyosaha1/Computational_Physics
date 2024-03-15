import numpy as np

A = np.array([[5, -2], [-2, 8]])

Q, R = np.linalg.qr(A)
eigenvalues_qr=np.diag(R)
eigenvalues_eigh=np.linalg.eigh(A)

print("Q matrix obtained by linlag.qr:")
print(Q)
print("\nR matrix obtained by linlag.qr:")
print(R)

print("Eigenvalues obtained by QR Decomposition:",eigenvalues_qr)
print("Eigenvalues obtained by linlag.eigh:",eigenvalues_eigh[0])


'''
OUTPUT:

Q matrix obtained by linlag.qr:
[[-0.92847669  0.37139068]
 [ 0.37139068  0.92847669]]

R matrix obtained by linlag.qr:
[[-5.38516481  4.82807879]
 [ 0.          6.68503217]]
 
Eigenvalues obtained by QR Decomposition: [-5.38516481  6.68503217]
Eigenvalues obtained by linlag.eigh: [4. 9.]
'''