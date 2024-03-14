import numpy as np

def solve_system(system):
    if system == 1:
        A = np.array([[3.0, -1.0, 1.0], [3.0, 6.2, 0], [3.0, 3.0, 7.0]])
        b = np.array([1, 0, 4])
    elif system == 2:
        A = np.array([[10, -1.0, 0], [-1, 10, -2], [0, -2, 10]])
        b = np.array([9, 7, 6])
    elif system == 3:
        A = np.array([[10, 5, 0, 0], [5, 10, -4, 0], [0, -4, 8, -1], [0, 0, -1, 5]])
        b = np.array([6, 25, -11, -11])
    elif system==4:
        A=np.array([[4,1,1,0,1],[-1,-3,1,1,0],[2,1,5,-1,-1],[-1,-1,-1,4,0],[0,2,-1,1,4]])
        b=np.array([6,6,6,6,6])
    else:
        print("Invalid system number")
        return None
    x=np.linalg.solve(A,b)      
    print("Solution for system",system, "is" ,x)

try:
    system_choice = int(input("Enter System number (1, 2,3, or 4): "))
    solve_system(system_choice)
except ValueError:
    print("Invalid input. Please enter a valid integer.")





""""
OUTPUT:
Enter System number (1, 2,3, or 4): 1
Solution for system 1 is [ 0.13135593 -0.06355932  0.54237288]

Enter System number (1, 2,3, or 4): 2
Solution for system 2 is [0.99578947 0.95789474 0.79157895]     

Enter System number (1, 2,3, or 4): 3
Solution for system 3 is [-0.79764706  2.79529412 -0.25882353 -2.25176471]1


Enter System number (1, 2,3, or 4): 4
Solution for system 4 is [ 0.78663239 -1.00257069  1.86632391  1.9125964   1.98971722]
"""