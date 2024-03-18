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

    x = np.zeros((1, len(b)))  # Initialize x with zeros
    tolerance_percentage = 0.1
    N = 100
    Indicator = 1
    n = len(b)

    for i in range(0, n):
        if A[i][i] == 0:
            print("Error: Jacobi method failed")
            Indicator = 0

    if Indicator:
        for k in range(0, N):
            x = np.vstack([x, np.zeros(n)])
            for i in range(0, n):
                x[k + 1][i] = b[i]

                for j in range(0, n):
                    if j != i:
                        x[k + 1][i] = x[k + 1][i] - A[i][j] * x[k][j]

                x[k + 1][i] = x[k + 1][i] / A[i][i]

            if k > 0:
                error = 100 * np.linalg.norm(x[k] - x[k-1], ord=np.inf) / (np.linalg.norm(x[k], ord=np.inf))
            else:
                error = None

            formatted_x = [f'{val:.3f}' for val in x[k]] if k > 0 else x[k]
            formatted_error = f'{error:.3f}' if error is not None else error

            print(f"After {k}th iteration x[{k}] =", formatted_x, "Error =", formatted_error)

            if error is not None and error < (tolerance_percentage):
                print("Converged after", k + 1, "iterations.")
                break

        print("Final solution:", [f'{val:.3f}' for val in x[-1]])

# Choose the system using a popup
try:
    system_choice = int(input("Enter System number (1, 2,3, or 4): "))
    solve_system(system_choice)
except ValueError:
    print("Invalid input. Please enter a valid integer.")
