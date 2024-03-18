#Gauss seidal


import numpy as np
tolerance=0.1
def gauss(A, b, tolerance=0.1, max_iterations=200):
    n = len(b)
    x = np.zeros(n)

    for iteration in range(max_iterations):
        x_old = x.copy()
        for i in range(n):
            x[i] = (b[i] - np.dot(A[i,:i], x[:i]) - np.dot(A[i,i+1:], x_old[i+1:])) / A[i,i]

        
        error=100*np.abs(np.linalg.norm(x - x_old))/np.abs(np.linalg.norm(x))
        print(f"Iteration {iteration + 1}: Solution = {x} ,error is {error}:")
        if error < tolerance:
            break

    return x

# Example usage
A = np.array([[3.0, -1.0, 1.0], [3.0, 6, 2], [3.0, 3.0, 7.0]])
b = np.array([1, 0, 4])
'''
A = np.array([[10, -1.0, 0], [-1, 10, -2], [0, -2, 10]])
b = np.array([9, 7, 6])
A=np.array([[4,1,1,0,1],[-1,-3,1,1,0],[2,1,5,-1,-1],[-1,-1,-1,4,0],[0,2,-1,1,4]])
b=np.array([6,6,6,6,6])
A = np.array([[10, 5, 0, 0], [5, 10, -4, 0], [0, -4, 8, -1], [0, 0, -1, 5]])
b = np.array([6, 25, -11, -11])'''
solution = gauss(A, b)
print("Final Solution:", solution)



'''
Output:
Iteration 1: Solution = [ 0.33333333 -0.16666667  0.5       ] ,error is 100.0:
Iteration 2: Solution = [ 0.11111111 -0.22222222  0.61904762] ,error is 38.70058131782371:
Iteration 3: Solution = [ 0.05291005 -0.23280423  0.64852608] ,error is 9.563849000381529:
Iteration 4: Solution = [ 0.03955656 -0.23595364  0.65559875] ,error is 2.2117539477915233:
Iteration 5: Solution = [ 0.0361492  -0.23660752  0.65733928] ,error is 0.5548678507844865:
Iteration 6: Solution = [ 0.03535107 -0.23678863  0.65775895] ,error is 0.1313981957482832:
Iteration 7: Solution = [ 0.03515081 -0.23682839  0.65786182] ,error is 0.032656548809614376:
Final Solution: [ 0.03515081 -0.23682839  0.65786182]

'''