import numpy as np

# Given matrix A and vector b
A = np.array([[0.2, 0.1, 1, 1, 0],
              [0.1, 4, -1, 1, -1],
              [1, -1, 60, 0, -2],
              [1, 1, 0, 8, 4],
              [0, -1, -2, 4, 700]])

b = np.array([1, 2, 3, 4, 5])
x_true = np.array([7.859713071, 0.422926408, -0.073592239, -0.540643016, 0.010626163])


# Gauss-Seidel method
def gauss_seidel(A, b, tolerance=0.01, max_iterations=200):
    n = len(b)
    x = np.zeros(n)

    for iteration in range(max_iterations):
        x_old = x.copy()
        for i in range(n):
            x[i] = (b[i] - np.dot(A[i, :i], x[:i]) - np.dot(A[i, i+1:], x_old[i+1:])) / A[i, i]
        
        error = 100 * np.abs(np.linalg.norm(x - x_true))
        print(f"Iteration {iteration + 1}: Solution = {x}, error is {error:.4f}")
        if error < tolerance:
            print(f"solution converged after {iteration+1} iteration")
            break

    return x

print("Gauss-Seidel method:")
x_gauss_seidel = gauss_seidel(A, b)

# Relaxation method (with Gauss-Seidel as a base)
def relaxation(A, b, w, tolerance=0.01, max_iterations=200):
    n = len(b)
    x = np.zeros(n)

    for iteration in range(max_iterations):
        x_old = x.copy()
        for i in range(n):
            x[i] = (1 - w) * x_old[i] + (w * (b[i] - np.dot(A[i, :i], x[:i]) - np.dot(A[i, i+1:], x_old[i+1:]))) / A[i, i]
        
        error = 100 * np.abs(np.linalg.norm(x - x_true))
        print(f"Iteration {iteration + 1}: Solution = {x}, error is {error:.4f}")
        if error < tolerance:
            print(f"solution converged after {iteration+1} iteration")
            break

    return x

print("\nRelaxation method:")
w = 1.25  # Relaxation parameter
x_relaxation = relaxation(A, b, w)

# Jacobi Method
def jacobi(A, b, tolerance=0.01, max_iterations=200):
    n = len(b)
    x = np.zeros(n)
    iteration = 0

    while iteration < max_iterations:
        x_old = x.copy()
        for i in range(n):
            x[i] = (b[i] - np.dot(A[i, :i], x_old[:i]) - np.dot(A[i, i+1:], x_old[i+1:])) / A[i, i]

        error = np.linalg.norm(x - x_true) 
        print(f"Iteration {iteration + 1}: Solution = {x}, error is {error:.4f}")
        if error < tolerance:
            print(f"solution converged after {iteration+1} iteration")
            break

        iteration += 1

    return x

print("\nJacobi method:")
x_jacobi = jacobi(A, b)


def conjugate_gradient(A, b, x_true, tolerance=0.01):
    n = len(b)
    x = np.zeros(n)
    r = b - A @ x
    p = np.copy(r)
    iteration = 0

    while True:
        alpha = np.dot(r, r) / np.dot(p, A @ p)
        x += alpha * p
        r_new = r - alpha * A @ p
        beta = np.dot(r_new, r_new) / np.dot(r, r)
        p = r_new + beta * p
        r = r_new
        error = np.linalg.norm(x - x_true)
        print(f"Iteration {iteration + 1}: Solution = {x}, error is {error:.4f}")
        if error < tolerance:
            print(f"solution converged after {iteration+1} iteration")
            break
        iteration += 1

    return x

print("\nConjugate Gradient method:")
x_conjugate_gradient = conjugate_gradient(A, b, x_true)



'''
Output:

Gauss-Seidel method:
Iteration 1: Solution = [ 5.          0.375      -0.02708333 -0.171875    0.00858333], error is 288.4166
Iteration 2: Solution = [ 5.80729167  0.39316146 -0.03994939 -0.27934831  0.00918665], error is 206.9475
Iteration 3: Solution = [ 6.39990777  0.4021487  -0.04965643 -0.35485038  0.0096032 ], error is 147.1923
Iteration 4: Solution = [ 6.82145972  0.40816279 -0.05656818 -0.40850441  0.00989863], error is 104.6871
Iteration 5: Solution = [ 7.12128154  0.41242668 -0.06148429 -0.44666284  0.01010873], error is 74.4561
Iteration 6: Solution = [ 7.33452235  0.41545876 -0.06498077 -0.473802    0.01025815], error is 52.9550
Iteration 7: Solution = [ 7.48618448  0.41761523 -0.06746755 -0.49310404  0.01036442], error is 37.6629
Iteration 8: Solution = [ 7.59405032  0.41914897 -0.06923621 -0.50683212  0.01044001], error is 26.7868
Iteration 9: Solution = [ 7.67076717  0.4202398  -0.07049412 -0.51659588  0.01049376], error is 19.0514
Iteration 10: Solution = [ 7.72533009  0.42101563 -0.07138878 -0.5235401   0.010532  ], error is 13.5498
Iteration 11: Solution = [ 7.76413658  0.42156741 -0.07202509 -0.528479    0.01055919], error is 9.6370
Iteration 12: Solution = [ 7.79173672  0.42195986 -0.07247764 -0.53199167  0.01057853], error is 6.8541
Iteration 13: Solution = [ 7.81136661  0.42223897 -0.07279951 -0.53448996  0.01059229], error is 4.8748
Iteration 14: Solution = [ 7.82532788  0.42243749 -0.07302843 -0.53626681  0.01060207], error is 3.4671
Iteration 15: Solution = [ 7.83525748  0.42257868 -0.07319124 -0.53753055  0.01060903], error is 2.4659
Iteration 16: Solution = [ 7.84231965  0.42267909 -0.07330704 -0.53842936  0.01061397], error is 1.7538
Iteration 17: Solution = [ 7.84734244  0.42275051 -0.0733894  -0.53906861  0.01061749], error is 1.2473
Iteration 18: Solution = [ 7.85091478  0.42280131 -0.07344797 -0.53952326  0.01062   ], error is 0.8871
Iteration 19: Solution = [ 7.85345551  0.42283743 -0.07348963 -0.53984662  0.01062178], error is 0.6310
Iteration 20: Solution = [ 7.85526254  0.42286313 -0.07351926 -0.5400766   0.01062304], error is 0.4487
Iteration 21: Solution = [ 7.85654774  0.4228814  -0.07354034 -0.54024017  0.01062394], error is 0.3192
Iteration 22: Solution = [ 7.85746181  0.4228944  -0.07355533 -0.5403565   0.01062459], error is 0.2270
Iteration 23: Solution = [ 7.85811192  0.42290364 -0.07356599 -0.54043924  0.01062504], error is 0.1614
Iteration 24: Solution = [ 7.8585743   0.42291022 -0.07357357 -0.54049808  0.01062536], error is 0.1148
Iteration 25: Solution = [ 7.85890315  0.42291489 -0.07357896 -0.54053994  0.0106256 ], error is 0.0817
Iteration 26: Solution = [ 7.85913703  0.42291822 -0.07358279 -0.5405697   0.01062576], error is 0.0581
Iteration 27: Solution = [ 7.85930338  0.42292058 -0.07358552 -0.54059088  0.01062588], error is 0.0413
Iteration 28: Solution = [ 7.85942169  0.42292227 -0.07358746 -0.54060593  0.01062596], error is 0.0294
Iteration 29: Solution = [ 7.85950584  0.42292346 -0.07358884 -0.54061664  0.01062602], error is 0.0209
Iteration 30: Solution = [ 7.85956568  0.42292431 -0.07358982 -0.54062426  0.01062606], error is 0.0149
Iteration 31: Solution = [ 7.85960825  0.42292492 -0.07359052 -0.54062968  0.01062609], error is 0.0106
Iteration 32: Solution = [ 7.85963852  0.42292535 -0.07359102 -0.54063353  0.01062611], error is 0.0075
solution converged after 32 iteration

Relaxation method:
Iteration 1: Solution = [ 6.25        0.4296875  -0.05875651 -0.41870117  0.01247675], error is 161.4409
Iteration 2: Solution = [ 7.40305583  0.40261432 -0.06813354 -0.49775864  0.00984042], error is 45.9149
Iteration 3: Solution = [ 7.68442818  0.42154101 -0.07136675 -0.52326829  0.01070397], error is 17.6163
Iteration 4: Solution = [ 7.78189882  0.42099463 -0.07256449 -0.53257501  0.0105493 ], error is 7.8262
Iteration 5: Solution = [ 7.82352548  0.42231612 -0.07311119 -0.53686231  0.010619  ], error is 3.6393
Iteration 6: Solution = [ 7.84250543  0.42258334 -0.07336146 -0.53884142  0.0106153 ], error is 1.7307
Iteration 7: Solution = [ 7.85152701  0.42277371 -0.07348303 -0.53978369  0.01062286], error is 0.8233
Iteration 8: Solution = [ 7.85580167  0.42285137 -0.07353976 -0.5402329   0.01062411], error is 0.3934
Iteration 9: Solution = [ 7.85784658  0.42289109 -0.0735673  -0.54044711  0.0106253 ], error is 0.1877
Iteration 10: Solution = [ 7.85882144  0.4229094  -0.07358029 -0.54054948  0.01062572], error is 0.0897
Iteration 11: Solution = [ 7.85928733  0.42291833 -0.07358654 -0.54059834  0.01062596], error is 0.0428
Iteration 12: Solution = [ 7.85950973  0.42292254 -0.07358952 -0.54062168  0.01062606], error is 0.0205
Iteration 13: Solution = [ 7.85961597  0.42292456 -0.07359094 -0.54063283  0.01062612], error is 0.0098
solution converged after 13 iteration

Jacobi method:
Iteration 1: Solution = [5.         0.5        0.05       0.5        0.00714286], error is 3.0467
Iteration 2: Solution = [ 2.          0.26428571 -0.0247619  -0.19107143  0.00514286], error is 5.8725
Iteration 3: Solution = [5.94702381 0.4928631  0.02124286 0.21439286 0.0085415 ], error is 2.0597
Iteration 4: Solution = [ 3.57538988  0.30517228 -0.04061796 -0.30925661  0.00668254], error is 4.2923
Iteration 5: Solution = [ 6.59678673e+00  4.79445550e-01 -4.28087540e-03  1.15884605e-02
  9.22994686e-03], error is 1.3813
Iteration 6: Solution = [ 4.7237393   0.33342048 -0.05164802 -0.38914401  0.00774933], error is 3.1410
Iteration 7: Solution = [ 7.03724991  0.46821785 -0.02291367 -0.13601964  0.00969529], error is 0.9191
Iteration 8: Solution = [ 5.56055761  0.35476907 -0.05916069 -0.44303111  0.00852353], error is 2.3023
Iteration 9: Solution = [ 7.33357449  0.45908455 -0.03647902 -0.2436776   0.01001225], error is 0.6064
Iteration 10: Solution = [ 6.17124084  0.37096334 -0.06424109 -0.4790885   0.00908691], error is 1.6904
Iteration 11: Solution = [ 7.53116629  0.45170256 -0.04636839 -0.32231898  0.01022691], error is 0.3965
Iteration 12: Solution = [ 6.61758558  0.38326522 -0.06765017 -0.50297206  0.00949749], error is 1.2433
Iteration 13: Solution = [ 7.66147852  0.44576521 -0.05358876 -0.37985509  0.01037122], error is 0.2570
Iteration 14: Solution = [ 6.94433665  0.39262243 -0.06991618 -0.51859108  0.00979715], error is 0.9162
Iteration 15: Solution = [ 7.74622507  0.4410096  -0.05886867 -0.42201846  0.01046736], error is 0.1658
Iteration 16: Solution = [ 7.18393084  0.39974866 -0.07140468 -0.52863801  0.01001621], error is 0.6763
Iteration 17: Solution = [ 7.80033914  0.43721412 -0.06273583 -0.45296804  0.0105307 ], error is 0.1074
Iteration 18: Solution = [ 7.3599123   0.40518225 -0.07236773 -0.53495951  0.01017659], error is 0.5001
Iteration 19: Solution = [ 7.83404505  0.43419429 -0.06557295 -0.47572511  0.01057184], error is 0.0712
Iteration 20: Solution = [ 7.48939317  0.40932987 -0.07297845 -0.53881583  0.01029421], error is 0.3706
Iteration 21: Solution = [ 7.85430649  0.43179807 -0.06765791 -0.49248749  0.01059805], error is 0.0496
Iteration 22: Solution = [ 7.58482797  0.41249924 -0.07335521 -0.5410621   0.01038062], error is 0.2751
Iteration 23: Solution = [ 7.86583689  0.42990118 -0.06919279 -0.50485621  0.01061434], error is 0.0372
Iteration 24: Solution = [ 7.65529442  0.41492352 -0.07357845 -0.54227443  0.0104442 ], error is 0.2046
Iteration 25: Solution = [ 7.87180263  0.42840268 -0.07032471 -0.51399934  0.01062409], error is 0.0299
Iteration 26: Solution = [ 7.70741891  0.41677962 -0.07370253 -0.54283771  0.01049107], error is 0.1524
Iteration 27: Solution = [ 7.87431139  0.42722109 -0.07116095 -0.52077035  0.01062961], error is 0.0251
Iteration 28: Solution = [ 7.74604598  0.41820197 -0.07376385 -0.54300636  0.01052569], error is 0.1138
Iteration 29: Solution = [ 7.87475009  0.4262909  -0.07177988 -0.52579384  0.01063243], error is 0.0215
Iteration 30: Solution = [ 7.77472312  0.41929284 -0.07378657 -0.54294634  0.01055129], error is 0.0851
Iteration 31: Solution = [ 7.87401813  0.42555969 -0.07223879 -0.52952764  0.01063358], error is 0.0184
Iteration 32: Solution = [ 7.79605234  0.42013015 -0.07378652 -0.54276402  0.01057028], error is 0.0638
Iteration 33: Solution = [ 7.87268761  0.42498563 -0.07257969 -0.53230795  0.01063373], error is 0.0156
Iteration 34: Solution = [ 7.8119454   0.42077331 -0.07377391 -0.54252602  0.01058437], error is 0.0479
Iteration 35: Solution = [ 7.871113    0.42453549 -0.07283339 -0.53438202  0.01063333], error is 0.0131
Iteration 36: Solution = [ 7.82380932  0.42126767 -0.07375518 -0.54227272  0.01059485], error is 0.0360
Iteration 37: Solution = [ 7.8695057   0.42418287 -0.07302253 -0.53593205  0.01063264], error is 0.0110
Iteration 38: Solution = [ 7.83268148  0.4216479  -0.07373429 -0.54202739  0.01060267], error is 0.0271
Iteration 39: Solution = [ 7.86798447  0.4239069  -0.0731638  -0.5370925   0.01063184], error is 0.0091
solution converged after 39 iteration

Conjugate Gradient method:
Iteration 1: Solution = [0.00300832 0.00601665 0.00902497 0.0120333  0.01504162], error is 7.8876
Iteration 2: Solution = [0.04647092 0.09363862 0.12985331 0.18422967 0.00632678], error is 7.8563
Iteration 3: Solution = [0.127921   0.26835779 0.05196309 0.493859   0.00484866], error is 7.8032
Iteration 4: Solution = [0.3059927  0.49147673 0.05351802 0.38951203 0.00577334], error is 7.6121
Iteration 5: Solution = [ 7.85971308  0.42292641 -0.07359224 -0.54064302  0.01062616], error is 0.0000
solution converged after 5 iteration

'''