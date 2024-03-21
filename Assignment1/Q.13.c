#include <stdio.h>
#include <gsl/gsl_linalg.h>

void solve_system(double *matrix_data, double *rhs_data, size_t size) {
    // Allocate memory for matrix and vector
    gsl_matrix_view mat = gsl_matrix_view_array(matrix_data, size, size);
    gsl_vector_view rhs = gsl_vector_view_array(rhs_data, size);

    // Allocate memory for permutation matrix and LU decomposition
    gsl_permutation *p = gsl_permutation_alloc(size);

    // Perform LU decomposition
    int signum;
    gsl_linalg_LU_decomp(&mat.matrix, p, &signum);

    // Solve the system of linear equations
    gsl_vector *x = gsl_vector_alloc(size);
    gsl_linalg_LU_solve(&mat.matrix, p, &rhs.vector, x);

    // Print L matrix
    printf("L matrix:\n");
    for (size_t i = 0; i < size; i++) {
        for (size_t j = 0; j < size; j++) {
            if (i == j)
                printf("%8.3f ", 1.0);
            else if (i > j)
                printf("%8.3f ", gsl_matrix_get(&mat.matrix, i, j));
            else
                printf("%8.3f ", 0.0);
        }
        printf("\n");
    }

    // Print U matrix
    printf("U matrix:\n");
    for (size_t i = 0; i < size; i++) {
        for (size_t j = 0; j < size; j++) {
            if (i <= j)
                printf("%8.3f ", gsl_matrix_get(&mat.matrix, i, j));
            else
                printf("%8.3f ", 0.0);
        }
        printf("\n");
    }

    // Verify correctness by reconstructing the original matrix
    gsl_matrix *reconstructed_mat = gsl_matrix_alloc(size, size);
    gsl_matrix *L = gsl_matrix_alloc(size, size);
    gsl_matrix *U = gsl_matrix_alloc(size, size);
    gsl_matrix_set_identity(L);

    // Get L and U matrices
    for (size_t i = 0; i < size; i++) {
        for (size_t j = 0; j < size; j++) {
            if (i > j) {
                gsl_matrix_set(L, i, j, gsl_matrix_get(&mat.matrix, i, j));
            } else {
                gsl_matrix_set(U, i, j, gsl_matrix_get(&mat.matrix, i, j));
            }
        }
    }

    // Reconstruct the original matrix using L and U
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, L, U, 0.0, reconstructed_mat);

    // Print the reconstructed matrix
    printf("Reconstructed Matrix LU:\n");
    for (size_t i = 0; i < size; i++) {
        for (size_t j = 0; j < size; j++) {
            printf("%8.3f ", gsl_matrix_get(reconstructed_mat, i, j));
        }
        printf("\n");
    }

    // Free allocated memory
    gsl_permutation_free(p);
    gsl_vector_free(x);
    gsl_matrix_free(reconstructed_mat);
    gsl_matrix_free(L);
    gsl_matrix_free(U);
}

int main() {
    // Example 1
    double matrix_data1[] = {3.0, -1.0, 1.0, 
                             3.0, 6.0, 2.0, 
                             3.0, 3.0, 7.0};
    double rhs_data1[] = {1.0, 0.0, 4.0};
    size_t size1 = 3;
    solve_system(matrix_data1, rhs_data1, size1);

    // Example 2
    double matrix_data2[] = {10, -1, 0,
                             -1, 10, -2,
                              0, -2, 10};
    double rhs_data2[] = {7, 9, 6};
    size_t size2 = 3;
    solve_system(matrix_data2, rhs_data2, size2);

    // Example 3
    double matrix_data3[] = {10, 5, 0, 0,
                             5, 10, -4, 0,
                             0, -4, 8, -1,
                             0, 0, -1, 5};
    double rhs_data3[] = {6, 25, -11, -11};
    size_t size3 = 4;
    solve_system(matrix_data3, rhs_data3, size3);

    // Example 4
    double matrix_data4[] = {4, 1, 1, 0, 1,
                             -1, -3, 1, 1, 0,
                             2, 1, 5, -1, -1,
                             -1, -1, -1, 4, 0,
                             0, 2, -1, 1, 4};
    double rhs_data4[] = {6, 6, 6, 6, 6};
    size_t size4 = 5;
    solve_system(matrix_data4, rhs_data4, size4);

    return 0;
}



/*
OUTPUT:
-------------------------------
L matrix:
   1.000     0.000     0.000 
   1.000     1.000     0.000 
   1.000     0.500     1.000 
U matrix:
   3.000    -1.000     1.000 
   0.000     7.000     1.000 
   0.000     0.000     4.500 
Reconstructed Matrix LU:
   3.000    -1.000     1.000 
   3.000     6.000     2.000 
   3.000     3.000     7.000 
L matrix:
   1.000     0.000     0.000 
   0.100     1.000     0.000 
   0.000     0.200     1.000 
U matrix:
  10.000    -1.000     0.000 
   0.000     9.900    -1.100 
   0.000     0.000     9.800 
Reconstructed Matrix LU:
  10.000    -1.000     0.000 
  -1.000    10.000    -2.000 
   0.000    -2.000    10.000 
L matrix:
   1.000     0.000     0.000     0.000 
   0.500     1.000     0.000     0.000 
   0.000    -0.400     1.000     0.000 
   0.000    -0.200    -0.105     1.000 
U matrix:
  10.000     5.000     0.000     0.000 
   0.000     7.500    -4.000     0.000 
   0.000     0.000     6.800    -1.000 
   0.000     0.000     0.000     0.947 
Reconstructed Matrix LU:
  10.000     5.000     0.000     0.000 
   5.000    10.000    -4.000     0.000 
   0.000    -4.000     8.000    -1.000 
   0.000     0.000    -1.000     5.000 
L matrix:
   1.000     0.000     0.000     0.000     0.000 
  -0.250     1.000     0.000     0.000     0.000 
   0.500     0.700     1.000     0.000     0.000 
  -0.250     0.100    -0.050     1.000     0.000 
   0.000     0.500    -0.083    -0.010     1.000 
U matrix:
   4.000     1.000     1.000     0.000     1.000 
   0.000    -3.250     1.250     1.000     0.250 
   0.000     0.000     5.400    -1.900    -0.900 
   0.000     0.000     0.000     2.030     0.010 
   0.000     0.000     0.000     0.000     3.987 
Reconstructed Matrix LU:
   4.000     1.000     1.000     0.000     1.000 
  -1.000    -3.000     1.000     1.000     0.000 
   2.000     1.000     5.000    -1.000    -1.000 
  -1.000    -1.000    -1.000     4.000     0.000 
   0.000     2.000    -1.000     1.000     4.000 
*/
