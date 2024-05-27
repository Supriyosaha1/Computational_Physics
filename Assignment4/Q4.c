#include <stdio.h>
#include <stdlib.h>
#include <math.h>

double exponential_distribution(double lambda) {
    double u = rand() / (RAND_MAX + 1.0); // Uniform random number between 0 and 1
    return -log(1 - u) / lambda; // Transformation method for exponential distribution
}

int main() {
    int i, n = 10000;
    double lambda = 0.5; // Mean of the exponential distribution
    double *random_numbers = malloc(n * sizeof(double));

    // Generate random numbers
    for (i = 0; i < n; i++) {
        random_numbers[i] = exponential_distribution(lambda);
    }

    // Write random numbers to a file
    FILE *fp;
    fp = fopen("random_numbers.txt", "w");
    if (fp == NULL) {
        printf("Error opening file.\n");
        return 1;
    }

    for (i = 0; i < n; i++) {
        fprintf(fp, "%lf\n", random_numbers[i]);
    }

    fclose(fp);
    free(random_numbers);

    return 0;
}

