#include <stdio.h>
#include <math.h>
#include <complex.h>
#include <fftw3.h>

#define N 1001
#define xmin -100
#define xmax 100

double gaussian(double x) {
    return exp(-x*x);
}

int main() {
    fftw_complex f[N], dft[N];
    fftw_plan p;
    double delta = (double) (xmax-xmin) / (N - 1);

    p = fftw_plan_dft_1d(N, f, dft, FFTW_FORWARD, FFTW_ESTIMATE);

    for (int i=0; i < N; i++) {
        f[i] = gaussian(xmin + i * delta) + I * 0.0;
    }

    fftw_execute(p);
    FILE *file = fopen("q4_1_data.csv","w");

    for (int i = 0; i < N; i++) {
        fprintf(file, "%g, %g\n", creal(dft[i]), cimag(dft[i]));
    }

    fclose(file);

    fftw_destroy_plan(p);
    fftw_cleanup();
    return 0;
}