#include <stdio.h>
#include <math.h>
#include <complex.h>
#include <fftw3.h>

#define N 200
#define xlim 50

double sinc(double x) {
    if (x == 0.0){
        return 1.0;
    } else {
        return sin(x)/x;
    }
}

int main() {
    double delta = (double) 2 * xlim / (N - 1);

    fftw_complex in[N], out[N];
    fftw_plan p;
    
    for (int i=0; i < N; i++) {
        in[i] = sinc(-xlim + i * delta) + I * 0.0;
    }

    p = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);

    fftw_execute(p);

    FILE *file = fopen("q2_data.csv","w");

    // Calculate normalization factor
    double dx = (double) 2 * xlim / (N - 1);
    double norm_factor = dx * sqrt(N / (2 * M_PI));

    for (int i = 0; i < N; i++) {
        fprintf(file, "%g, %g\n", creal(out[i]) * norm_factor, cimag(out[i]) * norm_factor);
    }

    fclose(file);

    fftw_destroy_plan(p);
    fftw_cleanup();

    printf("Fourier transform data has been written to 'q2_data.csv'.\n");

    return 0;
}
