#include <stdio.h>
#include <math.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_fft_complex.h>

#define REAL(z,i) ((z)[2*(i)])
#define IMAG(z,i) ((z)[2*(i)+1])
#define N 200
#define xmin -50
#define xmax 50


double sinc(double x) {
    if (x == 0.0){
        return 1.0;
    } else {
        return sin(x)/x;
    }
}

int main() {
    double Delta = (double)  (xmax-xmin) / (N - 1);
    double dft[2*N];

    gsl_fft_complex_wavetable * wavetable;
    gsl_fft_complex_workspace * workspace;

    for (int i=0; i < N; i++) {
        double x = xmin + i * Delta;
        REAL(dft,i) = sinc(x);
        IMAG(dft,i) = 0.0;
    }
    wavetable = gsl_fft_complex_wavetable_alloc(N);
    workspace = gsl_fft_complex_workspace_alloc(N);

    gsl_fft_complex_forward(dft, 1, N, wavetable, workspace);

    FILE *file = fopen("q3_1_data.csv","w");
    for (int i = 0; i < N; i++) {
        fprintf(file, "%g, %g\n", REAL(dft,i), IMAG(dft,i));
    }

    gsl_fft_complex_wavetable_free (wavetable);
    gsl_fft_complex_workspace_free (workspace);
    fclose(file);
    return 0;
}