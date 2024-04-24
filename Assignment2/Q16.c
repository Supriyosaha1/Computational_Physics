#include <stdio.h>
#include <math.h>

double exact_solution(double t) {
    return pow(t + 1, 2) - 0.5 * exp(t);
}

double f(double t, double y) {
    return y - pow(t, 2) + 1;
}

int main() {
    double t0 = 0.0; 
    double y0 = 0.5; 
    double h = 0.2;  
    double t, y, y_exact, error, error_bound;

    printf("t\t\tEuler's y\tExact y\t\tError\t\tError Bound\n");
    printf("-----------------------------------------------------------------------------\n");

    t = t0;
    y = y0;

    
    double max_second_derivative = 0.0;

    while (t <= 2.0) {
        y_exact = exact_solution(t);
        error = fabs(y - y_exact);

        
        double second_derivative = 2 - 0.5 * exp(t);
        if (fabs(second_derivative) > max_second_derivative) {
            max_second_derivative = fabs(second_derivative);
        }

        // Calculate Lipschitz constant L and maximum second derivative M
        double L = 1; // Example value for Lipschitz constant
        double M = max_second_derivative;

        
        error_bound = (h / 2) * (M / L) * (exp(L * (t - t0)) - 1);

        printf("%.2f\t\t%.5f\t\t%.5f\t\t%.5f\t\t%.5f\n", t, y, y_exact, error, error_bound);

      
        y = y + h * f(t, y);
        t = t + h;
    }

    return 0;
}
