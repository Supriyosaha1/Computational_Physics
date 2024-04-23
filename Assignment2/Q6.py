import numpy as np
import matplotlib.pyplot as plt


g = 10  
t1 = 10  

def f(t, x, v):
    return v, -g


def euler(f, t_span, x0, v0, dt):
    t = np.arange(t_span[0], t_span[1] + dt, dt)
    x = np.zeros_like(t)
    v = np.zeros_like(t)
    x[0] = x0
    v[0] = v0

    for i in range(1, len(t)):
        dxdt, dvdt = f(t[i-1], x[i-1], v[i-1])
        x[i] = x[i-1] + dxdt * dt
        v[i] = v[i-1] + dvdt * dt

    return t, x

def shooting_method(f, t_span, x0, xf, guess_range, num_guesses, dt):
    guess_values = np.linspace(guess_range[0], guess_range[1], num_guesses)
    solutions = []

    for guess in guess_values:
        _, x = euler(f, t_span, x0, guess, dt)
        solutions.append(x[-1])

    # Find the index of the solution closest to xf using numpy.argmin
    idx = np.argmin(np.abs(np.array(solutions) - xf))
    optimal_guess = guess_values[idx]

    return optimal_guess


def exact_solution(t):
    return -0.5 * g * t**2 + 5*g*t


x0 = 0
xf = 0
guess_range = [1,50]
num_guesses = 10
dt = 0.01
t_span = [0, t1]

optimal_guess = shooting_method(f, t_span, x0, xf, guess_range, num_guesses, dt)
print("Optimal initial velocity:", optimal_guess)

plt.figure(figsize=(10, 6))

t_exact = np.linspace(0, t1, 100)
plt.plot(t_exact, exact_solution(t_exact), label='Exact Solution', color='black',ls='--')

for guess in np.linspace(guess_range[0], guess_range[1], 5):
    _, x = euler(f, t_span, x0, guess, dt)
    plt.plot(_, x, label=f'Numerical Solution (Guess: {guess:.2f})')

plt.title('Shooting Method for Boundary Value Problem')
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.grid(True)
plt.legend()
plt.show()
