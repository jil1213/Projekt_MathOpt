import numpy as np
from scipy.optimize import minimize, approx_fprime
from numpy import sin, cos

# Systemparameter
m, J, a, d, g = 0.3553, 0.0361, 0.42, 0.005, 9.81

# Diskretisierung
N = 50
tau = np.linspace(0, 1, N)
dtau = 1 / (N - 1)

# Anfangs- und Zielbedingung
x0 = np.array([np.pi, 0, -0.5, 0])
x_target = np.array([0, 0, 0, 0])  # Ziel: System in Ruhelage

# Systemdynamik
def f(x, u):
    theta, omega, s, v = x
    dxdt = np.zeros(4)
    dxdt[0] = omega
    dxdt[1] = (m * g * a * sin(theta) - d * omega + m * a * u * cos(theta)) / (J + m * a**2)
    dxdt[2] = v
    dxdt[3] = u
    return dxdt

# Kostenfunktion: minimiere T
def cost_function(y):
    return y[-1]  # letzte Variable ist T

# Nebenbedingungen (Trapezregel, Anfangsbedingung, Endbedingung)
def constraint_vector(y):
    T = y[-1]
    constraints = []
    
    for i in range(1, N):
        xi_prev = y[(i - 1)*5:(i - 1)*5 + 4]
        ui_prev = y[(i - 1)*5 + 4]
        xi = y[i*5:i*5 + 4]
        ui = y[i*5 + 4]

        fi_prev = f(xi_prev, ui_prev)
        fi = f(xi, ui)

        dyn = xi - xi_prev - (T * dtau / 2) * (fi_prev + fi)
        constraints.append(dyn)

    x_start = y[0:4]
    x_end = y[(N - 1)*5:(N - 1)*5 + 4]

    constraints.insert(0, x_start - x0)
    constraints.append(x_end - x_target)

    return np.concatenate(constraints)

# Startwert y0 (Zustände, Steuerung, Zeit)
y0 = np.zeros(N * 5 + 1)
for i in range(N):
    y0[i*5 + 0] = np.pi
    y0[i*5 + 2] = -0.5
y0[-1] = 2.5  # Startwert für T

# Bounds für T und optional für s, u
bounds = [(None, None), (None, None), (-0.8, 0.8), (None, None), (-12, 12)]* N + [(0.1, 10.0)]  # T in [0.1, 10]


# ...existing code...

# Constraint-Dictionary
constraints = {'type': 'eq', 'fun': constraint_vector}

# Optimierung mit trust-constr
result_trust = minimize(cost_function, y0, method='trust-constr',
                        bounds=bounds,
                        constraints=[constraints],
                        options={'disp': True, 'maxiter': 500})

y_opt_trust = result_trust.x
T_opt_trust = y_opt_trust[-1]
x_opt_trust = y_opt_trust[:-1].reshape(N, 5)[:, 0:4]
u_opt_trust = y_opt_trust[:-1].reshape(N, 5)[:, 4]
t_opt_trust = T_opt_trust * tau  # echte Zeitachse

# Optimierung mit SLSQP
result_slsqp = minimize(cost_function, y0, method='SLSQP',
                        bounds=bounds,
                        constraints=[constraints],
                        options={'disp': True, 'maxiter': 500})

y_opt_slsqp = result_slsqp.x
T_opt_slsqp = y_opt_slsqp[-1]
x_opt_slsqp = y_opt_slsqp[:-1].reshape(N, 5)[:, 0:4]
u_opt_slsqp = y_opt_slsqp[:-1].reshape(N, 5)[:, 4]
t_opt_slsqp = T_opt_slsqp * tau  # echte Zeitachse

# Plot (Vergleich)
import matplotlib.pyplot as plt
labels = [r'$\theta$', r'$\omega$', r'$s$', r'$v$']

plt.figure(figsize=(10, 8))
for i in range(4):
    plt.subplot(5, 1, i + 1)
    plt.plot(t_opt_trust, x_opt_trust[:, i], label=f'trust-constr (T={T_opt_trust:.3f}s)')
    plt.plot(t_opt_slsqp, x_opt_slsqp[:, i], label=f'SLSQP (T={T_opt_slsqp:.3f}s)', linestyle='--')
    plt.ylabel(labels[i])
    plt.grid()
    plt.legend()
plt.subplot(5, 1, 5)
plt.plot(t_opt_trust, u_opt_trust, label='trust-constr')
plt.plot(t_opt_slsqp, u_opt_slsqp, label='SLSQP', linestyle='--')
plt.ylabel('u(t)')
plt.xlabel('Zeit [s]')
plt.grid()
plt.legend()
plt.suptitle('Zeitoptimale Steuerung — Vergleich trust-constr vs. SLSQP')
plt.tight_layout()
plt.show()
