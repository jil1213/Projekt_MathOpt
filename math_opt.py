import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
from scipy.optimize import approx_fprime
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from numpy import sin, cos
import os

# Systemparameter
m, J, a, d, g = 0.3553, 0.0361, 0.42, 0.005, 9.81

# Diskretisierung
N = 50
t0 = 0
t1 = 4
t = np.linspace(t0, t1, N)
dt = (t1 - t0) / (N - 1)

# Anfangsbedingung
x0 = np.array([np.pi, 0, -0.5, 0])

# Gewichtungsmatrizen
Q = np.eye(4)
R = 0.1
S = np.diag([0, 0, 5, 0])
M = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1]
])

# Dynamik des Systems f(x, u)
def f(x, u):
    theta, omega, s, v = x

    dxdt = np.zeros_like(x)
    dxdt[0] = omega
    dxdt[1] = (m * g * a * sin(theta) - d * omega + m * a * u * cos(theta)) / (J + m * a ** 2)
    dxdt[2] = v
    dxdt[3] = u

    return dxdt

# Kostenfunktionen
def l(x, u):
    return 1 + 0.5 * (x @ Q @ x + R * u ** 2)

def phi(x):
    return 0.5 * x @ S @ x

def constraint_vector(y):
    constraints = []
    x1 = y[0:4]
    constraints.append(x1 - x0)

    for i in range(1, N):
        xi_prev = y[(i - 1) * 5:(i - 1) * 5 + 4]
        ui_prev = y[(i - 1) * 5 + 4]

        xi = y[i * 5:i * 5 + 4]
        ui = y[i * 5 + 4]

        fi_prev = f(xi_prev, ui_prev)
        fi = f(xi, ui)

        trapz_dyn = xi - xi_prev - (dt / 2) * (fi_prev + fi)
        constraints.append(trapz_dyn)

    xN = y[-5:-1]
    constraints.append(M @ xN)

    # Neue Bedingung: u(0) = 0
    u0 = y[4]
    constraints.append(np.array([u0]))

    return np.concatenate(constraints)


# Gradient als Jacobi Matrix von den Nebenbedingungen
def jac_constraint_vector(y):
    epsilon = np.sqrt(np.finfo(float).eps)
    m = constraint_vector(y).size
    n = y.size
    J = np.zeros((m, n))
    for i in range(m):
        # Funktion für die i-te Komponente
        def gi(y_local):
            return constraint_vector(y_local)[i]
        J[i, :] = approx_fprime(y, gi, epsilon)
    return J

# Kostenfunktion J_d(y)
def cost_function(y):
    J_sum = 0

    for i in range(N - 1):
        xi = y[i * 5:i * 5 + 4]
        ui = y[i * 5 + 4]

        xi_next = y[(i + 1) * 5:(i + 1) * 5 + 4]
        ui_next = y[(i + 1) * 5 + 4]

        J_sum += 0.5 * dt * (l(xi, ui) + l(xi_next, ui_next))
    xN = y[-5:-1]
    return phi(xN) + J_sum

# Gradient der Kostenfunktion
def grad_cost_function(y):
    epsilon = np.sqrt(np.finfo(float).eps)
    return approx_fprime(y, cost_function, epsilon)

#Startwerte y0
y0 = np.zeros(5 * N)
for i in range(N):
    y0[i * 5 + 0] = np.pi  
    y0[i * 5 + 2] = -0.5 


constraints = {'type': 'eq', 'fun': constraint_vector}
# Minimierung ohne Bounds SLSQP
if os.path.exists('result_slsqp.npz'):
    data = np.load('result_slsqp.npz')
    x_optimal_slsqp = data['x_opt']
    u_optimal_slsqp = data['u_opt']
    t = data['t']
else:
    result = minimize(cost_function, y0, constraints=[constraints], method='SLSQP', options={'disp': True, 'maxiter': 1000})
    y_optimal_slsqp = result.x
    x_optimal_slsqp = y_optimal_slsqp.reshape(N, 5)[:, 0:4]
    u_optimal_slsqp = y_optimal_slsqp.reshape(N, 5)[:, 4]
    np.savez('result_slsqp.npz', x_opt=x_optimal_slsqp, u_opt=u_optimal_slsqp, t=t)

# Minimierung ohne Bounds trust-constr
if os.path.exists('result_trust.npz'):
    data = np.load('result_trust.npz')
    x_optimal_trust = data['x_opt']
    u_optimal_trust = data['u_opt']
else:
    result = minimize(cost_function, y0, constraints=[constraints], method='trust-constr', options={'disp': True, 'maxiter': 1000})
    y_optimal_trust = result.x
    x_optimal_trust = y_optimal_trust.reshape(N, 5)[:, 0:4]
    u_optimal_trust = y_optimal_trust.reshape(N, 5)[:, 4]
    np.savez('result_trust.npz', x_opt=x_optimal_trust, u_opt=u_optimal_trust, t=t)

# Minimierung mit Bounds SLSQP
if os.path.exists('result_bounded_slsqp.npz'):
    data = np.load('result_bounded_slsqp.npz')
    x_bounded_slsqp = data['x_opt']
    u_bounded_slsqp = data['u_opt']
else:
    bounds = [(None, None), (None, None), (-0.8, 0.8), (None, None), (-12, 12)]* N
    result_bounded = minimize(cost_function, y_optimal_slsqp, constraints=[constraints], method='SLSQP', bounds = bounds , options={'disp': True, 'maxiter': 1000})
    y_bounded = result_bounded.x
    x_bounded_slsqp = y_bounded.reshape(N, 5)[:, 0:4]
    u_bounded_slsqp = y_bounded.reshape(N, 5)[:, 4]
    np.savez('result_bounded_slsqp.npz', x_opt=x_bounded_slsqp, u_opt=u_bounded_slsqp, t=t)

# Minimierung mit Bounds trust-constr
if os.path.exists('result_bounded_trust.npz'):
    data = np.load('result_bounded_trust.npz')
    x_bounded_trust = data['x_opt']
    u_bounded_trust = data['u_opt']
else:
    result_bounded = minimize(cost_function, y_optimal_trust, constraints=[constraints], method='trust-constr', bounds = bounds , options={'disp': True, 'maxiter': 1000})
    y_bounded = result_bounded.x
    x_bounded_trust = y_bounded.reshape(N, 5)[:, 0:4]
    u_bounded_trust = y_bounded.reshape(N, 5)[:, 4]
    np.savez('result_bounded_trust.npz', x_opt=x_bounded_trust, u_opt=u_bounded_trust, t=t)



###########################################
# Zeit Optimierung
###########################################

# Diskretisierung
N = 50
tau = np.linspace(0, 1, N)
dtau = 1 / (N - 1)

x_target = np.array([0, 0, 0, 0])  # Ziel: System in Ruhelage

# Kostenfunktion: minimiere T
def cost_function_zeit(y):
    return y[-1]  # letzte Variable ist T

# Nebenbedingungen (Trapezregel, Anfangsbedingung, Endbedingung)
def constraint_vector_zeit(y):
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


# Constraint-Dictionary
constraints = {'type': 'eq', 'fun': constraint_vector_zeit}

# Optimierung mit trust-constr
if os.path.exists('result_time_trust.npz'):
    data = np.load('result_time_trust.npz')
    x_opt_trust = data['x_opt']
    u_opt_trust = data['u_opt']
    t_opt_trust = data['t']
    T_opt_trust = data['T_opt']
else:
    result_trust = minimize(cost_function_zeit, y0, method='trust-constr',
                            bounds=bounds,
                            constraints=[constraints],
                            options={'disp': True, 'maxiter': 500})
    y_opt_trust = result_trust.x
    T_opt_trust = y_opt_trust[-1]
    x_opt_trust = y_opt_trust[:-1].reshape(N, 5)[:, 0:4]
    u_opt_trust = y_opt_trust[:-1].reshape(N, 5)[:, 4]
    t_opt_trust = T_opt_trust * tau  # echte Zeitachse
    np.savez('result_time_trust.npz', x_opt=x_opt_trust, u_opt=u_opt_trust, t=t_opt_trust, T_opt=T_opt_trust)

# Optimierung mit SLSQP
if os.path.exists('result_time_slsqp.npz'):
    data = np.load('result_time_slsqp.npz')
    x_opt_slsqp = data['x_opt']
    u_opt_slsqp = data['u_opt']
    t_opt_slsqp = data['t']
    T_opt_slsqp = data['T_opt']
else:
    result_slsqp = minimize(cost_function_zeit, y0, method='SLSQP',
                            bounds=bounds,
                            constraints=[constraints],
                            options={'disp': True, 'maxiter': 500})
    y_opt_slsqp = result_slsqp.x
    T_opt_slsqp = y_opt_slsqp[-1]
    x_opt_slsqp = y_opt_slsqp[:-1].reshape(N, 5)[:, 0:4]
    u_opt_slsqp = y_opt_slsqp[:-1].reshape(N, 5)[:, 4]
    t_opt_slsqp = T_opt_slsqp * tau  # echte Zeitachse
    np.savez('result_time_slsqp.npz', x_opt=x_opt_slsqp, u_opt=u_opt_slsqp, t=t_opt_slsqp, T_opt=T_opt_slsqp)



###########################################
#Plotten
###########################################

# Einzelplots für jeden Zustand und die Steuerung
state_labels = [r'$\theta$', r'$\omega$', r'$s$', r'$v$']


# Vergleich SLSQP vs. trust-constr (ohne Bounds) — 2:1 Subplot
fig, axs = plt.subplots(5, 2, figsize=(16, 14), sharex=True)
for i in range(4):
    axs[i, 0].plot(t, x_optimal_slsqp[:, i], color='b', linestyle='--')
    axs[i, 0].set_ylabel(state_labels[i])
    axs[i, 0].set_title('SLSQP (ohne Bounds)')
    axs[i, 0].grid(True)
    axs[i, 1].plot(t, x_optimal_trust[:, i], color='r', linestyle='--')
    axs[i, 1].set_title('trust-constr (ohne Bounds)')
    axs[i, 1].grid(True)
for j in range(2):
    axs[4, j].plot(t, u_optimal_slsqp if j==0 else u_optimal_trust, color='b' if j==0 else 'r', linestyle='--')
    axs[4, j].set_ylabel('u(t)')
    axs[4, j].set_xlabel('Zeit [s]')
    axs[4, j].grid(True)
fig.suptitle('Vergleich: SLSQP vs. trust-constr (ohne Bounds) — nebeneinander')
plt.tight_layout()
plt.savefig('vergleich_ohne_bounds_subplot_2x1.pdf', dpi=500)
plt.savefig('vergleich_ohne_bounds_subplot_2x1.png', dpi=200)
plt.show()

# Vergleich SLSQP vs. trust-constr (mit Bounds) — 2:1 Subplot
fig, axs = plt.subplots(5, 2, figsize=(16, 14), sharex=True)
for i in range(4):
    axs[i, 0].plot(t, x_bounded_slsqp[:, i], color='b', linestyle='--')
    axs[i, 0].set_ylabel(state_labels[i])
    axs[i, 0].set_title('SLSQP (mit Bounds)')
    axs[i, 0].grid(True)
    if i == 2:
        axs[i, 0].axhline(0.8, color='orange', linestyle=':')
        axs[i, 0].axhline(-0.8, color='orange', linestyle=':')
    axs[i, 1].plot(t, x_bounded_trust[:, i], color='r', linestyle='--')
    axs[i, 1].set_title('trust-constr (mit Bounds)')
    axs[i, 1].grid(True)
    if i == 2:
        axs[i, 1].axhline(0.8, color='orange', linestyle=':')
        axs[i, 1].axhline(-0.8, color='orange', linestyle=':')
for j in range(2):
    axs[4, j].plot(t, u_bounded_slsqp if j==0 else u_bounded_trust, color='b' if j==0 else 'r', linestyle='--')
    axs[4, j].set_ylabel('u(t)')
    axs[4, j].set_xlabel('Zeit [s]')
    axs[4, j].grid(True)
    axs[4, j].axhline(12, color='orange', linestyle=':')
    axs[4, j].axhline(-12, color='orange', linestyle=':')
fig.suptitle('Vergleich: SLSQP vs. trust-constr (mit Bounds) — nebeneinander')
plt.tight_layout()
plt.savefig('vergleich_mit_bounds_subplot_2x1.pdf', dpi=500)
plt.savefig('vergleich_mit_bounds_subplot_2x1.png', dpi=200)
plt.show()

# Vergleich Zeitoptimierung SLSQP vs. trust-constr (freie Endzeit) — Subplot
fig, axs = plt.subplots(5, 1, figsize=(12, 14), sharex=True)
for i in range(4):
    axs[i].plot(t_opt_slsqp, x_opt_slsqp[:, i], label='SLSQP (freie Endzeit)', color='b', linestyle='--')
    axs[i].plot(t_opt_trust, x_opt_trust[:, i], label='trust-constr (freie Endzeit)', color='r', linestyle='--')
    if i == 2:
        axs[i].axhline(0.8, color='orange', linestyle=':')
        axs[i].axhline(-0.8, color='orange', linestyle=':')
    axs[i].set_ylabel(state_labels[i])
    axs[i].grid(True)
    axs[i].legend(loc='best', fontsize=9)
axs[4].plot(t_opt_slsqp, u_opt_slsqp, label='SLSQP (freie Endzeit)', color='b', linestyle='--')
axs[4].plot(t_opt_trust, u_opt_trust, label='trust-constr (freie Endzeit)', color='r', linestyle='--')
axs[4].axhline(12, color='orange', linestyle=':')
axs[4].axhline(-12, color='orange', linestyle=':')
axs[4].set_ylabel('u(t)')
axs[4].set_xlabel('Zeit [s]')
axs[4].grid(True)
axs[4].legend(loc='best', fontsize=9)
fig.suptitle('Vergleich: SLSQP vs. trust-constr (freie Endzeit)')
plt.tight_layout()
plt.savefig('vergleich_zeitopt_subplot.pdf', dpi=500)
plt.savefig('vergleich_zeitopt_subplot.png', dpi=200)
plt.show()

# Gemeinsamer 2x1-Subplot: links ohne Bounds, rechts mit Bounds
fig, axs = plt.subplots(5, 2, figsize=(16, 14), sharex=True)
# Achsenlimits für alle Subplots (anpassen nach Bedarf)
ylim_theta = (-4, 4)
ylim_omega = (-8, 8)
ylim_s = (-1.2, 1.2)
ylim_v = (-8, 8)
ylim_u = (-14, 14)
for i in range(4):
    # Links: ohne Bounds
    axs[i, 0].plot(t, x_optimal_slsqp[:, i], color='b', linestyle='--', label='SLSQP')
    axs[i, 0].plot(t, x_optimal_trust[:, i], color='r', linestyle='--', label='trust-constr')
    if i == 2:
        axs[i, 0].axhline(0.8, color='orange', linestyle=':')
        axs[i, 0].axhline(-0.8, color='orange', linestyle=':')
        axs[i, 0].set_ylim(ylim_s)
    elif i == 0:
        axs[i, 0].set_ylim(ylim_theta)
    elif i == 1:
        axs[i, 0].set_ylim(ylim_omega)
    elif i == 3:
        axs[i, 0].set_ylim(ylim_v)
    axs[i, 0].set_ylabel(state_labels[i])
    axs[i, 0].set_title('Ohne Bounds')
    axs[i, 0].grid(True)
    axs[i, 0].legend(loc='best', fontsize=9)
    # Rechts: mit Bounds
    axs[i, 1].plot(t, x_bounded_slsqp[:, i], color='b', linestyle='--', label='SLSQP')
    axs[i, 1].plot(t, x_bounded_trust[:, i], color='r', linestyle='--', label='trust-constr')
    if i == 2:
        axs[i, 1].axhline(0.8, color='orange', linestyle=':')
        axs[i, 1].axhline(-0.8, color='orange', linestyle=':')
        axs[i, 1].set_ylim(ylim_s)
    elif i == 0:
        axs[i, 1].set_ylim(ylim_theta)
    elif i == 1:
        axs[i, 1].set_ylim(ylim_omega)
    elif i == 3:
        axs[i, 1].set_ylim(ylim_v)
    axs[i, 1].set_ylabel(state_labels[i])
    axs[i, 1].set_title('Mit Bounds')
    axs[i, 1].grid(True)
    axs[i, 1].legend(loc='best', fontsize=9)
# Steuerung
for j in range(2):
    axs[4, j].plot(t, u_optimal_slsqp if j==0 else u_bounded_slsqp, color='b', linestyle='--', label='SLSQP')
    axs[4, j].plot(t, u_optimal_trust if j==0 else u_bounded_trust, color='r', linestyle='--', label='trust-constr')
    axs[4, j].axhline(12, color='orange', linestyle=':')
    axs[4, j].axhline(-12, color='orange', linestyle=':')
    axs[4, j].set_ylim(ylim_u)
    axs[4, j].set_ylabel('u(t)')
    axs[4, j].set_xlabel('Zeit [s]')
    axs[4, j].set_title('Ohne Bounds' if j==0 else 'Mit Bounds')
    axs[4, j].grid(True)
    axs[4, j].legend(loc='best', fontsize=9)
fig.suptitle('Vergleich: SLSQP vs. trust-constr — Ohne und Mit Bounds')
plt.tight_layout()
plt.savefig('vergleich_ohne_mit_bounds_subplot_2x1.pdf', dpi=500)
plt.savefig('vergleich_ohne_mit_bounds_subplot_2x1.png', dpi=200)
plt.show()
