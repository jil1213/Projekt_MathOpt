import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
from scipy.optimize import approx_fprime
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from numpy import sin, cos

# Systemparameter
m = 0.3553
J = 0.0361
a = 0.42
d = 0.005
g = 9.81

# Diskretisierung
N = 50
t0 = 0
t1 = 2.5
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

# Diskrete Nebenbedingungen mit impliziter Trapezregel
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

# Minimierung ohne Bounds
constraints = {'type': 'eq', 'fun': constraint_vector}
result = minimize(cost_function, y0, constraints=[constraints], method='SLSQP', options={'disp': True, 'maxiter': 500})
y_optimal = result.x

x_optimal = y_optimal.reshape(N, 5)[:, 0:4]
u_optimal = y_optimal.reshape(N, 5)[:, 4]

# Minimierung mit Bounds
bounds = [(None, None), (None, None), (-0.8, 0.8), (None, None), (-12, 12)]* N
result_bounded = minimize(cost_function, y_optimal, constraints=[constraints], method='SLSQP', bounds = bounds , options={'disp': True, 'maxiter': 500})
y_bounded = result_bounded.x

x_bounded = y_bounded.reshape(N, 5)[:, 0:4]
u_bounded = y_bounded.reshape(N, 5)[:, 4]

# Interpolation Eingang u
u_interp = interp1d(t, u_bounded, kind='linear')

# Dynamisches System fuer Solve IVP
def dyn(ti, xi):
    return f(xi, u_interp(ti))

sol = solve_ivp(dyn, [t0, t1], x0, t_eval=t)

# Linearisierung um die Trajektorie
def get_A_lin(x, u):
    return np.array([
        [0.0, 1.0, 0.0, 0.0],
        [(m * g * a * cos(x[0]) - m * a * u * sin(x[0])) / (J + m * a ** 2), -d / (J + m * a ** 2), 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 0.0]
    ])

def get_B_lin(x, u):
    return np.array([
        [0.0],
        [m * a * cos(x[0]) / (J + m * a ** 2)],
        [0.0],
        [1.0]
    ])



x_opt = y_bounded.reshape(N, 5)[:, 0:4]
u_opt = y_bounded.reshape(N, 5)[:, 4]

B_list = [get_B_lin(x, u) for x, u in zip(x_opt, u_opt)]

# Referenztrajektorie für Zustand und Eingang
x_ref = interp1d(t, x_opt, kind='linear', axis=0)
u_ref = interp1d(t, u_opt, kind='linear')

def riccati_rvs(t, P_flat):
    x_t = x_ref(t)
    u_t = u_ref(t)
    A = get_A_lin(x_t, u_t)
    B = get_B_lin(x_t, u_t)
    P = P_flat.reshape(4, 4)
    dP = -P @ A - A.T @ P - Q + P @ B @ np.linalg.inv([[R]]) @ B.T @ P
    return dP.flatten()

sol_P = solve_ivp(riccati_rvs, [t1, t0], S.flatten(), t_eval=t[::-1])

P_list = sol_P.y.T.reshape(-1, 4, 4)[::-1]  #Reinfolge umdrehen

K_list = [-(1/R) * (B.T @ P).flatten() for B, P in zip(B_list, P_list)]
k_ref = interp1d(t, K_list, kind='linear', axis=0)



# Regelung + System
def system_regler(t, x):
    theta, omega, s, v = x

    u = u_ref(t) + k_ref(t) @ (x - x_ref(t))

    dx1 = omega
    dx2 = (m * g * a * sin(theta) - d * omega + m * a * u * cos(theta)) / (J + m * a**2)
    dx3 = v
    dx4 = u
    return np.array([dx1, dx2, dx3, dx4])


# Simulation des geregelten Systems
sol_regler = solve_ivp(system_regler, [t0, t1], x0, t_eval=t)

x_sim = sol.y.T
x_regler = sol_regler.y.T

# Berechne u(t) aus der Reglersimulation
u_regler_traj = np.array([
    float(u_ref(ti) + k_ref(ti) @ (xi - x_ref(ti)))
    for ti, xi in zip(t, x_regler)
])

# Berechne Abweichung u_regler - u_bounded (Referenz)
u_diff = u_regler_traj - u_bounded

# Plotten
# Labels für Zustände
state_labels = [r'$\theta$', r'$\omega$', r'$s$', r'$v$']

# Erstelle 6 Subplots (4 Zustände + u(t) + u-Abweichung)
fig, axs = plt.subplots(6, 1, figsize=(12, 16), sharex=True)

# Platz rechts schaffen für die Legenden
fig.subplots_adjust(right=0.78)  # ← wichtig!

# Zustandstitel
state_labels = [r'$\theta$', r'$\omega$', r'$s$', r'$v$']

# Zustände 0–3
for i in range(4):
    axs[i].plot(t, x_optimal[:, i], label='Optimal (ohne Bounds)', linestyle='--')
    axs[i].plot(t, x_bounded[:, i], label='Optimal (mit Bounds)', linestyle='-')
    axs[i].plot(t, x_sim[:, i], label='Simulation ohne Regler', linestyle=':')
    axs[i].plot(t, x_regler[:, i], label='Simulation mit Regler', linestyle='-.')

    if i == 2:  # s hat Bounds
        axs[i].axhline(0.8, color='red', linestyle='--')
        axs[i].axhline(-0.8, color='red', linestyle='--')

    axs[i].set_ylabel(state_labels[i])
    axs[i].grid(True)
    axs[i].legend(loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize=9)

# u(t) Plot
axs[4].plot(t, u_optimal, label='u Optimal (ohne Bounds)', linestyle='--')
axs[4].plot(t, u_bounded, label='u Optimal (mit Bounds)', linestyle='-')
axs[4].plot(t, u_regler_traj, label='u mit Regler', linestyle='-.')

axs[4].axhline(12, color='red', linestyle='--')
axs[4].axhline(-12, color='red', linestyle='--')

axs[4].set_ylabel('u(t)')
axs[4].grid(True)
axs[4].legend(loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize=9)

# u-Abweichung Plot
axs[5].plot(t, u_diff, label=r'$u_{\mathrm{Regler}} - u_{\mathrm{optimal}}$', color='black')
axs[5].axhline(0, color='gray', linestyle='--')
axs[5].set_ylabel('u-Abweichung')
axs[5].set_xlabel('Zeit [s]')
axs[5].grid(True)
axs[5].legend(loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize=9)

# Gesamttitel und Layout
fig.suptitle('Vergleich der Trajektorien und Simulationsverläufe', fontsize=15)
plt.tight_layout(rect=[0, 0.03, 0.95, 0.97])  # Platz für Titel & rechtsbündige Legenden
plt.savefig("trajektorien_vergleich.pdf", dpi=500,)
plt.show()