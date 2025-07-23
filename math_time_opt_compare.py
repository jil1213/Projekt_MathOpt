import numpy as np
from scipy.optimize import minimize
from numpy import sin, cos
import matplotlib.pyplot as plt

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
    """Systemdynamik für das Pendel auf dem Wagen."""
    theta, omega, s, v = x
    dxdt = np.zeros(4)
    dxdt[0] = omega
    dxdt[1] = (m * g * a * sin(theta) - d * omega + m * a * u * cos(theta)) / (J + m * a**2)
    dxdt[2] = v
    dxdt[3] = u
    return dxdt

# Kostenfunktion für freie Endzeit
def cost_function_free(y):
    """Minimiere die Endzeit T (letzte Variable in y)."""
    return y[-1]

# Kostenfunktion für feste Endzeit
def cost_function_fixed(y):
    """Dummy-Kostenfunktion für feste Endzeit (z.B. minimiere Steuerenergie)."""
    return np.sum(y[4::5]**2) * dtau  # Summe u^2

# Nebenbedingungen für freie Endzeit
def constraint_vector_free(y):
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

# Nebenbedingungen für feste Endzeit
def constraint_vector_fixed(y, T_fixed):
    constraints = []
    for i in range(1, N):
        xi_prev = y[(i - 1)*5:(i - 1)*5 + 4]
        ui_prev = y[(i - 1)*5 + 4]
        xi = y[i*5:i*5 + 4]
        ui = y[i*5 + 4]
        fi_prev = f(xi_prev, ui_prev)
        fi = f(xi, ui)
        dyn = xi - xi_prev - (T_fixed * dtau / 2) * (fi_prev + fi)
        constraints.append(dyn)
    x_start = y[0:4]
    x_end = y[(N - 1)*5:(N - 1)*5 + 4]
    constraints.insert(0, x_start - x0)
    constraints.append(x_end - x_target)
    return np.concatenate(constraints)

# Optimierung mit fixer Endzeit für beide Methoden
def optimize_fixed_endtime(T_fixed=2.5, method='SLSQP'):
    """Optimiert das System mit fixer Endzeit T_fixed und gewähltem Optimierer."""
    y0 = np.zeros(N * 5)
    for i in range(N):
        y0[i*5 + 0] = np.pi
        y0[i*5 + 2] = -0.5
    bounds = [(None, None), (None, None), (-0.8, 0.8), (None, None), (-12, 12)] * N
    constraints = {'type': 'eq', 'fun': lambda y: constraint_vector_fixed(y, T_fixed)}
    result = minimize(cost_function_fixed, y0, method=method, bounds=bounds, constraints=[constraints], options={'disp': True, 'maxiter': 500})
    y_opt = result.x
    x_opt = y_opt.reshape(N, 5)[:, 0:4]
    u_opt = y_opt.reshape(N, 5)[:, 4]
    t_opt = T_fixed * tau
    np.savez(f'traj_fixed_{method.lower()}.npz', x_opt=x_opt, u_opt=u_opt, t_opt=t_opt, T_fixed=T_fixed)
    return x_opt, u_opt, t_opt, T_fixed

# Optimierung mit freier Endzeit für beide Methoden
def optimize_free_endtime(method='SLSQP'):
    """Optimiert das System mit freier Endzeit T und gewähltem Optimierer."""
    y0 = np.zeros(N * 5 + 1)
    for i in range(N):
        y0[i*5 + 0] = np.pi
        y0[i*5 + 2] = -0.5
    y0[-1] = 2.5  # Startwert für T
    bounds = [(None, None), (None, None), (-0.8, 0.8), (None, None), (-12, 12)] * N + [(0.1, 10.0)]
    constraints = {'type': 'eq', 'fun': constraint_vector_free}
    result = minimize(cost_function_free, y0, method=method, bounds=bounds, constraints=[constraints], options={'disp': True, 'maxiter': 500})
    y_opt = result.x
    T_opt = y_opt[-1]
    x_opt = y_opt[:-1].reshape(N, 5)[:, 0:4]
    u_opt = y_opt[:-1].reshape(N, 5)[:, 4]
    t_opt = T_opt * tau
    np.savez(f'traj_free_{method.lower()}.npz', x_opt=x_opt, u_opt=u_opt, t_opt=t_opt, T_opt=T_opt)
    return x_opt, u_opt, t_opt, T_opt

# Vergleichsfunktion für Trajektorien
def compare_trajectories(x1, x2, t, labels=None, title=None):
    """Vergleicht zwei Trajektorien x1 und x2 über die Zeit t."""
    if labels is None:
        labels = [r'$\theta$', r'$\omega$', r'$s$', r'$v$']
    plt.figure(figsize=(10, 8))
    for i in range(4):
        plt.subplot(4, 1, i + 1)
        plt.plot(t, x1[:, i], label='Trajektorie 1')
        plt.plot(t, x2[:, i], label='Trajektorie 2', linestyle='--')
        plt.ylabel(labels[i])
        plt.grid()
        plt.legend()
    if title:
        plt.suptitle(title)
    plt.xlabel('Zeit [s]')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Fixe Endzeit: SLSQP und trust-constr
    print("Starte Optimierung mit fixer Endzeit (SLSQP)...")
    x_fixed_slsqp, u_fixed_slsqp, t_fixed_slsqp, T_fixed = optimize_fixed_endtime(T_fixed=2.5, method='SLSQP')
    print("Starte Optimierung mit fixer Endzeit (trust-constr)...")
    x_fixed_trust, u_fixed_trust, t_fixed_trust, _ = optimize_fixed_endtime(T_fixed=2.5, method='trust-constr')
    # Freie Endzeit: SLSQP und trust-constr
    print("Starte Optimierung mit freier Endzeit (SLSQP)...")
    x_free_slsqp, u_free_slsqp, t_free_slsqp, T_free_slsqp = optimize_free_endtime(method='SLSQP')
    print("Starte Optimierung mit freier Endzeit (trust-constr)...")
    x_free_trust, u_free_trust, t_free_trust, T_free_trust = optimize_free_endtime(method='trust-constr')
    # Beispiel: Trajektorien laden und vergleichen
    data_fixed_slsqp = np.load('traj_fixed_slsqp.npz')
    data_fixed_trust = np.load('traj_fixed_trust-constr.npz')
    data_free_slsqp = np.load('traj_free_slsqp.npz')
    data_free_trust = np.load('traj_free_trust-constr.npz')
    # Vergleich: Fixe Endzeit SLSQP vs. trust-constr
    compare_trajectories(data_fixed_slsqp['x_opt'], data_fixed_trust['x_opt'], data_fixed_slsqp['t_opt'], title='Fixe Endzeit: SLSQP vs. trust-constr')
    # Vergleich: Freie Endzeit SLSQP vs. trust-constr
    compare_trajectories(data_free_slsqp['x_opt'], data_free_trust['x_opt'], data_free_slsqp['t_opt'], title='Freie Endzeit: SLSQP vs. trust-constr')
    # Optional: Steuerungen vergleichen
    plt.figure()
    plt.plot(data_fixed_slsqp['t_opt'], data_fixed_slsqp['u_opt'], label='Fixe SLSQP')
    plt.plot(data_fixed_trust['t_opt'], data_fixed_trust['u_opt'], label='Fixe trust-constr', linestyle='--')
    plt.xlabel('Zeit [s]')
    plt.ylabel('u(t)')
    plt.legend()
    plt.grid()
    plt.title('Steuerung: Fixe Endzeit')
    plt.show()
    plt.figure()
    plt.plot(data_free_slsqp['t_opt'], data_free_slsqp['u_opt'], label='Freie SLSQP')
    plt.plot(data_free_trust['t_opt'], data_free_trust['u_opt'], label='Freie trust-constr', linestyle='--')
    plt.xlabel('Zeit [s]')
    plt.ylabel('u(t)')
    plt.legend()
    plt.grid()
    plt.title('Steuerung: Freie Endzeit')
    plt.show()
