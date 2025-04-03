import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp # numerische Berechnung der Zustaende
from scipy.optimize import approx_fprime, minimize # numerischer Gradient und minimize
import control # Riccati-Gleichung

#############################################################
# Auswahl der auszufuehrenden Aufgaben
#############################################################
# Abschlieszende Simulation mit Plotten des Systems
plot = True
# Aufgabe d, Adaptive Schrittweite UseBacktracking
use_backtracking = True
# Aufgabe e, Riccati-Gleichung
Riccati = True
# Aufgabe f, scipy minimize
scipyMinimize = True
##############################################################

# Zeitspanne der Simulation
t_span = (0, 15)
t_eval = np.linspace(t_span[0], t_span[1], 500)

# Systemparameter
m = 1
omega = 2
q1, q2, q3, r1 = 3, 4, 5, 6

# Anfangsbedingungen der Zustaende
x0 = [1, 0, 0, 0]

# PID-Parameter (Startwerte -1)
Kp = -1.0
Kd = -1.0
Ki = -1.0
k_params = np.array([Kp, Kd, Ki], dtype=float)

# Epsilon Schranken
max_iterations = 1000  # Maximale Sicherheit gegen endlose Schleifen
epsilon = 1e-6  # Konvergenzschranke Gradientenabstieg
epsilon_fprime = 1e-6 # Konvergenzschranke numerische Gradeinten berechnung
epsilon_BFGS = 1e-6 # Konvergenzschranke BFGS Methode
epsilon_NelderMead = 1e-6 # Konvergenzschranke Nelder Mead Methode

# Stepsize Gradientenabstieg
alpha = 0.01  # Schrittweite, falls diese Konstant gehalten wird 
epsilon_alpha = 1e-6 # minimale Schrittweite, damit die Schrittweite nicht zu klein wird


# Dynamik des Systems
def system_dynamics(t, state, Kp, Kd, Ki):
    x1, x2, x3, x4 = state
    u = Kp * x1 + Kd * x2 + Ki * x3
    dx1_dt = x2
    dx2_dt = (u - omega**2 * x1) / m
    dx3_dt = x1
    dx4_dt = 0.5 * (q1 * x1**2 + q2 * x2**2 + q3 * x3**2 + r1 * u**2)
    return [dx1_dt, dx2_dt, dx3_dt, dx4_dt]

# Kostenfunktion basierend auf x4
def cost_function(k_params):
    Kp, Kd, Ki = k_params
    sol = solve_ivp(system_dynamics, t_span, x0, t_eval=t_eval, args=(Kp, Kd, Ki))
    return sol.y[3, -1]  # Endwert des Kostenfunktionals (x4)


# Backtracking-Liniensuche
def backtracking_line_search(k_params, grad, alpha=1.0, rho=0.5, c=1e-4):

    cost_old = cost_function(k_params)
    while cost_function(k_params - alpha * grad) > cost_old - c * alpha * np.dot(grad, grad):
        alpha *= rho  # Reduziere Schrittweite
        if alpha < epsilon_alpha:
            break  # Stoppe, wenn Schrittweite zu klein wird
    return alpha

# Gradientenabstieg mit Abbruchbedingung
iteration = 0
while iteration < max_iterations:
    grad = approx_fprime(k_params, cost_function, epsilon_fprime)
    
    # Wähle Schrittweitenregelung
    if use_backtracking:
        alpha = backtracking_line_search(k_params, grad)
    
    gradient_step = alpha * grad
    k_params -= gradient_step
    current_cost = cost_function(k_params)
    iteration += 1

    print(f"Iteration {iteration}: Kosten = {current_cost}, K = {k_params}, Schrittweite = {alpha}")

    # Abbruchbedingung: wenn alle Änderungen kleiner als epsilon sind
    if np.all(np.abs(gradient_step) < epsilon):
        print(f"Konvergenz erreicht nach {iteration} Iterationen.")
        break

    # Sicherheitsbedingung gegen endlose Schleifen falls es nicht konvergiert
    if iteration >= max_iterations:
        print("Maximale Iterationsanzahl erreicht.")
        break

if scipyMinimize:
    # Optimierung mit SciPy
    # Methode BFGS
    result_BFGS = minimize(cost_function, k_params, method='BFGS', jac=False, tol = epsilon_BFGS)
    print(f"SciPy-Ergebnis BFGS Methode:  Kosten = {result_BFGS.fun}, K = {result_BFGS.x}")
    # Methode Nelder Mead
    result_NelderMead = minimize(cost_function, k_params, method='Nelder-Mead', tol = epsilon_NelderMead)
    print(f"SciPy-Ergebnis Nelder-Mead Methode: Kosten = {result_NelderMead.fun}, K = {result_NelderMead.x}")

if Riccati:
    # Riccati-LQR Methode zur Berechnung der optimalen Reglerparameter
    A = np.array([[0, 1, 0],
              [-omega**2, 0, 0],
              [1, 0, 0]])                           
    B = np.array([[0],[1],[0]])
    Q = np.diag([q1, q2, q3])
    R = np.array([[r1]])
    # loesen mitttels control.lqr
    K_lqr, S, E = control.lqr(A, B, Q, R)
    K_lqr = (-1) * K_lqr[0]
    print(f"LQR-Ergebnis: K = {K_lqr}")


# Ergebnisse nach Optimierung plotten
if plot:
    solution = solve_ivp(system_dynamics, t_span, x0, t_eval=t_eval, args=tuple(k_params))

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.grid()
    plt.plot(solution.t, solution.y[0])
    plt.title('Zustand x1: Position')
    plt.xlabel('Zeit (s)')
    plt.ylabel('x1 (Position)')

    plt.subplot(2, 2, 2)
    plt.grid()
    plt.plot(solution.t, solution.y[1])
    plt.title('Zustand x2: Geschwindigkeit')
    plt.xlabel('Zeit (s)')
    plt.ylabel('x2 (Geschwindigkeit)')

    plt.subplot(2, 2, 3)
    plt.grid()
    plt.plot(solution.t, solution.y[2])
    plt.title('Zustand x3: Integrierter Zustand')
    plt.xlabel('Zeit (s)')
    plt.ylabel('x3 (Integrierter Zustand)')

    plt.subplot(2, 2, 4)
    plt.grid()
    plt.plot(solution.t, solution.y[3])
    plt.title('Zustand x4: Kostenfunktional')
    plt.xlabel('Zeit (s)')
    plt.ylabel('x4 (Kosten)')

    plt.tight_layout()
    plt.show()
