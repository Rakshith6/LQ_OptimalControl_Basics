# LQ optimal control for a point mass model of the hand to trace a predefined trajectory i.e. trajectory tracking
# Second order filter based muscle activation model for force output in the dimensions x and y

import numpy as np
import matplotlib.pyplot as plt

N = 100
h = 0.01 # time step for discrete time system
t = np.arange(N + 1) * h * 1000 # time in ms
m = 1.0 # mass of point mass
tau1, tau2 = 0.04, 0.04  # time constants of the second order muscle filter
A = np.array([[1, 0, h, 0, 0, 0, 0, 0], [0, 1, 0, h, 0, 0, 0, 0],
                   [0, 0, 1, 0, h / m, 0, 0, 0], [0, 0, 0, 1, 0, 0, h / m, 0],
                   [0, 0, 0, 0, (1 - h / tau2), h / tau2, 0, 0], [0, 0, 0, 0, 0, (1 - h / tau2), 0, 0],
                   [0, 0, 0, 0, 0, 0, (1 - h / tau1), h / tau1], [0, 0, 0, 0, 0, 0, 0, (1 - h / tau1)]])
A = np.block([[A, np.zeros((8, 1))], [np.zeros((1, 8)), 1]])
B = np.block([[np.zeros((5, 2))], [np.array([[h / tau1, 0], [0, 0], [0, h / tau1]])], [np.zeros((1, 2))]])

# target trajectory i.e. x and y target positions
x_targ = 0.1 * np.cos(2 * np.pi / N * np.arange(N + 1))
y_targ = 0.1 * np.sin(2 * np.pi / N * np.arange(N + 1))

# noise characterstics
wp = 1
wv = 0.1
wf = 0.01

# cost matrices
D = np.block([[-1, 0, 0, 0, 0, 0, 0, 0, x_targ[-1]],
              [0, -1, 0, 0, 0, 0, 0, 0, y_targ[-1]],
              [np.zeros((6, 2)), np.diag([wv, wv, wf, wf, wf, wf]), np.zeros((6,1))]])
Qf = 1 / (N + 2) * D.transpose() @ D
Q = np.diag([wp, wp, wv, wv, wf, 0, wf, 0, 0]) * 0
R = 0.000000002 / N  * np.eye(B.shape[1])

# determining the control feedback gains from backward recursion
L = np.zeros((N, R.shape[0], A.shape[1]))
S = Qf
for step in np.arange(N - 1, -1, -1):
    L[step, :, :] = np.linalg.inv(R + B.transpose() @ S @ B) @ B.transpose() @ S @ A

    if step < N - 1:
        P = np.array([[-1, 0, 0, 0, 0, 0, 0, 0, x_targ[step]], [0, -1, 0, 0, 0, 0, 0, 0, y_targ[step]]])
        Q = 1 / (N + 2) * P.transpose() @ P

    S = Q + A.transpose() @ S @ (A - B @ L[step, :, :])

# forwarding the states from the initial state using computed feedback gains
X = np.zeros((A.shape[0], N + 1))
X[:, 0] = np.array([x_targ[0], y_targ[0], 0, 0, 0, 0, 0, 0, 1])
for step in range(N):
    u = - L[step, :, :] @ X[:, step]
    X[:, step + 1] = A @ X[:, step] + B @ u

# plotting the target and observed trajectories
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_aspect('equal')
ax.plot(x_targ, y_targ, 'tab:grey', linewidth=4, linestyle = 'dashed', label = 'target trajec')
ax.plot(X[0, :], X[1, :], 'tab:blue', label = 'observed trajec')
ax.set_ylim((-0.2, 0.2))
ax.set_xlim((-0.2, 0.2))
ax.legend()

# evolution of states
fig2, ax = plt.subplots(2, 1)
ax[0].plot(t, X[0, :], color = 'tab:blue', label = 'x-pos')
ax[0].plot(t, X[1, :], color = 'tab:orange',label = 'y-pos')
ax[0].legend()
ax[1].plot(t, X[2, :], color = 'tab:blue', label = 'x-vel')
ax[1].plot(t, X[3, :], color = 'tab:orange', label = 'y-vel')
ax[1].legend()







