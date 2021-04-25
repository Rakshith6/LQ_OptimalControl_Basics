# Redundant system: a cursor inbetween two hands modeled as point masses.
# LQ optimal control for the two hands to move the cursor along a predefined trajectory i.e. trajectory tracking
# Second order filter based muscle activation model for force output in the dimensions x and y

import numpy as np
import matplotlib.pyplot as plt

N = 100 # total number of time steps
h = 0.01 # time step for discrete time system
t = np.arange(N + 1) * h * 1000 # time in ms
m = 1.0 # mass of point mass

# linear system matrices
Ah = np.array([[1, 0, h, 0], [0, 1, 0, h], [0, 0, 1, 0], [0, 0, 0, 1]])
A = np.block([[1, 0, 0, 0, 0.5 * h, 0, 0, 0, 0.5 * h, 0, 0],
              [0, 1, 0, 0, 0, 0.5 * h, 0, 0, 0, 0.5 * h, 0],
              [np.zeros((Ah.shape[0], 2)), Ah, np.zeros((Ah.shape[0], 5))],
              [np.zeros((Ah.shape[0], 6)), Ah, np.zeros((Ah.shape[0], 1))],
              [np.zeros((1, 10)), 1]])
B = np.block([[np.zeros((4, 4))],
              [h / m, 0, 0, 0],
             [0, h / m, 0, 0],
              [np.zeros((2,4))],
              [0, 0, h / m, 0],
              [0, 0, 0, h / m],
              [0, 0, 0, 0]])

# target trajectory i.e. x and y target positions
x_targ = 0.1 * np.cos(2 * np.pi / N * np.arange(N + 1))
y_targ = 0.1 * np.sin(2 * np.pi / N * np.arange(N + 1))

# noise characterstics
wp = 1
wv = 0.1

# cost matrices
D = np.block([[-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, x_targ[-1]],
              [0, -1, 0, 0, 0, 0, 0, 0, 0, 0, y_targ[-1]],
              [np.zeros((2, 11))],
              [0, 0, 0, 0, wv, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, wv, 0, 0, 0, 0, 0],
              [np.zeros((2, 11))],
              [0, 0, 0, 0, 0, 0, 0, 0, wv, 0, 0]])
Qf = 1 / 4 * D.transpose() @ D
Q = np.zeros_like(Qf)
R = 0.000002 / N  * np.eye(B.shape[1])

# determining the control feedback gains from backward recursion
L = np.zeros((N, R.shape[0], A.shape[1]))
S = Qf
for step in np.arange(N - 1, -1, -1):
    L[step, :, :] = np.linalg.inv(R + B.transpose() @ S @ B) @ B.transpose() @ S @ A

    if step < N - 1:
        P = np.array([[-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, x_targ[step]], [0, -1, 0, 0, 0, 0, 0, 0, 0, 0, y_targ[step]]])
        Q = 1 / (N + 2) * P.transpose() @ P

    S = Q + A.transpose() @ S @ (A - B @ L[step, :, :])

# forwarding the states from the initial state using computed feedback gains
X = np.zeros((A.shape[0], N + 1))
X[:, 0] = np.array([x_targ[0], y_targ[0], x_targ[0] + 0.05, 0, 0, 0, x_targ[0] - 0.05, 0, 0, 0, 1])
for step in range(N):
    u = - L[step, :, :] @ X[:, step]
    X[:, step + 1] = A @ X[:, step] + B @ u

# plotting the target and observed trajectories
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_aspect('equal')
ax.plot(x_targ, y_targ, 'tab:cyan', linewidth=4, linestyle = 'dashed', label = 'target trajec')
ax.plot(X[0, :], X[1, :], 'tab:gray', label = 'observed trajec')
ax.plot(X[2, :], X[3, :], 'tab:blue', label = 'R Hand')
ax.plot(X[6, :], X[7, :], 'tab:orange', label = 'L Hand')
ax.set_ylim((-0.4, 0.4))
ax.set_xlim((-0.3, 0.3))
ax.legend()

# evolution of states
fig2, ax = plt.subplots(2, 1)
ax[0].plot(t, X[0, :], color = 'tab:blue', label = 'x-pos')
ax[0].plot(t, X[1, :], color = 'tab:orange',label = 'y-pos')
ax[0].legend()
ax[1].plot(t, X[2, :], color = 'tab:blue', label = 'x-vel')
ax[1].plot(t, X[3, :], color = 'tab:orange', label = 'y-vel')
ax[1].legend()







