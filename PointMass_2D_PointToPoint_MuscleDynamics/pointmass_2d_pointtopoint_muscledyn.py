# Optimal control (Linear Quadratic- LQ) of a point mass in 2-D
# Point mass model of the human hand. Point mass controlled by two muscles (1 per dimension). The muscles are modelled as a second order filter.
import numpy as np
import matplotlib.pyplot as plt

N = 100
h = 0.01 # time step for discrete time system
t = np.arange(N + 1) * h * 1000 # simulation time in ms
m = 1.0 # mass of point mass
tau1, tau2 = 0.04, 0.04  # time constants of the second order muscle filter
A = np.array([[1, 0, h, 0, 0, 0, 0, 0], [0, 1, 0, h, 0, 0, 0, 0],
                   [0, 0, 1, 0, h / m, 0, 0, 0], [0, 0, 0, 1, 0, 0, h / m, 0],
                   [0, 0, 0, 0, (1 - h / tau2), h / tau2, 0, 0], [0, 0, 0, 0, 0, (1 - h / tau2), 0, 0],
                   [0, 0, 0, 0, 0, 0, (1 - h / tau1), h / tau1], [0, 0, 0, 0, 0, 0, 0, (1 - h / tau1)]])
A = np.block([[A, np.zeros((8, 1))], [np.zeros((1, 8)), 1]])
B = np.block([[np.zeros((5, 2))], [np.array([[h / tau1, 0], [0, 0], [0, h / tau1]])], [np.zeros((1, 2))]])

x_targ = 0.1 # target x-coordinate
y_targ = 0.2 # target y-coordinate
wp = 1 # Position cost weighting factor
wv = 1.0 # Velocity cost weighting factor
wf = 0.01 # Force cost weighting factor
D = np.block([[np.array([[-1, 0, 0, 0, 0, 0, 0, 0, x_targ], [0, -1, 0, 0, 0, 0, 0, 0, y_targ]])],
                           [np.zeros((6, 2)), np.diag([wv, wv, wf, wf, wf, wf]), np.zeros((6,1))]])
Qf = 1 / 4 * D.transpose() @ D # Final time state cost matrix.
Q = np.diag([wp, wp, wv, wv, wf, 0, wf, 0, 0]) * 0 # State cost set to 0 for all time steps other than final.
R = 0.00002 / N  * np.eye(B.shape[1]) # Control cost matrix

L = np.zeros((N, R.shape[0], A.shape[1])) # feedback gain matrix memory
S = Qf
# backward iteration to find the feedback gain matrices
for step in np.arange(N - 1, -1, -1):
    L[step, :, :] = np.linalg.inv(R + B.transpose() @ S @ B) @ B.transpose() @ S @ A

    S = Q + A.transpose() @ S @ (A - B @ L[step, :, :])

X = np.zeros((A.shape[0], N + 1)) # State memory
X[:, 0] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1])
# forward iteration of the state using calculated feedback gains
for step in range(N):
    u = - L[step, :, :] @ X[:, step]
    X[:, step + 1] = A @ X[:, step] + B @ u

# Plotting the position of the point mass in 2-D, X-Y plane
fig, ax = plt.subplots()
ax.plot(x_targ, y_targ, 'tab:cyan', marker = 'o', markersize = 10)
ax.plot(X[0, :], X[1, :], 'tab:gray')
ax.set_ylim((-0.1, 0.3))
ax.set_xlim((-0.2, 0.2))

# Plotting the evolution of positions and velocities of the point mass versus time
fig2, ax = plt.subplots(2, 1)
ax[0].plot(t, X[0, :], color = 'tab:blue', label = 'x-pos')
ax[0].plot(t, X[1, :], color = 'tab:orange',label = 'y-pos')
ax[0].legend()
ax[1].plot(t, X[2, :], color = 'tab:blue', label = 'x-vel')
ax[1].plot(t, X[3, :], color = 'tab:orange', label = 'y-vel')
ax[1].legend()







