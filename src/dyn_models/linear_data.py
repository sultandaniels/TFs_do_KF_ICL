import numpy as np
from tqdm import tqdm

def generate_linear_sample(n_positions, nx, ny, num_traces):
    w = np.random.multivariate_normal(np.zeros(nx), np.eye(nx)/ny, size=ny)
    x = np.random.multivariate_normal(np.zeros(nx), np.eye(nx)/nx, size=(num_traces, int(n_positions/2)))
    y = x @ w.T
    ones = np.ones((x.shape[0], x.shape[1], 1))
    x = np.concatenate((ones, x), axis=2)
    y = np.concatenate((ones, y), axis=2)

    zeros_x = np.zeros((x.shape[0], x.shape[1], y.shape[2]))
    zeros_y = np.zeros((y.shape[0], y.shape[1], x.shape[2]))
    x = np.concatenate((x, zeros_x), axis=2)
    y = np.concatenate((zeros_y, y), axis=2)

    # Interleave x and y observations to create alternating sequence
    combined = np.zeros((num_traces, n_positions, nx+ny+2))
    combined[:, 0::2, :] = x  # Even positions (0, 2, 4, ...) get x observations
    combined[:, 1::2, :] = y  # Odd positions (1, 3, 5, ...) get y observations

    return {"obs": combined}