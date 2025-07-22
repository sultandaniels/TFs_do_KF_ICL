import numpy as np
from tqdm import tqdm

def generate_linear_sample(n_positions, nx, ny, num_traces):
    w = np.random.multivariate_normal(np.zeros(nx), np.eye(nx), size=ny)
    x = np.random.multivariate_normal(np.zeros(nx), np.eye(nx), size=(num_traces, n_positions))
    y = x @ w.T
    return {"x": x, "y": y}