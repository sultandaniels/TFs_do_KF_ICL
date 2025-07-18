import numpy as np

def generate_linear_sample(n_positions, nx, ny, num_traces):
    w = np.random.multivariate_normal(np.zeros(nx), np.eye(nx), size=ny)
    x = np.random.multivariate_normal(np.zeros(nx), np.eye(nx), size=(n_positions, num_traces))
    y = x @ w.T
    return {"x": x, "y": y}