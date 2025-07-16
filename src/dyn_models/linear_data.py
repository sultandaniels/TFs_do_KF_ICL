import numpy as np

def generate_linear_sample(n_positions, nx, ny):
    w = np.random.multivariate_normal(np.zeros(nx), np.eye(nx), size=ny)
    x = np.random.multivariate_normal(np.zeros(nx), np.eye(nx), size=n_positions)
    y = x @ w.T
    
    return {"x": x, "y": y}