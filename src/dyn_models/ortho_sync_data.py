import numpy as np
import numpy.linalg as lin
from .filtering_lti import gen_rand_ortho_haar_real, FilterSim


def gen_sync_powers(Q, sync_ind, context):
    powers = np.empty((context, *Q.shape), dtype=Q.dtype)
    for i in range(context):
        p = i - sync_ind
        if p < 0:
            mat = Q.T
        else:
            mat = Q
        powers[i] = lin.matrix_power(mat, abs(p))
    return powers

def gen_sync_trace(powers, x):
    return powers @ x

def gen_ortho_mats(num_sys, n):
    ortho_mats = []
    sim_objs = []
    for i in range(num_sys):
        # Generate a random orthogonal matrix using the Haar measure
        Q = gen_rand_ortho_haar_real(n)
        fsim = FilterSim(n, n, 0.0, 0.0, tri="ortho", C_dist="_ident_C", n_noise=1, new_eig = False)
        fsim.A = Q
        ortho_mats.append(Q)
        sim_objs.append(fsim)
    return ortho_mats, sim_objs

def gen_ortho_sync_data(num_sys, n, context, sync_ind, num_traces):
    """
    Generate synchronized data for multiple systems with orthogonal matrices.
    
    Parameters:
    num_sys (int): Number of systems.
    n (int): Dimension of the state space.
    context (int): Context length for the data.
    sync_ind (int): Synchronization index.
    num_traces (int): Number of traces to generate for each system.
    
    Returns:
    list: List of generated synchronized data for each system.
    """
    ortho_mats, sim_objs = gen_ortho_mats(num_sys, n)

    rng = np.random.default_rng()
    x0s = rng.standard_normal((num_traces, n))/np.sqrt(n)
    print(f"x0s.shape: {x0s.shape}")
    
    # Generate powers and traces
    powers = [gen_sync_powers(Q, sync_ind, context) for Q in ortho_mats]
    print(f"len(powers): {len(powers)}")
    print(f"len(powers[0]): {len(powers[0])}")
    observations = []
    for i in range(num_sys):
        for j in range(num_traces):
            obs = gen_sync_trace(powers[i], x0s[j])
            observations.append({"obs": obs})
    
    return observations, ortho_mats, sim_objs


if __name__ == "__main__":
    num_sys = 3
    n = 5
    context = 12
    sync_ind = 10
    num_traces = 6

    traces, ortho_mats, sim_objs = gen_ortho_sync_data(num_sys, n, context, sync_ind, num_traces)
    





