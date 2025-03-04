import numpy as np
import scipy.linalg as la
import linalg_helpers as la_help
import control as ct



def K_k(Kn, ny, k):
    # The k-th Covariance matrix for the observation process
    return Kn[-ny*(k+1):, -ny*(k+1):]


def compute_cov_inv(A,C,V,Pi, context):
    ny = C.shape[0]

    A_powers = la_help.compute_powers(A, context)

    values = C@A_powers@Pi@C.T
    # #threshold every value below 1e-3 to 0
    # for i in range(len(values)):
    #     values[i] = la_help.lower_threshold_matrix(values[i], 1e-3)

    Kn = la_help.block_toeplitz(values) + la_help.create_repeated_block_diagonal(V, context)

    Kinvs = []
    Finvs = []
    K0 = K_k(Kn, ny, 0)
    bad_k = np.inf
    for k in range(context):

        # #compute condition number of K_k
        # print("k:", k)
        # print("cond(K_k):", np.linalg.cond(K_k(Kn, ny, k)))
        if k == 0:
            Kinvs.append(la.inv(K0))
            Finvs.append(K0)
        else:
            # la_help.print_matrix(K_k(Kn, ny,k), f"K_{k}")
            
            Koff = K_k(Kn, ny, k)[:ny, ny:]

            Finv = (K0 - Koff@Kinvs[k-1]@Koff.T)
            Finvs.append(Finv)

            Fk = la.inv(Finv)
            if not np.allclose(Fk@Finv, np.eye(ny), atol=1e-4):
                print("k:", k)
                la_help.print_matrix(Fk@Finv, "Fk @ Finv")
                # raise ValueError("Fk is not the inverse of Finv")
            
            Gk = -Fk@Koff@Kinvs[k-1]
            # Hk = Kinvs[k-1]@(np.eye(ny*k) - Koff.T@Gk, 1e-3)
            Hk = Kinvs[k-1] - Kinvs[k-1]@Koff.T@Gk

            
            Kinvs.append(np.block([[Fk, Gk], [Gk.T, Hk]]))

            # if not np.allclose(K_k(Kn, ny, k)@Kinvs[k], np.eye(ny*(k+1)), atol=1e-1):
            if not la.norm(K_k(Kn, ny, k)@Kinvs[k]- np.eye(ny*(k+1))) <= 1e-3*((ny*k)**2):
                if bad_k >= np.inf:
                    bad_k = k
                # print("k:", k)
                # print("eigvals of Kinvs[k]:", np.sort(np.real(la.eigvals(Kinvs[k])))[::-1])
                # print("eigvals of K_k", np.sort(np.real(la.eigvals(K_k(Kn, ny, k))))[::1])
                # # la_help.print_matrix(Koff, "Koff")
                # # la_help.print_matrix(Fk, "Fk")
                # # la_help.print_matrix(Finv, "Finv")
                # # la_help.print_matrix(Gk, "Gk")
                # la_help.print_matrix(K_k(Kn, ny, k)@Kinvs[k], "K_k @ K_k^inv")
                # # print("\n\n\n")
                # # la_help.print_matrix(Koff.T@Gk, "Koff.T@Gk")
                # la_help.print_matrix(np.eye(ny*k) - Koff.T@Gk, "np.eye(ny*k) - Koff.T@Gk")
                # raise ValueError("K_k^inv is not the inverse of K_K")
        
    print("ran for context:", context)
    la_help.print_matrix(K_k(Kn, ny, k)@Kinvs[k], "K_k @ K_k^inv")
    return Kn, Kinvs, Finvs, bad_k