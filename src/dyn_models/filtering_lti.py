import numpy as np
import scipy as sc
from numpy import linalg as lin
from scipy import linalg as la
from filterpy.kalman import KalmanFilter
import random
from decimal import Decimal, getcontext
import control as ct


def softplus(x):
    return np.log(1 + np.exp(x))


####################################################################################################

def is_symmetric(matrix, tol=1e-8):
    return np.allclose(matrix, matrix.T, atol=tol)

# # code that I added
# def solve_ricc(A, W):  # solve the Riccati equation for the steady state solution
#     # Ensure input arrays are of high precision type
#     A = np.array(A, dtype=np.float64)
#     W = np.array(W, dtype=np.float64)

#     L, V = np.linalg.eig(A)
#     Vinv = np.linalg.inv(V)
#     Pi = (V @ (
#             (Vinv @ W @ Vinv.T) / (1 - L[:, None] * L)
#     ) @ V.T).real
    
#     # if np.trace(Pi) < 0:

#     #     print("\n\ntrace of Pi:", np.trace(Pi))
#     #     print("is symmetric:", is_symmetric((Vinv @ W @ Vinv.T) / (1 - L[:, None] * L)))
#     #     print("eigs of inner mat:", lin.eig((Vinv @ W @ Vinv.T) / (1 - L[:, None] * L)))
#     #     print("\nVinv:", Vinv)
#     #     print("\nL:", L)
#     #     print("\nLinv:", 1/L)

#     # Pi = np.array(Pi, dtype=np.float32)
#     return Pi

def gen_A(elow, ehigh, n): # generates a 2d A matrix with evalue magnitudes in between elow and ehigh. For matrices larger than 2d, when there are complex evalues it just ensures that all evalues are below ehigh and at least one evalue is above elow.
    if elow > ehigh:
        raise ValueError("elow must be less than ehigh")

    # bound = np.random.uniform(elow, ehigh, size=(2,)) #sample a random bound for the evalues
    # #sort bound in descending order
    # bound = np.sort(bound)[::-1]
    a = ehigh #bound[0]
    b = elow #bound[1]

    # mat = np.random.normal(loc=0, scale = 1, size=(n,n)) #produce random square matrix with normal distribution
    mat = np.random.uniform(-1, 1, (n,n)) #produce random square matrix with uniform distribution

    # get eigenvalues of mat
    eigs = lin.eigvals(mat)
    # sort eignvalues by magnitude in descending order
    sorted_indices = np.argsort(np.abs(eigs))[::-1] #changed from np.sort(eigs)[::-1]
    eigs = eigs[sorted_indices] #sort the evalues

    eps = 1e-10 #small number to check if evalues are out of bounds
    if np.iscomplex(eigs).any(): #if there are complex evalues
        # scale the eigenvalues to the bound
        alpha = a #a + 0.5*(b-a) #number to scale the evalues to
        out = (alpha/np.abs(eigs[0]))*mat #scale the matrix

        if (np.sum(np.abs(lin.eigvals(out)) > elow - eps) < 1) or np.any(np.abs(lin.eigvals(out)) > ehigh+eps): #if there are no evalues above elow or there are evalues above ehigh
            print("np.abs(lin.eigvals(out)):", np.abs(lin.eigvals(out)))
            raise ValueError("evalues out of bounds (complex)")
    
    else: #if there are no complex evalues
        print("all real evalues")
        A = ((b-a)/(eigs[0] - eigs[-1]))*(mat - eigs[-1]*np.eye(n)) + a*np.eye(n) #subtract the smallest evalue from the matrix
        T,Z = la.schur(A) #get the schur decomposition
        sgn = 2*np.random.randint(2, size=n) - np.ones((n,)) #randomly choose a sign for each evalue
        for i in range(n):
            T[i,i] = T[i,i]*sgn[i] #multiply the diagonal by the sign
        out = Z@T@Z.T #reconstruct the matrix
        if (np.any(np.abs(lin.eigvals(out)) < elow-eps) or np.any(np.abs(lin.eigvals(out)) > ehigh+eps)): #if there are evalues below elow or above ehigh
            print("np.abs(lin.eigvals(out)):", np.abs(lin.eigvals(out)))
            raise ValueError("evalues out of bounds (real)")

    print("np.abs(lin.eigvals(out)):", np.abs(lin.eigvals(out))) 
    return out

def generate_random_list(lowest, highest):
    # Generate 8 random numbers between lowest and highest
    random_numbers = [random.uniform(lowest, highest) for _ in range(8)]
    
    # Add the specified lowest and highest numbers
    random_numbers.append(lowest)
    random_numbers.append(highest)
    
    #sort the list from lowest to highest
    random_numbers.sort(reverse=False)
    
    return random_numbers

def generate_random_rotation_matrix(n):
    # Generate a random 10x10 matrix
    random_matrix = np.random.randn(n, n)

    # Use QR decomposition to get a random rotation matrix
    Q, R = np.linalg.qr(random_matrix)
    return Q

def generate_random_mat_cond_number(n, cond_number):
    top = 1 #set the top bound for the singular values
    low = top / cond_number
    s = generate_random_list(low, top)
    S = np.diag(s)
    U = generate_random_rotation_matrix(n)
    VT = generate_random_rotation_matrix(n)
    random_matrix = U @ S @ VT

    # print("cond number = ", np.linalg.cond(random_matrix))
    return random_matrix

def gen_rand_ortho_haar_real(n):
    """
    Generate a random orthogonal matrix 'uniformly' distributed over real orthogonal matrices using QR decomposition.
    :param n: Size of the matrix
    :return: Random orthogonal matrix of size n x n
    """
    # Generate a random matrix
    A = np.random.randn(n, n)
    
    # Perform QR decomposition
    Q, R = np.linalg.qr(A)

    d = np.diag(R)
    d = d/np.abs(d)

    # Construct the diagonal matrix D
    D = np.diag(d)
    Q = Q @ D
    
    return Q

####################################################################################################


class FilterSim:
    # def __init__(self, nx=3, ny=2, sigma_w=1e-1, sigma_v=1e-1, tri=False, n_noise=1):
    #     self.sigma_w = sigma_w
    #     self.sigma_v = sigma_v

    #     self.n_noise = n_noise

    #     if tri:
    #         A = np.diag(np.random.rand(nx) * 2 - 1) * 0.95
    #         A[np.triu_indices(nx, 1)] = np.random.rand((nx ** 2 + nx) // 2 - nx) * 2 - 1
    #         self.A = A
    #     else:
    #         A = np.random.rand(nx, nx)
    #         A /= np.max(np.abs(np.linalg.eigvals(A)))
    #         self.A = A * 0.95

    #     self.C = np.eye(nx) if nx == ny else self.construct_C(self.A, ny)

    # ####################################################################################################
    # #code that I added
    def __init__(self, nx, ny, sigma_w, sigma_v, tri, C_dist, n_noise, new_eig, cond_num=10, E = 1, specific_sim_obj=None):

        valid_system = False
        while not valid_system:
            #sets noise energy
            self.sigma_w = sigma_w
            self.sigma_v = sigma_v

            self.n_noise = n_noise

            if specific_sim_obj:
                self.A = specific_sim_obj.A
                self.C = specific_sim_obj.C
                self.S_state_inf = specific_sim_obj.S_state_inf
                self.S_observation_inf = specific_sim_obj.S_observation_inf
                valid_system = True
                continue
            elif tri == "upperTriA":

                A = np.diag(np.random.uniform(-1, 1, nx)) * 0.95
                A[np.triu_indices(nx, 1)] = np.random.uniform(-1, 1, (nx ** 2 + nx) // 2 - nx)
                self.A = A

            elif tri.startswith("upperTriA_gauss"):

                A = np.diag(np.sqrt(0.33)*np.random.randn(nx))
                A[np.triu_indices(nx, 1)] = np.sqrt(0.33)*np.random.randn((nx ** 2 + nx) // 2 - nx)
                A /= np.max(np.abs(np.linalg.eigvals(A)))
                self.A = A * 0.95 #scale the matrix

            elif tri.startswith("rotDiagA"):
                if tri == "rotDiagA":
                    A = np.diag([0.99,0.98,0.97,0.96,0.95,0.94,0.93,0.92,0.91,0.9]) * 0.95 #generate a random diagonal matrix #values are handpicked to be near 1
                elif tri == "rotDiagA_unif":
                    A = np.diag([-0.07679708, 0.42467112, 0.28797265, 0.08335881, -0.228187, 0.82329669, 0.75456837, 0.41401802, -0.93903797, -0.38618607]) #diagonal matrix, values are from unif(-0.95,0.95)
                elif tri == "rotDiagA_gauss":
                    A = np.diag([0.63952862, 0.65871765, -0.02630632, 0.69056979, -0.23617829, -0.14567006,
                    -0.28050732, 0.8654012, -0.29325309, 0.95])

                # Generate a random 10x10 matrix
                random_matrix = np.random.randn(10, 10)

                # Use QR decomposition to get a random rotation matrix
                Q, R = np.linalg.qr(random_matrix)
                            
                self.A = Q @ A @ Q.T 
            elif tri == "gaussA":
                A = np.sqrt(0.33)*np.random.randn(nx, nx) #same second moment as uniform(-1,1)
                A /= np.max(np.abs(np.linalg.eigvals(A)))
                self.A = A * 0.95 #scale the matrix
            elif tri == "gaussA_noscale":
                self.A = np.sqrt(0.33)*np.random.randn(nx, nx) #same second moment as uniform(-1,1)
            elif tri == "single_system":
                self.A = np.array([
                    [-0.19724629,  0.54337223, -0.08717161, -0.14148353, -0.09288597,  0.34414641,
                    0.05392701,  0.27692442, -0.17766965, -0.28864491],
                    [-0.01148732, -0.00661153, -0.08374766,  0.07595969,  0.13724617, -0.20550157,
                    -0.09138989,  0.00958794, -0.09178032, -0.18323221],
                    [-0.14038119, -0.26320061, -0.50151312, -0.04012689,  0.02803123, -0.14850721,
                    -0.36312298, -0.03846626, -0.12939845, -0.06366234],
                    [-0.04863701,  0.27376886, -0.09335658, -0.50563864,  0.34658393, -0.3194514,
                    -0.12360261, -0.32700136, -0.26411677, -0.47258459],
                    [ 0.24806126, -0.19616362, -0.2526967 , -0.1923523 ,  0.00420298, -0.00551049,
                    0.07132224, -0.32386641, -0.03343957,  0.04528459],
                    [-0.46343962,  0.06561287,  0.06586732, -0.27404857,  0.0206694 ,  0.22270968,
                    -0.16489038, -0.20793099, -0.19448377, -0.66842933],
                    [-0.23671054,  0.07699569,  0.03016573,  0.05328966,  0.1614495 ,  0.2018623,
                    0.41401545,  0.27257689,  0.0528767 ,  0.10823899],
                    [ 0.3793518 , -0.34086795,  0.14860611, -0.1258312 , -0.25835586, -0.13858694,
                    -0.08842134,  0.06286227, -0.24796707,  0.43837012],
                    [ 0.14686412,  0.15896492, -0.18453878, -0.09718014, -0.23884111,  0.13125948,
                    0.64762778, -0.0520824 , -0.28642175, -0.38521792],
                    [ 0.30933025, -0.40293411, -0.12426857,  0.01131631, -0.58637518, -0.2329357,
                    0.08639332, -0.12452823,  0.23418365, -0.04600605]
                ]) #generate a random specific matrix

            elif tri == "cond_num":
                A = generate_random_mat_cond_number(nx, cond_num) #generate a random matrix with a condition number of cond_number
                self.A = A

            elif tri == "ident":
                self.A = np.eye(nx)

            elif tri == "ortho":
                random_matrix = np.random.randn(nx, nx) #generate a Gaussian random square matrix
                Q, R = np.linalg.qr(random_matrix) #get the QR decomposition
                self.A = Q #set A to be the orthogonal matrix

            elif tri == "ortho_haar":
                self.A = gen_rand_ortho_haar_real(nx)

            else:
                if new_eig:
                    self.A = gen_A(0.97, 0.99, nx)
                else:
                    A = np.random.uniform(-1, 1, (nx, nx))  # fixed the sampling of A to be between -1 and 1
                    A /= np.max(np.abs(np.linalg.eigvals(A)))
                    self.A = A * 0.95
                    # A = np.zeros((nx, nx))
                    # A[0,0] = 0.95
                    # self.A = A

            
            self.C = self.construct_C(self.A, ny, C_dist)

            
            if tri == "ident" or tri == "ortho" or tri == "ortho_haar":
                self.S_state_inf = (1/nx)*np.eye(nx) # for ident: Pi = A^T Pi A + W = Pi so every sym pos def matrix is a solution. just choose identity
                #for ortho case there is no steady state covariance unless A is Identity.
                #we have Pi = U Lambda U^T = A U Lambda U^T A^T so U = AU.
            else:
                self.S_state_inf = ct.dlyap(self.A, np.eye(nx) * self.sigma_w ** 2) #solve the riccati equation for the steady state solution

            eval, evec = lin.eig(self.S_state_inf)

            if is_symmetric(self.S_state_inf, tol=1e-5) and np.all(np.greater(eval, 0)):
                valid_system = True


                if not (tri == "ident" or tri == "ortho" or tri == "ortho_haar"):

                    S_state_inf_intermediate = sc.linalg.solve_discrete_are(self.A.T, self.C.T, np.eye(nx) * self.sigma_w ** 2, np.eye(ny) * self.sigma_v ** 2) #solve the riccati equation for the steady-state state error covariance
                    self.S_observation_inf = self.C @ S_state_inf_intermediate @ self.C.T + np.eye(ny) * self.sigma_v ** 2 #steady state observation error covariance

                    # rescale C and V
                    V = np.eye(ny) * self.sigma_v ** 2
                    obs_tr = np.trace(self.C @ self.S_state_inf @ self.C.T + V)

                    if obs_tr < 0:
                        print("obs_tr negative:", obs_tr)
                        print("evals of Pi", eval)
                        print("evals greater than 0?", np.greater(eval, 0))
                        print("all positive", np.all(np.greater(eval, 0)))
                        print("eval of CPiCT:", lin.eig(self.C @ self.S_state_inf @ self.C.T))
                        raise ValueError("Didn't catch negative evals")
                    

                    alpha = np.sqrt(E / obs_tr)

                    self.C = alpha*self.C
                    self.sigma_v = alpha*self.sigma_v
                    V = np.eye(ny) * self.sigma_v ** 2#think about the identity case for this

                    S_state_inf_intermediate = sc.linalg.solve_discrete_are(self.A.T, self.C.T, np.eye(nx) * self.sigma_w ** 2, np.eye(ny) * self.sigma_v ** 2)
                    self.S_observation_inf = self.C @ S_state_inf_intermediate @ self.C.T + np.eye(ny) * self.sigma_v ** 2
                else:

                    self.S_observation_inf = np.zeros((ny, ny))
                    self.sigma_v = 0.0
                    self.sigma_w = 0.0

            else:
                print("steady-state covariance symmetric?:", is_symmetric(self.S_state_inf))
                print("steady-state covariance positive definite?:", np.all(np.greater(eval, 0)))

        


    # ####################################################################################################

    def simulate(self, traj_len, x0=None):
        ny, nx = self.C.shape
        n_noise = self.n_noise
        xs = [np.random.randn(nx) if x0 is None else x0]  # initial state of dimension nx
        vs = [np.random.randn(ny) * self.sigma_v for _ in range(n_noise)]  # output noise of dimension ny
        ws = [np.random.randn(nx) * self.sigma_w for _ in range(n_noise)]  # state noise of dimension nx
        ys = [self.C @ xs[0] + sum(vs)]  # output of dimension ny
        for _ in range(traj_len):
            x = self.A @ xs[-1] + sum(ws[-n_noise:])
            xs.append(x)
            ws.append(np.random.randn(nx) * self.sigma_w)

            vs.append(np.random.randn(ny) * self.sigma_v)
            y = self.C @ xs[-1] + sum(vs[-n_noise:])
            ys.append(y)
        return np.array(xs).astype("f"), np.array(ys).astype("f")

    ####################################################################################################
    # code that I added

    def simulate_steady(self, batch_size, traj_len):  # change x0 to the steady state distribution
        ny, nx = self.C.shape
        n_noise = self.n_noise
        x0 = np.stack([
            np.random.multivariate_normal(np.zeros(nx), self.S_state_inf)
            for _ in range(batch_size)
        ]) 

        ws = np.random.randn(batch_size, n_noise + traj_len, nx) * self.sigma_w    # state noise of dimension nx
        vs = np.random.randn(batch_size, n_noise + traj_len, ny) * self.sigma_v    # output noise of dimension ny

        xs = [x0]  # initial state of dimension nx
        ys = [xs[0] @ self.C.T + vs[:, :n_noise].sum(axis=1)]  # output of dimension ny

        # #check if self.A is upper triangular
        # # print("check if self.A is upper triangular")
        # if not np.all(np.triu(self.A) == self.A):
        #     print("self.A:", self.A)
        #     raise ValueError("A is not upper triangular")
        
        for i in range(1, traj_len + 1):
            x = xs[-1] @ self.A.T + ws[:, i:i + n_noise].sum(axis=1)
            xs.append(x)

            y = xs[-1] @ self.C.T + vs[:, i:i + n_noise].sum(axis=1)
            ys.append(y)


        # print("norm of ys", lin.norm(ys))
        return np.stack(xs, axis=1).astype("f"), np.stack(ys, axis=1).astype("f")
    
    #implement Kf preds in the above function to speed things up.

    ####################################################################################################

    @staticmethod
    def construct_C(A, ny, C_type): 
        nx = A.shape[0]
        _O = [np.eye(nx)]
        for _ in range(nx - 1):
            _O.append(_O[-1] @ A)
        while True:
            if C_type == "_gauss_C":
                C = np.random.normal(0, np.sqrt(0.333333333), (ny, nx))

                #scale C by the reciprocal of its frobenius norm
                # C = C/np.linalg.norm(C, ord='fro') #scale the matrix 
            elif C_type == "_unif_C":
                C = np.random.rand(ny, nx) #uniform(0,1)
            elif C_type == "_zero_C":
                C = np.zeros((ny,nx))
                break
            elif C_type == "_ident_C":
                C = np.eye(ny)
            elif C_type == "_single_system":
                arr = np.array([
                        [ 1.58306722,  0.08254489,  0.63563991, -0.75589141,  0.45347395,  1.52444272,
                        0.19327427, -1.45641797,  1.31582812, -0.54842482],
                        [-0.35336291, -1.66812014,  0.68527397, -1.33242619, -0.74728405, -0.42868647,
                        0.88728522,  1.31609117,  1.16449376, -1.34126255],
                        [ 0.08083017,  0.66527164,  0.94678362, -1.35538693, -0.42848464,  0.30631194,
                        1.07822856,  0.56396537,  1.00666173,  0.29519488],
                        [-1.05025071, -0.72655103, -1.37426337, -1.33609509, -1.43163071, -0.78321476,
                        0.57989037, -0.93342293, -1.21680371, -1.39550893],
                        [ 0.34049993, -1.39124344,  1.17060005, -0.52646281, -1.23408133, -0.67153237,
                        1.51040147,  0.91665462, -1.1607541, -0.10762072]
                    ]) #generate a random specific matrix
                if ny == arr.shape[0] and nx == arr.shape[1]:
                    C = arr          
                else:
                    raise ValueError(f"Single system C matrix is not the right dimension of {ny}x{nx}")
            else:
                raise ValueError("C_type did not match")
            O = np.concatenate([C @ o for o in _O], axis=0)
            if np.linalg.matrix_rank(O) == nx:  # checking if state is observable
                break
        return C.astype("f")

    # ####################################################################################################
    # #code that I added
    # @staticmethod
    # def construct_C(A, ny):
    #     nx = A.shape[0]
    #     C = np.random.rand(ny, nx)
    #     return C.astype("f")
    # ####################################################################################################


####################################################################################################
# code that I added
def apply_kf(fsim, ys, sigma_w=None, sigma_v=None, return_obj=False):
    ny, nx = fsim.C.shape

    sigma_w = fsim.sigma_w if sigma_w is None else sigma_w
    sigma_v = fsim.sigma_v if sigma_v is None else sigma_v

    f = KalmanFilter(dim_x=nx, dim_z=ny)
    f.Q = np.eye(nx) * sigma_w ** 2
    f.R = np.eye(ny) * sigma_v ** 2
    f.P = fsim.S_state_inf
    f.x = np.zeros(nx)
    f.F = fsim.A
    f.H = fsim.C

    ls = [fsim.C @ f.x]
    count = 0
    for y in ys:
        f.update(y)
        f.predict()
        ls.append(fsim.C @ f.x)
        count += 1
    ls = np.array(ls)
    return (f, ls) if return_obj else ls


####################################################################################################
# code that I added


def _generate_lti_sample(C_dist, dataset_typ, batch_size, n_positions, nx, ny, sigma_w=1e-1, sigma_v=1e-1, n_noise=1, cond_num=None, specific_sim_obj=None):
    fsim = FilterSim(nx, ny, sigma_w, sigma_v, tri=dataset_typ, C_dist=C_dist, n_noise=n_noise, new_eig = False, cond_num=cond_num, specific_sim_obj=specific_sim_obj)
    states, obs = fsim.simulate_steady(batch_size, n_positions)
    # return fsim, {"states": states, "obs": obs, "A": fsim.A, "C": fsim.C}
    return fsim, {"obs": obs}

def generate_lti_sample(C_dist, dataset_typ, batch_size, n_positions, nx, ny, sigma_w=1e-1, sigma_v=1e-1, n_noise=1, cond_num=None, specific_sim_obj=None):
    while True:
        fsim, entry = _generate_lti_sample(C_dist, dataset_typ, batch_size, n_positions, nx, ny, sigma_w, sigma_v, n_noise=n_noise, cond_num=cond_num, specific_sim_obj=specific_sim_obj)
        if check_validity(entry):
            return fsim, entry


####################################################################################################


def generate_changing_lti_sample(n_positions, nx, ny, sigma_w=1e-1, sigma_v=1e-1, n_noise=1):
    fsim1 = FilterSim(nx=nx, ny=ny, sigma_w=sigma_w, sigma_v=sigma_v, n_noise=n_noise, new_eig=False)
    fsim2 = FilterSim(nx=nx, ny=ny, sigma_w=sigma_w, sigma_v=sigma_v, n_noise=n_noise, new_eig=False)

    _xs, _ys = fsim1.simulate(n_positions)
    while not check_validity({"states": _xs, "obs": _ys}):
        _xs, _ys = fsim1.simulate(n_positions)
    _xs_cont, _ys_cont = fsim2.simulate(n_positions, x0=_xs[-1])
    while not check_validity({"states": _xs_cont, "obs": _ys_cont}):
        _xs_cont, _ys_cont = fsim2.simulate(n_positions, x0=_xs[-1])
    y_seq = np.concatenate([_ys[:-1], _ys_cont], axis=0)
    return fsim1, {"obs": y_seq}


def check_validity(entry):
    if entry is None:
        return False
    obs = entry["obs"]
    return np.max(np.abs(obs)) < 50
