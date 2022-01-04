# ------ IMPORT LIBRARIES ------ #
import numpy as np
from scipy import linalg
from scipy.stats import random_correlation
import matplotlib.pyplot as plt
 
# define random data
rng = np.random.default_rng()
x = random_correlation.rvs((.5, .8, 1.2, 1.5), random_state=rng)
y = random_correlation.rvs((.2, .9, 1.1, 1.8), random_state=rng)

# generating 10 random values for each of the two variables
n = 50
p = 30
C_list = []

# computing the corrlation matrices
for i in range(n):
    # first corr matrix
    X1 = np.random.normal(-7, 1, p)
    Y1 = np.random.normal(5, 1, p)
    C1 = np.corrcoef(X1,Y1)
    # second corr matrix
    X2 = np.random.normal(3, 1.0, p)
    Y2 = np.random.normal(8, 1.0, p)
    C2 = np.corrcoef(X2,Y2)
    # add as list of pairs to compare distances
    C_list.append(np.array([C1, C2]))

# --------------------------------------------------
# ----- TEST CONDITIONS FOR SPD MATRICES -----------
# --------------------------------------------------

def is_spd(X, eps=1e-7):
    """Check matrix is symmetric & positive definite"""
    # X: Input n x n matrix 
    # Check X = X^T and min eigenvalue > 0
    if np.any(np.abs(X - X.T) > eps):
        raise ValueError('Error: input matrix must be symmetric')
    eigvals = linalg.eigvals(X)
    if min(eigvals) <= 0:
        raise ValueError('Error: input matrix has non-positive eigenvalue')
    return True


# --------------------------------------------------------
# ----- HELPER DISTANCE/GEOMETRY FUNCTIONS on SPD(M) -----
# --------------------------------------------------------

# source: https://github.com/pyRiemann/pyRiemann/blob/master/pyriemann/utils/geodesic.py

def distance_euclid(A, B):
    r"""Euclidean distance between two covariance matrices A and B.
    The Euclidean distance is defined by the Froebenius norm between the two
    matrices.
    .. math::
        d = \Vert \mathbf{A} - \mathbf{B} \Vert_F
    :param A: First covariance matrix
    :param B: Second covariance matrix
    :returns: Eclidean distance between A and B
    """
    return np.linalg.norm(A - B, ord='fro')


def distance_logeuclid(A, B):
    r"""Log Euclidean distance between two covariance matrices A and B.
    .. math::
        d = \Vert \log(\mathbf{A}) - \log(\mathbf{B}) \Vert_F
    :param A: First covariance matrix
    :param B: Second covariance matrix
    :returns: Log-Eclidean distance between A and B
    """
    return distance_euclid(linalg.logm(A), linalg.logm(B))


def distance_riemann(A, B):
    r"""Riemannian distance between two covariance matrices A and B.
    .. math::
        d = {\left( \sum_i \log(\lambda_i)^2 \right)}^{1/2}
    where :math:`\lambda_i` are the joint eigenvalues of A and B
    :param A: First covariance matrix
    :param B: Second covariance matrix
    :returns: Riemannian distance between A and B
    """
    return np.sqrt((np.log(linalg.eigvalsh(A, B))**2).sum())

def distance_logdet(A, B):
    r"""Log-det distance between two covariance matrices A and B.
    .. math::
        d = \sqrt{\log(\det(\frac{\mathbf{A}+\mathbf{B}}{2})) - \frac{1}{2} \log(\det(\mathbf{A}) \det(\mathbf{B}))}
    :param A: First covariance matrix
    :param B: Second covariance matrix
    :returns: Log-Euclid distance between A and B
    """  # noqa
    return np.sqrt(np.log(np.linalg.det(
        (A + B) / 2.0)) - 0.5 *
        np.log(np.linalg.det(A)*np.linalg.det(B)))


def distance_wasserstein(A, B):
    r"""Wasserstein distance between two covariances matrices.
    .. math::
        d = \left( {tr(A + B - 2(A^{1/2}BA^{1/2})^{1/2})} \right)^{1/2}
    :param A: First covariance matrix
    :param B: Second covariance matrix
    :returns: Wasserstein distance between A and B
    """
    B12 = sqrtm(B)
    C = sqrtm(np.dot(np.dot(B12, A), B12))
    return np.sqrt(np.trace(A + B - 2*C))


def distance(A, B, metric='riemann'):
    """Distance between two covariance matrices A and B according to the
    metric.
    :param A: First covariance matrix
    :param B: Second covariance matrix
    :param metric: the metric (Default value 'riemann'), can be : 'riemann' ,
        'logeuclid' , 'euclid' , 'logdet', 'kullback', 'kullback_right',
        'kullback_sym'.
    :returns: the distance between A and B
    """
    if callable(metric):
        distance_function = metric
    else:
        distance_function = distance_methods[metric]

    if len(A.shape) == 3:
        d = np.empty((len(A), 1))
        for i in range(len(A)):
            d[i] = distance_function(A[i], B)
    else:
        d = distance_function(A, B)

    return d

# ----------------
# -- TESTING -----
# ----------------
print(np.all([is_spd(c[0]) and is_spd(c[1]) for c in C_list]))
riemann_dists = np.array([distance_riemann(c[0], c[1]) for c in C_list])
logeuclid_dists = np.array([distance_logeuclid(c[0], c[1]) for c in C_list])
euclid_dists = np.array([distance_euclid(c[0], c[1]) for c in C_list])
plt.scatter(euclid_dists, riemann_dists)
plt.hist(riemann_dists)
plt.hist(euclid_dists)


# -------------------------------------------------------------
# ----- FINISH GOING THROUGH PAPER TO IMPLEMENT PRECISELY -----
# -------------------------------------------------------------

def spd_dist(X, Y, metric='intrinsic'):
    """Calculate geodesic distance for X,Y in SPD Manifold"""
    # X: Input n x n matrix
    # Y: Input n x n matrix
    # Intrinsic metric: Affine-invariant Riemannian Metric (AIRM)
    # Extrinsic metric: log-Euclidean Riemannian Metric (LERM)
    if metric == 'intrinsic':
        M = np.matmul(linalg.inv(X), Y)
        dist = np.sqrt(np.linalg.norm(linalg.logm(M)))
        return dist
    elif metric == 'extrinsic':
        M = linalg.logm(X) - linalg.logm(Y)
        dist = np.sqrt(np.linalg.norm(M))
        return dist
    else:
        raise ValueError('Error: must specify intrinsic or extrinsic metric')

        
def exp_map(X, V):
    """Exponential mapping from tangent space at X to SPD Manifold"""
    # X: n x n matrix in SPD Manfiold (M)
    # V: tangent "vector" (really a symmetric matrix) within Tx(M)
    # Output: a point Y in M (following shortest geodesic curve along M in direction v)
    # NOTE: tangent "vectors" in Tx(M) are n x n symmetric matrics

#   -- remember matlab docs on using eigenvals to calculate inverse square roots, 
#   -- cholesky/decomposition for SPD
#   -- Expai,X(U) = X1/2 exp(X−1U)X1/2 (new formula from ADHD pdf)
    
    
    
    
        

        
    
    
    
    
        

        