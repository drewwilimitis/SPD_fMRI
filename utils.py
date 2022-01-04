import numpy as np
from scipy import linalg

def is_spd(X):
    """Check matrix is symmetric & positive definite"""
    # X: Input p x p matrix 
    # Check X = X^T and min eigenvalue > 0
    if np.any(X != (X.T)):
        raise ValueError('Error: input matrix must be symmetric')
    eigvals = linalg.eigvals(X)
    if min(eigvals) <= 0:
        raise ValueError('Error: input matrix has non-positive eigenvalue')
    return True
    
def spd_dist(X, Y, metric='intrinsic'):
    """Calculate geodesic distance for X,Y in SPD Manifold"""
    # X: Input p x p matrix
    # Y: Input p x p matrix
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
        

        