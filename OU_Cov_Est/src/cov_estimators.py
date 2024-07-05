#Standard Library
from typing import NamedTuple, Tuple

#Numerical
import numpy as np
import scipy.linalg

class LinalgDecomposition(NamedTuple):
    vals: np.ndarray
    vecs: np.ndarray

def reduced_rank_fit(
        covariances: Tuple,
        reg: float,
        rank: int, 
    ) -> LinalgDecomposition:
    input_cov, cross_cov = covariances
    input_cov += reg*np.eye(input_cov.shape[0], dtype=input_cov.dtype)

    _inv_cov_crosscov = scipy.linalg.solve(input_cov, cross_cov, assume_a='pos') #C_reg^{-1}@T
    svals_sq, Q = np.linalg.eig(_inv_cov_crosscov@(cross_cov.T))

    #Ordering and truncating
    sort_perm = np.flip(np.argsort(svals_sq.real))
    svals_sq = (svals_sq[sort_perm][:rank]).real
    #Eigenvector normalization:    
    Q = Q[:, sort_perm][:,:rank]  
    evecs_norm = np.sqrt(np.sum((Q.conj())*(input_cov@Q), axis = 0))
    return LinalgDecomposition(svals_sq, Q@np.diag(evecs_norm**-1))

def pcr_fit(
        covariances: Tuple,
        reg: float,
        rank: int
    ) -> LinalgDecomposition:
    input_cov, cross_cov = covariances
    input_cov = input_cov + reg*np.eye(input_cov.shape[0], dtype=input_cov.dtype)
    vals, Q = scipy.linalg.eigh(input_cov)

    #Ordering and truncating
    sort_perm = np.flip(np.argsort(vals))
    vals = (vals[sort_perm][:rank])**-0.5
    Q = vals*Q[:, sort_perm][:,:rank]
       
    return LinalgDecomposition(vals, Q)

def eig(fitted_reducedrank: LinalgDecomposition, cross_cov: np.ndarray) -> LinalgDecomposition:
    U = fitted_reducedrank.vecs
    #U@(U.T)@Tw = v w -> (U.T)@T@Uq = vq and w = Uq 
    vals, Q = np.linalg.eig((U.T)@(cross_cov@U))
    return LinalgDecomposition(vals, U@Q)