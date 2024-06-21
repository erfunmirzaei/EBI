import numpy as np
from typing import NamedTuple, Optional
from numpy.typing import NDArray
from scipy.integrate import romb

class LinalgDecomposition(NamedTuple):
    values: NDArray
    vectors: NDArray

def eigfun_sign_phase(estimated, true):
    norm_p = np.linalg.norm(estimated + true)
    norm_m = np.linalg.norm(estimated - true)
    if norm_p <= norm_m:
        return -1.0
    else:
        return 1.0
# [configs, num_samples, domain_discretization, FEM_spectrum, MD_spectrum, evalue_errors, efunc_errors, metric_distortion, bias]

def standardize_evd(evd: LinalgDecomposition, dx:float, density: Optional[NDArray] = None) -> LinalgDecomposition:
    #Sorting and normalizing
    sort_perm = np.flip(np.argsort(evd.values.real))
    functions = (evd.vectors[:, sort_perm]).real
    abs2_eigfun = (np.abs(functions)**2).T
    if density is not None:
        abs2_eigfun *= density
    #Norms
    funcs_norm = np.sqrt(romb(abs2_eigfun, dx = dx, axis = -1))
    functions *= (funcs_norm**-1.0)
    values = (evd.values.real)[sort_perm]
    return LinalgDecomposition(values, functions)

def standardize_phase(evd:LinalgDecomposition, ref_evd: LinalgDecomposition, dx:float, density: Optional[NDArray] = None) -> LinalgDecomposition:
    #ref_evd is assumed already standardized
    evd = standardize_evd(evd, dx, density=density)
    phase_aligned_funcs = evd.vectors.copy()
    num_funcs = evd.vectors.shape[1]
    for r in range(num_funcs):
        estimated = evd.vectors[:, r]
        true = ref_evd.vectors[:, r]
        phase_aligned_funcs[:, r] = eigfun_sign_phase(estimated*density, true*density)*estimated
    return LinalgDecomposition(evd.values, phase_aligned_funcs)