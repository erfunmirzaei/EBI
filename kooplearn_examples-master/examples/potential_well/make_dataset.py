#Imports
#STDLIB
import json
from typing import Optional

#ESSENTIALS
import numpy as np
from numpy.typing import NDArray
from scipy.integrate import romb

#INTERNAL
from src.MD_utils import sample
from src.FEM_utils import koopman_spectrum
from src.internal_typing import LinalgDecomposition, Dataset
from src.physics import boltzmann_pdf

def eigfun_sign_phase(estimated, true):
    norm_p = np.linalg.norm(estimated + true)
    norm_m = np.linalg.norm(estimated - true)
    if norm_p <= norm_m:
        return -1.0
    else:
        return 1.0

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

def make_dataset(configs: dict) -> Dataset:
    pts = configs["num_samples"]["stop"]
    timesteps_between_samples = configs["timesteps_between_samples"]
    num_datasets = configs["num_simulations"]
    sampling_scheme = configs["sampling_scheme"]
    return sample(sampling_scheme, pts, timesteps_between_samples, num_datasets)

if __name__ == "__main__":
    with open("config.json", "r") as f:
        configs = json.load(f)

    timesteps_between_samples = configs["timesteps_between_samples"]
    num_nodes = 2**10 + 1
    FEM_spectrum, eigfun_sample = koopman_spectrum(timesteps_between_samples, num_nodes = num_nodes)

    dx = eigfun_sample[1] - eigfun_sample[0]
    boltzmann_pdf_sample = boltzmann_pdf(eigfun_sample)

    ref_evd = standardize_evd(FEM_spectrum, dx, density=boltzmann_pdf_sample)
    ref_evd_filename = f"data/ref_evd_{num_nodes}_points.npz"
    np.savez(ref_evd_filename, values=ref_evd.values, vectors=ref_evd.vectors, density=boltzmann_pdf_sample, domain_sample=eigfun_sample)

    data_dump = make_dataset(configs)

    file_name = f"data/{configs['num_samples']['stop']}_points_{configs['num_simulations']}_simulations.npz"
    np.savez(file_name, X=data_dump.X, Y=data_dump.Y)