#Imports
#STDLIB
import json
from typing import Optional
from datetime import datetime
import pickle
import importlib

#ESSENTIALS
import numpy as np
from numpy.typing import NDArray
from scipy.integrate import romb
from einops import einsum

#MISC
from tqdm import tqdm
from kooplearn.estimators import LowRankRegressor

#INTERNAL
from src.MD_utils import sample
from src.FEM_utils import koopman_spectrum
from src.internal_typing import LinalgDecomposition, SpectralData, Statistic
from src.physics import boltzmann_pdf

kernel_module = importlib.import_module('kooplearn.kernels')
estimator_module = importlib.import_module('kooplearn.estimators')

def metric_distortion(fitted_estimator: LowRankRegressor, right_vecs: NDArray, X_test: NDArray) -> NDArray:
    Uv = einsum(fitted_estimator.U_, right_vecs, "n r, r vec -> n vec")
    norm_L2 = (np.abs(fitted_estimator.kernel(X_test, fitted_estimator.X_fit_)@Uv)**2).sum(axis=0)
    norm_RKHS = np.abs((((Uv.conj())*(fitted_estimator.K_X_@Uv))).sum(axis=0))
    return np.sqrt(X_test.shape[0])*np.sqrt(norm_RKHS/norm_L2)

def empirical_metric_distortion(fitted_estimator: LowRankRegressor, right_vecs: NDArray) -> NDArray:
    return metric_distortion(fitted_estimator, right_vecs, fitted_estimator.X_fit_)

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

def spectral_bias(estimator: LowRankRegressor) -> float:
    rank = estimator.rank
    training_samples = estimator.X_fit_.shape[0]
    if estimator.__class__.__name__ == 'ReducedRank':
        return float(estimator.svals(k = rank + 3)[rank])
    else:
        K = (training_samples**-1.)*estimator.K_X_
        return float(np.flip(np.sort(np.linalg.eigvalsh(K)))[rank])

def estimate_koopman_spectrum(configs: dict, domain_sample: NDArray, ref_evd:LinalgDecomposition) -> SpectralData:
    sample_points = np.linspace(
        configs["num_samples"]["start"], 
        configs["num_samples"]["stop"],
        num = configs["num_samples"]["num"],
        dtype = int
    )
    num_sample_points = sample_points.shape[0]

    timesteps_between_samples = configs["timesteps_between_samples"]
    num_datasets = configs["num_simulations"]
    sampling_scheme = configs["sampling_scheme"]

    kernel = kernel_class(**configs["kernel_kwargs"])
    estimator = estimator_class(kernel, **configs["estimator_kwargs"])
    rank = configs["estimator_kwargs"]["rank"]

    dx = domain_sample[1] - domain_sample[0]
    boltzmann_pdf_sample = boltzmann_pdf(domain_sample)
    
    eigenvalues = np.zeros((num_sample_points, num_datasets, rank), dtype=np.float64)
    eigenfunctions = np.zeros((num_sample_points, num_datasets, domain_sample.shape[0], rank), dtype = np.float64)

    eigenvalues_errors = np.zeros_like(eigenvalues)
    eigenfunctions_errors = np.zeros_like(eigenvalues)
    distortion = np.zeros_like(eigenvalues)
    total_bias = np.zeros_like(eigenvalues)

    bias = np.zeros((num_sample_points, num_datasets), dtype = np.float64)

    for samples_idx, pts in (pbar := tqdm(enumerate(sample_points), total=num_sample_points)):
        dataset = sample(sampling_scheme, pts, timesteps_between_samples, num_datasets) 
        for ds_index in range(num_datasets):
            pbar.set_description(f"[{ds_index + 1}/{num_datasets}]")
            x, y = dataset.X[ds_index], dataset.Y[ds_index]
            estimator.fit(x, y)
            vals, vr = estimator._eig(return_type = 'eigenvalues_error_bounds')

            eta = empirical_metric_distortion(estimator, vr)
            distortion[samples_idx, ds_index] = eta[np.flip(np.argsort(vals.real))]

            dim_inv = (estimator.K_X_.shape[0])**(-1)
            sqrt_inv_dim = dim_inv**0.5

            sampled_eigfuns = sqrt_inv_dim*estimator.kernel(domain_sample[:, None], estimator.X_fit_, backend=estimator.backend)@estimator.U_@vr
            evd = LinalgDecomposition(vals, sampled_eigfuns)
            evd = standardize_phase(evd, ref_evd, dx, boltzmann_pdf_sample)

            #Estimation
            eigenvalues[samples_idx, ds_index] = evd.values
            eigenfunctions[samples_idx, ds_index] = evd.vectors
            #Error evaluation (koopman_eigenvalues are assumed to be sorted descendingly and koopman_eigenfunctions normalized to L^2(X, \pi))
            dist2 = romb(
                boltzmann_pdf_sample*((np.abs(evd.vectors - ref_evd.vectors)**2).T),
                dx = dx,
                axis = -1)
            #Errors
            eigenvalues_errors[samples_idx, ds_index] = np.abs(evd.values - ref_evd.values)
            eigenfunctions_errors[samples_idx, ds_index] = np.sqrt(dist2)
            #Bias
            bias[samples_idx, ds_index] = spectral_bias(estimator)
            total_bias[samples_idx, ds_index] = bias[samples_idx, ds_index]*distortion[samples_idx, ds_index]
    
    evalues_err_s = Statistic(eigenvalues_errors.mean(axis=1), eigenvalues_errors.std(axis=1))
    efuncs_err_s = Statistic(eigenfunctions_errors.mean(axis=1), eigenfunctions_errors.std(axis=1))
    distortion_s = Statistic(distortion.mean(axis=1), distortion.std(axis=1))
    bias_s = Statistic(bias.mean(axis=1), bias.std(axis=1))
    total_bias_s = Statistic(total_bias.mean(axis=1), total_bias.std(axis=1))
    
    return SpectralData(
        configs,
        sample_points,
        domain_sample,
        ref_evd,
        LinalgDecomposition(eigenvalues, eigenfunctions),
        evalues_err_s,
        efuncs_err_s,
        distortion_s,
        bias_s,
        total_bias_s
    )


if __name__ == "__main__":
    with open("config.json", "r") as f:
        configs = json.load(f)

    kernel_class = getattr(kernel_module, configs["kernel"])
    estimator_class = getattr(estimator_module, configs["estimator"])

    timesteps_between_samples = configs["timesteps_between_samples"]
    FEM_spectrum, eigfun_sample = koopman_spectrum(timesteps_between_samples)

    dx = eigfun_sample[1] - eigfun_sample[0]
    boltzmann_pdf_sample = boltzmann_pdf(eigfun_sample)

    ref_evd = standardize_evd(FEM_spectrum, dx, density=boltzmann_pdf_sample)
    data_dump = estimate_koopman_spectrum(configs, eigfun_sample, ref_evd)

    timestamp = datetime.now().strftime("%H%M_%-d_%-m_%-y")
    estimator_acronym = ''.join([char for char in configs["estimator"] if char.isupper()])
    file_name = estimator_acronym + "R_" + timestamp + ".pkl"
    with open('data/' + file_name, 'wb') as f:
        pickle.dump(data_dump, f)
