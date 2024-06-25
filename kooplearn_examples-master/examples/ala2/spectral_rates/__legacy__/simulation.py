#coding principle: everything defined outside of the __main__ scope should not depend on the configs.

#[IMPORTS]
import json
import os
import importlib
import pickle
from copy import deepcopy
from datetime import datetime
import numpy as np
from numpy.typing import NDArray
from kooplearn.estimators import LowRankRegressor
from einops import einsum
from tqdm import tqdm

from typing import Tuple

from src.io_utils import load_ala2_dataset, lagged_dataset
from src.custom_types import FittedModel, Simulation

kernel_module = importlib.import_module('kooplearn.kernels')
estimator_module = importlib.import_module('kooplearn.estimators')

def metric_distortion(fitted_estimator: LowRankRegressor, right_vecs: NDArray, X_test: NDArray) -> NDArray:
    Uv = einsum(fitted_estimator.U_, right_vecs, "n r, r vec -> n vec")
    norm_L2 = (np.abs(fitted_estimator.kernel(X_test, fitted_estimator.X_fit_)@Uv)**2).sum(axis=0)
    norm_RKHS = np.abs((((Uv.conj())*(fitted_estimator.K_X_@Uv))).sum(axis=0))
    return np.sqrt(X_test.shape[0])*np.sqrt(norm_RKHS/norm_L2)

def empirical_metric_distortion(fitted_estimator: LowRankRegressor, right_vecs: NDArray) -> NDArray:
    return metric_distortion(fitted_estimator, right_vecs, fitted_estimator.X_fit_)

def spectral_bias(estimator: LowRankRegressor) -> float:
    rank = estimator.rank
    training_samples = estimator.X_fit_.shape[0]
    if estimator.__class__.__name__ == 'ReducedRank':
        return float(estimator.svals(k = rank + 3)[rank])
    else:
        K = (training_samples**-1.)*estimator.K_X_
        return float(np.flip(np.sort(np.linalg.eigvalsh(K)))[rank])


if __name__ == "__main__":
    MAIN_DIR = os.path.dirname(__file__)
    CONFIGS_FILE = os.path.join(MAIN_DIR, 'configs/simulation_config.json')
    DATA_DIR = os.path.join(os.path.split(MAIN_DIR)[0], 'data')

    with open(CONFIGS_FILE, "r") as config_file:
        configs = json.load(config_file)
    kernel = getattr(kernel_module, configs["kernel"])(**configs["kernel_args"])
    
    if "base_tikhonov_reg" in configs:
        estimator = getattr(estimator_module, configs["estimator"])(
            kernel = kernel, 
            tikhonov_reg = configs["base_tikhonov_reg"], 
            **configs["estimator_args"]
        )
    else:
        estimator = getattr(estimator_module, configs["estimator"])(
            kernel = kernel, 
            **configs["estimator_args"]
        )
    lagtime = configs["lagtime"]
    num_B_svals = configs["num_B_svals"]
    
    training_points = np.linspace(
        configs["training_points"]["start"], 
        configs["training_points"]["stop"],
        num = configs["training_points"]["num"],
        dtype = int
    )

    #Load data and make dataset
    data = load_ala2_dataset(DATA_DIR)

    X_test, Y_test = lagged_dataset(data.test.distances, lagtime = configs["test_lagtime"])

    simulation_dict = {}
    
    for num_pts in (pbar := tqdm(training_points)):
        if "base_tikhonov_reg" in configs: 
            estimator.set_params(tikhonov_reg = configs["base_tikhonov_reg"]*(float(num_pts)**-1))
        X, Y = lagged_dataset(data.train.distances, lagtime = lagtime, num_points=num_pts)
        pbar.set_description(f"[{num_pts}] - FITTING")
        estimator.fit(X, Y)
        pbar.set_description(f"[{num_pts}] - HS NORM ERROR")
        op_norm_error = estimator.empirical_excess_risk(X_test, Y_test, norm = 'HS')
        pbar.set_description(f"[{num_pts}] - METRIC DISTORTION")
        evals, vr = estimator._eig(return_type = 'eigenvalues_error_bounds')
        eta = empirical_metric_distortion(estimator, vr)
        
        data_dump = FittedModel(
            data,
            deepcopy(estimator),
            eta,
            evals,
            float(op_norm_error)
        )
        
        simulation_dict[num_pts] = data_dump
        
    simulation_dump = Simulation(
        configs,
        simulation_dict
    )

    timestamp = datetime.now().strftime("%H%M_%-d_%-m_%-y")
    estimator_acronym = ''.join([char for char in estimator.__class__.__name__ if char.isupper()])
    file_name = estimator_acronym + "R_" + timestamp + ".pkl"
    with open('../data/eigenvalue_estimation/' + file_name, 'wb') as f:
        pickle.dump(simulation_dump, f)