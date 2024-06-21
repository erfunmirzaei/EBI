import importlib

from numpy.typing import NDArray
from sklearn.model_selection import ParameterGrid
from kooplearn.estimators import LowRankRegressor
from einops import einsum
import numpy as np
kernel_module = importlib.import_module('kooplearn.kernels')


#Defining scoring functions for sklearn cross validation.
def op_norm_score(estimator: LowRankRegressor, X: NDArray, Y: NDArray) -> float:
    error = estimator.empirical_excess_risk(X, Y , norm='op')
    return -float(error)

def HS_norm_score(estimator: LowRankRegressor, X: NDArray, Y: NDArray) -> float:
    error = estimator.empirical_excess_risk(X, Y , norm='HS')
    return -float(error)

def reconstruction_score(estimator: LowRankRegressor, X: NDArray, Y: NDArray) -> float:
    return -float(estimator.reconstruction_error(X, Y))

def metric_distortion(fitted_estimator: LowRankRegressor, right_vecs: NDArray, X_test: NDArray) -> NDArray:
    Uv = einsum(fitted_estimator.U_, right_vecs, "n r, r vec -> n vec")
    norm_L2 = (np.abs(fitted_estimator.kernel(X_test, fitted_estimator.X_fit_)@Uv)**2).sum(axis=0)
    norm_RKHS = np.abs((((Uv.conj())*(fitted_estimator.K_X_@Uv))).sum(axis=0))
    return np.sqrt(X_test.shape[0])*np.sqrt(norm_RKHS/norm_L2)

def empirical_metric_distortion_score(estimator: LowRankRegressor, X: NDArray, Y: NDArray) -> NDArray:
    _, vr = estimator._eig(return_type = 'eigenvalues_error_bounds')
    return metric_distortion(estimator, vr, X)

def spectral_bias_score(estimator: LowRankRegressor, X: NDArray, Y: NDArray) -> float:
    rank = estimator.rank
    training_samples = estimator.X_fit_.shape[0]
    if estimator.__class__.__name__ == 'ReducedRank':
        return -1.0*float(estimator.svals(k = rank + 3)[rank])
    else:
        K = (training_samples**-1.)*estimator.K_X_
        return -1.0*float(np.flip(np.sort(np.linalg.eigvalsh(K)))[rank])

def param_grid_from_custom_config(config_param_grid:dict, simulation_config:dict, num_train_points: int) -> dict:
    kernel_name = simulation_config["kernel"]
    kernel_class = getattr(kernel_module, kernel_name)

    kernel_kwargs_grid = list(ParameterGrid(config_param_grid[kernel_name]))
    kernel_list = [kernel_class(**kwargs) for kwargs in kernel_kwargs_grid]
    if "base_tikhonov_reg" in config_param_grid:
        reg_list = [float(num_train_points)**(optimal_reg_scaling(kernel_name))*base_reg for base_reg in config_param_grid["base_tikhonov_reg"]]
        params_grid =  {
            "kernel": kernel_list,
            "tikhonov_reg": reg_list
        }
        return params_grid
    else:
        return {"kernel": kernel_list}

def parse_param_grid(param_grid: dict, simulation_config:dict, num_train_points: int) -> dict:
    kernel_name = simulation_config["kernel"]
    kernel_obj = param_grid['kernel']
    _config = {
        kernel_name: {}
    }
    for k, v in kernel_obj.__dict__.items():
        if k[0] == '_':
            pass
        else:
            _config[kernel_name][k] = v
    scaling = -1.0*optimal_reg_scaling(kernel_name) #Minus sign to invert 
    if "tikhonov_reg" in param_grid:
        _config["base_tikhonov_reg"] = param_grid["tikhonov_reg"]*(float(num_train_points)**scaling)
    return _config

def optimal_reg_scaling(kernel_name: str) -> float:
    if kernel_name == 'RBF':
        return -1.0
    else:
        return -0.5