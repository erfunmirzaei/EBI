import os
import json
import pickle
import importlib
import numpy as np
from tqdm import tqdm
from einops import einsum
from sklearn.model_selection import ParameterGrid
from kooplearn.estimators import LowRankRegressor
from examples.ala2.spectral_rates.src.typing import Trajectory
from examples.ala2.spectral_rates.src.io import ala2_dataset, lagged_sampler

kernel_module = importlib.import_module('kooplearn.kernels')
estimator_module = importlib.import_module('kooplearn.estimators')

def hp_fit(configs:dict, current_file_path: os.PathLike):

    sp_rates_path, _ = os.path.split(current_file_path)
    results_path = os.path.join(sp_rates_path, 'results')
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    
    #Loading the distances data
    train_dist = load_data(current_file_path, split = 'train').dist
    train_pos = load_data(current_file_path, split = 'train').pos
    val_dist = load_data(current_file_path, split = 'validation').dist
    test_dist = load_data(current_file_path, split = 'test').dist
    test_pos = load_data(current_file_path, split = 'test').pos

    n_snapshots = train_dist.shape[0]
    lagtime = configs['data']['lag']
    shuffle = configs['data']['shuffle']
    train_ids = lagged_sampler(n_snapshots, lagtime = lagtime, num_points = configs['data']['n_train'], shuffle = shuffle)
    val_ids = lagged_sampler(n_snapshots, lagtime = lagtime, num_points = configs['data']['n_val'], shuffle = shuffle)
    test_ids = lagged_sampler(n_snapshots, lagtime = lagtime, num_steps = configs['data']['n_forecast'], num_points = configs['data']['n_test'], shuffle = shuffle)
    
    X_tr = train_dist[train_ids[0]]
    Y_tr = train_dist[train_ids[1]]
    X_val = val_dist[val_ids[0]]
    Y_val = val_dist[val_ids[1]]
    X_test = test_dist[test_ids[0]]

    Y_pos_tr = train_pos[train_ids[1]]
    pos_test = np.array([test_pos[idx] for idx in test_ids])
    
    #Estimator init
    estimator_args = configs['estimator']
    estimator_name = estimator_args.pop('name')
    estimator = getattr(estimator_module, estimator_name)(**estimator_args)
    
    #Kernel params grid
    kernels = configs['kernel']
    kernel_list = []
    for name in kernels.keys():
        k = getattr(kernel_module, name)
        for args in list(ParameterGrid(kernels[name])):
            kernel_list.append(k(**args))

    val_scores = np.zeros(len(kernel_list), dtype = np.float64)
    rmse = []
    for k_idx, kernel in tqdm(enumerate(kernel_list), total = len(kernel_list), desc = 'Fitting estimators'):
        estimator.set_params(kernel = kernel)
        estimator.fit(X_tr, Y_tr)
        mean_metric_distortion = empirical_metric_distortion_loss(estimator, X_val, Y_val).mean()
        spectral_bias = spectral_bias_score(estimator, X_tr, Y_tr)
        val_scores[k_idx] = mean_metric_distortion * spectral_bias
        pos_pred = estimator.forecast(
            X_test, 
            t = np.arange(1, configs['data']['n_forecast'] + 1),
            observable = lambda _: Y_pos_tr
            )
        dY = (pos_pred - pos_test[1:])**2
        dY = np.mean(dY, axis = (1, 2)) #Averaging on observations and features
        rmse.append(np.sqrt(dY))
    rmse = np.array(rmse)

    result = {
        'avg_bias': val_scores,
        'forecast_rmse': rmse,
        'kernels': kernel_list,
        'configs': configs
    }
    #Check if 'results' path exists
    
    #Pickling the results
    with open(os.path.join(results_path, 'results.pkl'), 'wb') as f:
        pickle.dump(result, f)

def metric_distortion(fitted_estimator: LowRankRegressor, right_vecs: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    num_test = X_test.shape[0]
    Uv = einsum(fitted_estimator.U_, right_vecs, "n r, r vec -> n vec")
    norm_L2 = (np.abs(fitted_estimator.kernel(X_test, fitted_estimator.X_fit_)@Uv)**2).sum(axis=0)
    norm_RKHS = np.abs((((Uv.conj())*(fitted_estimator.K_X_@Uv))).sum(axis=0))
    return np.sqrt(num_test*norm_RKHS/norm_L2)

def empirical_metric_distortion_loss(estimator: LowRankRegressor, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    _, vr = estimator._eig(return_type = 'eigenvalues_error_bounds')
    return metric_distortion(estimator, vr, X)

def spectral_bias_score(estimator: LowRankRegressor, X: np.ndarray, Y: np.ndarray) -> float:
    rank = estimator.rank
    training_samples = estimator.X_fit_.shape[0]
    if estimator.__class__.__name__ == 'ReducedRank':
        return float(estimator.svals(k = rank + 3)[rank])
    else:
        K = (training_samples**-1.)*estimator.K_X_
        return float(np.flip(np.sort(np.linalg.eigvalsh(K)))[rank])

def load_data(current_file_path:os.PathLike, split:str = 'train') -> Trajectory:
    sp_rates_path, _ = os.path.split(current_file_path)
    ala2_path, _ = os.path.split(sp_rates_path)
    data_path = os.path.join(ala2_path, 'data/')
    return ala2_dataset(data_path, split = split) 

if __name__ == "__main__":
    current_file_path = os.path.dirname(__file__)
    #Loading HP configs
    cfg_path, _ = os.path.split(current_file_path)
    cfg_path = os.path.join(cfg_path, 'configs/hp.json')
    with open(cfg_path, "r") as cfg_file:
        configs = json.load(cfg_file)
    hp_fit(configs, current_file_path)
