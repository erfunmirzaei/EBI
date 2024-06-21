import os
import json
import importlib
from datetime import datetime
import pickle
from typing import NamedTuple

import numpy as np
from sklearn.model_selection import PredefinedSplit, GridSearchCV
from examples.ala2.spectral_rates.src.io_utils import load_ala2_dataset, lagged_dataset
from examples.ala2.spectral_rates.src.hyperopt_utils import param_grid_from_custom_config, parse_param_grid

kernel_module = importlib.import_module('kooplearn.kernels')
estimator_module = importlib.import_module('kooplearn.estimators')
scores_module = importlib.import_module('src.hyperopt_utils')

class CVResults(NamedTuple):
    optimal_params_grid: dict
    hyperopt_config: dict
    simulation_config: dict
    sklearn_cvresults_dump: dict

if __name__ == "__main__":
    MAIN_DIR = os.path.dirname(__file__)
    CONFIGS_FILE = os.path.join(MAIN_DIR, 'configs/simulation_config.json')
    HYPEROPT_CONFIGS_FILE = os.path.join(MAIN_DIR, 'configs/hyperopt_config.json')
    DATA_DIR = os.path.join(os.path.split(MAIN_DIR)[0], 'data')

    with open(CONFIGS_FILE, "r") as config_file:
        configs = json.load(config_file)
    with open(HYPEROPT_CONFIGS_FILE, "r") as config_file:
        hyperopt_configs = json.load(config_file)
    
    #Init estimator with the default vaules from 'simulation_config.json'
    kernel = getattr(kernel_module, configs["kernel"])(**configs["kernel_args"])
    if "base_tikhonov_reg" in hyperopt_configs:
        estimator = getattr(estimator_module, configs["estimator"])(
            kernel = kernel, 
            tikhonov_reg = hyperopt_configs["base_tikhonov_reg"][0], 
            **configs["estimator_args"]
        )
    else:
        estimator = getattr(estimator_module, configs["estimator"])(
            kernel = kernel, 
            **configs["estimator_args"]
        )
    
    #Data loading
    data = load_ala2_dataset(DATA_DIR)

    X_train, Y_train = lagged_dataset(data.train.distances, lagtime = configs["lagtime"], num_points = hyperopt_configs["training_points"])
    X_val, Y_val = lagged_dataset(data.validation.distances, lagtime = hyperopt_configs["validation_lagtime"], num_points = hyperopt_configs["validation_points"])

    X = np.concatenate((X_train, X_val))
    Y = np.concatenate((Y_train, Y_val))
    
    #CV splits definition
    len_train = X_train.shape[0]
    len_validation = X_val.shape[0]

    predefined_cv = PredefinedSplit(np.array([-1]*len_train + [0]*len_validation))
    #Params grid
    custom_params_grid = param_grid_from_custom_config(
        hyperopt_configs["custom_params_grid"],
        configs,
        len_train
    )
    
    params_grid = hyperopt_configs["standard_params_grid"] | custom_params_grid

    #Hyperopt
    search = GridSearchCV(
        estimator,
        params_grid,
        scoring = getattr(scores_module, hyperopt_configs["scoring"]),
        n_jobs = 1, 
        refit = False,
        cv = predefined_cv,
        verbose = 3
    )

    search.fit(X, Y)
    best_params = search.best_params_

    parsed_best_params = {}

    #Update the configuration with the optimal parameters
    for k in hyperopt_configs["standard_params_grid"].keys():
        parsed_best_params[k] = best_params[k]
    
    parsed_best_params = parsed_best_params | parse_param_grid(best_params, configs, len_train)

    print(f"The best parameters, attaining a {hyperopt_configs['scoring']} of {-1.0*search.best_score_:.3f} are:")
    print(json.dumps(parsed_best_params, indent=4))

    cv_results = CVResults(
        parsed_best_params,
        hyperopt_configs,
        configs,
        search.cv_results_
    )  
    timestamp = datetime.now().strftime("%H%M_%-d_%-m_%-y")
    file_name = estimator.__class__.__name__ + "_cv_" + timestamp + ".pkl"
    with open('../data/eigenvalue_estimation/hyperopt_cv_results/' + file_name, 'wb') as f:
        pickle.dump(cv_results, f)