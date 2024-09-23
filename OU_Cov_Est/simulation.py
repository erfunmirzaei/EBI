import importlib
import subprocess
import sys
for module in ['kooplearn', 'datasets', 'matplotlib', 'ml-confs']: # !! Add here any additional module that you need to install on top of kooplearn
    try:
        importlib.import_module(module)
    except ImportError:
        if module == 'kooplearn':
            module = 'kooplearn[full]'
        # pip install -q {module}
        subprocess.check_call([sys.executable, "-m", "pip", "install", module])

from pathlib import Path
import random
import ml_confs
import numpy as np
import matplotlib.pyplot as plt
from data_pipeline import make_dataset
from cov_estimation import Covariance_Estimation_tau, Cov_Est_N

# Load configs
main_path = Path(__file__).parent
data_path = main_path / "__data__"
configs = ml_confs.from_file(main_path / "configs.yaml")
Path(main_path / "results").mkdir(parents=True, exist_ok=True)

# Set the seed
random.seed(configs.rng_seed)
np.random.seed(configs.rng_seed)

# Load data
data_points = make_dataset(configs)

# # Experiment 1: Plot the bounds for different values of tau
# length_scales = [0.05, 0.5, 5]
# for l in length_scales:
#     Pinelis_bound, Pinelis_emp_bound_biased_cov_est, Pinelis_emp_bound_unbiased_cov_est, M_bound, M_emp_bound_biased_cov_est, M_emp_bound_unbiased_cov_est, taus = Covariance_Estimation_tau(data_points, configs.n_plot_tau, configs.delta, l, configs)
#     np.save(str(main_path) + f'/results/Pinelis_bound_n_{configs.n_plot_tau}_delta_{configs.delta}_l_{l}.npy', Pinelis_bound)
#     np.save(str(main_path) + f'/results/Pinelis_emp_bound_biased_cov_est_n_{configs.n_plot_tau}_delta_{configs.delta}_l_{l}.npy', Pinelis_emp_bound_biased_cov_est)
#     np.save(str(main_path) + f'/results/Pinelis_emp_bound_unbiased_cov_est_n_{configs.n_plot_tau}_delta_{configs.delta}_l_{l}.npy', Pinelis_emp_bound_unbiased_cov_est)
#     np.save(str(main_path) + f'/results/M_bound_n_{configs.n_plot_tau}_delta_{configs.delta}_l_{l}.npy', M_bound)
#     np.save(str(main_path) + f'/results/M_emp_bound_biased_cov_est_n_{configs.n_plot_tau}_delta_{configs.delta}_l_{l}.npy', M_emp_bound_biased_cov_est)
#     np.save(str(main_path) + f'/results/M_emp_bound_unbiased_cov_est_n_{configs.n_plot_tau}_delta_{configs.delta}_l_{l}.npy', M_emp_bound_unbiased_cov_est)
#     np.save(str(main_path) + f'/results/taus_n_{configs.n_plot_tau}_delta_{configs.delta}_l_{l}.npy', taus)

# Experiment 2: Plot the bounds for different values of N

# data_points = data_points.astype('float16')

# Ns = np.arange(configs.n_train_first, configs.n_sample_est_tr, configs.n_train_step) 
Ns = [500,1000,2000,5000,10000] #20000,40000]
length_scales = [0.05, 0.5, 5]

for l in length_scales:
    Pinelis_bound, Pinelis_emp_bound_biased_cov_est, Pinelis_emp_bound_unbiased_cov_est, M_bound, M_emp_bound_biased_cov_est, M_emp_bound_unbiased_cov_est, True_value = Cov_Est_N(data_points, Ns, configs.delta, l, configs)
    np.save(str(main_path) + f'/results/Pinelis_bound_delta_{configs.delta}_l_{l}.npy', Pinelis_bound)
    np.save(str(main_path) + f'/results/Pinelis_emp_bound_biased_cov_est_delta_{configs.delta}_l_{l}.npy', Pinelis_emp_bound_biased_cov_est)
    np.save(str(main_path) + f'/results/Pinelis_emp_bound_unbiased_cov_est_delta_{configs.delta}_l_{l}.npy', Pinelis_emp_bound_unbiased_cov_est)
    np.save(str(main_path) + f'/results/M_bound_delta_{configs.delta}_l_{l}.npy', M_bound)
    np.save(str(main_path) + f'/results/M_emp_bound_biased_cov_est_delta_{configs.delta}_l_{l}.npy', M_emp_bound_biased_cov_est)
    np.save(str(main_path) + f'/results/M_emp_bound_unbiased_cov_est_delta_{configs.delta}_l_{l}.npy', M_emp_bound_unbiased_cov_est)
    np.save(str(main_path) + f'/results/True_value_delta_{configs.delta}_l_{l}.npy', True_value)