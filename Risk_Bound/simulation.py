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
from data_pipeline import make_dataset
from risk_bound import risk_bound_N_OU

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
train_dataset = data_points[0:configs.n_train_points]
val_dataset = data_points[configs.n_train_points:configs.n_train_points + configs.n_val_points]
test_dataset = data_points[configs.n_train_points + configs.n_val_points:]
# Experiment 1: Plot the bounds for different values of tau

# Ns = np.arange(configs.n_train_first, configs.n_sample_est_tr, configs.n_train_step) 
Ns = [500,1000,2000, 5000, 10000, 20000,40000, 80000]
length_scales = [0.05, 0.15, 0.25]

for l in length_scales:

    M_emp_bound_unbiased_cov_est, True_value = risk_bound_N_OU(train_dataset, val_dataset, test_dataset, Ns, lamda, l, configs)
    np.save(str(main_path) + f'/results/Pinelis_bound_delta_{configs.delta}_l_{l}.npy', Pinelis_bound)
    np.save(str(main_path) + f'/results/Pinelis_emp_bound_biased_cov_est_delta_{configs.delta}_l_{l}.npy', Pinelis_emp_bound_biased_cov_est)
