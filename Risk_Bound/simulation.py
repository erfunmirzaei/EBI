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
from hparam_tuning import parameter_tuning

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
train_dataset = data_points[0:configs.n_train_points+1]
val_dataset = data_points[configs.n_train_points+1:configs.n_train_points + configs.n_val_points+2]
test_dataset = data_points[configs.n_train_points + configs.n_val_points+2:]
# Experiment 1: Plot the bounds for different values of tau

# Ns = np.arange(configs.n_train_first, configs.n_sample_est_tr, configs.n_train_step) 
Ns = [500, 1000, 2000, 5000, 10000]
length_scales = [0.05, 0.15, 0.25]

# Perform parameter tuning
for l in length_scales:
    lamdas, _ = parameter_tuning(train_dataset, val_dataset, test_dataset, Ns, l, configs)
    np.save(str(main_path) + f'/results/lamdas_delta_{configs.delta}_l_{l}.npy', lamdas)

# # Compute the risk bounds
# for l in length_scales:
    emp_risk, risk_bound, test_emp_risk = risk_bound_N_OU(train_dataset, val_dataset, test_dataset, Ns, lamdas, l, configs)
    np.save(str(main_path) + f'/results/emp_risk_delta_{configs.delta}_l_{l}_reg_{configs.lamda}.npy', emp_risk)
    np.save(str(main_path) + f'/results/risk_bound_delta_{configs.delta}_l_{l}_reg_{configs.lamda}.npy', risk_bound)
    np.save(str(main_path) + f'/results/test_emp_risk_delta_{configs.delta}_l_{l}_reg_{configs.lamda}.npy', test_emp_risk)

