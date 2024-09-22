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

import pathlib as Path
import ml_confs
import numpy as np
import matplotlib.pyplot as plt
from utils import plot_OU_tau, plot_OU2

# Load configs
main_path = Path(__file__).parent
data_path = main_path / "__data__"
configs = ml_confs.from_file(main_path / "configs.yaml")

length_scales = [0.05, 0.5, 5]
Pinelis_bound = []
Pinelis_emp_bound_biased_cov_est = []
Pinelis_emp_bound_unbiased_cov_est = []
M_bound = []
M_emp_bound_biased_cov_est = []
M_emp_bound_unbiased_cov_est = []
taus = []

for l in length_scales:
    Pinelis_bound = np.load(f'Pinelis_bound_n_{configs.n_plot_tau}_delta_{configs.delta}_l_{l}.npy')
    Pinelis_emp_bound_biased_cov_est = np.load(f'Pinelis_emp_bound_biased_cov_est_n_{configs.n_plot_tau}_delta_{configs.delta}_l_{l}.npy')
    Pinelis_emp_bound_unbiased_cov_est = np.load(f'Pinelis_emp_bound_unbiased_cov_est_n_{configs.n_plot_tau}_delta_{configs.delta}_l_{l}.npy')
    M_bound = np.load(f'M_bound_n_{configs.n_plot_tau}_delta_{configs.delta}_l_{l}.npy')
    M_emp_bound_biased_cov_est = np.load(f'M_emp_bound_biased_cov_est_n_{configs.n_plot_tau}_delta_{configs.delta}_l_{l}.npy')
    M_emp_bound_unbiased_cov_est = np.load(f'M_emp_bound_unbiased_cov_est_n_{configs.n_plot_tau}_delta_{configs.delta}_l_{l}.npy')
    taus = np.load(f'taus_n_{configs.n_plot_tau}_delta_{configs.delta}_l_{l}.npy') 