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
        
import numpy as np
from utils import plot_noisy_ordered_MNIST, plot_oracle_metrics, plot_image_forecast, plot_TNSE, create_figure
from pathlib import Path
import ml_confs

# Load configs
main_path = Path(__file__).parent
configs = ml_confs.from_file(main_path / "configs.yaml")

biased_cov_ests = {}
unbiased_cov_ests = {}
ordered_acc = {}
Ns = np.arange(400, configs.train_samples, 400) # Ns = [500,1000,1500,2000,2500,3000,3500,4000,4500,5000,5500,6000,6500,7000,7500,8000]
delta = configs.delta

for i, model_name in enumerate(['Gaussian_RRR',"DPNets", "Classifier_Baseline"]):
    model_name = model_name.replace(" ", "")
    biased_cov_ests[model_name] = np.load(str(main_path) + f'/results/biased_cov_ests_{model_name}_eta_{configs.eta}.npy')
    unbiased_cov_ests[model_name] = np.load(str(main_path) + f'/results/unbiased_cov_ests_{model_name}_eta_{configs.eta}.npy')
    ordered_acc[model_name] = np.load(str(main_path) + f'/results/reports_{model_name}_eta_{configs.eta}.npy')

# Plot the results
# Assuming the required data structures are available
# TODO: Insted of transfer_operator_models and report use the models names 
create_figure(transfer_operator_models, biased_cov_ests, unbiased_cov_ests, Ns, delta, report, configs)

# Plot the image forecast for the first 16 examples in the test set
plot_image_forecast(Noisy_ordered_MNIST, report, configs, test_seed_idx=0)

# Plot the t-SNE of the feature functions for all the transfer operator models in the report dictionary
# TODO: Implement the function plot_TNSE into transfer_op.py
plot_TNSE(report, configs, test_data, test_labels, transfer_operator_models)