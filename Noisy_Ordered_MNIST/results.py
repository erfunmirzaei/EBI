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
from utils import plot_noisy_ordered_MNIST, plot_oracle_metrics, plot_image_forecast, plot_TNSE, Plot_first_figure
from pathlib import Path
import ml_confs

# Load configs
main_path = Path(__file__).parent
configs = ml_confs.from_file(main_path / "configs.yaml")

biased_cov_ests = {}
unbiased_cov_ests = {}
ordered_acc = {}
pred_labels = {}
pred_images = {}
fn_i = {}
fn_j = {}
Ns = np.arange(400, configs.train_samples, 400) # Ns = [500,1000,1500,2000,2500,3000,3500,4000,4500,5000,5500,6000,6500,7000,7500,8000]
n_train_plot = configs.n_train_acc_plot
delta = configs.delta
models_name = ['Gaussian_RRR',"DPNets", "Classifier_Baseline"]

for i, model_name in enumerate(models_name):
    model_name = model_name.replace(" ", "")
    biased_cov_ests[model_name] = np.load(str(main_path) + f'/results/biased_cov_ests_{model_name}_eta_{configs.eta}.npy')
    unbiased_cov_ests[model_name] = np.load(str(main_path) + f'/results/unbiased_cov_ests_{model_name}_eta_{configs.eta}.npy')
    ordered_acc[model_name] = np.load(str(main_path) + f'/results/reports_{model_name}_eta_{configs.eta}.npy')
    pred_labels[model_name] = np.load(str(main_path) + f'/results/pred_labels_{model_name}_eta_{configs.eta}.npy')
    pred_images[model_name] = np.load(str(main_path) + f'/results/pred_images_{model_name}_eta_{configs.eta}.npy')
    true_labels = np.load(str(main_path) + f'/results/true_labels_{model_name}_eta_{configs.eta}.npy')
    true_images = np.load(str(main_path) + f'/results/true_images_{model_name}_eta_{configs.eta}.npy')
    fn_i[model_name] = np.load(str(main_path) + f'/results/fn_i_{model_name}_eta_{configs.eta}.npy')
    fn_j[model_name] = np.load(str(main_path) + f'/results/fn_j_{model_name}_eta_{configs.eta}.npy')

# Plot the results
Plot_first_figure(models_name, biased_cov_ests, unbiased_cov_ests, ordered_acc, Ns, delta, n_train_plot, configs)

# Plot the image forecast for the first 16 examples in the test set
plot_image_forecast(true_labels, true_images, models_name, pred_labels, pred_images, configs)

# Plot the t-SNE of the feature functions for all the transfer operator models in the report dictionary
plot_TNSE(models_name, fn_i, fn_j, configs, true_labels)