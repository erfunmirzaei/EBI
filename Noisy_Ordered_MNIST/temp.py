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

import data_pipeline
import ml_confs
from datasets import load_from_disk
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import torch
from torch.utils.data import DataLoader
import lightning
# import logging
from kooplearn.data import traj_to_contexts
from sklearn.gaussian_process.kernels import RBF
from tqdm import tqdm
from utils import plot_noisy_ordered_MNIST, plot_oracle_metrics, plot_image_forecast, plot_TNSE
from transfer_op import fit_transfer_operator_models
from oracle_net import ClassifierFeatureMap, CNNEncoder, evaluate_model, Metrics
#%%
# Load configs
configs = ml_confs.from_file('configs.yaml') 

# Set the seed
random.seed(configs.rng_seed)
np.random.seed(configs.rng_seed)
torch.manual_seed(configs.rng_seed)

# Load the dataset
data_pipeline.main() # Run data download and preprocessing
ordered_MNIST = load_from_disk('__data__') # Load dataset (torch)
Noisy_ordered_MNIST = load_from_disk('__data__Noisy') # Load dataset (torch)

device = 'gpu' if torch.cuda.is_available() else 'cpu'

# # Plot the noisy ordered MNIST dataset for the first 16 examples in the train, test, and validation sets
# plot_noisy_ordered_MNIST(Noisy_ordered_MNIST, n=16)
#%%
# Setting up a validation scheme
# The validation of each model will be performed as follows: starting from a test image of the digit $c$, we will predict the next image by calling `model.predict`. 
# The prediction should be an MNIST-alike image of the digit $c+1$ (modulo `configs.classes`). We will feed this prediction to a very strong MNIST classifier, and evaluate how its accuracy degrades over time.

# Train the oracle
oracle_train_dl = DataLoader(ordered_MNIST['train'], batch_size=configs.oracle_batch_size, shuffle=True)
oracle_val_dl = DataLoader(ordered_MNIST['validation'], batch_size=len(ordered_MNIST['validation']), shuffle=False)

trainer_kwargs = {
    'accelerator': device,
    'max_epochs': configs.oracle_epochs,
    'log_every_n_steps': 2,
    'enable_progress_bar': False,
    'devices': 1
}

trainer = lightning.Trainer(**trainer_kwargs)

oracle = ClassifierFeatureMap(
    configs.classes,
    configs.oracle_lr,
    trainer,
    seed=configs.rng_seed
)

oracle.fit(train_dataloaders=oracle_train_dl, val_dataloaders=oracle_val_dl)

# # Plot the training and validation accuracy of the oracle
# plot_oracle_metrics(oracle.lightning_module.metrics)
#%% Fitting the transfer operator models

# Data preparation for the transfer operator models
train_data = traj_to_contexts(Noisy_ordered_MNIST['train']['image'], backend='numpy')
val_data = traj_to_contexts(Noisy_ordered_MNIST['validation']['image'], backend='numpy')
test_data = traj_to_contexts(Noisy_ordered_MNIST['test']['image'], backend='numpy')
test_labels = np.take(Noisy_ordered_MNIST['test']['label'], np.squeeze(test_data.idx_map.lookback(1))).detach().cpu().numpy()

transfer_operator_models, report = fit_transfer_operator_models(train_data, oracle, test_data, configs, device)

# Plot the image forecast for the first 16 examples in the test set
plot_image_forecast(Noisy_ordered_MNIST, report, configs, test_seed_idx=0)

# Plot the t-SNE of the feature functions for all the transfer operator models in the report dictionary
plot_TNSE(report, configs, test_data, test_labels, transfer_operator_models)