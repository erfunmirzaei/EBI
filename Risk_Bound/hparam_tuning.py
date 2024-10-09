import numpy as np
from tqdm import tqdm
import random
import torch
from corr_est_cov_est import biased_covariance_estimator, unbiased_covariance_estimator
from sklearn.gaussian_process.kernels import RBF
from kooplearn.models import Linear, Nonlinear, Kernel
from kooplearn.data import traj_to_contexts
from sklearn.model_selection import  ParameterGrid


def parameter_tuning(train_dataset, val_dataset, test_dataset, Ns, length_scale, configs):
    """
    Perform parameter tuning for the kernel model on the Noisy Ordered MNIST dataset
    """

    # Run data pipeline
    # Load configs
    device = 'gpu' if torch.cuda.is_available() else 'cpu'

    # Set the seed
    random.seed(configs.rng_seed)
    np.random.seed(configs.rng_seed)
    torch.manual_seed(configs.rng_seed)

    # Define the parameter grid
    tikhonov_regs = np.geomspace(1e-8, 1e-1, 20)
    params = list(
        ParameterGrid(
            {
                'tikhonov_reg': tikhonov_regs,
            }
        )
    )


    error = np.empty((len(Ns), len(params), 2))
    for iter_idx, iterate in tqdm(enumerate(params), total=len(params)):
        _err = [[] for _ in range(len(Ns))]
        for j in range(len(Ns)):
            n = Ns[j]
            for i in range(configs.n_repits):
                X_tr = train_dataset[0:n+1][:,i]
                X_tr = X_tr.reshape(X_tr.shape[0], -1)
                train_data = traj_to_contexts(X_tr, backend='numpy')
                
                X_val = val_dataset[:,i]
                X_val = X_val.reshape(X_val.shape[0], -1)
                val_data = traj_to_contexts(X_val, backend='numpy')
                
                X_ts = test_dataset[:,i]
                X_ts = X_ts.reshape(X_ts.shape[0], -1)
                test_data = traj_to_contexts(X_ts, backend='numpy')
                
                try :
                    model = Kernel(RBF(length_scale= length_scale), reduced_rank=configs.reduced_rank, rank = configs.rank, tikhonov_reg = iterate['tikhonov_reg']).fit(train_data)
                    _err[j].append(model.risk(val_data))
                except:
                    _err[j].append(np.inf)

            _err[j] = np.array(_err[j])
            error[j, iter_idx, 0] = np.mean(_err[j])
            error[j, iter_idx, 1] = np.std(_err[j])

    best_idxs = [np.argmin(error[j,:,0]) for j in range(len(Ns))]
    best_params = [params[best_idx] for best_idx in best_idxs]

    print(f"The best tikhonov reg is {[best_params[j]['tikhonov_reg'] for j in range(len(Ns))]}")
    print(f"The best error is {[error[j, best_idxs[j], 0] for j in range(len(Ns))]}")
        

    return best_params, error