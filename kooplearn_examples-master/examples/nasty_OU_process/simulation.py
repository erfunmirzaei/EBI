import json
from typing import NamedTuple, Tuple
import numpy as np
from datetime import datetime
from tqdm import tqdm
import pickle

from src.cov_estimators import pcr_fit, reduced_rank_fit, eig
from  src import OU_process
from src.kernel import HermitePoly

with open("config.json", "r") as f:
    config = json.load(f)

class SimulationData(NamedTuple):
    config: dict
    datasets: Tuple
    eigenvalues: dict

if __name__ == "__main__":
    rank = config["rank"]
    num_trajectories = config["num_trajectories"]
    num_points = config["num_points"]
    tikhonov_reg = config["tikhonov_reg"]
    oversamples = config["oversamples"]
    kind = config["kernel_kind"]

    dataset = OU_process.sample(num_points + 1, num_trajectories=num_trajectories)
    X, Y = dataset[:-1], dataset[1:] #Not elegant but working

    kernel = HermitePoly(rank, kind = kind, oversamples = oversamples)

    eigenvalues_storage = {
        'PCR': np.zeros((num_trajectories, rank), dtype = np.complex128),
        'RRR': np.zeros((num_trajectories, rank), dtype = np.complex128)
    }

    estimators = [
        (pcr_fit, 'PCR'),
        (reduced_rank_fit, 'RRR')
    ]

    for traj_idx in tqdm(range(num_trajectories), desc="Fitting PCR and RRR estimators"):
        _X = X[..., traj_idx]
        _Y = Y[..., traj_idx]
        covariances = (
            kernel.cov(_X), 
            kernel.cov(_X, _Y)
        )
        for fit_fn, estimator_name in estimators:        
            _tmp_res = fit_fn(covariances, tikhonov_reg, rank)
            eigenvalues_storage[estimator_name][traj_idx] = eig(_tmp_res, covariances[1]).vals
    
    data = SimulationData(
        config,
        (X, Y),
        eigenvalues_storage
        )

    timestamp = datetime.now().strftime("%H%M_%-d_%-m_%-y")
    file_name = kernel.__class__.__name__ + "_" +kind+ "_" + timestamp + ".pkl"

    with open('data/' + file_name, 'wb') as f:
        pickle.dump(data, f)