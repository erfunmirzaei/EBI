import os
import numpy as np
from typing import Tuple
from sklearn.preprocessing import robust_scale
from examples.ala2.spectral_rates.src.typing import Trajectory

def ala2_dataset(data_path:os.PathLike, split='train') -> Trajectory:
    files = {
        "dihedrals": "alanine-dipeptide-3x250ns-backbone-dihedrals.npz",
        "distances": "alanine-dipeptide-3x250ns-heavy-atom-distances.npz",
        "positions": "alanine-dipeptide-3x250ns-heavy-atom-positions.npz",
    }
    if split == 'train':
        aname = 'arr_0'
    elif split == 'test':
        aname = 'arr_1'
    elif split == 'validation':
        aname = 'arr_2'
    else:
        raise ValueError(f"Unknown split = '{split}'. The acceped values are ['train', 'test', 'validation'].") 

    return Trajectory(
        ang = np.load(os.path.join(data_path, files["dihedrals"]))[aname],
        dist = np.load(os.path.join(data_path, files["distances"]))[aname],
        pos = np.load(os.path.join(data_path, files["positions"]))[aname],
    )

def lagged_dataset(traj: np.ndarray, lagtime: int = 1, num_points: int = -1, normalize = True, shuffle = False)-> Tuple:
    n_snapshots = traj.shape[0]
    if normalize:
        traj = robust_scale(traj)
    if shuffle:
        p = np.random.permutation(n_snapshots - lagtime)
    else:
        p = np.arange(n_snapshots - lagtime, dtype=np.int32)
    
    X = traj[:-lagtime][p]
    Y = traj[lagtime:][p]

    if num_points < 0: #return all
        return (X, Y)
    else:
        assert num_points <= len(traj) - lagtime
        return (X[:num_points], Y[:num_points])

def lagged_sampler(n_snapshots: int, lagtime: int = 1, num_steps: int = 1, num_points: int = -1, shuffle = False)-> list:
    effective_len = n_snapshots - lagtime*num_steps
    if shuffle:
        p = np.random.permutation(effective_len)
    else:
        p = np.arange(effective_len, dtype=np.int32)
    
    if num_points > 0:
        assert num_points <= effective_len
        p = p[:num_points]
    
    idxs = [p + i*lagtime for i in range(num_steps + 1)]
    return idxs