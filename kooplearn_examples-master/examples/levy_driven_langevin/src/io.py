import os
from datetime import datetime
from pathlib import Path
from typing import Callable
import json
import inspect
import pickle

from examples.utils.typing import JsonNameSpace
def format_path_name(base_path:str):
    timestamp = datetime.now().strftime("%H%M_%-d%b%-y")
    return os.path.join(base_path, timestamp)

def save_md_runs(traj:list, potential: Callable, configs:JsonNameSpace, base_path:str) -> None:
    path = format_path_name(base_path)
    Path(path).mkdir(parents=True, exist_ok=True)

    traj_file = os.path.join(path, 'trajectories.pkl')
    potential_file = os.path.join(path, 'potential.py')   
    cfg_file = os.path.join(path, 'configs.json')
    
    with open(cfg_file, 'w') as outfile:
        json.dump(json.loads(configs.to_json_str()), outfile)
    with open(traj_file, 'wb') as f:
        pickle.dump(traj, f)
    with open(potential_file, 'w') as outfile:
        outfile.write(inspect.getsource(potential))
