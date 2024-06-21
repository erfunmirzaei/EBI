#Imports
import collections.abc
import os
from copy import deepcopy
import importlib

#MISC
import jax
import jax.numpy as jnp
from tqdm import tqdm
from examples.utils.io import load_json
from examples.utils.typing import JsonNameSpace

#INTERNAL
from examples.levy_driven_langevin.src.md import sample_markov, init_simulator
from examples.levy_driven_langevin.src.io import save_md_runs

potential_module = importlib.import_module('examples.levy_driven_langevin.src.potentials')
      

def simulate(configs: JsonNameSpace, potential: Callable) -> list:
    if not isinstance(configs.alpha, collections.abc.Iterable):
        _alphas = [configs.alpha]
    else:
        _alphas = configs.alpha

    timesteps_between_samples = configs.timesteps_between_samples
    num_datasets = configs.num_simulations

    _dsets = []

    for alpha in tqdm(_alphas):
        _cfgs = deepcopy(configs)
        _cfgs.alpha = alpha
        simulator = init_simulator(_cfgs, potential = potential)
        dataset = sample_markov(simulator, configs.num_samples, timesteps_between_samples, num_datasets)
        _dsets.append(dataset)
    return _dsets

if __name__ == "__main__":
    cfg_path = os.path.join(os.path.dirname(__file__), 'configs.json')
    base_path = os.path.join(os.path.dirname(__file__), 'data')
    configs = load_json(cfg_path)
    potential_fn = getattr(potential_module, configs.potential)
    
    if configs.get("potential_kwargs", None) is not None:
        potential = lambda x: jax.jit(potential_fn(x, **configs.potential_kwargs))
    else:
        potential = lambda x: jax.jit(potential_fn(x))
     
    data = simulate(configs, potential)
    save_md_runs(data,  potential_fn, configs, base_path)
