#Imports
#STDLIB
import json
from typing import TypeVar, Tuple, NamedTuple
from datetime import datetime
import pickle
import importlib

#ESSENTIALS
import numpy as np
import jax.numpy as jnp
import jax
import scipy.sparse.linalg
import scipy.sparse
import einops
import scipy.stats.sampling

#MISC
from tqdm import tqdm

import jax_md
from discretize import TensorMesh

kernel_module = importlib.import_module('kooplearn.kernels')
estimator_module = importlib.import_module('kooplearn.estimators')

with open("config.json", "r") as f:
    configs = json.load(f)

kernel_class = getattr(kernel_module, configs["kernel"])
estimator_class = getattr(estimator_module, configs["estimator"])

T = TypeVar('T')

class SimulationData(NamedTuple):
    configs_dump: dict
    datasets: Tuple
    eigenvalues: np.ndarray
    eta: np.ndarray #Eigenfunctions scalar products. Used to estimate data-dependent bounds
    sval_B_rp1: np.ndarray #\sigma_{r + 1}(B) (r + 1)-th singular value of B = C^{-1/2}T. Used to estimate data-dependent bounds.
    benchmark: np.ndarray
    timestamp: datetime

class BoltzmannDistribution():
    def __init__(self, beta: float):
        self.beta = beta
    def _potential(self, x: float) -> float:
        return potential(x).item()
    def pdf(self, x):
        return jnp.exp(-self.beta*self._potential(x))

def koopman_eigenvalues(
    timesteps_between_samples: int, 
    num_nodes: int = 1500, 
    domain: Tuple = (-1.15, 1.15)
    ) -> np.ndarray:
    width = (domain[1] - domain[0])/num_nodes
    mesh = TensorMesh([width*np.ones(num_nodes)], origin='C')
    mesh.set_cell_gradient_BC('neumann')
    format = 'csc'
    
    grad_Q = (mesh.average_face_x_to_cell.dot(mesh.cell_gradient_x)).asformat(format)
    lap_Q = (grad_Q.dot(grad_Q)).asformat(format)
    grad_V = scipy.sparse.diags(np.asarray(jax.vmap(grad_potential)(mesh.cell_centers)), format=format)
    generator = (configs['gamma']**-1)*configs['temperature']*lap_Q - (configs['gamma']**-1)*grad_V.dot(grad_Q)

    dt = configs['time_step']*timesteps_between_samples
    
    vals = np.exp(scipy.sparse.linalg.eigs(dt*generator, k = 15, return_eigenvectors=False, which='LR'))    
    return np.flip(np.sort(np.unique(vals.round(decimals=5))))[:configs["estimator_kwargs"]["rank"]]

@jax.jit
def potential(x):
    """
    See Example 1 of "Modeling Molecular Kinetics with tICA and the Kernel Trick" 10.1021/ct5007357
    """
    return 4*(x**8+ 0.8*jnp.exp(-80*(x**2)) +  0.2*jnp.exp(-80*((x - 0.5)**2)) + 0.5*jnp.exp(-40*((x + 0.5)**2)))

grad_potential = jax.grad(potential)

displacement_fn, shift_fn = jax_md.space.free()
init_simulation, _apply = jax_md.simulate.brownian(
    lambda x: jnp.sum(potential(x)), 
    shift_fn, 
    configs["time_step"], 
    configs["temperature"], 
    gamma= configs["gamma"]
)
apply = jax.jit(_apply)

def simulate(num_steps: int, state: T) -> T:
    def evolve_ith(i:int, state:T) -> T:
        return apply(state)
    state = jax.lax.fori_loop(0, num_steps, evolve_ith, state)
    return state

def sample(
    sampling_scheme: str,
    num_samples: int, 
    timesteps_between_samples: int = 1, 
    num_datasets: int = 1, 
    seed: int = 0
) -> Tuple:
    if sampling_scheme == 'markov':
        return sample_markov(
            num_samples, 
            timesteps_between_samples=timesteps_between_samples,
            num_datasets=num_datasets,
            seed = seed
            )
    elif sampling_scheme == 'iid':
        return sample_iid(
            num_samples, 
            timesteps_between_samples=timesteps_between_samples,
            num_datasets=num_datasets,
            seed = seed
            )
    else:
        raise ValueError(f"Invalid sampling scheme: available sampling schemes are 'markov' or 'iid', but {sampling_scheme} was selected.")

def sample_markov(
    num_samples: int, 
    timesteps_between_samples: int = 1, 
    num_datasets: int = 1, 
    seed: int = 0
    ) -> Tuple:

    key = jax.random.PRNGKey(seed)
    key_brownian, key_init_pos = jax.random.split(key)
    _initial_positions = jax.random.uniform(key_init_pos, (num_datasets, 1), minval=-1, maxval=1)
    state = init_simulation(key_brownian, _initial_positions)
    positions = jnp.zeros((num_samples + 1, num_datasets, 1))

    def _update_fn(i, val):
        state, positions = val
        state = simulate(timesteps_between_samples, state)
        return (state, positions.at[i].set(state.position))

    _, positions = jax.lax.fori_loop(0, num_samples, _update_fn, (state, positions))

    X = positions[:-1]
    Y = positions[1:]

    X = einops.rearrange(X, 's d f -> d s f') #[num_datasets, num_samples]
    Y = einops.rearrange(Y, 's d f -> d s f') #[num_datasets, num_samples]
    return (X, Y) 

def sample_iid(
    num_samples: int, 
    timesteps_between_samples: int = 1, 
    num_datasets: int = 1, 
    seed: int = 0
    ) -> Tuple:

    key = jax.random.PRNGKey(seed)
    rand_key, key_init_pos = jax.random.split(key)

    init_pos_seed = jax.random.randint(key_init_pos, (1,), 0, 1e6).item()

    boltzmann = BoltzmannDistribution(configs["temperature"]**-1)
    boltzmann_rng = scipy.stats.sampling.NumericalInversePolynomial(boltzmann, random_state = init_pos_seed)

    X = jnp.asarray(boltzmann_rng.rvs((num_samples, num_datasets, 1)))
    Y = jnp.zeros_like(X)

    def _update_fn(i, val):
        Y, rand_key = val
        rand_key, key_brownian = jax.random.split(rand_key)
        _initial_positions = X[i]
        state = init_simulation(key_brownian, _initial_positions)
        state = simulate(timesteps_between_samples, state)
        return (Y.at[i].set(state.position), rand_key)

    Y, _ = jax.lax.fori_loop(0, num_samples, _update_fn, (Y, rand_key))        
    
    X = einops.rearrange(X, 's d f -> d s f') #[num_datasets, num_samples, 1]
    Y = einops.rearrange(Y, 's d f -> d s f') #[num_datasets, num_samples, 1]
    return (X, Y)

def compute_eigenvalues(X: jnp.ndarray, Y: jnp.ndarray) -> Tuple:
    X = np.asarray(X)
    Y = np.asarray(Y)
    assert X.shape == Y.shape
    kernel = kernel_class(**configs["kernel_kwargs"])
    estimator = estimator_class(kernel, **configs["estimator_kwargs"])
    num_datasets = X.shape[0]
    eigenvalues = np.zeros((num_datasets, configs["estimator_kwargs"]["rank"]), dtype=np.complex128)
    eta = np.zeros((num_datasets, configs["estimator_kwargs"]["rank"]), dtype=np.complex128)
    sval_B_rp1 = np.zeros((num_datasets), dtype=np.float64)
    for ds_index in tqdm(range(num_datasets), desc="Computing eigenvalues", unit="dataset"):
        x, y = X[ds_index], Y[ds_index]
        estimator.fit(x, y)
        w, vr = estimator._eig(return_type='eigenvalues_error_bounds')
        eta[ds_index] = np.sum(vr.conj()*vr, axis=0)/(np.sum(vr.conj()*(estimator.U_.T@estimator.V_@vr), axis=0))
        eigenvalues[ds_index] = w
        try:
            sval_B_rp1[ds_index] = estimator.RRR_sq_svals_[configs["estimator_kwargs"]["rank"]]
        except AttributeError:
            sval_B_rp1[ds_index] = np.nan
    return (eigenvalues, eta, sval_B_rp1)

if __name__ == "__main__":
    num_samples = configs["num_samples"]
    timesteps_between_samples = configs["timesteps_between_samples"]
    num_datasets = configs["num_simulations"]
    sampling_scheme = configs["sampling_scheme"]

    X, Y = sample(sampling_scheme, num_samples, timesteps_between_samples, num_datasets)

    eigenvalues, eta, sval_B_rp1 = compute_eigenvalues(X, Y)
    benchmark = koopman_eigenvalues(timesteps_between_samples)

    data = SimulationData(
        configs,
        (X, Y),
        eigenvalues.real,
        eta,
        sval_B_rp1,
        benchmark.real,
        datetime.now()
    )

    with open('data/' + configs["file_name"], 'wb') as f:
        pickle.dump(data, f)
