from .misc import configs

from .internal_typing import T, Dataset
import numpy as np
import jax
import jax.numpy as jnp
import jax_md
import einops
import scipy

from .physics import potential, boltzmann_scipy_compliant

displacement_fn, shift_fn = jax_md.space.free()
init_simulation, _apply = jax_md.simulate.brownian(
    lambda x: jnp.sum(potential(x)), 
    shift_fn, 
    configs["time_step"], 
    configs["kBT"], 
    gamma = configs["gamma"]
)
apply = jax.jit(_apply)

def simulate(num_steps: int, state: T) -> T:
    def evolve_ith(i:int, state:T) -> T:
        return apply(state)
    state = jax.lax.fori_loop(0, num_steps, evolve_ith, state)
    return state

def sample_markov(
    num_samples: int, 
    timesteps_between_samples: int = 1, 
    num_datasets: int = 1, 
    seed: int = 0
    ) -> Dataset:

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

    X = einops.rearrange(X, 's d f -> d s f') #[num_datasets, num_samples, 1]
    Y = einops.rearrange(Y, 's d f -> d s f') #[num_datasets, num_samples, 1]

    X = np.asarray(X)
    Y = np.asarray(Y)
    return Dataset(X, Y)

def random_time_markov_sample(num_samples: int, mean_sample_timesteps: float, seed:int = 0):
    key = jax.random.PRNGKey(seed)
    key_brownian, key_init_pos, key_dt = jax.random.split(key, num = 3)
    x_0 = jax.random.uniform(key_init_pos, minval=-1, maxval=1)
    state = init_simulation(key_brownian, x_0)

    positions = jnp.zeros(num_samples)
    dt = jax.random.geometric(key_dt, p = 1/mean_sample_timesteps, shape = (num_samples,))
    times = jnp.cumsum(dt) - dt[0]

    def _update_fn(i, val):
        state, positions = val
        state = simulate(dt[i], state)
        return (state, positions.at[i].set(state.position))

    _, positions = jax.lax.fori_loop(0, num_samples, _update_fn, (state, positions))
    return positions, times

def sample_iid(
    num_samples: int, 
    timesteps_between_samples: int = 1, 
    num_datasets: int = 1, 
    seed: int = 0
    ) -> Dataset:

    key = jax.random.PRNGKey(seed)
    rand_key, key_init_pos = jax.random.split(key)

    init_pos_seed = jax.random.randint(key_init_pos, (1,), 0, 1e6).item()

    boltzmann = boltzmann_scipy_compliant(configs["kBT"]**-1)
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

    X = np.asarray(X)
    Y = np.asarray(Y)

    return Dataset(X, Y)

def sample(
    sampling_scheme: str,
    num_samples: int, 
    timesteps_between_samples: int = 1, 
    num_datasets: int = 1, 
    seed: int = 0
) -> Dataset:
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