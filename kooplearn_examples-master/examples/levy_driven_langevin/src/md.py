
from typing import Callable
from types import SimpleNamespace
import numpy as np
import jax
import jax.numpy as jnp
import jax_md
import einops

from examples.levy_driven_langevin.src.physics import alpha_stable
from examples.levy_driven_langevin.src.typing import T, Dataset

InitFn = Callable[..., T]
ApplyFn = Callable[[T], T]
Simulator = tuple[InitFn, ApplyFn]

def init_simulator(configs: SimpleNamespace, potential: Callable) -> Simulator:
    """
    Returns a simulator for Brownian or Lévy alpha stable dynamics.
    """
    _, shift_fn = jax_md.space.free()
    D = configs.noise_intensity
    gamma = configs.gamma
    mass = configs.mass
    if configs.alpha == 2: #Gaussian Noise
        kbT = D*mass*gamma*0.5
        print(f"The temperature corresponding to the noise level D={D:.2f} is kB_T = {kbT:.2f}")
        sim = jax_md.simulate.brownian
        kw = {
            "energy_or_force": lambda x: jnp.sum(potential(x)),
            "shift": shift_fn,
            "dt": configs.time_step,
            "kT": kbT,
            "gamma": gamma
        }
    else:
        sim = alpha_stable
        kw = {
            "energy_or_force": lambda x: jnp.sum(potential(x)),
            "shift": shift_fn,
            "dt": configs.time_step,
            "alpha": configs.alpha,
            "D": D,
            "gamma": gamma
        }
    init_fn, _apply = sim(**kw)
    apply_fn = jax.jit(_apply)
    return init_fn, apply_fn

def run_simulation(num_steps: int, state: T, apply_fn: ApplyFn) -> T:
    def evolve_ith(i:int, state:T) -> T:
        return apply_fn(state)
    state = jax.lax.fori_loop(0, num_steps, evolve_ith, state)
    return state

def sample_markov(
    simulator: Simulator,
    num_samples: int, 
    timesteps_between_samples: int = 1, 
    num_datasets: int = 1, 
    seed: int = 0
    ) -> Dataset:
    init_fn, apply_fn = simulator
    key = jax.random.PRNGKey(seed)
    key, key_init_pos = jax.random.split(key)
    _initial_positions = jax.random.uniform(key_init_pos, (num_datasets, 1), minval=-1, maxval=1)
    state = init_fn(key, _initial_positions)
    positions = jnp.zeros((num_samples + 1, num_datasets, 1))
    
    @jax.jit
    def _update_fn(i, val):
        state, positions = val
        state = run_simulation(timesteps_between_samples, state, apply_fn)
        return (state, positions.at[i].set(state.position))

    _, positions = jax.lax.fori_loop(0, num_samples, _update_fn, (state, positions))

    X = positions[:-1]
    Y = positions[1:]

    X = einops.rearrange(X, 's d f -> d s f') #[num_datasets, num_samples, 1]
    Y = einops.rearrange(Y, 's d f -> d s f') #[num_datasets, num_samples, 1]

    X = np.asarray(X)
    Y = np.asarray(Y)
    return Dataset(X, Y) 