from types import SimpleNamespace
from typing import Callable, TypeVar
import jax
import jax.numpy as jnp
from jax_md import dataclasses, util, space, quantity, simulate

import numpy as np
from scipy.integrate import quad

static_cast = util.static_cast

# JaxMD Types
Array = util.Array
f32 = util.f32
ShiftFn = space.ShiftFn

T = TypeVar('T')
InitFn = Callable[..., T]
ApplyFn = Callable[[T], T]
Simulator = tuple[InitFn, ApplyFn]

@jax.jit
def bistable_potential(x, width=1.0, depth=1.0):
   h = 2*depth/(width**2)
   g = 2*depth/(width**4)
   return 0.25*g*(x**4) - 0.5*h*(x**2)

grad_potential = jax.grad(bistable_potential)

def boltzmann_measure(x, configs: SimpleNamespace):
    D = configs.noise_intensity
    gamma = configs.gamma
    mass = configs.mass
    kbT = D*mass*gamma*0.5
    beta = kbT**-1
    return np.exp(-beta*bistable_potential(x, configs.potential_width, configs.potential_depth))

def partition_function(configs: SimpleNamespace):
  Z = quad(lambda x: boltzmann_measure(x, configs), -np.inf, np.inf, limit=1000)[0] #Returning a tuple (result, error)
  return Z

def boltzmann_pdf(x, partition_fn, configs: SimpleNamespace):
    return (partition_fn**-1.0)*boltzmann_measure(x, configs)

def L2_norm(func:Callable, configs: SimpleNamespace, partition_fn:float):
    func_nrm = quad(lambda x:(np.abs(func(x))**2)*boltzmann_pdf(x, partition_fn, configs), -10, 10, limit=1000)[0]
    return np.sqrt(func_nrm)

class boltzmann_scipy_compliant():
    def __init__(self, beta: float, configs: SimpleNamespace):
        self.beta = beta
        self.configs = configs
    def _potential(self, x: float) -> float:
        return bistable_potential(x, self.configs.potential_width, self.configs.potential_depth).item()
    def pdf(self, x):
        return jnp.exp(-self.beta*self._potential(x))

#The following is adapted from JaxMD's brownian dynamics.
@dataclasses.dataclass
class AlphaStableState:
  """A tuple containing state information for Lévy alpha-stable dynamics.

  Attributes:
    position: The current position of the particles. An ndarray of floats with
      shape `[n, spatial_dimension]`.
    mass: The mass of particles. Will either be a float or an ndarray of floats
      with shape `[n]`.
    rng: The current state of the random number generator.
  """
  position: Array
  mass: Array
  rng: Array

def sample_alpha_stable(key, alpha: float, shape = (), dtype = jnp.float32):
    u_key, exp_key = jax.random.split(key)
    gamma = jax.random.uniform(u_key, shape=shape, minval=-0.5*jnp.pi, maxval=0.5*jnp.pi, dtype=dtype)
    W = jax.random.exponential(exp_key, shape=shape, dtype=dtype)
    return alpha_stable_generator(alpha, gamma, W)

def alpha_stable_generator(alpha:float, gamma, W):
   #From "A Method for Simulating Stable RandomVariables, 1976" equation 2.3 with beta = 0
   return (jnp.sin(alpha*gamma)/jnp.cos(gamma)**(alpha**-1)) * (jnp.cos((1-alpha)*gamma)/W)**((1-alpha)/alpha)   

def alpha_stable(energy_or_force: Callable[..., Array],
             shift: ShiftFn,
             dt: float,
             alpha: float=1.5,
             D: float = 1.0,
             gamma: float=0.1) -> Simulator:
  """Simulation of Brownian dynamics.

  Simulates Lévy alpha stable dynamics which are synonymous with the overdamped
  regime of Langevin dynamics.

  Args:
    energy_or_force: A function that produces either an energy or a force from
      a set of particle positions specified as an ndarray of shape
      `[n, spatial_dimension]`.
    shift_fn: A function that displaces positions, `R`, by an amount `dR`.
      Both `R` and `dR` should be ndarrays of shape `[n, spatial_dimension]`.
    dt: Floating point number specifying the timescale (step size) of the
      simulation.
    alpha: A float specifying the alpha parameter of the Lévy alpha-stable distribution
    D: A float specifying the noise intensity of the Lévy alpha-stable distribution. In the Gaussian case, 2. * kB_T / (mass * gamma)
    gamma: A float specifying the friction coefficient between the particles
      and the solvent.

  Returns:
    See above.
  """

  force_fn = quantity.canonicalize_force(energy_or_force)

  dt, gamma = static_cast(dt, gamma)

  def init_fn(key, R, mass=f32(1)):
    state = AlphaStableState(R, mass, key)
    return simulate.canonicalize_mass(state)

  def apply_fn(state, **kwargs):
    R, mass, key = dataclasses.astuple(state)

    key, split = jax.random.split(key)

    F = force_fn(R, **kwargs)
    xi = sample_alpha_stable(split, alpha, R.shape, R.dtype)

    nu = f32(1) / (mass * gamma)
    #From "Barrier crossing driven by L´evy noise: Universality and the Role of Noise Intensity", Equation 8.
    dR = F * dt * nu + (D * dt)**(alpha**-1.) * xi
    R = shift(R, dR, **kwargs)

    return AlphaStableState(R, mass, key)  # pytype: disable=wrong-arg-count

  return init_fn, apply_fn