from .misc import configs
from typing import Callable
import jax
import jax.numpy as jnp

import numpy as np
from scipy.integrate import quad


@jax.jit
def potential(x):
    """
    See Example 1 of "Modeling Molecular Kinetics with tICA and the Kernel Trick" 10.1021/ct5007357
    """
    return 4*(x**8+ 0.8*jnp.exp(-80*(x**2)) +  0.2*jnp.exp(-80*((x - 0.5)**2)) + 0.5*jnp.exp(-40*((x + 0.5)**2)))

_dpot = jax.grad(potential) #Derivative of potential
dpot = jax.vmap(_dpot) #Batched derivative of potential
d2pot = jax.vmap(jax.grad(_dpot)) #Batched second derivative of potential

grad_potential = jax.grad(potential)

beta = configs["kBT"]**-1
def boltzmann_measure(x):
    return np.exp(-beta*potential(x))

boltzmann_mass = quad(boltzmann_measure, -np.inf, np.inf)[0] #Returning a tuple (result, error)

def boltzmann_pdf(x):
    return (boltzmann_mass**-1.0)*boltzmann_measure(x)

def L2_norm(func:Callable):
    func_nrm = quad(lambda x:(np.abs(func(x))**2)*boltzmann_pdf(x), -10, 10, limit=1000)[0]
    return np.sqrt(func_nrm)

class boltzmann_scipy_compliant():
    def __init__(self, beta: float):
        self.beta = beta
    def _potential(self, x: float) -> float:
        return potential(x).item()
    def pdf(self, x):
        return jnp.exp(-self.beta*self._potential(x))
