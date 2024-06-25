import jax.numpy as jnp

def schwantes_potential(x):
    """
    See Example 1 of "Modeling Molecular Kinetics with tICA and the Kernel Trick" 10.1021/ct5007357
    """
    return 4*(x**8+ 0.8*jnp.exp(-80*(x**2)) +  0.2*jnp.exp(-80*((x - 0.5)**2)) + 0.5*jnp.exp(-40*((x + 0.5)**2)))

def bistable_potential(x, width=1.0, depth=1.0):
   h = 4*depth/(width**2)
   g = 4*depth/(width**4)
   return -0.5*h*x**2 + 0.25*g*x**4

def harmonic_potential(x, width=1.0):
   return 0.5*width*x**2

def quartic_potential(x, width=1.0):
   return 0.25*width*x**4

