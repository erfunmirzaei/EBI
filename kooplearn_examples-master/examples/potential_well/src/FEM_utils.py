from .misc import configs

from typing import Tuple
from .internal_typing import LinalgDecomposition
from numpy.typing import NDArray

import jax
import numpy as np
import scipy
from scipy.integrate import romb

from discretize import TensorMesh
from .physics import grad_potential, boltzmann_pdf

def koopman_spectrum(
    timesteps_between_samples: int,
    num_eigenvalues: int = 4, 
    num_nodes: int = 2**11 + 1, #Odd number of nodes place a node at 0, Even number of nodes place a "center" at 0
    domain: Tuple = (-3.5, 3.5)
    ) -> Tuple[LinalgDecomposition, NDArray]:

    width = (domain[1] - domain[0])/num_nodes
    mesh = TensorMesh([width*np.ones(num_nodes)], origin='C')
    mesh.set_cell_gradient_BC('dirichlet') #Dirichlet boundary condition as we expect that that the density is (virtually) 0 at the domain boundaries

    grad_x = mesh.average_face_x_to_cell.dot(mesh.cell_gradient_x)
    lap_x = mesh.face_x_divergence.dot(mesh.cell_gradient_x)
    
    grad_potential_discretized = scipy.sparse.diags(np.asarray(jax.vmap(grad_potential)(mesh.cell_centers)))

    inv_gamma = configs['gamma']**-1
    kBT = configs['kBT']

    # Eq.(31) of https://doi.org/10.1007/978-3-642-56589-2_9 recalling that \sigma^2/(\gamma*2) = kBT
    generator = inv_gamma*kBT*lap_x - inv_gamma*grad_potential_discretized.dot(grad_x)
    
    dt = configs['time_step']*timesteps_between_samples #Discretization time
    generator = generator*dt

    vals, vecs = np.linalg.eig(np.asarray(generator.todense())) #Using the full decomposition here to ensure numerical consistency
    assert np.max(np.abs(np.exp(vals).imag)) < 1e-8, "The computed eigenvalues are not real"
    

    #Sorting (in decreasing size) and normalizing
    sort_perm = np.flip(np.argsort(np.exp(vals).real))[:num_eigenvalues]
    vecs = (vecs[:, sort_perm]).real

    #Normalizing
    boltzmann_pdf_sample = boltzmann_pdf(mesh.cell_centers)
    dx = mesh.cell_centers[1] - mesh.cell_centers[0]
    abs2_eigfun = (np.abs(vecs)**2).T
    eigfuns_norms = np.sqrt(romb(boltzmann_pdf_sample*abs2_eigfun, dx = dx, axis = -1))
    vecs = vecs*(eigfuns_norms**-1.0)
    vals = np.exp(vals).real[sort_perm]

    return (LinalgDecomposition(vals, vecs), mesh.cell_centers)



