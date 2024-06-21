# Lévy-driven Langevin equation - logbook.
**[Mar 13, 23]**: Today's objective is to have a working script to produce trajectories from Lévy (and Brownian) driven Langevin equations. I need to nail down a coherent definition of the parameter `noise_intensity`, taking the place of $k_{{\rm B}}T$ (or $\sigma$), which are only well defined in the Brownian case. We will set `noise_intensity` to the parameter $D$ appearing in the characteristic function of the random variable $L$. For symmetric and isotropic $\alpha$-stable distributions we denote the characteristic function as $\phi_{X}(u) = \exp \left[-D^{\alpha} |u|^{\alpha}\right]$ when $\alpha \in (1, 2]$.


In the Brownian $\alpha = 2$ case, therefore $D = \sigma/\sqrt{2}$, and in the overdamped Langevin dynamics $k_{{\rm B}}T = \frac{\sigma^{2}}{2\gamma m} =: \frac{D^2}{\gamma m}$.

