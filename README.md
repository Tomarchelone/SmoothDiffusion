# Smooth Variational Diffusion and Signal-to-Noise Ratio Equivalence
Code for research on diffusion generative modelling with smooth sample paths.

[Technical Report](https://zenodo.org/records/19867345)

Abstract:

Common diffusion generative models rely on stochastic processes with trajectories that are almost nowhere differentiable. This work is a proof of concept in which we construct a diffusion generative process whose trajectories are always differentiable. To do so, we use a two-dimensional Gaussian process in which one dimension is the derivative of the other. Building on prior phase-space diffusion approaches such as Critically-Damped Langevin Diffusion (Dockhorn et al., 2022) and the Acceleration Generative Model (Chen et al., 2023), we (i) consider a fully general second-order SDE with arbitrary time-dependent coefficients, and (ii) derive an exact ELBO in phase space. As a consequence, we show that the diffusion loss depends only on a scalar signal-to-noise ratio (SNR), which fully decouples the training of the denoising model from the parameterization of the noising process.

Below is a generation example. Model starts from standard gaussian and generates two gaussians with centers at +-1 and standard deviation of 0.1. Top image shows generation paths, bottom - derivatives of the paths.

<img width="2352" height="2352" alt="paths_and_derivatives" src="https://github.com/user-attachments/assets/c017e794-8c12-4b1c-83eb-042bcb18007a" />
