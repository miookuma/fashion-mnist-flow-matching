# Fashion-MNIST Flow Matching

A class-conditional generative modeling project using flow matching to generate Fashion-MNIST clothing images from Gaussian noise.

The project compares deterministic ODE sampling with a marginal-preserving stochastic SDE sampler.

## What I implemented

- Conditional U-Net in PyTorch
- Flow-matching training objective
- Classifier-free guidance
- Euler ODE sampling
- Euler-Maruyama SDE sampling
- ODE vs. SDE trajectory comparisons
- Diversity and total-variation analysis

## Dataset

Fashion-MNIST contains 28 × 28 grayscale images from ten clothing
categories.

## Main experiments

I investigated:

- ODE vs. SDE sampling
- Different diffusion strengths
- Different numbers of Euler steps
- Sampling trajectories
- Within-class diversity
- Image total variation

## Notebook

See `fashion_flow.ipynb` for the complete implementation, experiments, mathematical explanation, and results.
