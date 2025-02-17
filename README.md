# Flowfusion

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://MurrellGroup.github.io/Flowfusion.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://MurrellGroup.github.io/Flowfusion.jl/dev/)
[![Build Status](https://github.com/MurrellGroup/Flowfusion.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/MurrellGroup/Flowfusion.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/MurrellGroup/Flowfusion.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/MurrellGroup/Flowfusion.jl)



Flowfusion is a Julia package for learning and sampling from conditional diffusion processes across continuous, discrete, and manifold spaces. It provides a unified framework for:

- Learning conditional flows between states
- Sampling from learned distributions
- Working with various state types (continuous, discrete, manifold)
- Handling partial observations and masked states

## Features

### Multiple State Types
- Continuous states (Euclidean spaces)
- Discrete states (categorical variables)
- Manifold states (probability simplexes, tori, rotations)
- Masked states for partial conditioning

### Supported Processes
- Deterministic flows
- Brownian motion
- Ornstein-Uhlenbeck
- Discrete flows (InterpolatingDiscreteFlow, NoisyInterpolatingDiscreteFlow)
- Manifold-specific processes

### Core Operations
- `bridge(P, X0, X1, t)`: Sample intermediate states conditioned on start and end states
- `gen(P, X0, model, steps)`: Generate sequences using a learned model
- Support for both direct state prediction and tangent coordinate prediction

### Training
- Loss functions adapted to different state/process types
- Support for masked training (partial observations)
- Time scaling for improved training dynamics
- Integration with Flux.jl for neural network models

## Examples

The package includes several examples demonstrating different use cases:

- `continuous.jl`: Learning flows between clusters in continuous space
- `discrete.jl`: Learning categorical transitions
- `torus.jl`: Learning flows on a torus manifold
- `probabilitysimplex.jl`: Learning flows between probability distributions

## Installation

```julia
using Pkg
Pkg.add("Flowfusion")
```

## Quick Start

```julia
using Flowfusion, Flux
#To do.
```
