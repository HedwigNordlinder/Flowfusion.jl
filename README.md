# Flowfusion

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://MurrellGroup.github.io/Flowfusion.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://MurrellGroup.github.io/Flowfusion.jl/dev/)
[![Build Status](https://github.com/MurrellGroup/Flowfusion.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/MurrellGroup/Flowfusion.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/MurrellGroup/Flowfusion.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/MurrellGroup/Flowfusion.jl)



Flowfusion is a Julia package for training and sampling from diffusion and flow matching models (and some things in between), across continuous, discrete, and manifold spaces. It provides a unified framework for:

## Features

- Controllable noise (or fully deterministic for flow matching)
- Flexible initial $X_0$ distribution
- Conditioning via masking
- States: Continuous, discrete, and a wide variety of manifolds supported (via [Manifolds.jl](https://github.com/JuliaManifolds/Manifolds.jl))
- Compound states supported (e.g. jointly sampling from both continuous and discrete variables)

### Basic idea:
- Generate `X0` and `X1` states from your favorite distribution, and a random `t` between 0 and 1
- `Xt = bridge(P, X0, X1, t)`: Sample intermediate states conditioned on start and end states
- Train model to predict how to get to `X1` from `Xt`
- `gen(P, X0, model, steps)`: Generate sequences using a learned model

## Examples

The package includes several examples demonstrating different use cases:

- `continuous.jl`: Learning a continuous distribution
- `torus.jl`: Continous distributions on a manifold
- `discrete.jl`: Discrete distributions with discrete processes
- `probabilitysimplex.jl`: Discrete distributions with continuous processes via a probability simplex manifold
- `continuous_masked.jl`: Conditioning on partial observations
- `masked_tuple.jl`: Jointly sampling from continuous and discrete variables, with conditioning

## Installation

```julia
]add https://github.com/MurrellGroup/Flowfusion.jl
```

## A full example

```julia
using ForwardBackward, Flowfusion, Flux, RandomFeatureMaps, Optimisers, Plots

#Set up a Flux model: X̂1 = model(t,Xt)
struct FModel{A}
    layers::A
end
Flux.@layer FModel
function FModel(; embeddim = 128, spacedim = 2, layers = 3)
    embed_time = Chain(RandomFourierFeatures(1 => embeddim, 1f0), Dense(embeddim => embeddim, swish))
    embed_state = Chain(RandomFourierFeatures(2 => embeddim, 1f0), Dense(embeddim => embeddim, swish))
    ffs = [Dense(embeddim => embeddim, swish) for _ in 1:layers]
    decode = Dense(embeddim => spacedim)
    layers = (; embed_time, embed_state, ffs, decode)
    FModel(layers)
end
function (f::FModel)(t, Xt)
    l = f.layers
    tXt = tensor(Xt)
    tv = zero(tXt[1:1,:]) .+ expand(t, ndims(tXt))
    x = l.embed_time(tv) .+ l.embed_state(tXt)
    for ff in l.ffs
        x = x .+ ff(x)
    end
    tXt .+ l.decode(x) .* (1.05f0 .- expand(t, ndims(tXt))) 
end

model = FModel(embeddim = 256, layers = 3, spacedim = 2)

#Distributions for training:
T = Float32
sampleX0(n_samples) = rand(T, 2, n_samples) .+ 2
sampleX1(n_samples) = Flowfusion.random_literal_cat(n_samples, sigma = T(0.05))
n_samples = 400

#The process:
P = BrownianMotion(0.15f0)
#P = Deterministic()

#Optimizer:
eta = 0.001
opt_state = Flux.setup(AdamW(eta = eta), model)

iters = 4000
for i in 1:iters
    #Set up a batch of training pairs, and t:
    X0 = ContinuousState(sampleX0(n_samples))
    X1 = ContinuousState(sampleX1(n_samples))
    t = rand(T, n_samples)
    #Construct the bridge:
    Xt = bridge(P, X0, X1, t)
    #Gradient & update:
    l,g = Flux.withgradient(model) do m
        floss(P, m(t,Xt), X1, scalefloss(P, t))
    end
    Flux.update!(opt_state, model, g[1])
    (i % 10 == 0) && println("i: $i; Loss: $l")
end

#Generate samples by stepping from X0
n_inference_samples = 5000
X0 = ContinuousState(sampleX0(n_inference_samples))
samples = gen(P, X0, model, 0f0:0.005f0:1f0)

#Plotting
pl = scatter(X0.state[1,:],X0.state[2,:], msw = 0, ms = 1, color = "blue", alpha = 0.5, size = (400,400), legend = :topleft, label = "X0")
X1true = sampleX1(n_inference_samples)
scatter!(X1true[1,:],X1true[2,:], msw = 0, ms = 1, color = "orange", alpha = 0.5, label = "X1 (true)")
scatter!(samples.state[1,:],samples.state[2,:], msw = 0, ms = 1, color = "green", alpha = 0.5, label = "X1 (generated)")
```

## Tracking trajectories

```julia
#Generate samples by stepping from X0
n_inference_samples = 5000
X0 = ContinuousState(sampleX0(n_inference_samples))
paths = Tracker() #<- A tracker to record the trajectory
samples = gen(P, X0, model, 0f0:0.005f0:1f0, tracker = paths)

#Plotting:
pl = scatter(X0.state[1,:],X0.state[2,:], msw = 0, ms = 1, color = "blue", alpha = 0.5, size = (400,400), legend = :topleft, label = "X0")
tvec = stack_tracker(paths, :t)
xttraj = stack_tracker(paths, :xt)
for i in 1:50:1000
    plot!(xttraj[1,i,:], xttraj[2,i,:], color = "red", label = i==1 ? "Trajectory" : :none, alpha = 0.4)
end
X1true = sampleX1(n_inference_samples)
scatter!(X1true[1,:],X1true[2,:], msw = 0, ms = 1, color = "orange", alpha = 0.5, label = "X1 (true)")
scatter!(samples.state[1,:],samples.state[2,:], msw = 0, ms = 1, color = "green", alpha = 0.5, label = "X1 (generated)")
```