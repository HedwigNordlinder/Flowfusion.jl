using Pkg
Pkg.activate(".")
using Revise
Pkg.develop(path="../../ForwardBackward/")
Pkg.develop(path="../")
using ForwardBackward, Flowfusion, NNlib, Flux, RandomFeatureMaps, Optimisers, Plots

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
sampleX1(n_samples) = Flowfusion.random_literal_cat(n_samples, sigma = T(0.05))
sampleX0(n_samples) = rand(T, 2, n_samples) .+ 2
n_samples = 200

#The process:
P = BrownianMotion(0.1f0)
#P = Deterministic()

#Optimizer:
eta = 0.01
opt_state = Flux.setup(AdamW(eta = eta, lambda = 0.001), model)

iters = 5000
for i in 1:iters
    #Set up a batch of training pairs, and t:
    X1 = ContinuousState(sampleX1(n_samples))
    X0 = ContinuousState(sampleX0(n_samples))
    t = rand(T, n_samples)
    #Construct the bridge:
    Xt = bridge(P, X0, X1, t)
    #Gradient:
    l,g = Flux.withgradient(model) do m
        floss(P, m(t,Xt), X1, scalefloss(P, t, 2))
    end
    #Update:
    Flux.update!(opt_state, model, g[1])
    #Logging, and lr cooldown:
    if i % 10 == 0
        if i > iters - 2000
            eta *= 0.975
            Optimisers.adjust!(opt_state, eta)
        end
        println("i: $i; Loss: $l; eta: $eta")
    end
end

#Generate samples by stepping from X0
n_inference_samples = 5000
X0 = ContinuousState(sampleX0(n_inference_samples))
paths = Tracker()
samp = gen(P, X0, model, 0f0:0.005f0:1f0, tracker = paths)

#Plotting:
pl = scatter(X0.state[1,:],X0.state[2,:], msw = 0, ms = 1, color = "blue", alpha = 0.5, size = (400,400), legend = :topleft, label = "X0")
tvec = stack_tracker(paths, :t)
xttraj = stack_tracker(paths, :xt)
for i in 1:50:1000
    plot!(xttraj[1,i,:], xttraj[2,i,:], color = "red", label = i==1 ? "Trajectory" : :none, alpha = 0.4)
end
X1true = sampleX1(n_inference_samples)
scatter!(X1true[1,:],X1true[2,:], msw = 0, ms = 1, color = "orange", alpha = 0.5, label = "X1 (true)")
scatter!(samp.state[1,:],samp.state[2,:], msw = 0, ms = 1, color = "green", alpha = 0.5, label = "X1 (generated)")
display(pl)
savefig("continuous_cat_$P.svg")