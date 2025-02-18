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
    embed_mask = Dense(2 => embeddim) #<- The model should usually know which positions are masked
    embed_mask_discrete = Dense(1 => embeddim)
    embed_time = Chain(RandomFourierFeatures(1 => embeddim, 1f0), Dense(embeddim => embeddim, swish))
    embed_state = Chain(RandomFourierFeatures(2 => embeddim, 1f0), Dense(embeddim => embeddim, swish))
    embed_discrete_state = Dense(4 => embeddim)
    ffs = [Dense(embeddim => embeddim, swish) for _ in 1:layers]
    decode = Dense(embeddim => spacedim)
    decode_discrete = Dense(embeddim => 4)
    layers = (; embed_mask, embed_mask_discrete, embed_time, embed_state, embed_discrete_state, ffs, decode, decode_discrete)
    FModel(layers)
end

function (f::FModel)(t, Xt)
    cXt, dXt = Xt
    l = f.layers
    tXt = tensor(cXt)
    tv = zero(tXt[1:1,:]) .+ expand(t, ndims(tXt))
    x = l.embed_time(tv) .+ l.embed_state(tXt) .+ l.embed_mask(cXt.cmask) .+ l.embed_mask_discrete(reshape(dXt.cmask, 1, :)) .+ l.embed_discrete_state(tensor(dXt)) #<- Mask handling
    for ff in l.ffs
        x = x .+ ff(x)
    end
    scal = (1.05f0 .- expand(t, ndims(tXt)))
    (tXt .+ l.decode(x) .* scal), (l.decode_discrete(x) .* scal)
end

model = FModel(embeddim = 384, layers = 3, spacedim = 2)

#Distributions for training:
T = Float32
function sampleX1(n_samples)
    cstate = Flowfusion.random_literal_cat(n_samples, sigma = T(0.05))
    dstate = ones(Int64, n_samples)
    dstate[:] .+= cstate[1,:] .> 0
    dstate[:] .+= cstate[2,:] .> 0
    return cstate, dstate
end
sampleX0(n_samples) = (rand(T, 2, n_samples) .+ 2), rand(1:4, n_samples)
n_samples = 500

#The masking distribution - we'll only mask the continuous part of the state
X1mask(n_samples) = rand(2, n_samples) .< 0.75

#The process:
P = (BrownianMotion(0.4f0), NoisyInterpolatingDiscreteFlow(0.1))

#Optimizer:
eta = 0.01
opt_state = Flux.setup(AdamW(eta = eta, lambda = 0.0001), model)

iters = 6000
for i in 1:iters
    #Set up a batch of training pairs, and t, where X1 is a MaskedState:
    X1cm = X1mask(n_samples)
    X1dm = rand(n_samples) .< 0.33
    x1 = sampleX1(n_samples)
    X1 = (MaskedState(ContinuousState(x1[1]), X1cm, X1cm), MaskedState(onehot(DiscreteState(4, x1[2])), X1dm, X1dm))
    x0 = sampleX0(n_samples)
    X0 = (ContinuousState(x0[1]), onehot(DiscreteState(4, x0[2])))
    t = rand(T, n_samples)
    #Construct the bridge:
    Xt = bridge(P, X0, X1, t) #<- Only positions where mask=1 are noised because X1 is a MaskedState
    #Gradient:
    l,g = Flux.withgradient(model) do m
        floss(P, m(t,Xt), X1, scalefloss(P, t))
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

#Generate unconditional samples:
n_inference_samples = 5000
pl = plot(;size = (400,400), legend = :topleft)
Xcm = rand(2, n_inference_samples) .> 0
Xdm = rand(n_inference_samples) .> 0
x0 = sampleX0(n_inference_samples)
X0 = (MaskedState(ContinuousState(x0[1]), Xcm, Xcm), MaskedState(onehot(DiscreteState(4, x0[2])), Xdm, Xdm))
paths = Tracker()
samp = gen(P, X0, model, 0f0:0.005f0:1f0, tracker = paths)
cstate = tensor(samp[1])
dstate = tensor(unhot(samp[2].S))
scatter!(cstate[1,:],cstate[2,:], markerz = dstate, cmap = :brg, msw = 0, ms = 1, alpha = 0.3, label = :none, colorbar = :none)
savefig("tuple_cat_unconditioned_$P.svg")

#Generate conditioned samples:
n_inference_samples = 2000
pl = plot(;size = (400,400), legend = :topleft)
for dval in [1, 2, 3]
    Xcm = rand(2, n_inference_samples) .> 0
    Xdm = rand(n_inference_samples) .< 0
    x0 = sampleX0(n_inference_samples)
    x0[2] .= dval
    X0 = (MaskedState(ContinuousState(x0[1]), Xcm, Xcm), MaskedState(onehot(DiscreteState(4, x0[2])), Xdm, Xdm))
    paths = Tracker()
    samp = gen(P, X0, model, 0f0:0.005f0:1f0, tracker = paths)
    cstate = tensor(samp[1])
    dstate = tensor(unhot(samp[2].S))
    scatter!(cstate[1,:],cstate[2,:], msw = 0, ms = 1, alpha = 0.3, label = "D = $dval")
end
pl
savefig("tuple_cat_conditioned_$P.svg")
