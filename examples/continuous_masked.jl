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
    embed_time = Chain(RandomFourierFeatures(1 => embeddim, 1f0), Dense(embeddim => embeddim, swish))
    embed_state = Chain(RandomFourierFeatures(2 => embeddim, 1f0), Dense(embeddim => embeddim, swish))
    ffs = [Dense(embeddim => embeddim, swish) for _ in 1:layers]
    decode = Dense(embeddim => spacedim)
    layers = (; embed_mask, embed_time, embed_state, ffs, decode)
    FModel(layers)
end

function (f::FModel)(t, Xt)
    l = f.layers
    tXt = tensor(Xt)
    tv = zero(tXt[1:1,:]) .+ expand(t, ndims(tXt))
    x = l.embed_time(tv) .+ l.embed_state(tXt) .+ l.embed_mask(Xt.cmask) #<- Mask handling
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
n_samples = 500

#The masking distribution:
X1mask(n_samples) = rand(2, n_samples) .< 0.75

#The process:
P = BrownianMotion(0.2f0)

#Optimizer:
eta = 0.01
opt_state = Flux.setup(AdamW(eta = eta, lambda = 0.001), model)

iters = 6000
for i in 1:iters
    #Set up a batch of training pairs, and t, where X1 is a MaskedState:
    X1m = X1mask(n_samples)
    X1 = MaskedState(ContinuousState(sampleX1(n_samples)), X1m, X1m)
    X0 = ContinuousState(sampleX0(n_samples))
    t = rand(T, n_samples)
    #Construct the bridge:
    Xt = bridge(P, X0, X1, t) #<- Only positions where mask=1 are noised because X1 is a MaskedState
    #Gradient:
    l,g = Flux.withgradient(model) do m
        floss(P, m(t,Xt), X1, scalefloss(P, t)) #<- Only positions where mask=1 are included in the loss
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
X0m = zeros(2, n_inference_samples) .< Inf #<- A mask with no conditioned (all 1s)
X0 = MaskedState(ContinuousState(sampleX0(n_inference_samples)), X0m, X0m)
paths = Tracker()
samp = gen(P, X0, model, 0f0:0.005f0:1f0, tracker = paths)

#Generate conditional samples, where the mask gets encoded into X0
n_masked_inference_samples = 500
X0m = rand(2, n_masked_inference_samples) .< 0.5
X0m[1,:] .= 1 .- X0m[2,:] #Making sure we don't have any double-masked points
conditionalX0 = MaskedState(ContinuousState(sampleX0(n_masked_inference_samples)), X0m, X0m)
tensor(conditionalX0)[.!(X0m)] .= (rand(2, n_masked_inference_samples) .* 0.1f0 .+ [1f0, -1f0])[.!(X0m)] #<- Condition on these specific values
conditional_samp = gen(P, conditionalX0, model, 0f0:0.005f0:1f0)

#Plotting:
pl = scatter(tensor(X0)[1,:],tensor(X0)[2,:], msw = 0, ms = 1, color = "blue", alpha = 0.5, size = (400,400), legend = :topleft, label = "X0")
X1true = sampleX1(n_inference_samples)
scatter!(X1true[1,:],X1true[2,:], msw = 0, ms = 1, color = "orange", alpha = 0.5, label = "X1 (true)")
scatter!(tensor(samp)[1,:],tensor(samp)[2,:], msw = 0, ms = 1, color = "green", alpha = 0.5, label = "X1 (generated)")
scatter!(tensor(conditional_samp)[1,:],tensor(conditional_samp)[2,:], msw = 0, ms = 1, color = "red", alpha = 0.5, label = "X1 (conditioned)")
display(pl)
savefig("conditioned_cat_$P.svg")