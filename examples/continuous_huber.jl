#=
NOTE: This example is a demonstration of what can go wrong when you make a seemingly innocuous change to the loss function.
If the loss is not compatible with the bridging process used to construct training pairs, the model will not learn to sample from the target distribution.
Often the samples will wind up somewhere on the "data manifold", so it can be worth experimenting with, but be aware that you may no longer be learning the target distribution.
=#
using Pkg
Pkg.activate(".")
using Revise
Pkg.develop(path="../../ForwardBackward/")
Pkg.develop(path="../")
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

T = Float32
#Distributions for training:
sampleX0(n_samples) = rand(T, 2, n_samples) .+ 2
#Note: 95% of the samples should be in the left blob
X1draw() = rand() < 0.95 ? randn(T, 2) .* 0.5f0 .+ [-5, -4] : randn(T, 2) .* 0.5f0 .+ [6, -10]
sampleX1(n_samples) = stack([X1draw() for _ in 1:n_samples])
n_samples = 400

P = BrownianMotion(0.15f0)

#Alternative loss functions:
huber(x) = 2((abs(x) < 1) ? x^2/2 : abs(x) - 1/2)
L2(x) = x^2 #This is the "right" loss for the Brownian motion process
plot(-4:0.01:4, huber, label = "Huber loss", size = (400,400))
plot!(-4:0.01:4, L2, label = "L2")

for (lossname, lossf) in [("Huber", huber), ("L2", L2)]
    model = FModel(embeddim = 256, layers = 4, spacedim = 2)
    eta = initeta = 0.003
    opt_state = Flux.setup(AdamW(eta = eta), model)

    iters = 10000
    for i in 1:iters
        #Set up a batch of training pairs, and t:
        X0 = ContinuousState(sampleX0(n_samples))
        X1 = ContinuousState(sampleX1(n_samples))
        t = rand(T, n_samples)
        #Construct the bridge:
        Xt = bridge(P, X0, X1, t)
        #Gradient & update:
        l,g = Flux.withgradient(model) do m
            #floss(P, m(t,Xt), X1, scalefloss(P, t)) #Flowfusion.jl default loss - basically L2.
            sum(sum(lossf.(m(t,Xt) .- tensor(X1)), dims = 1) .* expand(t .+ 0.05f0, ndims(tensor(X1)))) #Either Huber or L2, scaled with time.
        end
        Flux.update!(opt_state, model, g[1])
        eta = eta - initeta/iters
        Optimisers.adjust!(opt_state, eta)
        (i % 10 == 0) && println("i: $i; Loss: $l")
    end

    #Generate samples by stepping from X0
    n_inference_samples = 5000
    X0 = ContinuousState(sampleX0(n_inference_samples))
    samples = gen(P, X0, model, 0f0:0.005f0:1f0)

    #Generate samples by stepping from X0
    n_inference_samples = 5000
    X0 = ContinuousState(sampleX0(n_inference_samples))
    paths = Tracker()
    samples = gen(P, X0, model, 0f0:0.005f0:1f0, tracker = paths)

    #Plotting:
    pl = scatter(X0.state[1,:],X0.state[2,:], msw = 0, ms = 1, color = "blue", alpha = 0.5, size = (400,400), legend = :topleft, label = "X0")
    tvec = stack_tracker(paths, :t)
    xttraj = stack_tracker(paths, :xt)
    for i in 1:50:1000
        plot!(xttraj[1,i,:], xttraj[2,i,:], color = "red", label = i==1 ? "Trajectory" : :none, alpha = 0.4)
    end

    ratio = sum(samples.state[1,:] .< 0)/n_inference_samples

    X1true = sampleX1(n_inference_samples)
    scatter!(X1true[1,:],X1true[2,:], msw = 0, ms = 1, color = "orange", alpha = 0.5, label = "X1 (true)", title = "$(lossname): Ratio=$ratio")
    scatter!(samples.state[1,:],samples.state[2,:], msw = 0, ms = 1, color = "green", alpha = 0.25, label = "X1 (generated)")
    display(pl)
    savefig("blob_$(lossname).svg")
end

