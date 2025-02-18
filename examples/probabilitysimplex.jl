using Pkg
Pkg.activate(".")
using Revise
Pkg.develop(path="../../ForwardBackward/")
Pkg.develop(path="../")
using ForwardBackward, Flowfusion, NNlib, Flux, RandomFeatureMaps, Optimisers, Plots, Manifolds

#Set up a Flux model: ξhat = model(t,Xt)
struct PSModel{A}
    layers::A
end
Flux.@layer PSModel
function PSModel(; embeddim = 128, l = 2, K = 33, layers = 3)
    embed_time = Chain(RandomFourierFeatures(1 => embeddim, 2.0f0), Dense(embeddim => embeddim, swish))
    embed_char = Dense(K => embeddim, bias = false)
    mix = Dense(l*embeddim => embeddim, swish)
    ffs = [Dense(embeddim => embeddim, swish) for _ in 1:layers]
    decode = Dense(embeddim => l*(K-1)) #Tangent coord is one less than the number of categories
    layers = (; embed_time, embed_char, mix, ffs, decode)
    PSModel(layers)
end

function (f::PSModel)(t, Xt)
    l = f.layers
    tXt = tensor(Xt)
    len = size(tXt)[end]
    tv = zero(similar(Float32.(tXt), 1, len)) .+ expand(t, 2)
    x = l.mix(reshape(l.embed_char(tXt), :, len))  .+ l.embed_time(tv)
    for ff in l.ffs
        x = x .+ ff(x)
    end
    return reshape(l.decode(x), :, 2, len) .* (1.05f0 .- expand(t, 3))
end

model = PSModel(embeddim = 128, l = 2, K = 33, layers = 2)

sampleX1(n_samples) = Flowfusion.random_discrete_cat(n_samples)
sampleX0(n_samples) = rand(25:32, 2, n_samples)

T = Float32
n_samples = 200

M = ProbabilitySimplex(32)
P = ManifoldProcess(0.5f0)

eta = 0.01
opt_state = Flux.setup(AdamW(eta = eta, lambda = 0.0001), model)

iters = 5000
for i in 1:iters
    #Set up a batch of training pairs, and t
    X0 = ManifoldState(T, M, sampleX0(n_samples)) #Note T when constructing a ManifoldState from discrete values
    X1 = ManifoldState(T, M, sampleX1(n_samples))
    t = rand(T, n_samples)
    #Construct the bridge:
    Xt = bridge(P, X0, X1, t)
    #Get the Xt->X1 tangent coordinates:
    ξ = Guide(Xt, X1)
    #Gradient:
    l,g = Flux.withgradient(model) do m
        floss(P, m(t,tensor(Xt)), ξ, scalefloss(P, t))
    end
    #Update:
    Flux.update!(opt_state, model, g[1])
    #Log and adjust learning rate:
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
X0 = ManifoldState(T, M, sampleX0(n_inference_samples));
paths = Tracker()
X1pred = (t,Xt) -> Guide(model(t,tensor(Xt)))
samp = gen(P, X0, X1pred, 0f0:0.002f0:1f0, tracker = paths)

#Plot the X0 and generated X1:
X0oc = Flowfusion.onecold(tensor(X0))
sampoc = Flowfusion.onecold(tensor(samp))
pl = scatter(X0oc[1,:],X0oc[2,:], msw = 0, color = "blue", label = :none, alpha = 0.02, size = (400,400), xlim = (0,34), ylim = (0, 34), title = "X0, X1 (sampled)", titlefontsize = 9)
scatter!([-10], [-10], msw = 0, color = "blue", alpha = 0.3, label = "X0")
scatter!(sampoc[1,:],sampoc[2,:], msw = 0, color = "green", alpha = 0.02, label = :none)
scatter!([-10], [-10], msw = 0, color = "green", alpha = 0.3, label = "gen X1")

#...compared to the true X1:
trueX1 = sampleX1(n_inference_samples)
pl = scatter(X0oc[1,:],X0oc[2,:], msw = 0, color = "blue", label = :none, alpha = 0.02, size = (400,400), xlim = (0,34), ylim = (0, 34))
scatter!([-10], [-10], msw = 0, color = "blue", alpha = 0.3, label = "X0")
scatter!(trueX1[1,:],trueX1[2,:], msw = 0, color = "green", alpha = 0.02, label = :none)
scatter!([-10], [-10], msw = 0, color = "green", alpha = 0.3, label = "true X1")

#Plot a random individual trajectoty:
tvec = stack_tracker(paths, :t)
xttraj = stack_tracker(paths, :xt)
plot(tvec, xttraj[:,1,rand(1:n_samples),:]', legend = :none)

#Animate trajectories as the product of evolving marginals (needs the above code to have run):
x1s = [j for i in 1:33, j in 1:33]
x2s = [i for i in 1:33, j in 1:33]
i = rand(1:n_samples)
gridtraj_vec = [reshape(xttraj[:,1,s,:], 1, 33, :) .* reshape(xttraj[:,2,s,:], 33, 1, :) for s in 1:30]
anim = @animate for i ∈ vcat(zeros(Int, 30), ones(Int, 10), collect(1:500), ones(Int, 10).*500, ones(Int, 30).*501)
    if i == 0
        scatter(X0oc[1,:],X0oc[2,:], msw = 0, color = "blue", label = :none, alpha = 0.02, size = (400,400), xlim = (0,34), ylim = (0, 34), title = "X0, X1 (true)", titlefontsize = 9)
        scatter!([-10], [-10], msw = 0, color = "blue", alpha = 0.3, label = "X0")
        scatter!(trueX1[1,:],trueX1[2,:], msw = 0, color = "green", alpha = 0.02, label = :none)
        scatter!([-10], [-10], msw = 0, color = "green", alpha = 0.3, label = "true X1")
    end
    if 1 <= i <= 500
        plot(; colorbar = :none, legend = :none, size = (400,400), title = "$(length(gridtraj_vec)) probability simplex trajectories, t = $(round(tvec[i], digits = 2))", titlefontsize = 9, xlim = (0,34), ylim = (0, 34))
        for g in gridtraj_vec
            scatter!(x1s, x2s, msw = 0, ms = sqrt.(150 .* g[:,:,i]), colorbar = :none, legend = :none, size = (400,400), alpha = 0.4)
        end
    end  
    if i > 500
        scatter(X0oc[1,:],X0oc[2,:], msw = 0, color = "blue", label = :none, alpha = 0.02, size = (400,400), xlim = (0,34), ylim = (0, 34), title = "X0, X1 (sampled)", titlefontsize = 9)
        scatter!([-10], [-10], msw = 0, color = "blue", alpha = 0.3, label = "X0")
        scatter!(sampoc[1,:],sampoc[2,:], msw = 0, color = "green", alpha = 0.02, label = :none)
        scatter!([-10], [-10], msw = 0, color = "green", alpha = 0.3, label = "gen X1")
    end
end
gif(anim, "probsimplex_$(P).mp4", fps = 30)
