using Pkg
Pkg.activate(".")
using Revise
Pkg.develop(path="../../ForwardBackward/")
Pkg.develop(path="../")
using ForwardBackward, Flowfusion, NNlib, Flux, RandomFeatureMaps, Optimisers, Plots, Manifolds

#Set up a Flux model: ξhat = model(t,Xt)
struct TModel{A}
    layers::A
end
Flux.@layer TModel
function TModel(; embeddim = 64, spacedim = 2, layers = 5)
    embed_time = Chain(RandomFourierFeatures(1 => embeddim, 2f0), Dense(embeddim => embeddim, swish))
    embed_state = Chain(RandomFourierFeatures(4 => embeddim, 1f0), Dense(embeddim => embeddim, swish))
    ffs = [Dense(embeddim => embeddim, swish) for _ in 1:layers]
    decode = Dense(embeddim => spacedim)
    layers = (; embed_time, embed_state, ffs, decode)
    TModel(layers)
end

function (f::TModel)(t, Xt)
    l = f.layers
    tXt = tensor(Xt)
    enc = vcat(sin.(tXt), cos.(tXt))
    tv = zero(tXt[1:1,:]) .+ expand(t, ndims(tXt))
    x = l.embed_time(tv) .+ l.embed_state(enc)
    for ff in l.ffs
        x = x .+ ff(x)
    end
    return (l.decode(x) .* (1.05f0 .- tv))
end

model = TModel(embeddim = 256, layers = 3, spacedim = 2)

T = Float32
sampleX0(n_samples) = rand(T, 2, n_samples) .+ [2.1f0, 1]
sampleX1(n_samples) = Flowfusion.random_literal_cat(n_samples, sigma = T(0.05))[[2,1],:] .* 0.4f0 .- [-0.1f0, 1.3f0]
n_samples = 500

M = Torus(2)
P = ManifoldProcess(0.2f0)
#P = Deterministic()

eta = 0.01
opt_state = Flux.setup(AdamW(eta = eta, lambda = 0.00001), model)

iters = 8000
for i in 1:iters
    #Set up a batch of training pairs, and t
    X1 = ManifoldState(M, eachcol(sampleX1(n_samples))) #Note: eachcol
    X0 = ManifoldState(M, eachcol(sampleX0(n_samples)))
    t = rand(T, n_samples)
    #Construct the bridge:
    Xt = bridge(P, X0, X1, t)
    #Compute the tangent coordinates:
    ξ = Flowfusion.tangent_coordinates(Xt, X1)
    #Gradient
    l,g = Flux.withgradient(model) do m
        tcloss(P, m(t,tensor(Xt)), ξ, scalefloss(P, t))
    end
    #Update
    Flux.update!(opt_state, model, g[1])
    #Logging, and lr cooldown:
    if i % 10 == 0
        if i > iters - 3000
            eta *= 0.975
            Optimisers.adjust!(opt_state, eta)
        end
        println("i: $i; Loss: $l; eta: $eta")
    end
end

#Generate samples by stepping from X0:
n_inference_samples = 2000
X0 = ManifoldState(M, eachcol(sampleX0(n_inference_samples)))
paths = Tracker()
#We wrap the model, because it was predicting tangent coordinates, not the actual state:
X1pred = (t,Xt) -> BackwardGuide(model(t,tensor(Xt)))
samp = gen(P, X0, X1pred, 0f0:0.002f0:1f0, tracker = paths)

#Plot the torus, with samples, and trajectories:
#Project Torus(2) into 3D (just for plotting)
function tor(p; R::Real=2, r::Real=0.5)
    u,v = p[1], p[2]
    x = (R + r*cos(u)) * cos(v)
    y = (R + r*cos(u)) * sin(v)
    z = r * sin(u)
    return [x, y, z]
end

R = 2
r = 0.5
u = range(0, 2π; length=100)
v = range(0, 2π; length=100)
pl = plot([(R + r*cos(θ))*cos(φ) for θ in u, φ in v], [(R + r*cos(θ))*sin(φ) for θ in u, φ in v], [r*sin(θ) for θ in u, φ in v],
    color = "grey", alpha = 0.3, label = :none, camera = (30,30))
torX0 = stack(tor.(eachcol(tensor(X0))))
torSamp = stack(tor.(eachcol(tensor(samp))))
scatter!(torX0[1,:], torX0[2,:], torX0[3,:], label = "X0", msw = 0, ms = 1, color = "blue", alpha = 0.3)
torTarget = stack(tor.(eachcol(sampleX1(n_inference_samples))))
scatter!(torTarget[1,:], torTarget[2,:], torTarget[3,:], label = "X1 (true)", msw = 0, ms = 1, color = "orange", alpha = 0.2)
scatter!(torSamp[1,:], torSamp[2,:], torSamp[3,:], label = "X1 (generated)", msw = 0, ms = 1, color = "green", alpha = 0.3)
tvec = stack_tracker(paths, :t)
xttraj = stack_tracker(paths, :xt)
for i in 1:50:1000
    tr = stack(tor.(eachcol(xttraj[:,i,:])))
    plot!(tr[1,:], tr[2,:], tr[3,:], color = "red", alpha = 0.3, linewidth = 0.5, label = i == 1 ? "Trajectory" : :none)
end
display(pl)
savefig("torus_$P.svg")