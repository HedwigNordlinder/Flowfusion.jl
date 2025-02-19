using Pkg
Pkg.activate(".")
using Revise
Pkg.develop(path="../../ForwardBackward/")
Pkg.develop(path="../")
using ForwardBackward, Flowfusion, NNlib, Flux, RandomFeatureMaps, Optimisers, Plots, Manifolds
using CSV, DataFrames

#Creating the logoinds.csv file:
#using FileIO, Images
#img = FileIO.load("fflogo.png")
#arr = [a.r for a in img] .< 0.5
#sampinds = (x -> (x[2], 265-x[1])).(CartesianIndices(arr)[arr])
#CSV.write("logoinds.csv", DataFrame(sampinds))

df = CSV.read("logoinds.csv", DataFrame) #https://github.com/user-attachments/files/18864465/logoinds.csv
sampinds = [Tuple(df[i,:]) for i in 1:size(df,1)]
flowinds = [s ./ 200 for s in sampinds if s[2] > 0]
fusioninds = [s ./ 200 for s in sampinds if s[2] <= 0]

#Set up a Flux model: X̂1 = model(t,Xt)
struct FModel{A}
    layers::A
end
Flux.@layer FModel
function FModel(; embeddim = 128, layers = 3)
    embed_time = Chain(RandomFourierFeatures(1 => embeddim, 1f0), Dense(embeddim => embeddim))
    embed_state = Chain(RandomFourierFeatures(2 => embeddim, 3f0), Dense(embeddim => embeddim))
    embed_angle = Chain(RandomFourierFeatures(2 => embeddim, 1f0), Dense(embeddim => embeddim))
    ffs = [Dense(embeddim => embeddim, swish) for _ in 1:layers]
    decode = Dense(embeddim => 2)
    decode_angle = Dense(embeddim => 1)
    layers = (;embed_time, embed_state, embed_angle, ffs, decode, decode_angle)
    FModel(layers)
end

function (f::FModel)(t, Xt)
    tXt, aXt = tensor.(Xt)
    l = f.layers
    aenc = vcat(sin.(aXt), cos.(aXt))
    tv = zero(tXt[1:1,:]) .+ expand(t, ndims(tXt))
    x = l.embed_time(tv) .+ l.embed_state(tXt) .+ l.embed_angle(aenc)
    for ff in l.ffs
        x = x .+ ff(x)
    end
    scal = (1.05f0 .- expand(t, ndims(tXt)))
    (tXt .+ l.decode(x) .* scal), (l.decode_angle(x) .* scal)
end

T = Float32
n_samples = 1000
M = Torus(1)
ManifoldState(M, Array{Float32}.(rand(M, n_samples)))
sampleX0(n_samples) = ContinuousState(T.(stack(rand(flowinds,   n_samples))) .+ rand(T, 2, n_samples) .* 0.01f0), ManifoldState(M, fill([0.6f0], n_samples))
sampleX1(n_samples) = ContinuousState(T.(stack(rand(fusioninds, n_samples))) .+ rand(T, 2, n_samples) .* 0.01f0), ManifoldState(M, fill([-2.54159f0], n_samples))

model = FModel(embeddim = 384, layers = 4)
n_samples = 500

#The process:
P = (FProcess(BrownianMotion(0.1f0), t -> 1-(1-t)^2), ManifoldProcess(0.1f0))
#P = (FProcess(Deterministic(), t -> 1-(1-t)^2), ManifoldProcess(0.1f0))

#Optimizer:
eta = 0.001
opt_state = Flux.setup(AdamW(eta = eta, lambda = 0.001), model)

iters = 8000
for i in 1:iters
    #Set up a batch of training pairs, and t, where X1 is a MaskedState: 
    X0 = sampleX0(n_samples)
    X1 = sampleX1(n_samples)
    t = rand(T, n_samples)
    #Construct the bridge:
    Xt = bridge(P, X0, X1, t)
    ξ = Guide(Xt[2], X1[2])
    #Gradient:
    l,g = Flux.withgradient(model) do m
        hat = m(t,Xt)
        floss(P[1], hat[1], X1[1], scalefloss(P[1], t)) + floss(P[2], hat[2], ξ, scalefloss(P[2], t))
    end
    #Update:
    Flux.update!(opt_state, model, g[1])
    #Logging, and lr cooldown:
    if i % 10 == 0
        if i > iters - 3000
            eta *= 0.98
            Optimisers.adjust!(opt_state, eta)
        end
        println("i: $i; Loss: $l; eta: $eta")
    end
end

function smodel(t, Xt)
    hat = model(t,Xt)
    return hat[1], Guide(hat[2])
end

#Generate unconditional samples:
n_inference_samples = 5000
X0 = sampleX0(n_inference_samples)
paths = Tracker()
samp = gen(P, X0, smodel, 0f0:0.005f0:1f0, tracker = paths)

postraj = stack(vcat([tensor(X0[1])], [tensor(p[1]) for p in paths.xt], [tensor(samp[1])]))
angtraj = stack(vcat([tensor(X0[2])], [tensor(p[2]) for p in paths.xt], [tensor(samp[2])]))

anim = @animate for i ∈ vcat([1 for i in 1:20], 1:size(postraj, 3), [size(postraj, 3) for i in 1:20])
    scatter(postraj[1,:,i], postraj[2,:,i], msw = 0, ms = 1, markerz = angtraj[1,:,i], cmap = :hsv, label = :none, xlim = (-0.0, 5.2), ylim = (-1.3, 1.3), size = (400, 200))
    scatter!([-100,-100],[-100,-100], markerz = [-pi,pi], label = :none, colorbar = :none, axis=([], false))
end
gif(anim, "logo_$(P).mp4", fps = 30)
gif(anim, "logo_$(P).gif", fps = 30)

#=
#To create a side-by-side animation (useful for comparing schedules, noise, etc), run the above once, then store the trajectories, then run it again, and then merge, with an offset:
#diffpostraj = copy(postraj)
#diffangtraj = copy(angtraj)

flowpostraj = copy(postraj)
flowangtraj = copy(angtraj)

mergedpostraj = hcat(diffpostraj .+ reshape([6, 0], 2, 1, 1), flowpostraj)
mergedangtraj = hcat(diffangtraj, flowangtraj)

anim = @animate for i ∈ vcat([1 for i in 1:20], 1:size(mergedpostraj, 3), [size(mergedpostraj, 3) for i in 1:20])
    scatter(mergedpostraj[1,:,i], mergedpostraj[2,:,i], msw = 0, ms = 1, markerz = mergedangtraj[1,:,i], cmap = :hsv, label = :none, xlim = (-0.0, 11.2), ylim = (-1.3, 1.3), size = (800, 200))
    scatter!([-100,-100],[-100,-100], markerz = [-pi,pi], label = :none, colorbar = :none, axis=([], false))
end
gif(anim, "mergedlogo_$(P).mp4", fps = 30)
gif(anim, "mergedlogo_$(P).gif", fps = 30)
=#