using Pkg
Pkg.activate(".")
using Revise
Pkg.develop(path="../../ForwardBackward/")
Pkg.develop(path="../")
using ForwardBackward, Flowfusion, NNlib, Flux, RandomFeatureMaps, Optimisers, Plots

struct DModel{A}
    layers::A
end

Flux.@layer DModel

function DModel(; embeddim = 64, l = 2, K = 32, layers = 5)
    embed_time = Chain(RandomFourierFeatures(1 => embeddim, 2.0f0), Dense(embeddim => embeddim, leakyrelu))
    embed_char = Dense(K => embeddim, bias = false)
    mix = Dense(l*embeddim => embeddim, leakyrelu)
    ffs = [Dense(embeddim => embeddim, leakyrelu) for _ in 1:layers]
    decode = Dense(embeddim => l*K)
    layers = (; embed_time, embed_char, mix, ffs, decode)
    DModel(layers)
end

function (f::DModel)(t, Xt)
    l = f.layers
    tXt = tensor(Xt)
    len = size(tXt)[end]
    tv = zero(similar(Float32.(tXt), 1, len)) .+ expand(t, 2)
    x = l.mix(reshape(l.embed_char(tXt), :, len))  .+ l.embed_time(tv)
    for ff in l.ffs
        x = x .+ ff(x)
    end
    reshape(l.decode(x), :, 2, len)
end

T = Float32
n_samples = 1000

sampleX1(n_samples) = Flowfusion.random_discrete_cat(n_samples)
sampleX0(n_samples) = rand(25:32, 2, n_samples)
#sampleX0(n_samples) = [33 for _ in zeros(2, n_samples)] #Required if you want to use a UniformUnmasking process

P = NoisyInterpolatingDiscreteFlow(0.1)
#P = InterpolatingDiscreteFlow()
#P = UniformUnmasking()
    
model = DModel(embeddim = 128, l = 2, K = 33, layers = 2)

eta = 0.005
opt_state = Flux.setup(AdamW(eta = eta, lambda = 0.0001), model)

iters = 400
for i in 1:iters
    #Set up a batch of training pairs, and t
    X1 = onehot(DiscreteState(33, sampleX1(n_samples)))
    X0 = onehot(DiscreteState(33, sampleX0(n_samples)))
    t = rand(T, 1, n_samples)
    #Construct the bridge:
    Xt = dense(bridge(P, X0, X1, t)) #Zygote doesn't like the onehot input, so we make it dense.
    #Gradient
    l,g = Flux.withgradient(model) do m
        floss(P, m(t,Xt), X1, scalefloss(P,t,1)) #I prefer pow = 1 for discrete.
    end
    #Update
    Flux.update!(opt_state, model, g[1])
    if i % 10 == 0
        if i > iters - 1000
            eta *= 0.975
            Optimisers.adjust!(opt_state, eta)
        end
        println("i: $i; Loss: $l; eta: $eta")
    end
end


n_inference_samples = 10000
X0 = DiscreteState(33, sampleX0(n_inference_samples))
paths = Tracker()
@time samp = gen(P, X0, (t,Xt) -> softmax(model(t,onehot(Xt))), 0f0:0.01f0:1f0, tracker = paths) #<- Note the softmax, and onehot here

pl = scatter(X0.state[1,:],X0.state[2,:], msw = 0, color = "blue", alpha = 0.4, label = "Initial", size = (400,400), legend = :topleft, xlim = (1,33), ylim = (1,33))
scatter!(samp.state[1,:],samp.state[2,:], msw = 0, color = "green", alpha = 0.04, label = :none)
scatter!([-10],[-10], msw = 0, color = "green", alpha = 0.3, label = "Sampled")
tvec = stack_tracker(paths, :t)
xttraj = stack_tracker(paths, :xt)
for i in 1:200:n_inference_samples
    plot!(xttraj[1,i,:], xttraj[2,i,:], color = "red", label = :none, alpha = 0.15)
end
plot!([-10],[-10], color = "red", label = "Trajectory", alpha = 0.4)
pl
savefig("discrete_$P.svg")

#=
#Another way to do this is to make the X0 onehot, and then it'll stay onehot through the gen:
X0 = onehot(DiscreteState(33, sampleX0(n_inference_samples)))
paths = Tracker()
@time samp = gen(P, X0, (t,Xt) -> softmax(model(t,Xt)), 0f0:0.01f0:1f0, tracker = paths)
=#

