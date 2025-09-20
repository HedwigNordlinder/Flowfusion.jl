using Flowfusion, ForwardBackward, Flux, RandomFeatureMaps, Optimisers, Plots

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

#Distributions for training:
T = Float32
sampleX0(n_samples) = rand(T, 2, n_samples) .+ 2
sampleX1(n_samples) = Flowfusion.random_literal_cat(n_samples, sigma = T(0.05))
n_samples = 400

#Look over three different process settings:
for (θ, v_at_0, v_at_1, dec) in [(10f0, 5f0, 0.01f0, -2f0),(2f0, 2f0, 0.1f0, -2f0),(10f0, 2f0, 0.1f0, -2f0)]
    #The process:
    P = OUFlow(θ, v_at_0, v_at_1, dec)

    #Optimizer:
    eta = 0.001
    model = FModel(embeddim = 256, layers = 3, spacedim = 2)
    opt_state = Flux.setup(AdamW(eta = eta), model)

    iters = 4000
    for i in 1:iters
        #Set up a batch of training pairs, and t:
        X0 = ContinuousState(sampleX0(n_samples))
        X1 = ContinuousState(sampleX1(n_samples))
        t = rand(T, n_samples).*0.999f0
        #Construct the bridge:
        Xt = bridge(P, X0, X1, t)
        #Gradient & update:
        l,g = Flux.withgradient(model) do m
            floss(P, m(t,Xt), X1, scalefloss(P, t))
        end
        Flux.update!(opt_state, model, g[1])
        (i % 10 == 0) && println("i: $i; Loss: $l")
    end

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
    X1true = sampleX1(n_inference_samples)
    scatter!(X1true[1,:],X1true[2,:], msw = 0, ms = 1, color = "orange", alpha = 0.5, label = "X1 (true)")
    scatter!(samples.state[1,:],samples.state[2,:], msw = 0, ms = 1, color = "green", alpha = 0.5, label = "X1 (generated)")
    display(pl)
    savefig("OU_continuous_cat_$P.svg")
    pl = plot()
    for i in 1:50:1000
        plot!(xttraj[1,i,:], color = "red", alpha = 0.4, label = :none)
    end
    pl
    savefig("OU_continuous_traj_$P.svg")
end
