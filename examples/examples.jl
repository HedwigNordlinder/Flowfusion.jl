using Pkg
Pkg.activate(".")
using Revise
Pkg.develop(path="../../ForwardBackward/")
Pkg.develop(path="../")
using ForwardBackward, Flowfusion, NNlib, Flux, RandomFeatureMaps, Optimisers



using Plots
function circle(n, σ; r = 10)
    T = eltype(σ)
    r = T(r)
    θ = 2 * T(π) .* rand(T, n)
    x = @. cos(θ) * r + σ * $(randn(T, n))
    y = @. sin(θ) * r + σ * $(randn(T, n))
    x, y
end

function spiral(n, σ)
    T = eltype(σ)
    u = 4 * T(π) .* rand(T, n)
    x = @. u * cos(u) + σ * $(randn(T, n))
    y = @. u * sin(u) + σ * $(randn(T, n))
    x, y 
end

function crescent(n, σ; r = 10, t = oftype(σ, π) / 3)
    T = eltype(σ)
    θ = T(π) / 3 .* randn(T, n)
    s = @. (cos(mod2pi(θ)) + 1) / 2
    x = @. cos(θ - t) * r + s * σ * $(randn(T, n))
    y = @. sin(θ - t) * r + s * σ * $(randn(T, n))
    x, y
end

function discretize(x, d, lo, hi)
    for (i, v) in enumerate(range(lo, hi, length = d - 1))
        x < v && return i
    end
    d
end

d = 32  # discretization level
σ = 1.0f0  # noise level
n_samples = 100
x, y = crescent(n_samples, σ)
lo, hi = -12, 12


struct FModel{A}
    layers::A
end

Flux.@layer FModel

function FModel(; embeddim = 64, spacedim = 2, layers = 5)
    embed_time = RandomFourierFeatures(1 => embeddim, 2.0f0)
    embed_state = RandomFourierFeatures(2 => embeddim, 0.5f0)
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
    l.decode(x)
end


T = Float32
n_samples = 1000

#P = Deterministic()
#P = FProcess(BrownianMotion(20), t -> sin(t*pi/2))
#P = OrnsteinUhlenbeck(0, 1, 10) #Trying to break it.
P = BrownianMotion(3)

model = FModel(embeddim = 256, layers = 2, spacedim = 2)

eta = 0.005
opt_state = Flux.setup(Adam(eta), model)

iters = 5000
for i in 1:iters
    #Set up a batch of training pairs, and t
    X1 = ContinuousState(Array(hcat(crescent(n_samples, σ)...)'))
    X0 = ContinuousState(rand(T, 2, n_samples) .* 10)
    t = rand(T, n_samples)
    #Construct the bridge:
    Xt = bridge(P, X0, X1, t)
    #Gradient
    l,g = Flux.withgradient(model) do m
        floss(P, m(t,Xt), X1, scalefloss(P, t))
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
X0 = ContinuousState(rand(T, 2, n_inference_samples) .* 10)
paths = Tracker()
samp = gen(P, X0, model, 0f0:0.002f0:1f0, tracker = paths)

pl = scatter(X0.state[1,:],X0.state[2,:], msw = 0, color = "blue", alpha = 0.05, size = (400,400), legend = :topleft)
scatter!(samp.state[1,:],samp.state[2,:], msw = 0, color = "green", alpha = 0.05)
tvec = stack_tracker(paths, :t)
xttraj = stack_tracker(paths, :xt)
for i in 1:20:1000
    plot!(xttraj[1,i,:], xttraj[2,i,:], color = "red", label = :none)
end
pl





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

discrete_target(n_samples) = Array(stack([discretize.(a, d, lo, hi) for a in crescent(n_samples, σ)])')

#This example shows an interesting effect where many of the points don't migrate off their starting values.
#Does the model just struggle to learn the actual conditionals? Need to set up a closed-form solution to test this.
T = Float32
n_samples = 1000
P = UniformDiscrete(1)
model = DModel(embeddim = 386, l = 2, K = 32, layers = 3)
eta = 0.005
opt_state = Flux.setup(Adam(eta), model)

iters = 6000
for i in 1:iters
    #Set up a batch of training pairs, and t
    X1 = DiscreteState(32, discrete_target(n_samples))
    X0 = DiscreteState(32, [rand(1:10) for _ in zeros(2, n_samples)])
    t = rand(T, n_samples)
    #Construct the bridge:
    Xt = stochastic(Float32, bridge(P, X0, X1, t)) #Note the stochastic here - because Zygote doesn't like the onehot.
    #Gradient
    l,g = Flux.withgradient(model) do m
        floss(P, m(t,Xt), onehot(X1), scalefloss(P, t))
        #Flowfusion.mean(Flowfusion.lce(m(t,Xt), onehot(X1)))
    end
    #Update
    Flux.update!(opt_state, model, g[1])
    if i % 10 == 0
        if i > iters - 3000
            eta *= 0.975
            Optimisers.adjust!(opt_state, eta)
        end
        println("i: $i; Loss: $l; eta: $eta")
    end
end

n_inference_samples = 2000
X0 = DiscreteState(32, [rand(1:10) for _ in zeros(2, n_inference_samples)])
paths = Tracker()
samp = gen(P, X0, (t,Xt) -> softmax(model(t,onehot(Xt))), 0f0:0.0005f0:1f0, tracker = paths, midpoint = true) #Note the softmax here

pl = scatter(X0.state[1,:],X0.state[2,:], msw = 0, color = "blue", alpha = 0.4)
scatter!(samp.state[1,:],samp.state[2,:], msw = 0, color = "green", alpha = 0.4)
#scatter(samp.state[1,:],samp.state[2,:], msw = 0, color = "green", alpha = 0.4)
tvec = stack_tracker(paths, :t)
xttraj = stack_tracker(paths, :xt)
for i in 1:20:n_samples
    plot!(xttraj[1,i,:], xttraj[2,i,:], color = "red", label = :none)
end
pl




#P = UniformUnmasking()
#P = PiQ(2.0f0, vcat(ones(Float32,32)./32, [0.0f0]))
#P = UniformDiscrete(0.0001f0)
P = UniformUnmasking(0.00001f0)
P = PiQ(1f0, vcat(ones(Float32,32)./32, [0.0f0]))
X0 = DiscreteState(33, [33])
X1 = DiscreteState(33, [1])
f(t) = (forward(X0, P, t) ⊙ backward(X1, P, 1-t)).dist[[1,2,33]]
plot(0:0.01:1.0,x -> f(x)[1], label = "P(Terminal)", xlabel = "t", legend = :top)
plot!(0:0.01:1.0,x -> f(x)[2], label = "P(One Other)")
plot!(0:0.01:1.0,x -> f(x)[3], label = "P(Initial)")





T = Float32
n_samples = 1000

P = UniformUnmasking(1)
#P = PiQ(0.1f0, vcat(ones(Float32,32)./32, [0.0f0]))
#P = UniformDiscrete(0.5f0)
#P = UniformDiscrete(0.001f0)
#P = UniformUnmasking(0.001f0)

model = DModel(embeddim = 256, l = 2, K = 33, layers = 2)

eta = 0.005
opt_state = Flux.setup(Adam(eta), model)

iters = 10000
for i in 1:iters
    #Set up a batch of training pairs, and t
    X1 = DiscreteState(33, discrete_target(n_samples))
    X0 = DiscreteState(33, [33 for _ in zeros(2, n_samples)])
    t = rand(T, 1, n_samples)
    #Construct the bridge:
    Xt = stochastic(Float32, bridge(P, X0, X1, t))
    #Gradient
    l,g = Flux.withgradient(model) do m
        floss(P, m(t,Xt), onehot(X1), scalefloss(P, t))
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

#Useful if you want to structurally prevent the model from predicting the masked state during inference.
function masked_softmax(x)
    h = copy(x)
    selectdim(h,1,size(h,1)) .= -Inf
    return softmax(h)
end


n_inference_samples = 10000
X0 = DiscreteState(33, [33 for _ in zeros(2, n_inference_samples)])
paths = Tracker()
samp = gen(P, X0, (t,Xt) -> masked_softmax(model(t,onehot(Xt))), 0f0:0.002f0:1f0, tracker = paths) #Note the softmax here

pl = scatter(X0.state[1,:],X0.state[2,:], msw = 0, color = "blue", alpha = 0.4, label = "Initial", size = (400,400), legend = :topleft)
scatter!(samp.state[1,:],samp.state[2,:], msw = 0, color = "green", alpha = 0.04, label = "Sampled")
tvec = stack_tracker(paths, :t)
xttraj = stack_tracker(paths, :xt)
for i in 1:20:n_samples
    plot!(xttraj[1,i,:], xttraj[2,i,:], color = "red", label = :none)
end
pl



n_inference_samples = 10000
X0 = DiscreteState(33, [33 for _ in zeros(2, n_inference_samples)])
samp = discrete_target(n_inference_samples)
pl = scatter(X0.state[1,:],X0.state[2,:], msw = 0, color = "blue", alpha = 0.4, label = "Initial", size = (400,400), legend = :topleft)
scatter!(samp[1,:],samp[2,:], msw = 0, color = "green", alpha = 0.04, label = "Target")




#Investigating weirdness - with a very very low rate, sometimes we see no switches. Maybe some underflow issue?
X0 = DiscreteState(33, [33 for _ in zeros(2, n_samples)])
t = 0.9f0
fb = forward(X0, P, t)  ⊙ backward(CategoricalLikelihood(masked_softmax(model(t,onehot(X0)))), P, 1-t)
fb.dist



using LinearAlgebra
#SymTridiagonal Q matrix version, for neighbour walks:
diago = -vcat([1], ones(Int, 31) .* 2, [1])
offdiago = ones(Int, 32)
Q = SymTridiagonal(Tridiagonal(offdiago, diago, offdiago))
P = GeneralDiscrete(Q .* 100.0)

T = Float32
n_samples = 1000

model = DModel(embeddim = 384, l = 2, K = 33, layers = 3)

eta = 0.005
opt_state = Flux.setup(Adam(eta), model)

iters = 10000
for i in 1:iters
    #Set up a batch of training pairs, and t
    X1 = DiscreteState(33, discrete_target(n_samples))
    X0 = DiscreteState(33, [33 for _ in zeros(2, n_samples)])
    t = rand(T)
    #Construct the bridge:
    Xt = stochastic(Float32, bridge(P, X0, X1, t))
    #Gradient
    l,g = Flux.withgradient(model) do m
        floss(P, m(t,Xt), onehot(X1), scalefloss(P, t))
    end
    #Update
    Flux.update!(opt_state, model, g[1])
    if i % 10 == 0
        if i > iters - 2000
            eta *= 0.975
            Optimisers.adjust!(opt_state, eta)
        end
        println("i: $i; Loss: $l; eta: $eta")
    end
end

#Useful if you want to structurally prevent the model from predicting the masked state during inference.
function masked_softmax(x)
    h = copy(x)
    selectdim(h,1,size(h,1)) .= -Inf
    return softmax(h)
end

n_inference_samples = 10000
X0 = DiscreteState(33, [33 for _ in zeros(2, n_inference_samples)])
paths = Tracker()
samp = gen(P, X0, (t,Xt) -> masked_softmax(model(t,onehot(Xt))), 0f0:0.0005f0:1f0, tracker = paths) #Note the softmax here

pl = scatter(X0.state[1,:],X0.state[2,:], msw = 0, color = "blue", alpha = 0.4, label = "Initial", size = (400,400), legend = :topleft)
scatter!(samp.state[1,:],samp.state[2,:], msw = 0, color = "green", alpha = 0.04, label = "Sampled")
tvec = stack_tracker(paths, :t)
xttraj = stack_tracker(paths, :xt)
for i in 1:5
    plot!(xttraj[1,i,:], xttraj[2,i,:], color = "red", label = :none, alpha = 0.5)
end
pl




#Project Torus(2) into 3D (just for plotting)
function tor(p; R::Real=2, r::Real=0.5)
    u,v = p[1], p[2]
    x = (R + r*cos(u)) * cos(v)
    y = (R + r*cos(u)) * sin(v)
    z = r * sin(u)
    return [x, y, z]
end

torus_target(n_samples) = Array(hcat(crescent(n_samples, 0.1f0, r = 2f0)...)')
#sam = torus_target(1000)
#scatter(sam[1,:], sam[2,:], msw = 0, alpha = 0.4)

struct TModel{A}
    layers::A
end

Flux.@layer TModel

function TModel(; embeddim = 64, spacedim = 4, layers = 5)
    embed_time = RandomFourierFeatures(1 => embeddim, 2f0)
    embed_state = RandomFourierFeatures(4 => embeddim, 1f0)
    ffs = [Dense(embeddim => embeddim, swish) for _ in 1:layers]
    decode = Dense(embeddim => spacedim)
    layers = (; embed_time, embed_state, ffs, decode)
    TModel(layers)
end

function (f::TModel)(t, Xt)
    l = f.layers
    tXt = tensor(Xt)
    enc = vcat(sin.(tXt), cos.(tXt))
    #@show size(enc)
    tv = zero(tXt[1:1,:]) .+ expand(t, ndims(tXt))
    x = l.embed_time(tv) .+ l.embed_state(enc)
    for ff in l.ffs
        x = x .+ ff(x)
    end
    return mod.((l.decode(x) .* 0.05f0 .*(1.1f0 .- tv)) .+ tXt .+ π, 2π) .- π
end


T = Float32
n_samples = 1000
P = ManifoldProcess(0.2f0)
#P = Deterministic()
M = Torus(2)

model = TModel(embeddim = 384, layers = 3, spacedim = 2)

#model(t,tensor(Xt))

#=
X1 = ManifoldState(M, eachcol(torus_target(n_samples)))
X0 = ManifoldState(M, Vector{Float32}.(rand(M, n_samples)))
t = rand(T, n_samples)
Xt = bridge(P, X0, X1, t)
floss(P, model(t,tensor(Xt)), X1, scalefloss(P, t))
=#

eta = 0.005
opt_state = Flux.setup(AdamW(eta = eta, lambda = 0.00001), model)

#Issue: how do I control the eltype of rand(M, 100)? Defaults to Float64.

iters = 10000
for i in 1:iters
    #Set up a batch of training pairs, and t
    X1 = ManifoldState(M, eachcol(torus_target(n_samples)))
    X0 = ManifoldState(M, eachcol(1 .+ 0.5f0 .* randn(Float32, 2, n_samples))) #Gross type control. We should help with this.
    #X0 = ManifoldState(M, Vector{Float32}.(rand(M, n_samples))) #Gross type control. We should help with this.
    t = rand(T, n_samples)
    #Construct the bridge:
    Xt = bridge(P, X0, X1, t)
    #Gradient
    l,g = Flux.withgradient(model) do m
        floss(P, m(t,tensor(Xt)), X1, scalefloss(P, t, 0.5f0))
    end
    #Update
    Flux.update!(opt_state, model, g[1])
    if i % 10 == 0
        if i > iters - 2000
            eta *= 0.975
            Optimisers.adjust!(opt_state, eta)
        end
        println("i: $i; Loss: $l; eta: $eta")
    end
end

n_inference_samples = 2000
X0 = ManifoldState(M, eachcol(1 .+ 0.5f0 .* randn(Float32, 2, n_inference_samples)))
#X0 = ManifoldState(M, Vector{Float32}.(rand(M, n_inference_samples)))
paths = Tracker()
samp = gen(P, X0, model, 0f0:0.002f0:1f0, tracker = paths)

pl = scatter(tensor(X0)[1,:],tensor(X0)[2,:], msw = 0, color = "blue", alpha = 0.05, size = (400,400), legend = :topleft)
scatter!(tensor(samp)[1,:],tensor(samp)[2,:], msw = 0, color = "green", alpha = 0.05)

targ = torus_target(n_samples)
scatter!(targ[1,:], targ[2,:], msw = 0, color = "red", alpha = 0.05)

tvec = stack_tracker(paths, :t)
xttraj = stack_tracker(paths, :xt)
for i in 1:20:1000
    plot!(xttraj[1,i,:], xttraj[2,i,:], color = "red", label = :none)
end
pl




X1 = ManifoldState(M, eachcol(torus_target(n_samples)))
X0 = ManifoldState(M, Vector{Float32}.(rand(M, n_samples))) #Gross type control. We should help with this.
t = ones(T, n_samples) .- 0.01f0
Xt = bridge(P, X0, X1, t)

scatter(tensor(Xt)[1,:], tensor(Xt)[2,:], msw = 0, color = "blue", alpha = 0.05)


X1 = ManifoldState(M, eachcol(torus_target(n_samples)))
X0 = ManifoldState(M, Vector{Float32}.(rand(M, n_samples))) #Gross type control. We should help with this.
t = rand(T, n_samples)
Xt = bridge(P, X0, X1, t)
floss(P, model(t,tensor(Xt)), X1, scalefloss(P, t))

X0 = ManifoldState(M, Vector{Float32}.(rand(M, n_samples)))
t = 0.5f0
X1 = copy(X0)
floss(P, X0, X1, scalefloss(P, t))



Flowfusion.msta(X1, X0)











#When non-zero, the process will diffuse. When 0, the process is deterministic:
for P in [ManifoldProcess(0), ManifoldProcess(0.05)]
    #When non-zero, the endpoints will be slightly noised:
    for perturb_var in [0.0, 0.0001] 
     
        #We'll generate endpoint-conditioned samples evenly spaced over time:
        t_vec = 0:0.001:1

        #Set up the X0 and X1 states, just repeating the endpoints over and over:
        X0 = ManifoldState(M, [perturb(M, p0, perturb_var) for _ in t_vec])
        X1 = ManifoldState(M, [perturb(M, p1, perturb_var) for _ in t_vec])

        #Independently draw endpoint-conditioned samples at times t_vec:
        Xt = endpoint_conditioned_sample(X0, X1, P, t_vec)

        #Plot the torus:
        R = 2
        r = 0.5
        u = range(0, 2π; length=100)  # angle around the tube
        v = range(0, 2π; length=100)  # angle around the torus center
        pl = plot([(R + r*cos(θ))*cos(φ) for θ in u, φ in v], [(R + r*cos(θ))*sin(φ) for θ in u, φ in v], [r*sin(θ) for θ in u, φ in v],
            color = "grey", alpha = 0.3, label = :none, camera = (30,30))

        #Map the points to 3D and plot them:
        endpts = stack(tor.([p0,p1]))
        smppts = stack(tor.(eachcol(tensor(Xt))))
        scatter!(smppts[1,:], smppts[2,:], smppts[3,:], label = :none, msw = 0, ms = 1.5, color = "blue", alpha = 0.5)
        scatter!(endpts[1,:], endpts[2,:], endpts[3,:], label = :none, msw = 0, ms = 2.5, color = "red")
        savefig("torus_$(perturb_var)_$(P.v).svg")
    end
end









t = rand(T, n_samples)
Xt = bridge(P, X0, X1, t)


floss(P, X0, X1, scalefloss(P, t))






#=
function (aln::AdaLN)(x::AbstractArray, cond::AbstractArray)
    normalized = aln.norm(x)
    shift = aln.shift(cond)
    scale = aln.scale(cond)
    # Change to original transformation
    return normalized .* (1 .+ reshape(scale, :, 1, size(shift)[end])) .+ reshape(shift, :, 1, size(shift)[end])
end
=#



target_conc = 5
current_conc = 0.1
target_conc/current_conc = 2^cycles
cycles = log2(5/conc)


upperconc = 10
lowerconc = 0.1




M = Rotations(3)
X0 = rand(M, 100)
X1 = rand(M, 100)

 .- inverse_retract.((M,), X0, X1)

p = X0[1]
q = X1[1]

X = log(M, p, q)
C = get_coordinates(M, p, X)


M = Torus(2)
Xt = ManifoldState(M, [randn(2) for _ in zeros(2,3)])
X1 = ManifoldState(M, [randn(2) for _ in zeros(2,3)])



c = tangent_coordinates(Xt, X1)

X̂₁ = apply_tangent_coordinates(Xt, c)

X̂₁.state .- X1.state


grd = inverse_retract.((M,), X0, X1) #A faster (sometimes approx, sometimes defined) version of logarithmic map.

v = vee(M, X0[1], grd[1])
hat(M, X0[1], v)


basis = get_basis(M, X0[1], DefaultOrthonormalBasis())
vecX = get_coordinates(M, X0[1], grd[1], basis)

get_vector(M, X0[1], vecX)

function 


















    
#torus_target(n_samples) = Array(hcat(crescent(n_samples, 0.1f0, r = 2f0)...)')
torus_target(n_samples) = 0.08f0 .*Array(hcat(spiral(n_samples, 0.2f0)...)') .- [0f0, 1.5f0]
#sam = torus_target(1000)
#scatter(sam[1,:], sam[2,:], msw = 0, alpha = 0.05)

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
    #return l.decode(x)
    return (l.decode(x) .* (1.05f0 .- tv))
end


T = Float32
n_samples = 1000
P = ManifoldProcess(0.3f0)
#P = Deterministic()
M = Torus(2)

model = TModel(embeddim = 256, layers = 3, spacedim = 2)

eta = 0.005
opt_state = Flux.setup(AdamW(eta = eta, lambda = 0.00001), model)

iters = 5000
for i in 1:iters
    #Set up a batch of training pairs, and t
    X1 = ManifoldState(M, eachcol(torus_target(n_samples)))
    X0 = ManifoldState(M, eachcol(1 .+ 0.5f0 .* randn(Float32, 2, n_samples))) #Gross type control. We should help with this.
    #X0 = ManifoldState(M, Vector{Float32}.(rand(M, n_samples))) #Gross type control. We should help with this.
    t = rand(T, n_samples)
    #Construct the bridge:
    Xt = bridge(P, X0, X1, t)
    #Compute the tangent coordinates:
    ξ = Flowfusion.tangent_coordinates(Xt, X1)
    #Gradient
    l,g = Flux.withgradient(model) do m
        tcloss(P, m(t,tensor(Xt)), ξ, scalefloss(P, t, 1))
    end
    #Update
    Flux.update!(opt_state, model, g[1])
    if i % 10 == 0
        if i > iters - 2000
            eta *= 0.975
            Optimisers.adjust!(opt_state, eta)
        end
        println("i: $i; Loss: $l; eta: $eta")
    end
end

n_inference_samples = 2000
X0 = ManifoldState(M, eachcol(1 .+ 0.5f0 .* randn(Float32, 2, n_inference_samples)))
paths = Tracker()
#We wrap the model, because it was predicting tangent coordinates, not the actual state:
X1pred = (t,Xt) -> apply_tangent_coordinates(Xt, model(t,tensor(Xt)))
samp = gen(P, X0, X1pred, 0f0:0.002f0:1f0, tracker = paths)

pl = scatter(tensor(X0)[1,:],tensor(X0)[2,:], msw = 0, color = "blue", alpha = 0.05, size = (400,400), legend = :topleft)
scatter!(tensor(samp)[1,:],tensor(samp)[2,:], msw = 0, color = "green", alpha = 0.05)

#targ = torus_target(n_samples)
#scatter!(targ[1,:], targ[2,:], msw = 0, color = "red", alpha = 0.05)

tvec = stack_tracker(paths, :t)
xttraj = stack_tracker(paths, :xt)
for i in 1:20:1000
    plot!(xttraj[1,i,:], xttraj[2,i,:], color = "red", label = :none, alpha = 0.1)
end
pl


#Project Torus(2) into 3D (just for plotting)
function tor(p; R::Real=2, r::Real=0.5)
    u,v = p[1], p[2]
    x = (R + r*cos(u)) * cos(v)
    y = (R + r*cos(u)) * sin(v)
    z = r * sin(u)
    return [x, y, z]
end


#Plot the torus, with samples, and trajectories
R = 2
r = 0.5
u = range(0, 2π; length=100)  # angle around the tube
v = range(0, 2π; length=100)  # angle around the torus center
pl = plot([(R + r*cos(θ))*cos(φ) for θ in u, φ in v], [(R + r*cos(θ))*sin(φ) for θ in u, φ in v], [r*sin(θ) for θ in u, φ in v],
    color = "grey", alpha = 0.3, label = :none, camera = (30,30))
torX0 = stack(tor.(eachcol(tensor(X0))))
torSamp = stack(tor.(eachcol(tensor(samp))))
scatter!(torX0[1,:], torX0[2,:], torX0[3,:], label = :none, msw = 0, ms = 1.5, color = "blue", alpha = 0.1)
scatter!(torSamp[1,:], torSamp[2,:], torSamp[3,:], label = :none, msw = 0, ms = 1.5, color = "green", alpha = 0.2)
tvec = stack_tracker(paths, :t)
xttraj = stack_tracker(paths, :xt)
for i in 1:50:1000
    tr = stack(tor.(eachcol(xttraj[:,i,:])))
    plot!(tr[1,:], tr[2,:], tr[3,:], color = "red", label = :none, alpha = 0.3)
end
pl


#Overlay samples and true target dist:
pl = plot([(R + r*cos(θ))*cos(φ) for θ in u, φ in v], [(R + r*cos(θ))*sin(φ) for θ in u, φ in v], [r*sin(θ) for θ in u, φ in v],
    color = "grey", alpha = 0.3, label = :none, camera = (30,30))
torX0 = stack(tor.(eachcol(tensor(X0))))
torTarget = stack(tor.(eachcol(torus_target(n_inference_samples))))
torSamp = stack(tor.(eachcol(tensor(samp))))
scatter!(torX0[1,:], torX0[2,:], torX0[3,:], label = :none, msw = 0, ms = 1.5, color = "blue", alpha = 0.3)
scatter!(torTarget[1,:], torTarget[2,:], torTarget[3,:], label = :none, msw = 0, ms = 1.5, color = "green", alpha = 0.2)
scatter!(torSamp[1,:], torSamp[2,:], torSamp[3,:], label = :none, msw = 0, ms = 1.5, color = "red", alpha = 0.2)




M = Rotations(3)

Xt = ManifoldState(M, rand(M, 2))
X1 = ManifoldState(M, rand(M, 2))

caas = Flowfusion.tangent_coordinates(Xt, X1)

p = Xt.state[1]
get_vector(M, p, [1.0, 0.0, 0.0])
get_vector(M, p, [0.0, 1.0, 0.0])
get_vector(M, p, [0.0, 0.0, 1.0])

aas = angleaxis_stack(batched_mul(batched_transpose(tensor(Xt.state)),tensor(X1.state)))
(1 ./(2 .* sin.(aas[1]))) .* aas[2]

tra = (Xt.state[1]' * X1.state[1])
tra[3,2] - tra[2,3]

axis_x = reshape(coeff .* (R[3,2,:] .- R[2,3,:]), 1, :)
axis_y = reshape(coeff .* (R[1,3,:] .- R[3,1,:]), 1, :)
axis_z = reshape(coeff .* (R[2,1,:] .- R[1,2,:]), 1, :)


so3_tangent(Xt.state[1], X1.state[1])

function so3_tangent(Rt::AbstractMatrix, R1::AbstractMatrix)
    # Relative rotation
    Delta = Rt' * R1
    
    # Clamp trace to the valid domain of acos to handle round-off
    tr = (Delta[1,1] + Delta[2,2] + Delta[3,3] - 1) / 2
    tr_clamped = min(max(tr, -1.0), 1.0)
    
    # Angle of rotation
    theta = acos(tr_clamped)
    
    # If the angle is near zero, we return the zero vector (they are almost the same rotation)
    if abs(theta) < 1e-12
        return zeros(3)
    end
    
    # Compute rotation axis using the skew-symmetric part
    # [Delta[3,2] - Delta[2,3], Delta[1,3] - Delta[3,1], Delta[2,1] - Delta[1,2]]
    denom = 2 * sin(theta)
    axis = [Delta[3,2] - Delta[2,3],
            Delta[1,3] - Delta[3,1],
            Delta[2,1] - Delta[1,2]] / denom
    
    # Tangent vector = angle * axis
    return theta * axis
end






M = ProbabilitySimplex(3)

rand(M, 2)

Xt = ManifoldState(M, rand(M, 2))
X1 = ManifoldState(M, rand(M, 2))

caas = Flowfusion.tangent_coordinates(Xt, X1)








struct PSModel{A}
    layers::A
end

Flux.@layer PSModel

function PSModel(; embeddim = 64, l = 2, K = 32, layers = 2)
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




discrete_X1(n_samples) = Array(stack([discretize.(a, d, lo, hi) for a in crescent(n_samples, σ)])')
discrete_X0(n_samples) = rand(10:15, 2, n_samples)

M = ProbabilitySimplex(32)
T = Float32
n_samples = 1000

P = ManifoldProcess(0.005f0)
#P = Deterministic()

model = PSModel(embeddim = 256, l = 2, K = 33, layers = 5)
eta = 0.005
opt_state = Flux.setup(Adam(eta), model)

iters = 10000
for i in 1:iters
    #Set up a batch of training pairs, and t
    #X0 = ManifoldState(M, discrete_X0(n_samples))
    X0 = ManifoldState(M, eachslice(stack([Float32.(rand(M)) for _ in zeros(2,1000)]), dims = (2,3)))
    X1 = ManifoldState(M, discrete_X1(n_samples))
    t = rand(T, n_samples)
    #Construct the bridge:
    Xt = bridge(P, X0, X1, t)
    #Get the Xt->X1 tangent coordinates:
    ξ = Flowfusion.tangent_coordinates(Xt, X1)
    #Gradient:
    l,gr = Flux.withgradient(model) do m
        tcloss(P, m(t,tensor(Xt)), ξ, scalefloss(P, t, 1))
    end
    #Update:
    Flux.update!(opt_state, model, gr[1])
    #Log and adjust learning rate:
    if i % 10 == 0
        if i > iters - 3000
            eta *= 0.975
            Optimisers.adjust!(opt_state, eta)
        end
        println("i: $i; Loss: $l; eta: $eta")
    end
end


n_inference_samples = 5000
#X0 = ManifoldState(M, discrete_X0(n_inference_samples));
X0 = ManifoldState(M, eachslice(stack([Float32.(rand(M)) for _ in zeros(2,n_inference_samples)]), dims = (2,3)))
uncorner!(tensor(X0))
paths = Tracker()
X1pred = (t,Xt) -> apply_tangent_coordinates(Xt, model(t,tensor(Xt)))
samp = gen(P, X0, X1pred, 0f0:0.002f0:1f0, tracker = paths)

#Plot the X0 and generated X1:
X0oc = Flowfusion.onecold(tensor(X0))
sampoc = Flowfusion.onecold(tensor(samp))
pl = scatter(X0oc[1,:],X0oc[2,:], msw = 0, color = "blue", label = :none, alpha = 0.02, size = (400,400), xlim = (0,34), ylim = (0, 34), title = "X0, X1 (sampled)", titlefontsize = 9)
scatter!([-10], [-10], msw = 0, color = "blue", alpha = 0.3, label = "X0")
scatter!(sampoc[1,:],sampoc[2,:], msw = 0, color = "green", alpha = 0.02, label = :none)
scatter!([-10], [-10], msw = 0, color = "green", alpha = 0.3, label = "gen X1")

#...compared to the true X1:
trueX1 = discrete_target(n_inference_samples)
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
gif(anim, "evol_deterministic_randsimplex.mp4", fps = 30)
#gif(anim, "evol_v$(P.v)_randsimplex.mp4", fps = 30)





X0 = ManifoldState(M, eachslice(stochastic(DiscreteState(33, 33 .+ zeros(Int, 2, n_samples))).dist, dims = (2,3)));
    X1 = ManifoldState(M, eachslice(stochastic(DiscreteState(33, discrete_target(n_samples))).dist, dims = (2,3)));
    uncorner!(tensor(Xt))
    uncorner!(tensor(X1))
    t = rand(T, n_samples)
    #Construct the bridge:
    Xt = bridge(P, X0, X1, t)



    X0 = ManifoldState(M, eachslice(stochastic(DiscreteState(33, rand(10:15, 2, n_samples))).dist, dims = (2,3)));
    X1 = ManifoldState(M, eachslice(stochastic(DiscreteState(33, discrete_target(n_samples))).dist, dims = (2,3)));
    uncorner!(tensor(X0))
    uncorner!(tensor(X1))
    t = rand(T, n_samples)
    #Construct the bridge:
    @time Xt = bridge(P, X0, X1, t)