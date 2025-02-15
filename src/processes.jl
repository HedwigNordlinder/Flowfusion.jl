##########################################
#For processes that aren't used elsewhere
##########################################
"""
    InterpolatingDiscreteFlow(κ::Function, κ̇::Function)
    InterpolatingDiscreteFlow() - Uses default Cosine scheduler.


A Discrete process that interpolates between two states (equation 9 from https://arxiv.org/pdf/2407.15595)
κ controls the interpolation schedule, κ̇ is the derivative of κ.
Works when model predicts `X̂₁` with cross-entropy loss (`floss` will do this).
"""
struct InterpolatingDiscreteFlow <: DiscreteProcess
    κ::Function
    κ̇::Function
end

InterpolatingDiscreteFlow() = InterpolatingDiscreteFlow(t -> 1-cos((pi/2)*t), t -> (pi/2)*sin((pi/2)*t))

function bridge(p::InterpolatingDiscreteFlow, x0::DiscreteState, x1::DiscreteState, t)
    ts = expand(t, ndims(x0.state))
    i = p.κ.(ts) .≥ rand(size(x0.state)...)
    xt = copy(x0)
    xt.state[i] .= x1.state[i]
    return xt
end

function step(P::InterpolatingDiscreteFlow, Xₜ::DiscreteState, X̂₁, s₁, s₂)
    step = s₂ .- s₁
    ohXₜ = onehot(Xₜ)
    velo = (P.κ̇.(s₁) ./ (1 - P.κ.(s₁))) .* (tensor(X̂₁) - tensor(ohXₜ))
    newXₜ = CategoricalLikelihood(eltype(s₁).(tensor(ohXₜ) .+ (step .* velo)))
    clamp!(tensor(newXₜ), 0, Inf) #Because one velo will be < 0 and a large step might push Xₜ < 0
    return rand(newXₜ)
end


"""
    NoisyInterpolatingDiscreteFlow(κ₁, κ₂, dκ₁, dκ₂)
    NoisyInterpolatingDiscreteFlow(noise) - Uses default cosine schedule, where `noise` is the maximum amplitude of the uniform noise component.
    NoisyInterpolatingDiscreteFlow() - Uses default cosine schedule and noise = 0.2.

A convex mixture of X0, uniform noise, and X1. Equation 10 in https://arxiv.org/pdf/2407.15595
Compared to InterpolatingDiscreteFlow, it encourages the model to make multiple switches during inference.
κ₁, κ₂ are the schedules for target token interpolation and uniform noise probability.
dκ₁, dκ₂ are the derivatives of κ₁, κ₂.
Defaults to using a cosine schedule.
"""
struct NoisyInterpolatingDiscreteFlow <: DiscreteProcess
    κ₁::Function    # schedule for target token interpolation
    κ₂::Function    # schedule for uniform noise probability
    dκ₁::Function   # derivative of κ₁
    dκ₂::Function   # derivative of κ₂
end

NoisyInterpolatingDiscreteFlow(noise) = NoisyInterpolatingDiscreteFlow(
    t -> oftype(t,(1 - cos((π/2)*t))),
    t -> oftype(t,(noise * sin(π*t))),
    t -> oftype(t,((π/2)*sin((π/2)*t))),
    t -> oftype(t,(noise*π*cos(π*t))),
)
NoisyInterpolatingDiscreteFlow() = NoisyInterpolatingDiscreteFlow(0.2)

function bridge(p::NoisyInterpolatingDiscreteFlow, x0::DiscreteState, x1::DiscreteState, t)
    D = size(x0.state)
    ts = expand(t, ndims(x0.state))
    Xt = copy(x0)
    rands = rand(D...) 
    x1bool = p.κ₁.(ts) .> rands
    uniformbool = (p.κ₂.(ts) .+ p.κ₁.(ts)) .> rands
    for idx in eachindex(rands)
        if x1bool[idx]
            Xt.state[idx] = x1.state[idx]
        elseif uniformbool[idx]
            Xt.state[idx] = rand(1:x0.K)
        else
            Xt.state[idx] = x0.state[idx]
        end
    end
    return Xt
end

function step(P::NoisyInterpolatingDiscreteFlow, Xₜ::DiscreteState, X̂₁, s₁, s₂)
    T = eltype(s₁)
    Δt = s₂ .- s₁
    ohXₜ = onehot(Xₜ)
    pu = T(1/Xₜ.K)
    eps = T(1e-10)
    κ1 = P.κ₁.(s₁)
    κ2 = P.κ₂.(s₁)
    κ3 = (1 .- (κ1 .+ κ2))  # κ₃(t)=1-κ₁(t)-κ₂(t)
    dκ1 = P.dκ₁.(s₁)
    dκ2 = P.dκ₂.(s₁)
    dκ3 = .- (dκ1 .+ dκ2)  # Because dκ₃ = - (dκ₁+dκ₂)
    bt = dκ3 ./ (eps .+ κ3)
    #Theorem 3 applied to equation 10 in https://arxiv.org/pdf/2407.15595
    velo = (dκ1 .- κ1 .* bt) .* tensor(X̂₁) .+ (dκ2 .- κ2 .* bt) .* pu .+ bt .* tensor(ohXₜ)
    newXₜ = CategoricalLikelihood(eltype(s₁).(tensor(ohXₜ) .+ (Δt .* velo)))
    clamp!(tensor(newXₜ), 0, Inf)
    return rand(newXₜ)
end
