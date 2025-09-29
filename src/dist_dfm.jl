# A DFM that uses κ(t) = CDF(dist, t), κ̇ = PDF(dist, t)
struct DistInterpolatingDiscreteFlow <: ConvexInterpolatingDiscreteFlow
    D::UnivariateDistribution  # support [0,1]
end

# Single-time bridge (unchanged logic; just uses κ = CDF(D,·))
function bridge(p::DistInterpolatingDiscreteFlow,
                x0::DiscreteState{<:AbstractArray{<:Signed}},
                x1::DiscreteState{<:AbstractArray{<:Signed}},
                t)
    ts = expand(t, ndims(x0.state))
    κt = cdf.(Ref(p.D), ts)
    i  = κt .≥ rand(size(x0.state)...)
    xt = copy(x0)
    xt.state[i] .= x1.state[i]
    return xt
end

# Two-time bridge: κ_{t|t0} = (κ(t)-κ(t0))/(1-κ(t0)); t_eff = quantile(D, κ_{t|t0})
function bridge(p::DistInterpolatingDiscreteFlow,
                x0::DiscreteState{<:AbstractArray{<:Signed}},
                x1::DiscreteState{<:AbstractArray{<:Signed}},
                t0, t)
    ts0 = expand(t0, ndims(x0.state))
    tst = expand(t,  ndims(x0.state))
    κ0  = cdf.(Ref(p.D), ts0)
    κt  = cdf.(Ref(p.D), tst)
    ϵ = eps(eltype(κ0))
    κcond = clamp.((κt .- κ0) ./ max.(1 .- κ0, ϵ), 0.0, 1.0)
    t_eff = quantile.(Ref(p.D), κcond)
    return bridge(p, x0, x1, t_eff)
end

# Euler step for distribution-based DFM (same as your step, with κ̇/ (1-κ))
function step(P::DistInterpolatingDiscreteFlow,
              Xₜ::DiscreteState{<:AbstractArray{<:Signed}},
              X̂₁, s₁, s₂)
    step = s₂ .- s₁
    ohXₜ = onehot(Xₜ)
    κ1  = cdf.(Ref(P.D), s₁)
    κ̇1 = pdf.(Ref(P.D), s₁)
    velo = (κ̇1 ./ max.(1 .- κ1, eps(eltype(κ1)))) .* (tensor(X̂₁) - tensor(ohXₜ))
    newXₜ = CategoricalLikelihood(eltype(s₁).(tensor(ohXₜ) .+ (step .* velo)))
    clamp!(tensor(newXₜ), 0, Inf)
    return rand(newXₜ)
end

export DistInterpolatingDiscreteFlow


"""
    DistNoisyInterpolatingDiscreteFlow(; D1=Beta(2,2), D2=Beta(2,2), ωu=0.2, dummy_token=nothing)

Convex 3-way path over {X0, Uniform, X1} with distribution-backed schedules:
  κ₁(t) = cdf(D1,t)
  κ̃₂(t) = cdf(D2,t)
  κ₂(t) = ωu * (1 - κ₁(t)) * κ̃₂(t)        # uniform amplitude scaled by ωu ∈ [0,1)
  κ₃(t) = 1 - κ₁(t) - κ₂(t)

Derivatives:
  dκ₂(t) = ωu * ( -(dκ₁) * κ̃₂ + (1 - κ₁) * dκ̃₂ ),   dκ₃ = -(dκ₁ + dκ₂)

`ωu` directly controls the uniform noise amount; set ωu=0 for no-uniform, ωu→1 for max-gated uniform.
"""
struct DistNoisyInterpolatingDiscreteFlow{D1<:UnivariateDistribution,
                                          D2<:UnivariateDistribution, T} <: ConvexInterpolatingDiscreteFlow
    D1::D1
    D2::D2
    ωu::Float64
    dummy_token::T
end
DistNoisyInterpolatingDiscreteFlow(; D1=Beta(2,2), D2=Beta(2,2), ωu=0.2, dummy_token=nothing) =
    DistNoisyInterpolatingDiscreteFlow{typeof(D1),typeof(D2),typeof(dummy_token)}(D1, D2, ωu, dummy_token)

# Helpers (broadcast-friendly)
κ1(p::DistNoisyInterpolatingDiscreteFlow, t)    = cdf.(Ref(p.D1), t)
dκ1(p::DistNoisyInterpolatingDiscreteFlow, t)   = pdf.(Ref(p.D1), t)
κ2til(p::DistNoisyInterpolatingDiscreteFlow, t) = cdf.(Ref(p.D2), t)
dκ2til(p::DistNoisyInterpolatingDiscreteFlow,t) = pdf.(Ref(p.D2), t)

κ2(p::DistNoisyInterpolatingDiscreteFlow, t) = p.ωu .* (1 .- κ1(p,t)) .* κ2til(p,t)
function dκ2(p::DistNoisyInterpolatingDiscreteFlow, t)
    κ1t   = κ1(p,t);   dκ1t   = dκ1(p,t)
    κ2tlt = κ2til(p,t); dκ2tlt = dκ2til(p,t)
    p.ωu .* ( .-(dκ1t .* κ2tlt) .+ (1 .- κ1t) .* dκ2tlt )
end

κ3(p::DistNoisyInterpolatingDiscreteFlow, t)  = 1 .- κ1(p,t) .- κ2(p,t)
dκ3(p::DistNoisyInterpolatingDiscreteFlow, t) = .-(dκ1(p,t) .+ dκ2(p,t))

# -------------------
# One-time bridge
# -------------------
function bridge(p::DistNoisyInterpolatingDiscreteFlow,
                x0::DiscreteState{<:AbstractArray{<:Signed}},
                x1::DiscreteState{<:AbstractArray{<:Signed}},
                t)
    D  = size(x0.state)
    ts = expand(t, ndims(x0.state))
    Xt = copy(x0)

    κ1t = κ1(p, ts)
    κ2t = κ2(p, ts)
    r   = rand(D...)

    x1bool      = κ1t .> r
    uniformbool = (κ1t .+ κ2t) .> r

    @inbounds for I in eachindex(r)
        if x1bool[I]
            Xt.state[I] = x1.state[I]
        elseif uniformbool[I]
            Xt.state[I] = rand(1:x0.K)
        else
            Xt.state[I] = x0.state[I]
        end
    end
    return Xt
end

# -------------------
# Two-time bridge
# -------------------
# Conditional weights given we're on X0-track at t0:
#   denom = κ₃(t0)
#   w1 = (κ₁(t)-κ₁(t0)) / denom
#   w2 = (κ₂(t)-κ₂(t0)) / denom
#   w3 =  κ₃(t)         / denom
function bridge(p::DistNoisyInterpolatingDiscreteFlow,
                x0::DiscreteState{<:AbstractArray{<:Signed}},
                x1::DiscreteState{<:AbstractArray{<:Signed}},
                t0, t)
    D   = size(x0.state)
    ts0 = expand(t0, ndims(x0.state))
    tst = expand(t,  ndims(x0.state))

    κ1_0, κ2_0 = κ1(p, ts0), κ2(p, ts0)
    κ1_t, κ2_t = κ1(p, tst), κ2(p, tst)
    κ3_0, κ3_t = 1 .- κ1_0 .- κ2_0, 1 .- κ1_t .- κ2_t

    ϵ = eps(eltype(κ3_0))
    denom = max.(κ3_0, ϵ)

    w1 = clamp.((κ1_t .- κ1_0) ./ denom, 0.0, 1.0)
    w2 = clamp.((κ2_t .- κ2_0) ./ denom, 0.0, 1.0)
    w3 = clamp.( κ3_t           ./ denom, 0.0, 1.0)

    s = w1 .+ w2 .+ w3
    w1 ./= s; w2 ./= s; w3 ./= s

    r = rand(D...)
    Xt = copy(x0)
    @inbounds for I in eachindex(r)
        u = r[I]
        if u < w1[I]
            Xt.state[I] = x1.state[I]
        elseif u < (w1[I] + w2[I])
            Xt.state[I] = rand(1:x0.K)
        else
            Xt.state[I] = x0.state[I]
        end
    end
    return Xt
end

# -------------------
# Euler step (Eq. 10)
# -------------------
function step(P::DistNoisyInterpolatingDiscreteFlow,
              Xₜ::DiscreteState{<:AbstractArray{<:Signed}},
              X̂₁, s₁, s₂)
    T    = eltype(s₁)
    Δt   = s₂ .- s₁
    ohXₜ = onehot(Xₜ)
    pu   = T(1 / Xₜ.K)
    ϵ    = T(1e-10)

    κ1_ = κ1(P, s₁)
    κ2_ = κ2(P, s₁)
    κ3_ = 1 .- κ1_ .- κ2_

    dκ1_ = dκ1(P, s₁)
    dκ2_ = dκ2(P, s₁)
    dκ3_ = .- (dκ1_ .+ dκ2_)

    βt = dκ3_ ./ max.(κ3_, ϵ)

    # v = (dκ1 - κ1*β) * X̂₁ + (dκ2 - κ2*β) * pu + β * oh(X_t)
    velo = (dκ1_ .- κ1_ .* βt) .* tensor(X̂₁) .+
           (dκ2_ .- κ2_ .* βt) .* pu .+
           βt .* tensor(ohXₜ)

    newXₜ = CategoricalLikelihood(eltype(s₁).(tensor(ohXₜ) .+ (Δt .* velo)))
    clamp!(tensor(newXₜ), 0, Inf)
    return rand(newXₜ)
end

export DistNoisyInterpolatingDiscreteFlow