# A DFM that uses κ(t) = CDF(dist, t), κ̇ = PDF(dist, t)
"""
    DistInterpolatingDiscreteFlow(D::UnivariateDistribution)

D controls the schedule.
Note: both training and inference expect the model to output logits, unlike the other `InterpolatingDiscreteFlow` (where the user needs a manual `softmax` for inference).
"""
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
              X̂₁logits, s₁, s₂)
    X̂₁ = LogExpFunctions.softmax(X̂₁logits, dims = 1)
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
Note: both training and inference expect the model to output logits, unlike the other `NoisyInterpolatingDiscreteFlow` (where the user needs a manual `softmax` for inference).
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

   # ---- Gauss–Legendre constants (8-pt) ----
const _GL8_X = (-0.9602898564975363, -0.7966664774136267, -0.5255324099163290,
                -0.1834346424956498,  0.1834346424956498,  0.5255324099163290,
                 0.7966664774136267,  0.9602898564975363)
const _GL8_W = ( 0.10122853629037626, 0.22238103445337448, 0.31370664587788727,
                 0.36268378337836200, 0.36268378337836200, 0.31370664587788727,
                 0.22238103445337448, 0.10122853629037626)

# Scalar hazard integrals on [a,b]
function _W12_scalar(p::DistNoisyInterpolatingDiscreteFlow, a::T, b::T, ωu::T) where {T}
    if b <= a
        return zero(T), zero(T)
    end
    c1 = (b - a) / 2
    c2 = (b + a) / 2
    W1 = zero(T); W2 = zero(T)
    for k in 1:8
        s = c1 * _GL8_X[k] + c2
        κ1s    = cdf(p.D1, s)
        dκ1s   = pdf(p.D1, s)
        κ2tls  = cdf(p.D2, s)          # \tilde κ2(s)
        dκ2tls = pdf(p.D2, s)          # \dot{\tilde κ2}(s)

        # κ3 a1 and κ3 a2
        f1 = (one(T) - ωu*κ2tls) * dκ1s + κ1s * ωu * (one(T) - κ1s) * dκ2tls
        f2 = ωu * (one(T) - κ1s)^2 * dκ2tls

        W1 += _GL8_W[k] * f1
        W2 += _GL8_W[k] * f2
    end
    return c1 * W1, c1 * W2
end

# Vectorized over the *time* broadcast shape only (often the batch dim)
function _hazard_integrals(p::DistNoisyInterpolatingDiscreteFlow, ts0, tst, ωu::T) where {T}
    if ts0 isa AbstractArray
        W1 = similar(ts0, T); W2 = similar(ts0, T)
        for i in eachindex(ts0)
            a = T(ts0[i]); b = T(tst[i])
            W1[i], W2[i] = _W12_scalar(p, min(a,b), max(a,b), ωu)
        end
        return W1, W2
    else
        # scalar times
        a = T(ts0); b = T(tst)
        W1, W2 = _W12_scalar(p, min(a,b), max(a,b), ωu)
        return W1, W2
    end
end

# -------------------
# Two-time bridge (hazard integrals; integrates only over time shape)
# -------------------
function bridge(p::DistNoisyInterpolatingDiscreteFlow,
                x0::DiscreteState{<:AbstractArray{<:Signed}},
                x1::DiscreteState{<:AbstractArray{<:Signed}},
                t0, t)

    D   = size(x0.state)
    ts0 = expand(t0, ndims(x0.state))   # scalar or broadcastable array (e.g., batch dim)
    tst = expand(t,  ndims(x0.state))

    κ1_0, κ2_0 = κ1(p, ts0), κ2(p, ts0)
    κ1_t, κ2_t = κ1(p, tst),  κ2(p, tst)
    κ3_0, κ3_t = 1 .- κ1_0 .- κ2_0, 1 .- κ1_t .- κ2_t

    # numeric types
    T  = κ3_0 isa AbstractArray ? eltype(κ3_0) : typeof(κ3_0)
    ϵ  = eps(T)
    ωu = T(p.ωu)

    # integrate only across the time/batch broadcast shape
    W1, W2 = _hazard_integrals(p, ts0, tst, ωu)     # matches shape(ts0)
    denom  = κ3_0

    # conditional weights given X0-track at t0
    w1_raw = W1 ./ max.(denom, ϵ)
    w2_raw = W2 ./ max.(denom, ϵ)
    w3_raw = κ3_t ./ max.(denom, ϵ)

    s   = w1_raw .+ w2_raw .+ w3_raw
    ok  = denom .> ϵ                                  # well-posed conditioning
    w1c = w1_raw ./ s
    w2c = w2_raw ./ s

    # fallback to 1-time mixture where κ3(t0) ~ 0
    w1 = ifelse.(ok, w1c, κ1_t)
    w2 = ifelse.(ok, w2c, κ2_t)

    # --- sampling; randomness drawn over state dims, thresholds broadcast over time dims ---
    r  = rand(D...)
    Xt = copy(x0)

    x1bool      = w1 .> r
    uniformbool = (w1 .+ w2) .> r

    for I in eachindex(r)
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
# Euler step (Eq. 10)
# -------------------
function step(P::DistNoisyInterpolatingDiscreteFlow,
              Xₜ::DiscreteState{<:AbstractArray{<:Signed}},
              X̂₁logits, s₁, s₂)
    X̂₁ = LogExpFunctions.softmax(X̂₁logits, dims = 1)
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
