mse(X̂₁, X₁) = abs2.(tensor(X̂₁) .- tensor(X₁)) #Mean Squared Error
lce(X̂₁, X₁) = -sum(tensor(X₁) .* logsoftmax(tensor(X̂₁)), dims=1) #Logit Cross Entropy
kl(P,Q) = sum(softmax(tensor(P)) .* (logsoftmax(tensor(P)) .- log.(tensor(Q))), dims=1) #Kullback-Leibler Divergence
rkl(P,Q) = sum(tensor(Q) .* (log.(tensor(Q)) .- logsoftmax(tensor(P))), dims=1) #Reverse Kullback-Leibler Divergence

function scaledmaskedmean(l::AbstractArray{T}, c::Union{AbstractArray, Real}, m::Union{AbstractArray, Real}) where T
    expanded_m = expand(m, ndims(l))
    T(mean(l .* expand(c, ndims(l)) .* expanded_m) / ((sum(expanded_m)/T(length(expanded_m))) + T(1e-6)))
end

scaledmaskedmean(l::AbstractArray, c::Union{AbstractArray, Real}, m::Nothing) = mean(l .* expand(c, ndims(l)))

#Need to consider loss scaling for discrete case, since we're predicting a distribution instead of a point.
#Similar to if we were allowed to predict a Gaussian variance, the scaling would semi-automatically compensate.
scalefloss(P::UProcess, t, pow = 2, eps = eltype(t)(0.05)) = 1 ./ ((1 + eps) .- tscale(P, t)) .^ pow
scalefloss(P::Tuple, t::Union{AbstractArray, Real}, pow = 2, eps = eltype(t)(0.05)) = scalefloss.(P, (t,), (pow,), (eps,))
scalefloss(P::Tuple, t::Tuple, pow = 2, eps = eltype(t)(0.05)) = scalefloss.(P, t, (pow,), (eps,))

"""
    floss(Xₜ::UState, X̂₁, X₁::UState, c)
    floss(P::UProcess, Xₜ::UState, X̂₁, X₁::UState, c)
    floss(P::UProcess, X̂₁, X₁::UState, c)
    
Where c is shaped like t, and scales the loss. Typical call with default loss scaling would be like: `floss(P, X̂₁, X₁, scalefloss(P, t))`
"""

fbu(T) = Union{T, FProcess{<:T}}
msu(T) = Union{T, MaskedState{<:T}}

floss(P::fbu(Deterministic),                X̂₁, X₁::msu(ContinuousState), c) = scaledmaskedmean(mse(X̂₁, X₁), c, getlmask(X₁))
floss(P::fbu(BrownianMotion),               X̂₁, X₁::msu(ContinuousState), c) = scaledmaskedmean(mse(X̂₁, X₁), c, getlmask(X₁))
floss(P::fbu(OrnsteinUhlenbeck),            X̂₁, X₁::msu(ContinuousState), c) = scaledmaskedmean(mse(X̂₁, X₁), c, getlmask(X₁))
floss(P::fbu(ManifoldProcess{<:Euclidean}), X̂₁, X₁::msu(ContinuousState), c) = scaledmaskedmean(mse(X̂₁, X₁), c, getlmask(X₁))
#For a discrete process, X̂₁ will be a distribution, and X₁ will have to be a onehot before going onto the gpu.
floss(P::fbu(DiscreteProcess), X̂₁, X₁::msu(DiscreteState{<:AbstractArray{<:Integer}}), c) = error("X₁ needs to be onehot encoded with `onehot(X₁)`. You might need to do this before moving it to the GPU.")
floss(P::fbu(DiscreteProcess), X̂₁, X₁::msu(DiscreteState{<:OneHotArray}), c) = scaledmaskedmean(lce(X̂₁, X₁), c, getlmask(X₁))
floss(P::Tuple, X̂₁::Tuple, X₁::Tuple, c::Union{AbstractArray, Real}) = sum(floss.(P, X̂₁, X₁, (c,)))
floss(P::Tuple, X̂₁::Tuple, X₁::Tuple, c::Tuple) = sum(floss.(P, X̂₁, X₁, c))
floss(P::Union{fbu(ManifoldProcess), fbu(Deterministic)}, ξhat, ξ::Guide, c) = scaledmaskedmean(mse(ξhat, ξ.H), c, getlmask(ξ))

#I should make a self-balancing loss that tracks the running mean/std and adaptively scales to balance against target weights.

########################################################################
#Manifold-specific helper functions
########################################################################

########################################################################
#SO(3): Rotations
########################################################################

#It is sometimes useful to be able to compute the tangent coordinates from the predicted endpoint,
#eg. if the model needs to reason about its final location during the forward pass.
function so3_tangent_coordinates_stack(Rt::AbstractArray{T,3}, R1::AbstractArray{T,3}) where T
    eps = T(0.00001)
    R = batched_mul(batched_transpose(R1), Rt)
    tr_R = R[1,1,:] .+ R[2,2,:] .+ R[3,3,:]
    theta = acos.(clamp.((tr_R .- 1) ./ 2,T(-0.99),T(0.99)))
    sin_theta = sin.(theta)
    coeff = T(0.5) ./ (sin_theta .+ eps)
    axis_x = reshape(coeff .* (R[3,2,:] .- R[2,3,:]), 1, :)
    axis_y = reshape(coeff .* (R[1,3,:] .- R[3,1,:]), 1, :)
    axis_z = reshape(coeff .* (R[2,1,:] .- R[1,2,:]), 1, :)
    axis = vcat(axis_x, axis_y, axis_z)
    theta = reshape(theta, 1, :)
    tangent = sqrt(T(2)) .* (theta .* axis)
    return tangent
end

function so3_tangent_coordinates_stack(rhat::AbstractArray{T,4}, r::AbstractArray{T,4}) where T
    return reshape(so3_tangent_coordinates_stack(reshape(rhat, 3, 3, :), reshape(r, 3, 3, :)), 3, size(rhat,3), size(rhat,4))
end
floss(P::fbu(SwitchingBM), X̂₁, X₁::msu(ContinuousState), c) =
    scaledmaskedmean(mse(X̂₁, X₁), c, getlmask(X₁))
floss(P::fbu(SDEProcess), X̂₁, X₁::msu(ContinuousState), c) =
    scaledmaskedmean(mse(X̂₁, X₁), c, getlmask(X₁))
# Add loss for SwitchingSDEProcess: predict continuous endpoint only; Q not learned
floss(P::fbu(SwitchingSDEProcess), X̂₁, X₁::msu(ContinuousState), c) =
    scaledmaskedmean(mse(X̂₁, X₁), c, getlmask(X₁))



# ==========================
# CTMC-marginalized direction loss for ConditionalBridgeProcess (Euclidean)
# ==========================

# Unit vector utility (robust to near-zero norms)
@inline function _unit_vec(v::AbstractVector{T}; eps::T=T(1e-12)) where {T<:Real}
    n = sqrt(sum(abs2, v))
    n > eps && return (@. v / n)
    # Return a fresh unit vector e1 of appropriate length without mutation
    return vcat(one(T), zeros(T, max(length(v) - 1, 0)))
end

# Deterministic alternate anchor for loss: along (a - x) direction
@inline function _alt_anchor_det(a::AbstractVector{T}, x::AbstractVector{T}, r::T) where {T<:Real}
    u = _unit_vec(@. a - x)
    @. a + r * u
end

# Local hazard to true endpoint (mirrors process hazard)
@inline _λ_to_a_loss(κ::T, τ::T, T_end::T) where {T<:Real} = κ / max(T(1e-12), T_end - τ)

# One-column CTMC-marginalized direction
@inline function _ctmc_marginal_direction_col(P::ConditionalBridgeProcess{T},
                                              x::AbstractVector{T},
                                              a::AbstractVector{T},
                                              s_current::Int,
                                              τ::T, T_end::T, ε::T) where {T<:Real}
    v_a = @. a - x
    b   = _alt_anchor_det(a, x, P.ε * P.σ)
    v_b = @. b - x

    # Instantaneous mixture at current time τ (no forward ε step)
    λ_ba = _λ_to_a_loss(P.κ, τ, T_end)   # b -> a
    λ_ab = P.λ_b                         # a -> b
    denom = max(T(1e-12), λ_ba + λ_ab)
    p_a = λ_ba / denom                   # instantaneous mixture weight for anchor a

    @. (p_a * v_a) + ((one(T) - p_a) * v_b)
end

# Batched CTMC-marginalized direction (D×N)
function _ctmc_marginal_direction(P::ConditionalBridgeProcess{T},
                                  Xτ::ConditionalBridgeState{T},
                                  X₁::ContinuousState{T},
                                  τ, T_end, ε::T) where {T<:Real}
    x = tensor(Xτ.continuous_state)  # D×N
    a = tensor(X₁)                   # D×N
    s = Xτ.anchor_state              # N (1=>a, 2=>b)
    D, N = size(x)
    τn(n)    = (τ isa AbstractVector ? τ[n] : τ)
    Tendn(n) = (T_end isa AbstractVector ? T_end[n] : T_end)
    cols = map(1:N) do n
        _ctmc_marginal_direction_col(P, view(x, :, n), view(a, :, n), s[n], τn(n), Tendn(n), ε)
    end
    # hcat the columns into a D×N matrix (handles N>=1). For N==0, return x (empty).
    return N > 0 ? reduce(hcat, cols) : x
end

# Convert model output to direction if it is an endpoint
@inline function _to_direction_from_endpoint(d̂_or_X̂₁, Xτ_cont)
    X̂ = tensor(d̂_or_X̂₁)
    Xt = tensor(Xτ_cont)
    if size(X̂) == size(Xt)
        return X̂ .- Xt
    else
        return X̂
    end
end

# Loss comparing predicted direction to CTMC-marginalized direction (or true direction)
function floss(P::fbu(ConditionalBridgeProcess),
               Xτ::msu(ConditionalBridgeState),
               d̂_or_X̂₁,
               X₁::msu(ContinuousState),
               c; τ,
               T_end=nothing,
               ε=nothing,
               normalize::Bool=false,
               target::Symbol=:ctmc)
    # Handle FProcess wrapper
    baseP = process(P)
    T = eltype(tensor(X₁))
    τval = τ
    Tend = isnothing(T_end) ? one(T) : T_end
    εval = isnothing(ε) ? T(1e-3) : ε

    # Predicted direction (D×N)
    Xτu = unmask(Xτ)
    d̂ = _to_direction_from_endpoint(d̂_or_X̂₁, Xτu.continuous_state)

    # Target direction
    target_dir = if target === :ctmc
        _ctmc_marginal_direction(baseP, Xτu, unmask(X₁), τval, Tend, εval)
    elseif target === :true
        tensor(unmask(X₁)) .- tensor(Xτu.continuous_state)
    else
        throw(ArgumentError("Unsupported target=:$(target). Use :ctmc or :true"))
    end

    if normalize
        epsn = T(1e-12)
        nd = sqrt.(sum(abs2, d̂; dims=1)) .+ epsn
        nt = sqrt.(sum(abs2, target_dir; dims=1)) .+ epsn
        d̂ = d̂ ./ nd
        target_dir = target_dir ./ nt
    end

    scaledmaskedmean(mse(d̂, target_dir), c, getlmask(X₁))
end
