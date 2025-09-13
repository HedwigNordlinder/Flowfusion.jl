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
function floss(p::fbu(SwitchBridgeProcess), X̂₁, X₁::msu(ContinuousState), c)
    # Weighted loss between aiming for the true endpoint and an alternative endpoint.
    # Notes/assumptions:
    # - ForwardBackward's SwitchBridgeProcess switches between two Brownian bridges with
    #   constant rates λ_alt (orig→alt) and λ_orig (alt→orig).
    # - We marginalise the loss by weighting the two targets by their switching probabilities
    #   over the remaining time. In absence of explicit time here, we infer t from the
    #   default `scalefloss` schedule if `c` was produced by it; otherwise this reduces to
    #   stationary weights λ/(λ+λ).
    # - If no alternative endpoint is provided, we default to zeros of X₁'s shape, which
    #   matches the common case where X₀ is the origin. See the 2-arg Tuple overload below
    #   to pass an explicit alternative endpoint.

    # Helper to evaluate λ fields whether they are Functions or scalars
    λ_alt_val(x) = (p.λ_alt isa Function) ? p.λ_alt(x) : p.λ_alt
    λ_orig_val(x) = (p.λ_orig isa Function) ? p.λ_orig(x) : p.λ_orig

    T = eltype(tensor(unmask(X₁)))
    eps = T(1e-12)

    # Try to infer current time t from the loss scale c, assuming default scalefloss: 1/((1+ϵ)-t)^2
    # If `c` did not come from scalefloss, clamp to a safe mid value.
    infer_t(cval) = begin
        ϵ = T(0.05)
        pow = T(2)
        tc = (T(1) + ϵ) .- (cval .^ (-T(1) / pow))
        clamp.(tc, T(0), T(1))
    end

    # Broadcast t to the loss tensor's dimensionality
    t_like_c = isa(c, Real) ? fill(T(c), 1) : T.(c)
    t_est = infer_t(t_like_c)
    # Remaining time r = 1 - t
    r = max.(T(0), T(1) .- t_est)

    # Instantaneous mixture weights: proportional to incoming rates into each regime.
    # Scale the rate toward the true endpoint by 1/(1 - t).
    # Broadcast remaining time to the state shape for safe elementwise ops.
    Xtrue = tensor(unmask(X₁))
    r_safe = max.(expand(r, ndims(Xtrue)), eps)
    λα = λ_alt_val(unmask(X₁))
    λo = λ_orig_val(unmask(X₁))
    λo_eff = λo ./ r_safe
    Z = λα .+ λo_eff .+ eps
    w_orig = λo_eff ./ Z
    w_alt = λα ./ Z

    # Targets: true endpoint and alternative endpoint (defaults to zeros)
    Xalt = zero.(Xtrue)

    # Compute weighted MSE and apply masking/scaling
    L_true = mse(X̂₁, Xtrue)
    L_alt = mse(X̂₁, Xalt)
    w_orig_e = expand(w_orig, ndims(L_true))
    w_alt_e = expand(w_alt, ndims(L_alt))
    return scaledmaskedmean(w_orig_e .* L_true .+ w_alt_e .* L_alt, c, getlmask(X₁))
end

# Overload that allows passing an explicit alternative endpoint alongside the true endpoint.
function floss(p::fbu(SwitchBridgeProcess), X̂₁, X::Tuple{<:msu(ContinuousState),<:msu(ContinuousState)}, c)
    X₁, Xalt = X
    # Reuse the primary method but substitute the internal alternative endpoint.
    # We temporarily compute the weighted loss components here for clarity.
    λ_alt_val(x) = (p.λ_alt isa Function) ? p.λ_alt(x) : p.λ_alt
    λ_orig_val(x) = (p.λ_orig isa Function) ? p.λ_orig(x) : p.λ_orig

    T = eltype(tensor(unmask(X₁)))
    eps = T(1e-12)

    infer_t(cval) = begin
        ϵ = T(0.05)
        pow = T(2)
        tc = (T(1) + ϵ) .- (cval .^ (-T(1) / pow))
        clamp.(tc, T(0), T(1))
    end

    t_like_c = isa(c, Real) ? fill(T(c), 1) : T.(c)
    t_est = infer_t(t_like_c)
    r = max.(T(0), T(1) .- t_est)

    Xtrue = tensor(unmask(X₁))
    r_safe = max.(expand(r, ndims(Xtrue)), eps)
    λα = λ_alt_val(unmask(X₁))
    λo = λ_orig_val(unmask(X₁))
    λo_eff = λo ./ r_safe
    Z = λα .+ λo_eff .+ eps
    w_orig = λo_eff ./ Z
    w_alt = λα ./ Z

    Xtrue = tensor(unmask(X₁))
    Xalternative = tensor(unmask(Xalt))

    L_true = mse(X̂₁, Xtrue)
    L_alt = mse(X̂₁, Xalternative)
    w_orig_e = expand(w_orig, ndims(L_true))
    w_alt_e = expand(w_alt, ndims(L_alt))
    return scaledmaskedmean(w_orig_e .* L_true .+ w_alt_e .* L_alt, c, getlmask(X₁))
end
