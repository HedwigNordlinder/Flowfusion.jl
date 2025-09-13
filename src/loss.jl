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
    # Loss marginalisation for ForwardBackward.SwitchBridgeProcess using Doob h-transform.
    # We condition on ending at the true endpoint (orig regime at t=1). Under this
    # conditioning, the switching rates are q^h_{i→j}(t) = q_{i→j} * h_j(t) / h_i(t),
    # where h(t) solves the backward equation ∂ₜ h(t) = -Q h(t), h(1) = [1, 0].
    # We convert these rates into mixture weights between the two endpoints using the
    # incoming rates into each regime under q^h.

    # Helper to evaluate λ fields whether they are Functions or scalars
    λ_alt_val(x) = (p.λ_alt isa Function) ? p.λ_alt(x) : p.λ_alt   # orig→alt
    λ_orig_val(x) = (p.λ_orig isa Function) ? p.λ_orig(x) : p.λ_orig # alt→orig

    T = eltype(tensor(unmask(X₁)))
    eps = T(1e-12)

    # Infer current time t from default scalefloss c ≈ 1/((1+ϵ)-t)^2, else clamp
    infer_t(cval) = begin
        ϵ = T(0.05)
        pow = T(2)
        tc = (T(1) + ϵ) .- (cval .^ (-T(1) / pow))
        clamp.(tc, T(0), T(1))
    end

    t_like_c = isa(c, Real) ? fill(T(c), 1) : T.(c)
    t_est = infer_t(t_like_c)
    r = max.(T(0), T(1) .- t_est) # remaining time

    Xtrue = tensor(unmask(X₁))

    # Evaluate base switching rates per-sample (shared across dims) to match the
    # process definition where the discrete regime is common to all coordinates.
    # Shape conventions: Xtrue ≈ (D, ...batch_axes)
    tail_shape = size(Xtrue)[2:end]
    # Build a helper to evaluate a possibly state-dependent rate on each sample vector
    function _eval_rate(param, X)
        if param isa Function
            if isempty(tail_shape)
                v = param(view(X, :,))
                reshape(T(v), ())
            else
                out = Array{T}(undef, tail_shape...)
                for J in CartesianIndices(Tuple(tail_shape))
                    out[J] = T(param(view(X, :, Tuple(J)...)))
                end
                out
            end
        else
            fill(T(param), tail_shape...)
        end
    end

    λα = _eval_rate(λ_alt_val, Xtrue)
    λo = _eval_rate(λ_orig_val, Xtrue)
    # Insert a leading singleton dim so rates broadcast across coordinates
    λα = reshape(λα, (ones(Int, max(1, ndims(Xtrue) - length(size(λα))))..., size(λα)...))
    λo = reshape(λo, (ones(Int, max(1, ndims(Xtrue) - length(size(λo))))..., size(λo)...))

    s = λα .+ λo .+ eps
    πo = λo ./ s
    # h for remaining time r: h_o = πo + (1-πo) e^{-s r}, h_a = πo - πo e^{-s r}
    er = exp.(-expand(r, ndims(Xtrue)) .* s)
    h_o = clamp.(πo .+ (1 .- πo) .* er, eps, T(Inf))
    h_a = clamp.(πo .- πo .* er, eps, T(Inf))

    # Doob-transformed rates
    qh_o2a = λα .* (h_a ./ h_o)            # orig → alt under conditioning
    qh_a2o = λo .* (h_o ./ h_a)            # alt  → orig under conditioning

    # Mixture weights proportional to incoming rates under q^h
    Z = qh_o2a .+ qh_a2o .+ eps
    w_orig = qh_a2o ./ Z
    w_alt  = qh_o2a ./ Z

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

    λ_alt_val(x) = (p.λ_alt isa Function) ? p.λ_alt(x) : p.λ_alt   # orig→alt
    λ_orig_val(x) = (p.λ_orig isa Function) ? p.λ_orig(x) : p.λ_orig # alt→orig

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
    tail_shape = size(Xtrue)[2:end]
    # Per-sample base rates, consistent with shared regime across dims
    function _eval_rate(param, X)
        if param isa Function
            if isempty(tail_shape)
                v = param(view(X, :,))
                reshape(T(v), ())
            else
                out = Array{T}(undef, tail_shape...)
                for J in CartesianIndices(Tuple(tail_shape))
                    out[J] = T(param(view(X, :, Tuple(J)...)))
                end
                out
            end
        else
            fill(T(param), tail_shape...)
        end
    end

    λα = _eval_rate(λ_alt_val, Xtrue)
    λo = _eval_rate(λ_orig_val, Xtrue)
    λα = reshape(λα, (ones(Int, max(1, ndims(Xtrue) - length(size(λα))))..., size(λα)...))
    λo = reshape(λo, (ones(Int, max(1, ndims(Xtrue) - length(size(λo))))..., size(λo)...))
    s = λα .+ λo .+ eps
    πo = λo ./ s
    er = exp.(-expand(r, ndims(Xtrue)) .* s)
    h_o = clamp.(πo .+ (1 .- πo) .* er, eps, T(Inf))
    h_a = clamp.(πo .- πo .* er, eps, T(Inf))

    qh_o2a = λα .* (h_a ./ h_o)
    qh_a2o = λo .* (h_o ./ h_a)
    Z = qh_o2a .+ qh_a2o .+ eps
    w_orig = qh_a2o ./ Z
    w_alt  = qh_o2a ./ Z

    Xtrue = tensor(unmask(X₁))
    Xalternative = tensor(unmask(Xalt))

    L_true = mse(X̂₁, Xtrue)
    L_alt = mse(X̂₁, Xalternative)
    w_orig_e = expand(w_orig, ndims(L_true))
    w_alt_e = expand(w_alt, ndims(L_alt))
    return scaledmaskedmean(w_orig_e .* L_true .+ w_alt_e .* L_alt, c, getlmask(X₁))
end
