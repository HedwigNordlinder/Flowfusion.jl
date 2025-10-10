#=#####################
Assumptions:
- t ∈ [0,1]. Any behavior can be controlled by manipulating the process parameters.
- FProcess.F is a monotonic function, with F(0) = 0 and F(1) = 1.
- Default sampling steps are FProcess.F(t) with even t intervals [NOTE TO SELF: Intervals should be F(t2)-F(t1)]
=#####################

process(P::FProcess) = P.P
process(P::Process) = P

tscale(P::Process, t) = t
tscale(P::FProcess, t) = P.F.(t)

Adapt.adapt_structure(to, S::ForwardBackward.DiscreteState) = ForwardBackward.DiscreteState(S.K, Adapt.adapt(to, S.state))
Adapt.adapt_structure(to, S::ForwardBackward.ContinuousState) = ForwardBackward.ContinuousState(Adapt.adapt(to, S.state))
Adapt.adapt_structure(to, S::ForwardBackward.CategoricalLikelihood) = ForwardBackward.CategoricalLikelihood(Adapt.adapt(to, S.dist), Adapt.adapt(to, S.log_norm_const))
Adapt.adapt_structure(to, S::ForwardBackward.ManifoldState) = ForwardBackward.ManifoldState(S.M, Adapt.adapt(to, S.state))

"""
    onehot(X)

Rerturns a state where `X.state` is a onehot array.
"""
onehot(X::DiscreteState{<:AbstractArray{<:Integer}}) = DiscreteState(X.K, onehotbatch(X.state, 1:X.K))
onehot(X::DiscreteState{<:Union{OneHotArray, OneHotMatrix}}) = X

"""
    unhot(X)

Returns a state where `X.state` is not onehot.
"""
unhot(X::DiscreteState{<:Union{OneHotArray, OneHotMatrix}}) = DiscreteState(X.K, onecold(X.state, 1:X.K))
unhot(X::DiscreteState{<:AbstractArray{<:Integer}}) = X
ForwardBackward.stochastic(T::Type, o::DiscreteState{<:Union{OneHotArray, OneHotMatrix}}) = CategoricalLikelihood(T.(o.state .+ 0), zeros(T, size(o.state)[2:end]...))

"""
    dense(X::DiscreteState; T = Float32)

Converts `X` to an appropriate dense representation. If `X` is a `DiscreteState`, then `X` is converted to a `CategoricalLikelihood` with default eltype Float32.
If `X` is a "onehot" CategoricalLikelihood then `X` is converted to a fully dense one.
"""
dense(X::DiscreteState; T = Float32) = stochastic(T, X)


function copytensor!(dest, src)
    tensor(dest) .= tensor(src)
    return dest
end

#resolveprediction exists to stop bridge from needing multiple definitions.
#Tuple broadcast:
resolveprediction(dest::Tuple, src::Tuple) = map(resolveprediction, dest, src)

#Default if X̂₁ is a plain tensor:
#I think these were serving processes with a faulty assumption, so I'm swapping them out to make Doob flows easier.
#resolveprediction(X̂₁, Xₜ::DiscreteState{<:AbstractArray{<:Signed}}) = copytensor!(stochastic(Xₜ), X̂₁) #Returns a Likelihood
#resolveprediction(X̂₁, Xₜ::DiscreteState{<:Union{OneHotArray, OneHotMatrix}}) = copytensor!(stochastic(unhot(Xₜ)), X̂₁) #Probably inefficient
resolveprediction(X̂₁, Xₜ::DiscreteState{<:AbstractArray{<:Signed}}) = X̂₁ #<-Need to test if this breaking anything else
resolveprediction(X̂₁, Xₜ::DiscreteState{<:Union{OneHotArray, OneHotMatrix}}) = X̂₁ #<-Need to test if this breaking anything else
function resolveprediction(pred::Tuple{AbstractArray,AbstractArray},
                                      X0::LatentJumpingState)
    cont_pred, logits_pred = pred                 # (d,B), (K,B)
    # Pick the most likely terminal discrete state per trajectory (greedy is fine here)
    # If you prefer stochastic rollout, sample from softmax(logits_pred; dims=1).
    _, argmax_idx = findmax.(eachcol(logits_pred))  # returns (val, idx) per column
    term_states   = collect(argmax_idx)             # Vector{Int} of length B

    K = X0.switching_state.k
    B = size(cont_pred, 2)

    # Build the new LatentJumpingState for the endpoint:
    X1_disc = DiscreteState(K, term_states)
    X1_cont = ContinuousState(cont_pred)            # just the continuous endpoint
    # For LatentJumpingState the third field is the "combined" continuous state used by the process;
    # it should mirror the continuous coordinate here — do NOT add expected jumps.
    return LatentJumpingState(ContinuousState(cont_pred), X1_disc, X1_cont)
end

resolveprediction(X̂₁, Xₜ::State) = copytensor!(copy(Xₜ), X̂₁) #Returns a State - Handles Continuous and Manifold cases
#Passthrough if the user returns a State or Likelihood
resolveprediction(X̂₁::State, Xₜ) = X̂₁
resolveprediction(X̂₁::State, Xₜ::State) = X̂₁
resolveprediction(X̂₁::StateLikelihood, Xₜ) = X̂₁
#Handles when the 
resolveprediction(G::Guide, Xₜ::ManifoldState) = apply_tangent_coordinates(Xₜ, G.H)



"""
    bridge(P, X0, X1, t)
    bridge(P, X0, X1, t0, t)

Samples `Xt` at `t` conditioned on `X0` and `X1` under the process `P`. Start time is `t0` (0 if not specified). End time is 1.
If `X1` is a `MaskedState`, then `Xt` will equal `X1` where the conditioning mask `X1.cmask` is 1.
`P`, `X0`, `X1` can also be tuples where the Nth element of `P` will be used for the Nth elements of `X0` and `X1`.
The same `t` and (optionally) `t0` will be used for all elements. If you need a different `t` for each Proces/State, broadcast with `bridge.(P, X0, X1, t0, t)`.
"""

function bridge(P::UProcess, X0, X1, t0, t)
    T = eltype(t)
    tF = T.(tscale(P,t) .- tscale(P,t0))
    tB = T.(tscale(P,1) .- tscale(P,t))
    endpoint_conditioned_sample(X0, X1, process(P), tF, tB)
end
bridge(P, X0, X1, t) = bridge(P, X0, X1, eltype(t)(0.0), t)
bridge(P::Tuple{Vararg{UProcess}}, X0::Tuple{Vararg{UState}}, X1::Tuple, t0, t) = bridge.(P, X0, X1, (t0,), (t, ))
bridge(P::Tuple{Vararg{UProcess}}, X0::Tuple{Vararg{UState}}, X1::Tuple, t) = bridge.(P, X0, X1, (t, ))

#Step is like bridge (and falls back to where possible). But sometimes we only have enough to take an Euler step (which is ok when `s₂-s₁` is small).
step(P, Xₜ, hat, s₁, s₂) = bridge(P, Xₜ, hat, s₁, s₂)
step(P::Tuple{Vararg{UProcess}}, Xₜ::Tuple{Vararg{UState}}, hat::Tuple, s₁, s₂) = step.(P, Xₜ, hat, (s₁,), (s₂, ))

#Bridge/step overload for onehot discrete states - if either is onehot, the result will be onehot
bridge(P::ConvexInterpolatingDiscreteFlow, X0::Union{DiscreteState, DiscreteState{<:Union{OneHotArray, OneHotMatrix}}}, X1::Union{DiscreteState, DiscreteState{<:Union{OneHotArray, OneHotMatrix}}}, t) = onehot(bridge(P, unhot(X0), unhot(X1), t))
step(P::ConvexInterpolatingDiscreteFlow, Xₜ::Union{DiscreteState, DiscreteState{<:Union{OneHotArray, OneHotMatrix}}}, hat, s₁, s₂) = onehot(step(P, unhot(Xₜ), hat, s₁, s₂))

Base.copy(X::DiscreteState{<:Union{OneHotArray, OneHotMatrix}}) = onehot(copy(unhot(X)))


"""
    gen(P, X0, model, steps; tracker=Returns(nothing), midpoint = false)

Constructs a sequence of (stochastic) bridges between `X0` and the predicted `X̂₁` under the process `P`.
`P`, `X0`, can also be tuples where the Nth element of `P` will be used for the Nth elements of `X0` and `model`.
model is a function that takes `t` (scalar) and `Xₜ` (optionally a tuple) and returns `hat` (a `UState`, a flat tensor with the right shape, or a tuple of either if you're combining processes).
If `X0` is a `MaskedState`, then anything in `X̂₁` will be conditioned on `X0` where the conditioning mask `X0.cmask` is 1.
"""
function gen(P::Tuple{Vararg{UProcess}}, X₀::Tuple{Vararg{UState}}, model, steps::AbstractVector; tracker::Function=Returns(nothing), midpoint = false)
    Xₜ = copy.(X₀)
    for (s₁, s₂) in zip(steps, steps[begin+1:end])
        t = midpoint ? (s₁ + s₂) / 2 : t = s₁
        hat = resolveprediction(model(t, Xₜ), Xₜ)
        Xₜ = mask(step(P, Xₜ, hat, s₁, s₂), X₀)
        tracker(t, Xₜ, hat)
    end
    return Xₜ
end

#                                                                         t[1]? or just t?
gen(P, X₀, model, args...; kwargs...) = gen((P,), (X₀,), (t, Xₜ) -> (model(t[1], Xₜ[1]),), args...; kwargs...)[1]

struct Tracker <: Function
    t::Vector
    xt::Vector
    x̂1::Vector
end

Tracker() = Tracker([], [], [])

function (tracker::Tracker)(t, xt, x̂1)
    push!(tracker.t, t)
    push!(tracker.xt, xt)
    push!(tracker.x̂1, x̂1)
    return nothing
end

function stack_tracker(tracker, field; tuple_index = 1)
    return stack([tensor(data[tuple_index]) for data in getproperty(tracker, field)])
end


#Todo: tesst Guide with MaskedState
Guide(Xt::ManifoldState, X1::ManifoldState; kwargs...) = Guide(tangent_guide(Xt, X1; kwargs...))
Guide(mXt::Union{MaskedState{<:ManifoldState}, ManifoldState}, mX1::MaskedState{<:ManifoldState}; kwargs...) = Guide(tangent_guide(mXt, mX1; kwargs...), mX1.cmask, mX1.lmask)

#=If we want the model to directly predict the tangent coordinates, we use:
- tangent_coordinates outside the gradient call to get the thing the model will predict
- apply_tangent_coordinates during gen, to provide X̂₁ when the model is predicting the tangent coordinates
- the loss should just be the mse between the predicted tangent coordinates and the true tangent coordinates
Note: this gives you an invariance for free, since the model is predicting the change from Xt that results in X1.
=#
"""
    tangent_guide(Xt::ManifoldState, X1::ManifoldState)

Computes the coordinate vector (in the default basis) pointing from `Xt` to `X1`.
"""
function tangent_guide(mXt::Union{MaskedState, ManifoldState}, mX1::Union{MaskedState, ManifoldState}; inverse_retraction_method=default_inverse_retraction_method(unmask(mX1).M))
    Xt = unmask(mXt)
    X1 = unmask(mX1)
    T = eltype(tensor(X1))
    d = manifold_dimension(X1.M)
    ξ = zeros(T, d, size(Xt.state)...)
    temp_retract = inverse_retract(X1.M, Xt.state[1], X1.state[1], inverse_retraction_method)
    for ind in eachindex(Xt.state)
        inverse_retract!(X1.M, temp_retract, Xt.state[ind], X1.state[ind], inverse_retraction_method)
        ξ[:,ind] .= get_coordinates(X1.M, Xt.state[ind], temp_retract)
    end
    return ξ
end

"""
    apply_tangent_coordinates(Xt::ManifoldState, ξ; retraction_method=default_retraction_method(Xt.M))

returns `X̂₁` where each point is the result of retracting `Xt` by the corresponding tangent coordinate vector `ξ`.
"""
function apply_tangent_coordinates(Xt::ManifoldState, ξ; retraction_method=default_retraction_method(Xt.M))
    X̂₁ = copy(Xt)
    for ind in eachindex(Xt.state)
        X = get_vector(Xt.M, Xt.state[ind], ξ[:,ind])
        retract!(Xt.M, X̂₁.state[ind], Xt.state[ind], X, retraction_method)
    end
    return X̂₁
end

#As some point we should move these to ForwardBackward.jl. Low prio because they're mostly not needed for other applications.
tensor_cat(Xs::Vector{T}; dims_from_end = 1) where T = cat(tensor.(Xs)..., dims = ndims(tensor(Xs[1])) - dims_from_end + 1)
tensor_cat(Xs::Vector{Nothing}) = nothing

"""
    batch(Xs::Vector{T}; dims_from_end = 1)

Doesn't handle padding. Add option to pad if batching along dims that don't have the same length.
"""
batch(Xs::Vector{T}; dims_from_end = 1) where T<:ContinuousState = T(tensor_cat(Xs; dims_from_end))
batch(Xs::Vector{T}; dims_from_end = 1) where T<:DiscreteState = T(Xs[1].K, tensor_cat(Xs; dims_from_end))
batch(Xs::Vector{<:ManifoldState{<:M,<:A}}; dims_from_end = 1) where {M, A} = ManifoldState(Xs[1].M, eachslice(tensor_cat(Xs; dims_from_end), dims = Tuple((ndims(Xs[1].state[1])+1:ndims(tensor(Xs[1])))))) #Only tested for rotations.
batch(Xs::Vector{<:Tuple{Vararg{UState}}}, dims_from_end = 1) = Tuple([batch([x[i] for x in Xs], dims_from_end = dims_from_end) for i in 1:length(Xs[1])])

#Should never move to ForwardBackward.jl
batch(Xs::Vector{<:MaskedState}; dims_from_end = 1) = MaskedState(batch(unmask.(Xs); dims_from_end), tensor_cat([X.cmask for X in Xs]; dims_from_end), tensor_cat([X.lmask for X in Xs]; dims_from_end))


# Implementation of gen for LatentJumpingProcess
