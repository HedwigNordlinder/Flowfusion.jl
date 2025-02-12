#=#####################
Assumptions:
- t ∈ [0,1]. Any behavior can be controlled by manipulating the process parameters.
- FProcess.F is a monotonic function, with F(0) = 0 and F(1) = 1.
- Default sampling steps are FProcess.F(t) with even t intervals [NOTE TO SELF: Intervals should be F(t2)-F(t1)]
=#####################

struct FProcess{A,B}
    P::A #Process
    F::B #Time transform
end

UProcess = Union{Process,FProcess}
process(P::FProcess) = P.P
process(P::Process) = P

tscale(P::Process, t) = t
tscale(P::FProcess, t) = P.F.(t)

struct MaskedState{A,B,C}
    S::A     #State
    cmask::B #Conditioning mask. 1 = Xt=X1
    lmask::C #Loss mask.         1 = included in loss
end

Adapt.adapt_structure(to, S::ForwardBackward.DiscreteState) = ForwardBackward.DiscreteState(S.K, Adapt.adapt(to, S.state))
Adapt.adapt_structure(to, S::ForwardBackward.ContinuousState) = ForwardBackward.ContinuousState(Adapt.adapt(to, S.state))
Adapt.adapt_structure(to, S::ForwardBackward.CategoricalLikelihood) = ForwardBackward.CategoricalLikelihood(Adapt.adapt(to, S.dist), Adapt.adapt(to, S.log_norm_const))
Adapt.adapt_structure(to, MS::MaskedState{<:State}) = MaskedState(Adapt.adapt(to, MS.S), Adapt.adapt(to, MS.cmask), Adapt.adapt(to, MS.lmask))
Adapt.adapt_structure(to, MS::MaskedState{<:CategoricalLikelihood}) = MaskedState(Adapt.adapt(to, MS.S), Adapt.adapt(to, MS.cmask), Adapt.adapt(to, MS.lmask))
Adapt.adapt_structure(to, S::ForwardBackward.ManifoldState) = ForwardBackward.ManifoldState(S.M, Adapt.adapt(to, S.state))

UState = Union{State,MaskedState}

ForwardBackward.tensor(X::MaskedState) = tensor(X.S)

import Base.copy
copy(X::MaskedState) = MaskedState(copy(X.S), copy(X.cmask), copy(X.lmask))

"""
    endslices(a,m)

Returns a view of `a` where slices specified by `m` are selected. `m` can be multidimensional, but the dimensions of m must match the last dimensions of `a`.
For example, if `m` is a boolean array, then `size(a)[ndims(a)-ndims(m):end] == size(m)`.
"""
endslices(a,m) = @view a[ntuple(Returns(:),ndims(a)-ndims(m))...,m]

"""
    cmask!(Xt_state, X1_state, cmask)
    cmask!(Xt, X1)

Applies, in place, a conditioning mask, forcing elements (or slices) of `Xt` to be equal to `X1`, where `cmask` is 1.
"""
function cmask!(Xt_state, X1_state, cmask)
    endslices(Xt_state,cmask) .= endslices(X1_state,cmask)
    return Xt_state
end

cmask!(Xt_state, X1_state, cmask::Nothing) = Xt_state
cmask!(Xt, X1::State) = Xt
cmask!(Xt, X1::StateLikelihood) = Xt
cmask!(Xt, X1::MaskedState) = cmask!(Xt.S.state, X1.S.state, X1.cmask)
cmask!(Xt, X1::MaskedState{<:CategoricalLikelihood}) = error("Cannot condition on a CategoricalLikelihood")
cmask!(x̂₁::Tuple, x₀::Tuple) = map(cmask!, x̂₁, x₀)

"""
    bridge(P, X0, X1, t)
    bridge(P, X0, X1, t0, t)

Samples `Xt` at `t` conditioned on `X0` and `X1` under the process `P`. Start time is `t0` (0 if not specified). End time is 1.
If `X1` is a `MaskedState`, then `Xt` will equal `X1` where the conditioning mask `X1.cmask` is 1.
`P`, `X0`, `X1` can also be tuples where the Nth element of `P` will be used for the Nth elements of `X0` and `X1`.
The same `t` and (optionally) `t0` will be used for all elements. If you need a different `t` for each Proces/State, broadcast with `bridge.(P, X0, X1, t0, t)`.
"""

function bridge(P::UProcess, X0::UState, X1, t0, t)
    T = eltype(t)
    tF = T.(tscale(P,t) .- tscale(P,t0))
    tB = T.(tscale(P,1) .- tscale(P,t))
    endpoint_conditioned_sample(cmask!(X0,X1), X1, process(P), tF, tB)
end
bridge(P, X0, X1, t) = bridge(P, X0, X1, eltype(t)(0.0), t)
bridge(P::Tuple{Vararg{UProcess}}, X0::Tuple{Vararg{UState}}, X1::Tuple, t0, t) = bridge.(P, X0, X1, (t0,), (t, ))



#copytensor! and predictresolve are used handle the state translation that happens in gen(...).
#We want the user's X̂₁predictor, which is a DL model, to return a plain tensor (since that will be on the GPU, in the loss, etc).
#This means we need to automagically create a State (typical for the continuous case) or Likelihood (typical for the discrete case) from the tensor.
#But the user may return a State in the Discrete case (for massive state spaces with sub-linear sampling), and a Likelihood in the Continuous case (for variance matching models)
#This also needs to handle MaskedStates (needs testing).
#We need: X̂₁ =  fix(X̂₁predictor(t, Xₜ))
#Plan: When X̂₁predictor(t, Xₜ) is a State or Likelihood, just pass through.
#When X̂₁predictor(t, Xₜ) is a plain tensor, we apply default conversion rules.

function copytensor!(dest, src)
    tensor(dest) .= tensor(src)
    return dest
end
#copytensor!(dest::Tuple, src::Tuple) = map(copytensor!, dest, src)

#Tuple broadcast:
resolveprediction(dest::Tuple, src::Tuple) = map(resolveprediction, dest, src)
#Default if X̂₁ is a plain tensor:
resolveprediction(X̂₁, X₀::DiscreteState) = copytensor!(stochastic(X₀), X̂₁) #Returns a Likelihood
resolveprediction(X̂₁, X₀::State) = copytensor!(copy(X₀), X̂₁) #Returns a State - Handles Continuous and Manifold cases
#Passthrough if the user returns a State or Likelihood
resolveprediction(X̂₁::State, X₀) = X̂₁
resolveprediction(X̂₁::State, X₀::State) = X̂₁
resolveprediction(X̂₁::StateLikelihood, X₀) = X̂₁
#####Add MaskedState case(s)######

##################################



"""
    gen(P, X0, X̂₁predictor, steps; tracker=Returns(nothing), midpoint = false)

Constructs a sequence of (stochastic) bridges between `X0` and the predicted `X̂₁` under the process `P`.
`P`, `X0`, can also be tuples where the Nth element of `P` will be used for the Nth elements of `X0` and `X̂₁predictor`.
X̂₁predictor is a function that takes `t` (scalar) and `Xₜ` (optionally a tuple) and returns `X̂₁` (a `UState`, a flat tensor with the right shape, or a tuple of either).
If `X0` is a `MaskedState` (or has a ), then anything  `X̂₁` will be conditioned on `X0` where the conditioning mask `X0.cmask` is 1.
"""
function gen(P::Tuple{Vararg{UProcess}}, X₀::Tuple{Vararg{UState}}, X̂₁predictor, steps::AbstractVector; tracker::Function=Returns(nothing), midpoint = false)
    Xₜ = copy.(X₀)
    for (s₁, s₂) in zip(steps, steps[begin+1:end])
        t = midpoint ? (s₁ + s₂) / 2 : t = s₁
        X̂₁ = resolveprediction(X̂₁predictor(t, Xₜ), X₀)
        cmask!(X̂₁, X₀)
        Xₜ = bridge(P, Xₜ, X̂₁, s₁, s₂)
        tracker(t, Xₜ, X̂₁)
    end
    return Xₜ
end

gen(P, X₀, X̂₁predictor, args...; kwargs...) = gen((P,), (X₀,), (t, Xₜ) -> (X̂₁predictor(t[1], Xₜ[1]),), args...; kwargs...)[1]

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