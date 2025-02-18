#Need to test key mask functions on ContinuousState, DiscreteState, CategoricalLikelihood, ManifoldState

#=#####################
Conditioning mask behavior:
The typical use is that it makes sense, during training, to construct the conditioning mask on the training observation, X1.
During inference, the conditioning mask (and conditioned-upon state) has to be present on X1.
This dictates the behavior of the masking:
- When bridge() is called, the mask, and the state where mask=1, are inherited from X1.
- When gen is called, the state and mask will be propogated from X0 through all of the Xts.
=#####################


#import Base.copy
ForwardBackward.tensor(X::MaskedState) = tensor(X.S)
Base.copy(X::MaskedState) = MaskedState(copy(X.S), copy(X.cmask), copy(X.lmask))

"""
    endslices(a,m)

Returns a view of `a` where slices specified by `m` are selected. `m` can be multidimensional, but the dimensions of m must match the last dimensions of `a`.
For example, if `m` is a boolean array, then `size(a)[ndims(a)-ndims(m):end] == size(m)`.
"""
endslices(a,m) = @view a[ntuple(Returns(:),ndims(a)-ndims(m))...,m]


#=
Need to handle:
Xt = stochastic(Float32, bridge(P, X0, X1, t))
which means "stochastic" needs to preserve mask, and CategoricalLikelihoods need to be able to be masked.
onehot too
=#



onehot(X::MaskedState{<:DiscreteState{<:AbstractArray{<:Integer}}}) = MaskedState(onehot(X.S), X.cmask, X.lmask)
ForwardBackward.stochastic(T::Type, o::MaskedState) = MaskedState(stochastic(T, o.S), o.cmask, o.lmask)



getlmask(X1::UState) = X1.lmask
getlmask(X1::State) = nothing
getcmask(X1::UState) = X1.cmask
getcmask(X1::State) = nothing
getlmask(ξ::Guide) = ξ.lmask
getcmask(ξ::Guide) = ξ.cmask

"""
    unwrap(X)

Returns the underlying state or dist of `X` (`X.state` if `X` is a `State`, `X.dist` if `X` is a `StateLikelihood`, and `X.S.state` if `X` is a `MaskedState`, etc).
Unlike `tensor(X)` this does not flatten the state.
"""
unwrap(X::State) = X.state
unwrap(X::StateLikelihood) = tensor(X)
unwrap(X::MaskedState) = unwrap(X.S)

"""
   unmask(X)

`unmask(X) = X`, unless X is a `MaskedState`, in which case `X.S` is returned.
"""
unmask(X::MaskedState) = X.S
unmask(X) = X

"""
    cmask!(Xt_state, X1_state, cmask)
    cmask!(Xt, X1)

Applies, in place, a conditioning mask, where only elements (or slices) of `Xt` where `cmask` is 1 are noised. When `cmask` is 0, the elements are forced to be equal to `X1`.
"""
function cmask!(Xt_state, X1_state, cmask)
    endslices(Xt_state,.!cmask) .= endslices(X1_state,.!cmask)
    return Xt_state
end

#THIS IS NOT MODIFYING - NEED TO RETHINK
#=
function cmask!(ohXt_state::Union{OneHotArray, OneHotMatrix}, ohX1_state::Union{OneHotArray, OneHotMatrix}, cmask)
    K = size(ohXt_state, 1)
    Xt_state, X1_state = onecold(ohXt_state, 1:K), onecold(ohX1_state, 1:K)
    endslices(Xt_state,.!cmask) .= endslices(X1_state,.!cmask)
    return onehotbatch(Xt_state, 1:K)
end
=#

cmask!(Xt::Union{State, MaskedState{<:State}}, X1::MaskedState{<:StateLikelihood}) = error("Cannot condition a state on a Likelihood")

#=
function cmask!(Xt::MaskedState, X1::MaskedState)
    cmask!(Xt.S.state, X1.S.state, X1.cmask)
    size(Xt.cmask) != size(X1.cmask) && error("cmask dimensions must match")
    Xt.cmask .= X1.cmask
    return Xt
end

cmask!(Xt_state, X1_state, cmask::Nothing) = Xt_state
cmask!(Xt, X1::State) = Xt
cmask!(Xt, X1::StateLikelihood) = Xt
cmask!(Xt::Tuple, X1::Tuple) = map(cmask!, Xt, X1)
=#

"""
    mask(X, Y)

If `Y` is a `MaskedState`, `mask(X, Y)` returns a `MaskedState` with the content of `X` where elements of `Y.cmask` are 1, and `Y` where `Y.cmask` is 0.
`cmask` and `lmask` are inherited from `Y`.
If `Y` is not a `MaskedState`, `mask(X, Y)` returns `X`.
"""
mask(Xt::Tuple, X1::Tuple) = map(mask, Xt, X1)
mask(Xt::State, X1::State) = Xt
mask(Xt::StateLikelihood, X1::StateLikelihood) = Xt

function mask(Xt, X1::MaskedState)
    cmask!(unwrap(Xt), unwrap(X1), X1.cmask)
    return MaskedState(unmask(Xt), X1.cmask, X1.lmask)
end

#Needed custom handling for onehots:
unhot(X::MaskedState{<:DiscreteState{<:Union{OneHotArray, OneHotMatrix}}}) = MaskedState(unhot(X.S), X.cmask, X.lmask)
unhot(X) = X
mask(Xt, X1::MaskedState{<:DiscreteState{<:Union{OneHotArray, OneHotMatrix}}}) = onehot(mask(unhot(Xt), unhot(X1)))

bridge(P::UProcess, X0, X1::MaskedState, t0, t) = mask(bridge(P, unmask(X0), X1.S, t0, t), X1)
bridge(P::UProcess, X0, X1::MaskedState, t) = mask(bridge(P, unmask(X0), X1.S, t), X1)

#Mask passthroughs, because the masking gets handled elsewhere:
step(P::UProcess, Xₜ::MaskedState, hat, s₁, s₂) = step(P, unmask(Xₜ), unmask(hat), s₁, s₂) #step is only called in gen, which handles the masking itself
resolveprediction(X, Xₜ) = resolveprediction(unmask(X), unmask(Xₜ))
#resolveprediction(X, Xₜ::MaskedState) = resolveprediction(unmask(X), unmask(Xₜ))

print("?")

#REMOVE?:
#⊙ itself doesn't force the masks - it just propogates them. The forcing happens elsewhere.
#ForwardBackward.:⊙(a::MaskedState, b::MaskedState; kwargs...) = MaskedState(⊙(a.S, b.S; kwargs...), a.cmask .* b.cmask, a.lmask .* b.lmask)