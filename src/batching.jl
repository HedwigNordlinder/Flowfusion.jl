#This will be in Flowfusion:
element(state, seqindex) = selectdim(state, ndims(state), seqindex:seqindex)
element(state, seqindex, batchindex) = element(selectdim(state, ndims(state), batchindex), seqindex)
element(S::MaskedState, inds...) = element(S.S, inds...)
element(S::ContinuousState, inds...) = ContinuousState(element(S.state, inds...))
element(S::ManifoldState, inds...) = ManifoldState(S.M, element(S.state, inds...))
element(S::DiscreteState, inds...) = DiscreteState(S.K, element(S.state, inds...))
element(S::Tuple{Vararg{Flowfusion.UState}}, inds...) = element.(S, inds...)


#Create a "zero" state appropriate for the type. Tricky for manifolds, but we just want rotations working for now I think.
zerostate(element::T, expandsize...) where T <: ContinuousState = T(similar(tensor(element), size(tensor(element))[1:end-1]..., expandsize...) .= 0)
zerostate(element::DiscreteState{<:AbstractArray{<:Signed}}, expandsize...) = DiscreteState(element.K,similar(tensor(element), size(tensor(element))[1:end-1]..., expandsize...) .= element.K)
zerostate(element::DiscreteState, expandsize...) = Flowfusion.onehot(DiscreteState(element.K,zeros(Int,expandsize...) .= element.K))
function zerostate(element::T, expandsize...) where T <: Union{ManifoldState{<:Rotations},ManifoldState{<:SpecialOrthogonal}}
    newtensor = similar(tensor(element), size(tensor(element))[1:end-1]..., expandsize...) .= 0
    for i in 1:manifold_dimension(element.M)
        selectdim(selectdim(newtensor, 1,i),1,i) .= 1
    end
    return ManifoldState(element.M, eachslice(newtensor, dims=ntuple(i -> 2+i, length(expandsize))))
end
#Pls test this general version with other manifolds? Not sure this will handle the various underlying representations
function zerostate(element::ManifoldState, expandsize...)
    # Use the first available point as a placeholder for padded slots; they will be masked out
    exemplar = first(element.state)
    arr = fill(exemplar, expandsize...)
    return ManifoldState(element.M, arr)
end

#In general, these will be different lengths, so we use an array of arrays as input.
#Doesn't work for onehot states yet.
function regroup(elarray::AbstractArray{<:AbstractArray})
    example_tuple = elarray[1][1]
    maxlen = maximum(length.(elarray))
    b = length(elarray)
    newstates = [zerostate(example_tuple[i],maxlen,b) for i in 1:length(example_tuple)]
    for i in 1:b
        for j in 1:length(elarray[i])
            for k in 1:length(example_tuple)
                element(tensor(newstates[k]),j,i) .= tensor(elarray[i][j][k])
            end
        end
    end
    return Tuple(newstates)
end


# ─────────────────────────────────────────────────────────────────────────────
# Generic padding + batching into a MaskedState
# ─────────────────────────────────────────────────────────────────────────────

_seqlen(S::State) = size(tensor(S))[end]
_seqlen(S::MaskedState) = _seqlen(S.S)

"""
    batch(Xs::Vector{<:State}) -> MaskedState

Pads a vector of same-typed states to the maximum sequence length and stacks them
along a new batch dimension as the last dimension. Returns a `MaskedState` where:
 - lmask[i, b] = true for real elements and false for padded ones
 - cmask defaults to lmask for plain state inputs

Works for `ContinuousState`, `DiscreteState` (both integer and onehot-backed), and
`ManifoldState` (via `zerostate` and `element` helpers).
"""
function batch(Xs::Vector{T}) where {T<:State}
    @assert !isempty(Xs)
    lens = _seqlen.(Xs)
    maxlen = maximum(lens)
    longest = argmax(lens)
    if maxlen == 0
        error("At least one state must have a non-zero length to be batched.") #We should find a way to make this not be a restriction because it can kill a training run.
    end
    b = length(Xs)

    # Create an appropriately-typed zero/pad state of shape (..., maxlen, b)
    ex = element(Xs[longest], 1)
    Sbat = zerostate(ex, maxlen, b)

    # Fill real elements
    for i in 1:b
        li = lens[i]
        for j in 1:li
            element(tensor(Sbat), j, i) .= tensor(element(Xs[i], j))
        end
    end

    # Masks: real positions true, padded false. Use lmask for cmask by default.
    lmask = falses(maxlen, b)
    for i in 1:b
        lmask[1:lens[i], i] .= true
    end
    cmask = copy(lmask)
    return MaskedState(Sbat, cmask, lmask)
end

"""
    batch(Xs::Vector{<:MaskedState}) -> MaskedState

Pads and batches masked states, preserving each element's cmask and lmask where
defined, and masking out padded positions.
"""
function batch(Xs::Vector{<:MaskedState})
    @assert !isempty(Xs)
    lens = _seqlen.(Xs)
    maxlen = maximum(lens)
    b = length(Xs)

    # Build underlying stacked state from unmasked contents
    ex = element(unmask(Xs[1]), 1)
    Sbat = zerostate(ex, maxlen, b)
    for i in 1:b
        li = lens[i]
        for j in 1:li
            element(tensor(Sbat), j, i) .= tensor(element(unmask(Xs[i]), j))
        end
    end

    # Pad/stack masks; expect per-sequence masks (length = lens[i])
    lmask = falses(maxlen, b)
    cmask = falses(maxlen, b)
    for i in 1:b
        li = lens[i]
        # Support vector or any array with last-dim matching sequence
        ml = getlmask(Xs[i])
        mc = getcmask(Xs[i])
        if ml !== nothing
            @assert size(ml)[end] == li
            lmask[1:li, i] .= vec(selectdim(ml, ndims(ml), :))
        else
            lmask[1:li, i] .= true
        end
        if mc !== nothing
            @assert size(mc)[end] == li
            cmask[1:li, i] .= vec(selectdim(mc, ndims(mc), :))
        else
            cmask[1:li, i] .= lmask[1:li, i]
        end
    end
    return MaskedState(Sbat, cmask, lmask)
end

"""
    batch(Xs::Vector{<:Tuple{Vararg{Flowfusion.UState}}})

Batch a vector of tuples component-wise, returning a tuple where each component
is a `MaskedState` produced by `batch` on that component across the vector.
"""
function batch(Xs::Vector{<:Tuple{Vararg{Flowfusion.UState}}})
    @assert !isempty(Xs)
    ncomp = length(Xs[1])
    return Tuple([batch([x[i] for x in Xs]) for i in 1:ncomp])
end
