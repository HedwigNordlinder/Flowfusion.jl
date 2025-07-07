element(state, seqindex) = selectdim(state, ndims(state), seqindex:seqindex)
element(state, seqindex, batchindex) = element(selectdim(state, ndims(state), batchindex), seqindex)

element(S::MaskedState, inds...) = element(S.S, inds...)
element(S::ContinuousState, inds...) = ContinuousState(element(S.state, inds...))
element(S::ManifoldState, inds...) = ManifoldState(S.M, element(S.state, inds...))
element(S::DiscreteState, inds...) = DiscreteState(S.K, element(S.state, inds...))

element(S::Tuple{Vararg{Flowfusion.UState}}, inds...) = element.(S, inds...)

#Create a "zero" state appropriate for the type. Tricky for manifolds, but we just want rotations working for now I think.
zerostate(element::T, expandsize...) where T <: ContinuousState = T(similar(tensor(element), size(tensor(element))..., expandsize...) .= 0)
zerostate(element::DiscreteState{<:AbstractArray{<:Signed}}, expandsize...) = DiscreteState(element.K,similar(tensor(element), size(tensor(element))..., expandsize...) .= element.K)
zerostate(element::DiscreteState, expandsize...) = Flowfusion.onehot(DiscreteState(element.K,zeros(Int,expandsize...) .= element.K))
function zerostate(element::T, expandsize...) where T <: Union{ManifoldState{<:Rotations},ManifoldState{<:SpecialOrthogonal}}
    newtensor = similar(tensor(element), size(tensor(element))..., expandsize...) .= 0
    for i in 1:manifold_dimension(element.M)
        selectdim(selectdim(newtensor, 1,i),1,i) .= 1
    end
    return ManifoldState(element.M, eachslice(newtensor, dims=ntuple(i -> 2+i, length(expandsize))))
end

#In general, these will be different lengths, so we use an array of arrays as input.
#Doesn't work for onehot states yet.
function regroup(elarray::AbstractArray{<:AbstractArray})
    example_tuple = elarray[1][1]
    maxlen = maximum(length.(elarray))
    b = length(elarray)
    @show maxlen, b
    newstates = [zerostate(example_tuple[i],maxlen,b) for i in 1:length(example_tuple)]
    for ne in newstates
        @show size(tensor(ne))
    end
    for i in 1:b
        for j in 1:length(elarray[i])
            for k in 1:length(example_tuple)
                element(tensor(newstates[k]),j,i) .= tensor(elarray[i][j][k])
            end
        end
    end
    return Tuple(newstates)
end
