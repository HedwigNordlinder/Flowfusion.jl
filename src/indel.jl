function combine(
    elements::A, new_elements::A,
    counts::AbstractVector{Int}=[size(new_elements)[end]],
    positions::AbstractVector{Int}=[size(elements)[end]],
    deletions::AbstractVector{Bool}=fill(false, size(elements)[end])
) where A<:AbstractArray
    D = ndims(elements)
    @assert length(deletions) == size(elements)[end]
    @assert ndims(new_elements) == D

    insertions = Dict{Int,Int}()
    for (p, n) in zip(positions, counts)
        insertions[p] = get(insertions, p, 0) + n
    end

    num_kept = count(!, deletions)
    num_new = sum(values(insertions))
    combined_len = num_kept + num_new

    other_dims = size(elements)[1:end-1]
    combined_size = (other_dims..., combined_len)
    combined_elements = similar(elements, combined_size)

    dst_idx = 1
    new_idx = 1

    if haskey(insertions, 0)
        n = insertions[0]
        if new_idx + n - 1 <= size(new_elements)[end] && dst_idx + n - 1 <= combined_len
            selectdim(combined_elements, D, dst_idx:dst_idx+n-1) .= selectdim(new_elements, D, new_idx:new_idx+n-1)
            dst_idx += n
            new_idx += n
        end
    end

    for src_idx in 1:size(elements)[end]
        if !deletions[src_idx] && dst_idx <= combined_len
            selectdim(combined_elements, D, dst_idx) .= selectdim(elements, D, src_idx)
            dst_idx += 1
        end

        if haskey(insertions, src_idx)
            n = insertions[src_idx]
            if new_idx + n - 1 <= size(new_elements)[end] && dst_idx + n - 1 <= combined_len
                selectdim(combined_elements, D, dst_idx:dst_idx+n-1) .= selectdim(new_elements, D, new_idx:new_idx+n-1)
                dst_idx += n
                new_idx += n
            end
        end
    end

    if dst_idx <= combined_len
        return selectdim(combined_elements, D, 1:dst_idx-1)
    else
        return combined_elements
    end
end

function combine(elements::DiscreteState, new_elements::DiscreteState, args...)
    @assert elements.K == new_elements.K "Number of categories must match"
    combined_state = combine(elements.state, new_elements.state, args...)
    return DiscreteState(elements.K, combined_state)
end

function combine(elements::ContinuousState, new_elements::ContinuousState, args...)
    combined_state = combine(elements.state, new_elements.state, args...)
    return ContinuousState(combined_state)
end

# Method for ManifoldState
function combine(elements::ManifoldState, new_elements::ManifoldState, args...)
    @assert elements.M == new_elements.M "Manifolds must match"
    combined_state = combine(elements.state, new_elements.state, args...)
    return ManifoldState(elements.M, combined_state)
end
