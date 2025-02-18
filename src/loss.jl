#=######
NOTES on what works:
- Euclidean state:
- - any compatible process, using floss
- Manifold state:
- - any compatible process, using tcloss
- Discrete state:
- - for a DiscreteProcess, only UniformUnmasking works properly. The rest have issues.
- - works to using the ProbabilitySimplex in a ManifoldProcess.
- - Either:
- - - The process must have non-zero variance
- - - or X0 must be a continuous distribution (ie. not discrete "corners") on the ProbabilitySimplex (in which case a deterministic process also works)
=#######

#This is badness that doesn't work:
#rotangle(rots::AbstractArray{T,3}) where T = acos.(clamp.((rots[1,1,:] .+ rots[2,2,:] .+ rots[3,3,:] .- 1) ./ 2, T(-0.99), T(0.99)))
#rotangle(rots::AbstractArray) = reshape(rotangle(reshape(rots, 3, 3, :)), 1, size(rots)[3:end]...)
#torangle(x, y) = mod.(y .- x .+ π, 2π) .- π
#msra(X̂₁, X₁) = rotangle(batched_mul(batched_transpose(tensor(X̂₁)), tensor(X₁))).^2 #Mean Squared Angle
#msta(X̂₁, X₁) = sum(torangle(tensor(X̂₁), tensor(X₁)), dims=1).^2 #Mean Squared Toroidal Angle

mse(X̂₁, X₁) = abs2.(tensor(X̂₁) .- tensor(X₁)) #Mean Squared Error
lce(X̂₁, X₁) = -sum(tensor(X₁) .* logsoftmax(tensor(X̂₁)), dims=1) #Logit Cross Entropy
kl(P,Q) = sum(softmax(tensor(P)) .* (logsoftmax(tensor(P)) .- log.(tensor(Q))), dims=1) #Kullback-Leibler Divergence
rkl(P,Q) = sum(tensor(Q) .* (log.(tensor(Q)) .- logsoftmax(tensor(P))), dims=1) #Reverse Kullback-Leibler Divergence

function scaledmaskedmean(l::AbstractArray{T}, c::Union{AbstractArray, Real}, m::Union{AbstractArray, Real}) where T
    expanded_m = expand(m, ndims(l))
    mean(l .* expand(c, ndims(l)) .* expanded_m) / (mean(expanded_m) + T(1e-6))
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
#floss(P::fbu(ManifoldProcess{Rotations(3)}), X̂₁, X₁::msu(ManifoldState{Rotations(3)}), c) = scaledmaskedmean(msra(X̂₁, X₁), c, getlmask(X₁))
#floss(P::fbu(ManifoldProcess{SpecialOrthogonal(3)}), X̂₁, X₁::msu(ManifoldState{SpecialOrthogonal(3)}), c) = scaledmaskedmean(msra(X̂₁, X₁), c, getlmask(X₁))
#floss(P::fbu(ManifoldProcess), X̂₁, X₁::msu(ManifoldState{<:Torus}), c) = scaledmaskedmean(msta(X̂₁, X₁), c, getlmask(X₁))

floss(P::Tuple, X̂₁::Tuple, X₁::Tuple, c::Union{AbstractArray, Real}) = sum(floss.(P, X̂₁, X₁, (c,)))
floss(P::Tuple, X̂₁::Tuple, X₁::Tuple, c::Tuple) = sum(floss.(P, X̂₁, X₁, c))

#I should make a self-balancing loss that tracks the running mean/std and adaptively scales to balance against target weights.

"""
    tcloss(P::Union{fbu(ManifoldProcess), fbu(Deterministic)}, ξhat, ξ, c, mask = nothing)

Where `ξhat` is the predicted tangent coordinates, and `ξ` is the true tangent coordinates.
"""
floss(P::Union{fbu(ManifoldProcess), fbu(Deterministic)}, ξhat, ξ::Guide, c) = scaledmaskedmean(mse(ξhat, ξ.H), c, getlmask(ξ))
#tcloss(P::fbu(DiscreteProcess), ξhat, ξ, c, mask = nothing) = scaledmaskedmean(rkl(ξhat, ξ), c, mask)




#=
#Doesn't help to do it this way
"""
    tangent_coordinates(P::DiscreteProcess, Xt::DiscreteState, X1)

Computes (a weighted mixture of) Doob's h-transform(s) that would condition the current state Xt (which must be a discrete value)
to end at X1 (which can be a distribution) under P. Maybe.
"""
function tangent_coordinates(P::DiscreteProcess, X1::DiscreteState, t)
    #(for a single column) for state=i at 1-t, H_j(t)/H_i(t) is the rate scaling ratio per Doob's h-transform.
    #If the model can learn this directly, we can gen.
    H = backward(X1, P, 1 .- t)
    scale = sum(H.dist, dims = 1)
    H.dist ./= scale
    return H
end
=#



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
