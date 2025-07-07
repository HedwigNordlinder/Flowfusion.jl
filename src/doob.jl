#ToDo: Incorporate FProcesses, with their schedules. The bridge behavior should already be correct,
#and the fallback doob should be correct if delta is passed through the schedule.
#But for the closed form we'll need to mod the velocities per the gradient, etc, and the same when stepping.

struct DoobMatchingFlow{Proc} <: Process
    P::Proc
end
export DoobMatchingFlow

Flowfusion.bridge(p::DoobMatchingFlow, x0::DiscreteState{<:AbstractArray{<:Signed}}, x1::DiscreteState{<:AbstractArray{<:Signed}}, t) = bridge(p.P, x0, x1, t)
#Finite diff fallback for when we don't have a closed form for the forward positive velocities:
function fallback_doob(P::DiscreteProcess, t, Xt::DiscreteState, X1::DiscreteState; delta = eltype(t)(1e-5))
    return (tensor(forward(Xt, P, delta) ⊙ backward(X1, P, (1 .- t) .- delta)) .- tensor(onehot(Xt))) ./ delta;
end

doob_guide(P::DiscreteProcess, t, Xt::DiscreteState, X1::DiscreteState) = fallback_doob(P, t, Xt, X1)

function closed_form_doob(P::DiscreteProcess, t, Xt::DiscreteState, X1::DiscreteState)
    tenXt = tensor(onehot(Xt))
    bk = tensor(backward(X1, P, 1 .- t))
    fv = forward_positive_velocities(onehot(Xt), P)
    positive_doob = (fv .* bk) ./ sum(bk .* tenXt, dims = 1)
    return positive_doob .- tenXt .* sum(positive_doob, dims = 1)
end

forward_positive_velocities(Xt::DiscreteState, P::PiQ)= (P.r .* (P.π ./ sum(P.π))) .* (1 .- tensor(onehot(Xt)))
doob_guide(P::PiQ, t, Xt::DiscreteState, X1::DiscreteState) = closed_form_doob(P, t, Xt, X1)

forward_positive_velocities(Xt::DiscreteState, P::UniformUnmasking{T}) where T = (P.μ .* T((1 ./ (Xt.K-1)))) .* (1 .- tensor(onehot(Xt)))
doob_guide(P::UniformUnmasking, t, Xt::DiscreteState, X1::DiscreteState) = closed_form_doob(P, t, Xt, X1)

forward_positive_velocities(Xt::DiscreteState, P::UniformDiscrete{T}) where T = (P.μ * T(1/(Xt.K*(1-1/Xt.K)))) .* (1 .- tensor(onehot(Xt)))
doob_guide(P::UniformDiscrete, t, Xt::DiscreteState, X1::DiscreteState) = closed_form_doob(P, t, Xt, X1)

Guide(P::DoobMatchingFlow, t, Xt::DiscreteState, X1::DiscreteState) = Flowfusion.Guide(doob_guide(P.P, t, Xt, X1))
Guide(P::DoobMatchingFlow, t, mXt::Union{MaskedState{<:DiscreteState}, DiscreteState}, mX1::MaskedState{<:DiscreteState}) = Guide(doob_guide(P.P, t, mXt, mX1), mX1.cmask, mX1.lmask)

function velo_step(Xₜ::DiscreteState{<:AbstractArray{<:Signed}}, delta_t, velocity)
    ohXₜ = onehot(Xₜ)
    newXₜ = CategoricalLikelihood(eltype(delta_t).(tensor(ohXₜ) .+ (delta_t .* velocity)))
    clamp!(tensor(newXₜ), 0, Inf) #Because one velo will be < 0 and a large step might push Xₜ < 0
    return rand(newXₜ)
end

step(P::DoobMatchingFlow, Xₜ::DiscreteState{<:AbstractArray{<:Signed}}, veloX̂₁::Flowfusion.Guide, s₁, s₂) = velo_step(Xₜ, s₂ .- s₁, veloX̂₁.H)
step(P::DoobMatchingFlow, Xₜ::DiscreteState{<:AbstractArray{<:Signed}}, veloX̂₁, s₁, s₂) = velo_step(Xₜ, s₂ .- s₁, veloX̂₁)

poisson_loss(mu, count, mask) = sum(mask .* (mu .- xlogy.(count, mu))) / sum(mask)
