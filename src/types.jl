struct FProcess{A,B}
    P::A #Process
    F::B #Time transform
end

UProcess = Union{Process,FProcess}

"""
    MaskedState(S::State, cmask, lmask)

Wraps a `State` with a conditioning mask (`cmask`) and a loss mask (`lmask`).

Conditioning mask behavior:

The typical use is that it makes sense, during training, to construct the conditioning mask on the training observation, `X1``.
During inference, the conditioning mask (and conditioned-upon state) has to be present on `X1`.
This dictates the behavior of the masking:
- When `bridge()` is called, the mask, and the state where `cmask=1`, are inherited from `X1`.
- When `gen()` is called, the state and mask will be propogated from `X0` through all of the `Xt`s.

Loss mask behavior:
- Where `lmask=0`, that observation (where the shape/size of the observation is determined by the difference in dimensions between the mask and the state) is not included in the loss.
"""
struct MaskedState{A,B,C}
    S::A     #State
    cmask::B #Conditioning mask. 1 = Xt=X1
    lmask::C #Loss mask.         1 = included in loss
end

#For when we want to predict the transitions instead of X1hat
"""
    Guide(H::AbstractArray)

Wrapping a model prediction in Guide instructs the solver that the prediction points to X1 from the current state, instead of being a prediction of X1 itself.
Used for ManifoldStates where the prediction is a tangent 
"""
struct Guide{A}
    H::A
end

UState = Union{State,MaskedState, Guide}