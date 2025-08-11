using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using Flowfusion, ForwardBackward
using Flux, Optimisers, CannotWaitForTheseOptimisers
using Random
using NNlib
using OneHotArrays
using Onion
using RandomFeatureMaps
using Zygote

# Alphabet and encoding (include tokens used by the PIP implementation)
const AAs = collect("->ACDEFGHIKLMNPQRSTVWY.#")
const TOK2ID = Dict(c => i for (i, c) in enumerate(AAs))
const ID2TOK = Dict(v => k for (k, v) in TOK2ID)
START_ID = TOK2ID['-']
encode(s::AbstractString) = [TOK2ID[c] for c in collect(s)]
decode(v::Vector{Int}) = join(ID2TOK[i] for i in v)

# Process
K = length(AAs)
P = UniformDiscretePoissonIndelProcess(K; lambda = 0.1, mu = 0.1, alpha = 0.1)

# Load real data and sample prefixes for x1
data = let
    lines = readlines(joinpath(@__DIR__, "abs.txt"))
    allowed = Set("ACDEFGHIKLMNPQRSTVWY")
    [String(filter(c -> c in allowed, d)) for d in lines if !occursin("X", d)]
end

#Truncated samples for quicker testing
function sample_pair()
    j = rand(1:length(data))
    s = data[j]
    N = rand(5:15)
    sN = s[1:N]
    base_id = TOK2ID['#']
    len0 = rand(1:25)
    x0 = DiscreteState(K, [base_id for _ in 1:len0])
    # Prepend explicit start marker token to X1 so model can learn sequence start
    x1 = DiscreteState(K, vcat([TOK2ID['>']], encode(sN), [TOK2ID['.']]))
    return x0, x1
end

#=
function sample_pair()
    j = rand(1:length(data))
    s = data[j]
    base_id = TOK2ID['#']
    len0 = rand(5:250)
    x0 = DiscreteState(K, [base_id for _ in 1:len0])
    # Prepend explicit start marker token to X1 so model can learn sequence start
    x1 = DiscreteState(K, vcat([TOK2ID['>']], encode(s), [TOK2ID['.']]))
    return x0, x1
end
=#

# Minimal model: token embedding + time embedding (RFF) + a few conditioned Transformer blocks + three heads
struct PIPModel{E,TEmb,Blocks,HS,HD,HI,Ro}
    embedding::E
    time_embed::TEmb
    blocks::Blocks
    head_sub::HS
    head_del::HD
    head_ins::HI
    rope::Ro
    K::Int
end

Flux.@layer PIPModel

function PIPModel(; d = 128, num_heads = 4, nlayers = 2, rff_dim = 64, cond_dim = 128, K::Int)
    embedding = Flux.Embedding(K => d)
    time_embed = Flux.Chain(RandomFourierFeatures(1 => rff_dim, 1.0f0), Dense(rff_dim => cond_dim, Flux.leakyrelu))
    blocks = [Onion.AdaTransformerBlock(d, cond_dim, num_heads) for _ in 1:nlayers]
    head_sub = Dense(d => K, bias = false)
    head_del = Dense(d => 1, bias = false)
    head_ins = Dense(d => K, bias = false)
    rope = RoPE(d ÷ num_heads, 4096)
    return PIPModel(embedding, time_embed, blocks, head_sub, head_del, head_ins, rope, K)
end

function (m::PIPModel)(t::Vector{Float32}, Xt_raw::Matrix{Int})
    # Prepend immortal START inside the model to align positions and gaps
    n, B = size(Xt_raw)
    Xt2_padded = vcat(fill(START_ID, 1, B), Xt_raw)  # (n+1, B)
    Lmax = n + 1
    H = m.embedding(Xt2_padded)            # (d, Lmax, B)
    cond = m.time_embed(reshape(t, 1, B))  # (cond_dim, B)
    for blk in m.blocks
        H = blk(H, cond, m.rope[1:Lmax], 0)
    end
    # exclude START for real positions
    H_real = H[:, 2:end, :]                # (d, n, B)
    sub_logits = m.head_sub(H_real)        # (K, n, B)
    del_logits = m.head_del(H_real)        # (1, n, B)
    # Insertion logits from positions including the immortal START and the rightmost
    ins_logits = m.head_ins(H)             # (K, n+1, B)
    # Return raw logits; process transform and masking handled in loss/rollout
    return (sub = sub_logits, del = del_logits, ins = ins_logits)
end


# Inference demo
# This needs to be done with "gen".
function rollout(P, model::PIPModel, x0::DiscreteState, x1::DiscreteState; dt=0.02f0, maxlength = 300)
    tgrid = collect(Float32(0.0):dt:Float32(1.0))
    Xt = x0
    # Ensure x0 begins with a real start symbol '>' just like training x1
    if isempty(tensor(Xt)) || tensor(Xt)[1] != TOK2ID['>']
        Xt = DiscreteState(Xt.K, vcat([TOK2ID['>']], tensor(Xt)))
    end
    traj = Vector{Vector{Int}}(undef, length(tgrid))
    traj[1] = tensor(Xt)
    for k in 1:length(tgrid)-1
        s1, s2 = tgrid[k], tgrid[k+1]
        # Model expects raw tokens; START is prepended internally
        Xt_raw = reshape(tensor(Xt), :, 1)
        if length(Xt_raw) > maxlength
            @warn "maxlength reached"
            return break;
        end
        preds = model([Float32(s1)], Xt_raw)  # raw logits
        # Apply process transform and zero self-substitutions before stepping
        sub2 = Array(P.transform(preds.sub)[:, :, 1])  # (K, n)
        del2 = vec(Array(P.transform(preds.del))[1, :, 1])  # (n,)
        ins2 = Array(P.transform(preds.ins)[:, :, 1])  # (K, n+1)
        # Zero self substitutions using current tokens
        current_mask = onehotbatch(Xt_raw, 1:K)   # (K, n, 1)
        sub2 .= sub2 .* (1 .- Array(current_mask)[:, :, 1])
        guide = Flowfusion.Guide((sub = sub2, del = del2, ins = ins2))
        Xt = Flowfusion.step(P, Xt, guide, s1, s2)
        traj[k+1] = tensor(Xt)
        println("Xt: ", decode(tensor(Xt)))
    end
    return traj
end


function demo_samples(P, model::PIPModel; N=3)
    for _ in 1:N
        x0, x1 = sample_pair()
        traj = rollout(P, model, x0, x1; dt=0.01f0)
        @show size(traj), typeof(traj)
        println("x1: ", decode(tensor(x1)))
        println("final Xt: ", decode(traj[end]))
    end
end

# Model, optimiser
model = PIPModel(; d=128, num_heads=8, nlayers=6, rff_dim=128, K=K)
opt_state = Flux.setup(Optimisers.AdamW(1e-3), model);

# Train (short by default; adjust with env vars)
batch_size = parse(Int, get(ENV, "BATCH", "8"))
nepochs = parse(Int, get(ENV, "EPOCHS", "10"))
steps_per_epoch = parse(Int, get(ENV, "STEPS", "500"))

for epoch in 1:nepochs
    for step in 1:steps_per_epoch
        pairs = [sample_pair() for _ in 1:batch_size]
        x0s = [p[1] for p in pairs]
        x1s = [p[2] for p in pairs]
        ts = rand(Float64, batch_size)
        Xts = [Flowfusion.bridge(P, x0s[b], x1s[b], ts[b]) for b in 1:batch_size]
        guide_tgt = Flowfusion.Guide(P, Float32.(ts), Xts, x1s)
        # Batch current states once; use as loss masks and model input
        Xt_ms = Flowfusion.batch(Xts)
        Xt_raw = tensor(Flowfusion.unmask(Xt_ms))
        c = Float32.(Flowfusion.scalefloss(P, reshape(Float32.(ts), 1, :), 1))
        l, grad = Flux.withgradient(model) do m
            preds = m(Float32.(ts), Xt_raw)
            Flowfusion.floss(P, Xt_ms, preds, guide_tgt, c)
        end
        Flux.update!(opt_state, model, grad[1])
        if step % 20 == 0
            @info "train" epoch step loss=Float32(l) eta=eta[]
        end
    end
    demo_samples(P, model; N=3)
end
