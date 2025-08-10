using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using Flowfusion, ForwardBackward
using Flux, Optimisers
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
const START_ID = TOK2ID['-']

encode(s::AbstractString) = [TOK2ID[c] for c in collect(s)]
decode(v::Vector{Int}) = join(ID2TOK[i] for i in v)

# Process
const K = length(AAs)
const P = UniformDiscretePoissonIndelProcess(K; lambda = 0.10, mu = 0.10, alpha = 0.10)

# Load real data and sample prefixes for x1
const _DATA = let
    lines = readlines(joinpath(@__DIR__, "abs.txt"))
    allowed = Set("ACDEFGHIKLMNPQRSTVWY")
    [String(filter(c -> c in allowed, d)) for d in lines if !occursin("X", d)]
end

function sample_pair()
    j = rand(1:length(_DATA))
    s = _DATA[j]
    N = rand(5:min(15, lastindex(s)))
    sN = s[begin:begin+N-1]
    base_id = TOK2ID['#']
    len0 = rand(2:5)
    x0 = DiscreteState(K, [base_id for _ in 1:len0])
    # Prepend explicit start marker token to X1 so model can learn sequence start
    x1 = DiscreteState(K, vcat([TOK2ID['>']], encode(sN), [TOK2ID['.']]))
    return x0, x1
end

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
    rope = RoPE(d ÷ max(num_heads, 1), 4096)
    return PIPModel(embedding, time_embed, blocks, head_sub, head_del, head_ins, rope, K)
end

function (m::PIPModel)(t::Vector{Float32}, Xt_raw::Matrix{Int})
    # Prepend immortal START inside the model to align positions and gaps
    n, B = size(Xt_raw)
    Xt2_padded = vcat(fill(START_ID, 1, B), Xt_raw)  # (n+1, B)
    Lmax = n + 1
    H = m.embedding(Xt2_padded)            # (d, Lmax, B)
    cond = m.time_embed(reshape(t, 1, B))  # (cond_dim, B)
    rmax = size(m.rope.cos, 2)
    useL = min(Lmax, rmax)
    for blk in m.blocks
        H = blk(H, cond, m.rope[1:useL], 0)
    end
    # exclude START for letter positions
    H_real = H[:, 2:end, :]                # (d, n, B)
    d, n, _ = size(H_real)
    # Heads keep all batch dims
    sub_logits = m.head_sub(H_real)        # (K, n, B)
    del_logits = m.head_del(H_real)        # (1, n, B)
    # Insertion logits from positions including the immortal START and the rightmost
    # This yields (K, Lmax, B) where Lmax = n+1, aligning gaps 0..n
    ins_logits = m.head_ins(H)                                      # (K, n+1, B)
    sub = NNlib.softplus(sub_logits)
    del = NNlib.softplus(del_logits)
    ins = NNlib.softplus(ins_logits)
    # leave token space unconstrained; only prevent self-substitution below
    # Zero self-substitution via one-hot mask of real tokens
    xt_real = Xt_raw
    current_mask = onehotbatch(xt_real, 1:m.K)   # (K, n, B)
    sub = sub .* (1 .- current_mask)
    return (sub = sub, del = del, ins = ins)
end

# Padded batch utilities
function build_padded_Xt2(Xts::Vector{<:DiscreteState}; start_id::Int)
    B = length(Xts)
    lens = length.(tensor.(Xts))
    nmax = maximum(lens)
    Lmax = nmax + 1
    Xt2 = fill(start_id, Lmax, B)
    lmask = falses(nmax, B)
    gapmask = falses(nmax + 1, B)
    for b in 1:B
        x = tensor(Xts[b])
        Xt2[2:(length(x)+1), b] .= x
        lmask[1:length(x), b] .= true
        gapmask[1:(length(x)+1), b] .= true
    end
    return Xt2, lmask, gapmask
end

# Ground-truth Doob targets
function compute_targets_batch(P, Xts::Vector{<:DiscreteState}, X1s::Vector{<:DiscreteState}, tvec::Vector{Float32})
    B = length(Xts)
    lens = length.(tensor.(Xts))
    nmax = maximum(lens)
    K = P.k
    sub = zeros(Float32, K, nmax, B)
    del = zeros(Float32, 1, nmax, B)
    ins = zeros(Float32, K, nmax + 1, B)
    lmask = falses(nmax, B)
    gapmask = falses(nmax + 1, B)
    for b in 1:B
        t = Float64(tvec[b])
        Xt = Xts[b]
        X1 = X1s[b]
        tgt = Zygote.@ignore Flowfusion.doob_ud_full_tensors_fast(P, tensor(Xt), tensor(X1), t)
        n = length(tensor(Xt))
        sub[:, 1:n, b] .= Float32.(tgt.sub)
        del[1, 1:n, b] .= Float32.(tgt.del)
        ins[:, 1:(n+1), b] .= Float32.(tgt.ins)
        lmask[1:n, b] .= true
        gapmask[1:(n+1), b] .= true
    end
    return (sub=sub, del=del, ins=ins, lmask=lmask, gapmask=gapmask)
end

# Positive Bregman divergence helper
pos_breg(y, μ; eps=1f-8) = y .* (log.(y .+ eps) .- log.(μ .+ eps)) .- y .+ μ

function rate_loss_from_preds(P, preds, tgts, lmask::AbstractArray{Bool,2}, gapmask::AbstractArray{Bool,2}, c)
    m_sub = reshape(lmask, 1, size(lmask,1), size(lmask,2))
    m_del = reshape(lmask, 1, size(lmask,1), size(lmask,2))
    m_ins = reshape(gapmask, 1, size(gapmask,1), size(gapmask,2))
    Lsub = Flowfusion.scaledmaskedmean(pos_breg(tgts.sub, preds.sub), c, m_sub)
    Ldel = Flowfusion.scaledmaskedmean(pos_breg(tgts.del, preds.del), c, m_del)
    Lins = Flowfusion.scaledmaskedmean(pos_breg(tgts.ins, preds.ins), c, m_ins)
    return Lsub + Ldel + Lins
end

# Diagnostics
function rate_means!(P, model::PIPModel, tvec::Vector{Float32}, Xts::Vector{<:DiscreteState}, X1s::Vector{<:DiscreteState})
    # Model now expects raw Xt without START padding
    Xt_raw = let B = length(Xts)
        nmax = maximum(length.(tensor.(Xts)))
        out = fill(START_ID, nmax, B)
        for b in 1:B
            x = tensor(Xts[b])
            out[1:length(x), b] .= x
        end
        out
    end
    lmask = falses(size(Xt_raw,1), size(Xt_raw,2))
    for b in 1:length(Xts)
        n = length(tensor(Xts[b]))
        lmask[1:n, b] .= true
    end
    gapmask = falses(size(Xt_raw,1)+1, size(Xt_raw,2))
    for b in 1:length(Xts)
        n = length(tensor(Xts[b]))
        gapmask[1:n+1, b] .= true
    end
    preds = model(tvec, Xt_raw)
    tgts = compute_targets_batch(P, Xts, X1s, tvec)
    m_sub = reshape(lmask, 1, size(lmask,1), size(lmask,2))
    m_del = reshape(lmask, 1, size(lmask,1), size(lmask,2))
    m_ins = reshape(gapmask, 1, size(gapmask,1), size(gapmask,2))
    Ktok = size(preds.sub, 1)
    ms = sum(preds.sub .* m_sub) / max(Ktok * sum(m_sub), 1)
    ts = sum(tgts.sub .* m_sub) / max(Ktok * sum(m_sub), 1)
    md = sum(preds.del .* m_del) / max(sum(m_del), 1)
    td = sum(tgts.del .* m_del) / max(sum(m_del), 1)
    mi = sum(preds.ins .* m_ins) / max(Ktok * sum(m_ins), 1)
    ti = sum(tgts.ins .* m_ins) / max(Ktok * sum(m_ins), 1)
    @info "means sub_pred=$(ms) sub_tgt=$(ts) | del_pred=$(md) del_tgt=$(td) | ins_pred=$(mi) ins_tgt=$(ti)"
end

# Model, optimiser
model = PIPModel(; d=128, num_heads=4, nlayers=2, rff_dim=64, K=K)
opt_state = Flux.setup(Optimisers.Adam(3e-4), model)

# Train (short by default; adjust with env vars)
batch_size = parse(Int, get(ENV, "BATCH", "8"))
nepochs = parse(Int, get(ENV, "EPOCHS", "1"))
steps_per_epoch = parse(Int, get(ENV, "STEPS", "20000"))

for epoch in 1:nepochs
    for step in 1:steps_per_epoch
        pairs = [sample_pair() for _ in 1:batch_size]
        x0s = [p[1] for p in pairs]
        x1s = [p[2] for p in pairs]
        ts = rand(Float64, batch_size)
        Xts = [Flowfusion.bridge(P, x0s[b], x1s[b], ts[b]) for b in 1:batch_size]
        # Build raw Xt batch and masks for model and loss
        Xt_raw = let B = length(Xts)
            nmax = maximum(length.(tensor.(Xts)))
            out = fill(START_ID, nmax, B)
            for b in 1:B
                x = tensor(Xts[b])
                out[1:length(x), b] .= x
            end
            out
        end
        lmask = falses(size(Xt_raw,1), size(Xt_raw,2))
        for b in 1:length(Xts)
            n = length(tensor(Xts[b]))
            lmask[1:n, b] .= true
        end
        gapmask = falses(size(Xt_raw,1)+1, size(Xt_raw,2))
        for b in 1:length(Xts)
            n = length(tensor(Xts[b]))
            gapmask[1:n+1, b] .= true
        end
        tgts = compute_targets_batch(P, Xts, x1s, Float32.(ts))
        c = Float32.(Flowfusion.scalefloss(P, reshape(Float32.(ts), 1, :), 1))
        l, grad = Flux.withgradient(model) do m
            preds = m(Float32.(ts), Xt_raw)
            rate_loss_from_preds(P, preds, tgts, lmask, gapmask, c)
        end
        Flux.update!(opt_state, model, grad[1])
        if step % 50 == 0
            @info "train" epoch step loss=Float32(l)
            rate_means!(P, model, Float32.(ts), Xts, x1s)
        end
    end
end

# Inference demo
function rollout(P, model::PIPModel, x0::DiscreteState, x1::DiscreteState; dt=0.02f0)
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
        rates = model([Float32(s1)], Xt_raw)
        # Collapse singleton batch dims for stepping API
        sub2 = Array(rates.sub[:, :, 1])
        del2 = vec(rates.del[1, :, 1])
        ins2 = Array(rates.ins[:, :, 1])
        guide = Flowfusion.Guide((sub = sub2, del = del2, ins = ins2))
        Xt = Flowfusion.step(P, Xt, guide, s1, s2)
        traj[k+1] = tensor(Xt)
    end
    return traj
end

function demo_samples(P, model::PIPModel; N=3)
    for _ in 1:N
        x0, x1 = sample_pair()
        traj = rollout(P, model, x0, x1; dt=0.01f0)
        println("x1: ", decode(tensor(x1)))
        println("final Xt: ", decode(traj[end]))
    end
end

demo_samples(P, model; N=3)


