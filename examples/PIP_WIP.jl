using Pkg
Pkg.activate(@__DIR__)
using Revise
#Pkg.instantiate()

using Flowfusion, ForwardBackward
using Flux, Optimisers, CannotWaitForTheseOptimisers
using Random
using NNlib
using OneHotArrays
using Onion
using RandomFeatureMaps
using Zygote

# Alphabet and encoding (include tokens used by the PIP implementation)
const AAs = collect("ACDEFGHIKLMNPQRSTVWY#-*")
const TOK2ID = Dict(c => i for (i, c) in enumerate(AAs))
const ID2TOK = Dict(v => k for (k, v) in TOK2ID)
LEFT_IMMORTAL_ID = TOK2ID['-']
RIGHT_IMMORTAL_ID = TOK2ID['*']
encode(s::AbstractString) = [TOK2ID[c] for c in collect(s)]
decode(v::Vector{Int}) = join(ID2TOK[i] for i in v)

# Process
K = length(AAs)-2 #Exclude left immortal token
P = UniformDiscretePoissonIndelProcess(K; lambda = 0.5f0, mu = 0.5f0, alpha = 0.2f0)

# Load real data and sample prefixes for x1
data = let
    lines = readlines(joinpath(@__DIR__, "abs.txt"))
    allowed = Set("ACDEFGHIKLMNPQRSTVWY")
    [String(filter(c -> c in allowed, d)) for d in lines if !occursin("X", d)]
end

#Default: truncated samples for quicker testing
function sample_pair(; lengthbound = Inf)
    x0 = DiscreteState(K, [TOK2ID['#'] for _ in 1:rand(1:60)])
    #x1 = DiscreteState(K, encode(rand(data)[1:rand(1:Int(min(lengthbound,end)))]))
    x1 = DiscreteState(K, encode(rand(data)))
    return x0, x1
end

struct PIPModel{L}
    layers::L
end

Flux.@layer PIPModel

function PIPModel(; d = 128, num_heads = 8, nlayers = 6, rff_dim = 128, cond_dim = 128, K::Int)
    embedding = Flux.Embedding(K => d) #We include the immortal tokens in the embedding
    time_embed = Flux.Chain(RandomFourierFeatures(1 => rff_dim, 1.0f0), Dense(rff_dim => cond_dim))
    blocks = [Onion.AdaTransformerBlock(d, cond_dim, num_heads) for _ in 1:nlayers]
    head_sub = Dense(d => K-2, bias = false) #We exclude the immortal tokens from the sub/ins heads
    head_del = Dense(d => 1, bias = false)
    head_ins = Dense(d => K-2, bias = false) #We exclude the immortal tokens from the sub/ins heads
    rope = RoPE(d ÷ num_heads, 4096)
    return PIPModel((;embedding, time_embed, blocks, head_sub, head_del, head_ins, rope, K))
end

#Need to add pad mask in here!
function (model::PIPModel)(t, Xt)
    m = model.layers
    n, B = size(tensor(Xt))
    pmask = Zygote.@ignore self_att_padding_mask(Flowfusion.getlmask(Xt))
    H = m.embedding(tensor(Xt))            # (d, Lmax, B)
    cond = m.time_embed(reshape(t, 1, B))  # (cond_dim, B)
    for blk in m.blocks
        H = blk(H, cond, m.rope[1:n], pmask)
    end
    # Insertion logits from positions including the immortal START and the rightmost
    ins_logits = m.head_ins(H[:, 1:end-1, :])             # (K, n, B)
    # Sub and delete logits from positions excluding the immortal START
    if size(H, 2) <= 1
        return (sub = similar(ins_logits, size(ins_logits,1), 0, B), del = similar(ins_logits, 1, 0, B), ins = ins_logits)
    end
    H_real = H[:, 2:end-1, :]
    sub_logits = m.head_sub(H_real)        # (K, n-1, B)
    del_logits = m.head_del(H_real)        # (1, n-1, B)
    return (sub = sub_logits, del = del_logits, ins = ins_logits)
end


# Model, optimiser
#model = PIPModel(; d=128, num_heads=8, nlayers=6, rff_dim=128, K=K+2)
model = PIPModel(; d=128, num_heads=8, nlayers=6, rff_dim=128, K=K+2)
eta = 1e-2
opt_state = Flux.setup(CannotWaitForTheseOptimisers.Muon(eta=eta), model);
#opt_state = Flux.setup(Optimisers.Adam(1e-4), model);

#Wrapper for generation:
function m(t,Xt)
    println(decode(tensor(Xt)))
    return model([t], Flowfusion.batch([Flowfusion.prefix(Xt, LEFT_IMMORTAL_ID, suffix = RIGHT_IMMORTAL_ID)]))
end

batch_size = 16
for epoch in 1:100
    for step in 1:500
        xpairs = [sample_pair() for _ in 1:batch_size]
        x0s = [p[1] for p in xpairs]
        x1s = [p[2] for p in xpairs]
        ts = rand(Float32, batch_size)
        Xts = [Flowfusion.bridge(P, x0s[b], x1s[b], ts[b]) for b in 1:batch_size]
        guide_tgt = Flowfusion.Guide(P, ts, Xts, x1s)
        # Batch current states once; use as loss masks and model input
        Xt_model = Flowfusion.batch(Flowfusion.prefix.(Xts, LEFT_IMMORTAL_ID, suffix = RIGHT_IMMORTAL_ID))
        Xt_loss = Flowfusion.batch(Flowfusion.prefix.(Xts, LEFT_IMMORTAL_ID))
        l, grad = Flux.withgradient(model) do m
            preds = m(ts, Xt_model)
            Flowfusion.floss(P, Xt_loss, preds, guide_tgt, scalefloss(P,ts,1))
        end
        Flux.update!(opt_state, model, grad[1])
        if step % 20 == 0
            eta = eta * 0.99975
            Flux.adjust!(opt_state, eta)
            @info "train" epoch step loss=Float32(l) eta=eta
        end
        
    end
    try
        gen(P, sample_pair()[1], m, 0f0:0.01f0:1f0)
        gen(P, sample_pair()[1], m, 0f0:0.01f0:1f0)
        gen(P, sample_pair()[1], m, 0f0:0.01f0:1f0)
    catch e
    end
end

decode(tensor(gen(P, sample_pair()[1], m, 0f0:0.01f0:1f0)))

@time draws = [tensor(gen(P, sample_pair()[1], m, 0f0:0.01f0:1f0)) for _ in 1:10000];
@time truedraws = [tensor(sample_pair()[2]) for _ in 1:250000];

using StatsBase

cm_samp = proportionmap(decode.(draws))
cm_true = proportionmap(decode.(truedraws))
sort(collect(cm_samp), by = last, rev = true)
sort(collect(cm_true), by = last, rev = true)

shared = intersect(union(decode.(draws)), union(decode.(truedraws)))
scatter([sqrt(cm_samp[s]) for s in shared], [sqrt(cm_true[s]) for s in shared], label = :none, msw = 0, color = :black, alpha = 0.5, xlabel = "sqrt(sampled)", ylabel = "sqrt(true)", size = (600,600))
