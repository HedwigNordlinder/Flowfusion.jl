
# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
survival(μ,t) = exp(-μ*t)
pow_sub(α,t)  = exp(-α*t)

"""
    rand_other_excl2(rng, k, a, b) -> Int

Sample uniformly from {1,…,k} excluding {a,b}.
Assumes 1 ≤ a,b ≤ k and a ≠ b.
"""
function rand_other_excl2(rng::AbstractRNG, k::Int, a::Int, b::Int)
  x = rand(rng, 1:k-2)
  x >= min(a,b) && (x += 1)
  x >= max(a,b) && (x += 1)
  return x
end


"""
    rand_other_excl1(rng, k, a) -> Int

Sample uniformly from {1,…,k} excluding {a}.
Assumes 1 ≤ a ≤ k.
"""
function rand_other_excl1(rng::AbstractRNG, k::Int, a::Int)
  x = rand(rng, 1:k-1)
  x >= a && (x += 1)
  return x
end

# ─────────────────────────────────────────────────────────────────────────────
# Two-branch kernels specialized to UniformDiscrete
# ─────────────────────────────────────────────────────────────────────────────
"""
    KernelsUD

Container for per-step posterior proportions used by the two-branch alignment DP:
- pA::T: probability of an A-step (x0-only) per letter, aggregated over tokens.
- pB::T: probability of a B-step (x1-only) per letter, aggregated over tokens.
- pR_diag::T: probability of an R-step when x0[i] == x1[j].
- pR_off::T: probability of an R-step when x0[i] != x1[j].
These are normalized so that, at any (i,j), the three out-going transitions sum appropriately.
"""
struct KernelsUD{T}
  pA::T
  pB::T
  pR_diag::T
  pR_off::T
end

"""
    PrecompUD

Precomputed branch- and time-specific constants at time t:
- t, s = t, 1 - t
- s1 = exp(-μ t), s2 = exp(-μ (1 - t))
- pow_t = exp(-α t), pow_s = exp(-α (1 - t))
- P1_diag, P1_off: substitution kernel on left branch (length t)
- P2_diag, P2_off: substitution kernel on right branch (length 1 - t)
- Iins_t, Iins_s: per-token insertion mass on left/right branches
"""
struct PrecompUD{T}
  t::T
  s::T               # 1 - t
  s1::T              # e^{-μ t}
  s2::T              # e^{-μ (1-t)}
  pow_t::T           # e^{-α t}
  pow_s::T           # e^{-α (1-t)}
  P1_diag::T; P1_off::T
  P2_diag::T; P2_off::T
  Iins_t::T          # λ * ((1 - e^{-μ t})/μ) / K
  Iins_s::T          # λ * ((1 - e^{-μ (1-t)})/μ) / K
end

"""
    make_precomp(p, t) -> PrecompUD

Build `PrecompUD` with survival, substitution kernels and per-token insertion masses
for the left (length t) and right (length 1 - t) branches.
"""
function make_precomp(p::UniformDiscretePoissonIndelProcess{T}, t::T) where T
  s  = one(T) - t
  s1 = survival(p.μ, t)
  s2 = survival(p.μ, s)
  powt = pow_sub(p.α, t)
  pows = pow_sub(p.α, s)
  P1_diag = powt + (1 - powt)/p.k
  P1_off  = (1 - powt)/p.k
  P2_diag = pows + (1 - pows)/p.k
  P2_off  = (1 - pows)/p.k
  Iins_t  = p.λ * ((1 - s1)/p.μ) / p.k
  Iins_s  = p.λ * ((1 - s2)/p.μ) / p.k
  PrecompUD(t, s, s1, s2, powt, pows, P1_diag, P1_off, P2_diag, P2_off, Iins_t, Iins_s)
end

"""
    make_kernels(p, C) -> KernelsUD

Build the two-branch event proportions (A, B, R) used by the alignment DP:
- Aggregates over token identities for A/B and splits R into diagonal vs off-diagonal cases.
- Normalizes by total mass Z so the three transitions at a grid cell (i,j) are comparable.
"""
function make_kernels(p::UniformDiscretePoissonIndelProcess{T}, C::PrecompUD{T}) where T
  K = p.k
  # eA,eB per-letter (before normalization)
  eA_per = C.s1*(1 - C.s2)/K + p.λ*((1 - C.s1)/p.μ)/K          # independent of α
  eB_per = C.s2*(1 - C.s1)/K + p.λ*((1 - C.s2)/p.μ)/K
  eA_tot = K*eA_per
  eB_tot = K*eB_per
  # eR for a==b and a!=b (before normalization)
  s12 = C.s1*C.s2
  eR_same = (1/K) * s12 * (C.P1_diag*C.P2_diag + (K-1)*C.P1_off*C.P2_off)
  eR_diff = (1/K) * s12 * (C.P1_diag*C.P2_off  + C.P1_off*C.P2_diag + (K-2)*C.P1_off*C.P2_off)
  sum_eR = K*eR_same + K*(K-1)*eR_diff
  Z = sum_eR + eA_tot + eB_tot
  KernelsUD(eA_per/Z, eB_per/Z, eR_same/Z, eR_diff/Z)
end

# ─────────────────────────────────────────────────────────────────────────────
# Forward DP and backward sample using KernelsUD
# ─────────────────────────────────────────────────────────────────────────────
"""
    Event

Alignment event in the two-branch DP:
- typ::Symbol: one of :A (x0-only), :B (x1-only), :R (paired).
- i::Int, j::Int: 1-based indices into x0 and x1 at the step (0 used for the “other side” of an indel).
"""
struct Event
  typ::Symbol
  i::Int
  j::Int
end


"""
    sample_alignment_ud(rng, A, B, KUD) -> (events::Vector{Event}, h::T)

Forward dynamic program followed by backward sampling of an alignment path between sequences A and B:
- Uses `KUD::KernelsUD` to weight A/B/R steps, with R split by token equality.
- Returns the sampled sequence of `Event`s and the total path mass `h = F[n+1, m+1]`.
"""
function sample_alignment_ud(rng::AbstractRNG,
                             A::Vector{Int}, B::Vector{Int},
                             KUD::KernelsUD{T}) where T
  n, m = length(A), length(B)
  F = zeros(T, n+1, m+1); F[1,1] = one(T)

  # forward DP
  for i in 0:n, j in 0:m
    f = F[i+1,j+1]
    if i < n
      F[i+2,j+1] += f * KUD.pA
    end
    if j < m
      F[i+1,j+2] += f * KUD.pB
    end
    if i < n && j < m
      pr = (A[i+1] == B[j+1]) ? KUD.pR_diag : KUD.pR_off
      F[i+2,j+2] += f * pr
    end
  end

  # backward sampling with explicit bounds guards
  events = Vector{Event}()
  i, j = n, m
  while i>0 || j>0
    wA = zero(T)
    if i > 0
      wA = F[i, j+1] * KUD.pA
    end

    wB = zero(T)
    if j > 0
      wB = F[i+1, j] * KUD.pB
    end

    wR = zero(T)
    if i > 0 && j > 0
      wR = F[i, j] * ((A[i] == B[j]) ? KUD.pR_diag : KUD.pR_off)
    end

    wTot = wA + wB + wR
    # If numerical underflow makes wTot==0, prefer any valid move deterministically
    if !(wTot > 0)
      if i > 0 && j > 0
        push!(events, Event(:R, i, j)); i -= 1; j -= 1
      elseif i > 0
        push!(events, Event(:A, i, 0)); i -= 1
      else
        push!(events, Event(:B, 0, j)); j -= 1
      end
      continue
    end

    u = rand(rng) * float(wTot)
    if u < wA
      push!(events, Event(:A, i, 0));  i -= 1
    elseif u < wA + wB
      push!(events, Event(:B, 0, j));  j -= 1
    else
      push!(events, Event(:R, i, j));  i -= 1;  j -= 1
    end
  end

  reverse!(events)
  return events, F[n+1,m+1]
end

# ─────────────────────────────────────────────────────────────────────────────
# Expand visible + invisible ⇒ X_t (ancestral r sampling in O(1))
# ─────────────────────────────────────────────────────────────────────────────
"""
    sample_Xt_ud(rng, p, x0, x1, C) -> (Xt::Vector{Int}, events::Vector{Event})

Samples an ancestral sequence X_t at time t given endpoints x0 (at 0) and x1 (at 1),
under the Uniform Discrete PIP bridge:
- Builds `KernelsUD` from `p` and `C::PrecompUD` at time t, samples an alignment path,
  then samples ancestral tokens r for each event as:
    - :R: proportional to P1(r→x0[i]) * P2(r→x1[j]).
    - :A or :B: mixture of “present at t” vs “inserted after t” on that branch; if present, r|descendant via P1/P2.
- Finally, adds “ghost” insertions in X_t: Poisson((λ/μ) (1 - s1) (1 - s2)) many, uniform placement and token.

Returns the sampled `Xt` and the alignment `events`.
"""
function sample_Xt_ud(rng::AbstractRNG, p::UniformDiscretePoissonIndelProcess{T},
                      x0::Vector{Int}, x1::Vector{Int}, C::PrecompUD{T}) where T
  KUD = make_kernels(p, C)
  events, _ = sample_alignment_ud(rng, x0, x1, KUD)

  Xt = Int[]
  K = p.k

  for ev in events
    if ev.typ === :R
      a = x0[ev.i]; b = x1[ev.j]
      if a == b
        w_a   = C.P1_diag * C.P2_diag
        w_oth = C.P1_off  * C.P2_off
        tot = w_a + (K-1)*w_oth
        if rand(rng) < w_a / tot
          push!(Xt, a)
        else
          push!(Xt, rand_other_excl1(rng, K, a))
        end
      else
        w_a   = C.P1_diag * C.P2_off
        w_b   = C.P1_off  * C.P2_diag
        w_oth = C.P1_off  * C.P2_off
        tot = w_a + w_b + (K-2)*w_oth
        u = rand(rng)*tot
        if u < w_a
          push!(Xt, a)
        elseif u < w_a + w_b
          push!(Xt, b)
        else
          push!(Xt, rand_other_excl2(rng, K, a, b))
        end
      end
    elseif ev.typ === :A
      a = x0[ev.i]
      # present-at-t vs inserted-after-t
      pre = C.s1*(1 - C.s2) / K
      ins = (p.λ/p.μ) * (1 - C.s1) / K
      if rand(rng) < pre/(pre+ins)
        # r=a vs other
        w_a   = C.P1_diag
        w_oth = C.P1_off
        tot = w_a + (K-1)*w_oth
        if rand(rng) < w_a/tot
          push!(Xt, a)
        else
          push!(Xt, rand_other_excl1(rng, K, a))
        end
      end
    else # :B
      b = x1[ev.j]
      pre = C.s2*(1 - C.s1) / K
      ins = (p.λ/p.μ) * (1 - C.s2) / K
      if rand(rng) < pre/(pre+ins)
        w_b   = C.P2_diag
        w_oth = C.P2_off
        tot = w_b + (K-1)*w_oth
        if rand(rng) < w_b/tot
          push!(Xt, b)
        else
          push!(Xt, rand_other_excl1(rng, K, b))
        end
      end
    end
  end

  # ghosts: (λ/μ) * (1 - s1) * (1 - s2); insert uniformly among |Xt|+1 slots
  Λghost = (p.λ/p.μ) * (1 - C.s1) * (1 - C.s2)
  Kghost = rand(Poisson(Λghost))
  for _ in 1:Kghost
    r   = rand(rng, 1:p.k)
    pos = rand(rng, 0:length(Xt))
    insert!(Xt, pos + 1, r)
  end

  return Xt, events
end

#Default RNG
sample_Xt_ud(P, x0, x1, C) = sample_Xt_ud(Random.default_rng(), P, x0, x1, C)

# ─────────────────────────────────────────────────────────────────────────────
# Single-branch DP specialized to UniformDiscrete (no K dependence)
# ─────────────────────────────────────────────────────────────────────────────
"""
    BranchUD

Right-branch (length s = 1 - t) constants for single-branch DP:
- surv::T: e^{-μ s}
- Pdiag, Poff: substitution kernel on the right branch
- Iins::T: per-token insertion mass on the right branch
- q_off::T: instantaneous off-diagonal rate α/K
"""
struct BranchUD{T}
  surv::T          # e^{-μ s}
  Pdiag::T         # pow_s + (1 - pow_s)/K
  Poff::T          # (1 - pow_s)/K
  Iins::T          # λ * ((1 - surv)/μ) / K
  q_off::T         # instantaneous substitution a→b, b≠a: α / K
end

"""
    make_branch_ud(p, C) -> BranchUD

Build right-branch constants from `UniformDiscretePoissonIndelProcess` and `PrecompUD` at time t.
"""
function make_branch_ud(p::UniformDiscretePoissonIndelProcess{T}, C::PrecompUD{T}) where T
  q_off = p.α / p.k
  BranchUD{T}(C.s2, C.P2_diag, C.P2_off, C.Iins_s, q_off)
end

"""
    branch_prob_ud!(F, root, leaf, B) -> h::T

Forward DP for P(leaf | root) on the right branch with uniform-discrete substitutions:
- root, leaf :: Vector{Int}
- F :: preallocated (n+1)×(m+1) matrix; overwritten
- B :: `BranchUD`
Returns the total probability h.
"""
function branch_prob_ud!(F::AbstractMatrix{T}, root::Vector{Int},
                         leaf::Vector{Int}, B::BranchUD{T}) where T
  n, m = length(root), length(leaf)
  F[1:n+1, 1:m+1] .= 0
  F[1,1] = one(T)
  for i in 0:n, j in 0:m
    f = F[i+1,j+1]
    if i < n
      F[i+2,j+1] += f*(1 - B.surv)
    end
    if j < m
      F[i+1,j+2] += f*B.Iins
    end
    if i < n && j < m
      P = (root[i+1] == leaf[j+1]) ? B.Pdiag : B.Poff
      F[i+2,j+2] += f*B.surv*P
    end
  end
  return F[n+1,m+1]
end

"""
    branch_prob_ud_delete!(F, root, delpos, leaf, B) -> h::T

Same as `branch_prob_ud!` but treats `root` with element at 1-based `delpos` removed
when aligning to `leaf`. Useful for marginalizing deletion-at-position proposals.
"""
function branch_prob_ud_delete!(F::AbstractMatrix{T}, root::Vector{Int}, delpos::Int,
                                leaf::Vector{Int}, B::BranchUD{T}) where T
  n0, m = length(root), length(leaf)
  n = n0 - 1
  F[1:n+1, 1:m+1] .= 0
  F[1,1] = one(T)
  for i in 0:n, j in 0:m
    f = F[i+1,j+1]
    if i < n
      F[i+2,j+1] += f*(1 - B.surv)
    end
    if j < m
      F[i+1,j+2] += f*B.Iins
    end
    if i < n && j < m
      idx = i + 1
      orig = (idx < delpos) ? idx : idx + 1
      P = (root[orig] == leaf[j+1]) ? B.Pdiag : B.Poff
      F[i+2,j+2] += f*B.surv*P
    end
  end
  return F[n+1,m+1]
end

"""
    branch_prob_ud_insert!(F, root, inspos, c, leaf, B) -> h::T

Same as `branch_prob_ud!` but treats `root` with a symbol `c` inserted after position `inspos`
(0-based gap index; i.e., `inspos=0` inserts before the first symbol, `inspos=n` after the last).
"""
function branch_prob_ud_insert!(F::AbstractMatrix{T}, root::Vector{Int}, inspos::Int, c::Int,
                                leaf::Vector{Int}, B::BranchUD{T}) where T
  n0, m = length(root), length(leaf)
  n = n0 + 1
  F[1:n+1, 1:m+1] .= 0
  F[1,1] = one(T)
  for i in 0:n, j in 0:m
    f = F[i+1,j+1]
    if i < n
      F[i+2,j+1] += f*(1 - B.surv)
    end
    if j < m
      F[i+1,j+2] += f*B.Iins
    end
    if i < n && j < m
      idx = i + 1
      a = (idx == inspos + 1) ? c : ((idx <= inspos) ? root[idx] : root[idx-1])
      P = (a == leaf[j+1]) ? B.Pdiag : B.Poff
      F[i+2,j+2] += f*B.surv*P
    end
  end
  return F[n+1,m+1]
end

# ─────────────────────────────────────────────────────────────────────────────
# Doob rates grouped by match-vs-other classes (no O(K))
# ─────────────────────────────────────────────────────────────────────────────
"""
    doob_ud_grouped_fast(p, Xt, x1, t)

Grouped Doob h-transform rates for Uniform Discrete on the right branch (length 1 - t).
Inputs:
- p::UniformDiscretePoissonIndelProcess, Xt::Vector{Int} (current ancestral sequence at time t), x1::Vector{Int} (leaf), t::T

Returns NamedTuple:
- hcur::T: current likelihood h(Xt → x1 | branch)
- del::Vector{T}               (n,)            deletion rates at each position i
- sub_match::Matrix{T}         (U, n)          substitution rates to tokens that appear in x1
- sub_other::Vector{T}         (n,)            substitution rate aggregated over “other” tokens
- sub_match_tokens::Vector{Int}(U,)            token IDs for the rows of `sub_match`
- sub_other_count::Vector{Int} (n,)            multiplicity of “other” class at each i
- ins_match::Matrix{T}         (U, n+1)        insertion rates for tokens in x1 at each gap
- ins_other::Vector{T}         (n+1,)          insertion rate aggregated over “other” tokens
- ins_other_count::Int                         multiplicity of “other” insertion class (K - U)

All rates are instantaneous hazards based on inside–outside ratios in log-space.
"""
function doob_ud_grouped_fast(p::UniformDiscretePoissonIndelProcess{T}, Xt::Vector{Int}, x1::Vector{Int}, t::T) where T

    local function logaddexp(a::T, b::T) where T
        if a == -Inf; return b; end
        if b == -Inf; return a; end
        if b > a; a, b = b, a; end
        return a + log1p(exp(b - a))
    end

    C = make_precomp(p, t)
    n, m, K = length(Xt), length(x1), p.k

    # Precompute logs of constants
    log_s2  = log(C.s2)
    log_del = log1p(-C.s2)                  # log(1 - s2)
    log_ins = log(p.λ) + log1p(-C.s2) - log(p.μ) - log(K)

    # Forward DP (log)
    logF = fill(-Inf, n+1, m+1); logF[1,1] = zero(T)
    for i in 0:n, j in 0:m
        f = logF[i+1, j+1]; f == -Inf && continue
        if i < n
            logF[i+2, j+1] = logaddexp(logF[i+2, j+1], f + log_del)
        end
        if j < m
            logF[i+1, j+2] = logaddexp(logF[i+1, j+2], f + log_ins)
        end
        if i < n && j < m
            P = (Xt[i+1] == x1[j+1]) ? C.P2_diag : C.P2_off
            logF[i+2, j+2] = logaddexp(logF[i+2, j+2], f + log_s2 + log(P))
        end
    end
    loghcur = logF[n+1, m+1]

    # Backward DP (log)
    logB = fill(-Inf, n+1, m+1); logB[n+1, m+1] = zero(T)
    for i in n:-1:0, j in m:-1:0
        (i == n && j == m) && continue
        acc = -Inf
        if i < n
            acc = logaddexp(acc, log_del + logB[i+2, j+1])
        end
        if j < m
            acc = logaddexp(acc, log_ins + logB[i+1, j+2])
        end
        if i < n && j < m
            P = (Xt[i+1] == x1[j+1]) ? C.P2_diag : C.P2_off
            acc = logaddexp(acc, log_s2 + log(P) + logB[i+2, j+2])
        end
        logB[i+1, j+1] = acc
    end

    # Token set in x1
    seen = Dict{Int,Int}(); toks = Int[]
    for b in x1
        haskey(seen,b) || (seen[b] = length(seen)+1; push!(toks,b))
    end
    U = length(toks)

    # Substitutions (inside–outside ratios)
    ΔP = C.P2_diag - C.P2_off
    q_off = p.α / K

    sub_match = zeros(T, U, n)
    sub_other = zeros(T, n)
    sub_other_count = Vector{Int}(undef, n)

    for i in 1:n
        # per-token accumulator over leaf positions, in log-space
        tmp = fill(-Inf, U)
        for j in 1:m
            idx = get(seen, x1[j], 0)
            if idx != 0
                val = logF[i, j] + log_s2 + logB[i+1, j+1]
                tmp[idx] = logaddexp(tmp[idx], val)
            end
        end
        # convert to ratios
        r_tok = similar(tmp)
        for u in 1:U
            r_tok[u] = tmp[u] == -Inf ? zero(T) : exp(tmp[u] - loghcur)
        end
        idx_a = get(seen, Xt[i], 0)
        r_a = (idx_a == 0 || tmp[idx_a] == -Inf) ? zero(T) : exp(tmp[idx_a] - loghcur)

        for u in 1:U
            ratio = max(zero(T), 1 + ΔP * (r_tok[u] - r_a))
            sub_match[u, i] = q_off * ratio
        end
        other_mult = K - 1 - (haskey(seen, Xt[i]) ? U-1 : U)
        sub_other_count[i] = max(other_mult, 0)
        base_ratio = max(zero(T), 1 - ΔP * r_a)
        sub_other[i] = q_off * base_ratio * sub_other_count[i]
    end

    # Deletions (ratios)
    del = zeros(T, n)
    for i in 1:n
        acc = -Inf
        for j in 0:m
            acc = logaddexp(acc, logF[i, j+1] + log_del + logB[i+1, j+1])
        end
        r = acc == -Inf ? zero(T) : exp(acc - loghcur)
        del[i] = p.μ * r
    end

    # Insertions (grouped): fix indexing for match branch: F[s+1,j] * B[s+1,j+1]
    D_ratio  = zeros(T, n+1)      # deletion-of-insert branch ratio
    M_ratio  = zeros(T, U, n+1)   # match-of-insert per token ratio
    for s in 0:n
        # deletion branch sum_j F[s,j] * B[s,j]
        accD = -Inf
        for j in 0:m
            accD = logaddexp(accD, logF[s+1, j+1] + logB[s+1, j+1])
        end
        D_ratio[s+1] = accD == -Inf ? zero(T) : exp(accD - loghcur)

        # match branch sum over j with correct indexing (no out-of-bounds at s=n)
        for j in 1:m
            idx = get(seen, x1[j], 0)
            if idx != 0
                val = logF[s+1, j] + logB[s+1, j+1]   # note: B[s+1, j+1], not s+2
                M_ratio[idx, s+1] += exp(val - loghcur)
            end
        end
    end
    S_ratio = vec(sum(M_ratio; dims=1))  # (n+1,)

    ins_match = zeros(T, U, n+1)
    ins_other = zeros(T, n+1)
    ins_other_count = K - U
    q_ins = p.λ / K

    for s in 0:n
        base_ratio = (1 - C.s2) * D_ratio[s+1] + C.s2 * S_ratio[s+1] * C.P2_off
        base_ratio = max(zero(T), base_ratio)
        ins_other[s+1] = q_ins * base_ratio * max(ins_other_count, 0)

        for (u, tok) in enumerate(toks)
            r_m = M_ratio[u, s+1]
            val_ratio = (1 - C.s2) * D_ratio[s+1] + C.s2 * (r_m * C.P2_diag + (S_ratio[s+1] - r_m) * C.P2_off)
            ins_match[u, s+1] = q_ins * max(zero(T), val_ratio)
        end
    end

    return (hcur = exp(loghcur),
            del = del,
            sub_match = sub_match,
            sub_other = sub_other,
            sub_match_tokens = toks,
            sub_other_count = sub_other_count,
            ins_match = ins_match,
            ins_other = ins_other,
            ins_other_count = ins_other_count)
end


"""
    doob_ud_full_tensors_fast(p, Xt, x1, t)

Full-tensor Doob h-transform rates for Uniform Discrete on the right branch (length 1 - t).
Inputs:
- p::UniformDiscretePoissonIndelProcess, Xt::Vector{Int}, x1::Vector{Int}, t::T

Returns NamedTuple:
- sub::Array{T,2}  (K, n): substitution hazards to each token at each position (self-substitution zeroed)
- del::Vector{T}   (n,):   deletion hazards per position
- ins::Array{T,2}  (K, n+1): insertion hazards to each token per gap
- hcur::T: current likelihood

Internally uses log-space forward-backward and token-first storage for cache efficiency.
"""
function doob_ud_full_tensors_fast(p::UniformDiscretePoissonIndelProcess{T}, Xt::Vector{Int}, x1::Vector{Int}, t::T) where T
    # log-space inside–outside and full tensors with token-first layout:
    # Returns:
    #   sub :: (K, n), del :: (n,), ins :: (K, n+1), hcur
    local function logaddexp(a::T, b::T) where T
        if a == -Inf; return b; end
        if b == -Inf; return a; end
        if b > a; a, b = b, a; end
        return a + log1p(exp(b - a))
    end

    C = make_precomp(p, t)
    n, m, K = length(Xt), length(x1), p.k

    log_s2  = log(C.s2)
    log_del = log1p(-C.s2)
    log_ins = log(p.λ) + log1p(-C.s2) - log(p.μ) - log(K)

    # Forward DP (log)
    logF = fill(-Inf, n+1, m+1); logF[1,1] = zero(T)
    for i in 0:n, j in 0:m
        f = logF[i+1, j+1]; f == -Inf && continue
        if i < n
            logF[i+2, j+1] = logaddexp(logF[i+2, j+1], f + log_del)
        end
        if j < m
            logF[i+1, j+2] = logaddexp(logF[i+1, j+2], f + log_ins)
        end
        if i < n && j < m
            P = (Xt[i+1] == x1[j+1]) ? C.P2_diag : C.P2_off
            logF[i+2, j+2] = logaddexp(logF[i+2, j+2], f + log_s2 + log(P))
        end
    end
    loghcur = logF[n+1, m+1]

    # Backward DP (log)
    logB = fill(-Inf, n+1, m+1); logB[n+1, m+1] = zero(T)
    for i in n:-1:0, j in m:-1:0
        (i == n && j == m) && continue
        acc = -Inf
        if i < n
            acc = logaddexp(acc, log_del + logB[i+2, j+1])
        end
        if j < m
            acc = logaddexp(acc, log_ins + logB[i+1, j+2])
        end
        if i < n && j < m
            P = (Xt[i+1] == x1[j+1]) ? C.P2_diag : C.P2_off
            acc = logaddexp(acc, log_s2 + log(P) + logB[i+2, j+2])
        end
        logB[i+1, j+1] = acc
    end

    # Token set in x1
    seen = Dict{Int,Int}(); toks = Int[]
    for b in x1
        haskey(seen,b) || (seen[b] = length(seen)+1; push!(toks,b))
    end
    U = length(toks)

    ΔP   = C.P2_diag - C.P2_off
    q_off = p.α / K
    q_ins = p.λ / K

    # Substitutions: token-first (K, n)
    sub = zeros(T, K, n)
    for i in 1:n
        # per-token accumulator over leaf positions, in log-space
        tmp = fill(-Inf, U)
        for j in 1:m
            idx = get(seen, x1[j], 0)
            if idx != 0
                val = logF[i, j] + log_s2 + logB[i+1, j+1]
                tmp[idx] = logaddexp(tmp[idx], val)
            end
        end
        # convert to ratios
        r_tok = similar(tmp)
        for u in 1:U
            r_tok[u] = tmp[u] == -Inf ? zero(T) : exp(tmp[u] - loghcur)
        end
        idx_a = get(seen, Xt[i], 0)
        r_a = (idx_a == 0 || tmp[idx_a] == -Inf) ? zero(T) : exp(tmp[idx_a] - loghcur)

        # base fill (other tokens, including those not in x1)
        base_ratio = max(zero(T), 1 - ΔP * r_a)
        fill!(view(sub, :, i), q_off * base_ratio)
        # overwrite specific tokens in x1
        for (u, tok) in enumerate(toks)
            ratio = max(zero(T), 1 + ΔP * (r_tok[u] - r_a))
            sub[tok, i] = q_off * ratio
        end
        # no self-substitution
        sub[Xt[i], i] = zero(T)
    end

    # Deletions: (n,)
    del = zeros(T, n)
    for i in 1:n
        acc = -Inf
        for j in 0:m
            acc = logaddexp(acc, logF[i, j+1] + log_del + logB[i+1, j+1])
        end
        r = acc == -Inf ? zero(T) : exp(acc - loghcur)
        del[i] = p.μ * r
    end

    # Insertions: token-first (K, n+1); fix match indexing to B[s+1, j+1]
    D_ratio = zeros(T, n+1)           # deletion-of-insert ratio
    M_ratio = zeros(T, U, n+1)        # match-of-insert per token ratio
    for s in 0:n
        # deletion branch sum_j F[s,j] * B[s,j]
        accD = -Inf
        for j in 0:m
            accD = logaddexp(accD, logF[s+1, j+1] + logB[s+1, j+1])
        end
        D_ratio[s+1] = accD == -Inf ? zero(T) : exp(accD - loghcur)
        # match branch: sum over j with correct indexing
        for j in 1:m
            idx = get(seen, x1[j], 0)
            if idx != 0
                val = logF[s+1, j] + logB[s+1, j+1]   # note: B[s+1, j+1]
                M_ratio[idx, s+1] += exp(val - loghcur)
            end
        end
    end
    S_ratio = vec(sum(M_ratio; dims=1))  # (n+1,)

    ins = zeros(T, K, n+1)
    for s in 0:n
        base_ratio = (1 - C.s2) * D_ratio[s+1] + C.s2 * S_ratio[s+1] * C.P2_off
        base_ratio = max(zero(T), base_ratio)
        fill!(view(ins, :, s+1), q_ins * base_ratio)
        for (u, tok) in enumerate(toks)
            r_m = M_ratio[u, s+1]
            val_ratio = (1 - C.s2) * D_ratio[s+1] + C.s2 * (r_m * C.P2_diag + (S_ratio[s+1] - r_m) * C.P2_off)
            ins[tok, s+1] = q_ins * max(zero(T), val_ratio)
        end
    end

    return (sub = sub, del = del, ins = ins, hcur = exp(loghcur))
end

# ─────────────────────────────────────────────────────────────────────────────
# Fused front-end: sample + grouped Doob
# ─────────────────────────────────────────────────────────────────────────────
#=
"""
    sample_and_doob_ud(rng, p, x0, x1, t) -> (Xt, events, rates)

Convenience wrapper:
1) Build precomputation at time t, 2) sample `Xt` and alignment events,
3) compute Doob rates (currently full-tensor).
"""

function sample_and_doob_ud(rng::AbstractRNG, p::UniformDiscretePoissonIndelProcess{T},
                            x0::Vector{Int}, x1::Vector{Int}, t::T) where T
  @time C = make_precomp(p, t)
  @time Xt, events = sample_Xt_ud(rng, p, x0, x1, C)
  #@time rates = doob_ud_grouped_fast(p, Xt, x1, t)
  @time rates = doob_ud_full_tensors_fast(p, Xt, x1, t)
  return Xt, events, rates
end
=#

function bridge(p::UniformDiscretePoissonIndelProcess, x0::DiscreteState{<:AbstractArray{<:Signed}}, x1::DiscreteState{<:AbstractArray{<:Signed}}, t)
    if ndims(x0.state) != 1
        error("bridge for UniformDiscretePoissonIndelProcess only implemented for 1D DiscreteState")
    end
    C = make_precomp(p, t)
    flat_xt, _ = sample_Xt_ud(p, tensor(x0), tensor(x1), C)
    xt = DiscreteState(x0.K, flat_xt)
    return xt
end

#=
P = UniformDiscretePoissonIndelProcess(20)
x0 = DiscreteState(20, [1,2,3,4,5])
x1 = DiscreteState(20, [6,7,8,9,10])
t = 0.5
bridge(P, x0, x1, t)
=#