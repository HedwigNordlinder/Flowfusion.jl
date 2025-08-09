using Flowfusion
using Test
using Manifolds
using ForwardBackward

@testset "Flowfusion.jl" begin

    @testset "Masking" begin
        siz = (5,6,7)
        XC() = ContinuousState(randn(5, siz...))
        XD() = DiscreteState(5, rand(1:5, siz...))
        MT = Torus(2) #Where the representation is a vector
        XT() = ManifoldState(MT, [rand(MT) for _ in zeros(siz...)])
        MR = SpecialOrthogonal(3) #Where the representation is a matrix
        XR() = ManifoldState(MR, [rand(MR) for _ in zeros(siz...)])
        XDL() = CategoricalLikelihood(rand(5, siz...))
        XGL() = GaussianLikelihood(randn(5, siz...), randn(5, siz...), zeros(siz...))

        for f in [XC, XD, XT, XR, XDL, XGL] 
            Xa = f()
            Xb = f()
            Xc = Flowfusion.mask(Xa, Xb)
            @test isapprox(tensor(Xa),tensor(Xc))
            @test typeof(Xc) == typeof(Xa)
            Xa = f()
            Xb = f()
            m = rand(Bool, siz...)
            XM = MaskedState(Xb, m, m)
            Xc = Flowfusion.mask(Xa, XM)

            @test typeof(Xc) == typeof(XM) #If you mask a regular State with a MaskedState, the result is a MaskedState.
            d = (tensor(Xb) .- tensor(Xc))
            @test isapprox(sum(d .* expand(.!m, ndims(d))),0)

            m = rand(Bool, siz...)
            Xa = MaskedState(f(), m, m)
            Xb = MaskedState(f(), m, m)
            Xc = Flowfusion.mask(Xa, Xb)
            @test typeof(Xc) == typeof(Xa)
            d = (tensor(Xb) .- tensor(Xc))
            @test isapprox(sum(d .* expand(.!m, ndims(d))),0)
        end
    end

    @testset "Bridge, step" begin

        siz = (5,6)
        XC() = ContinuousState(randn(5, siz...))
        XD() = DiscreteState(5, rand(1:5, siz...))
        MT = Torus(2)
        XT() = ManifoldState(MT, [rand(MT) for _ in zeros(siz...)])
        MR = SpecialOrthogonal(3)
        XR() = ManifoldState(MR, [rand(MR) for _ in zeros(siz...)])

        for (f,p) in [(XC, BrownianMotion()),
                    (XT, ManifoldProcess(1)),
                    (XR, ManifoldProcess(1)),
                    (XD, InterpolatingDiscreteFlow())]
            #bridge - propogates the mask
            Xa = f()
            Xb = f()
            m = rand(Bool, siz...)
            XM = MaskedState(Xb, m, m)
            Xt = Flowfusion.bridge(p, Xa, XM, 0.1)
            @test typeof(Xt) == typeof(XM)
            if !(p isa InterpolatingDiscreteFlow)
                @test isapprox(sum((tensor(Xt) .== tensor(Xb))), sum(.!m) * (length(tensor(Xb)) / length(m)))
            else
                @test sum((tensor(Xt) .== tensor(Xb))) >= sum(.!m) * (length(tensor(Xb)) / length(.!m))
            end

            #step - doesn't propogate the mask
            Xa = f()
            Xb = f()
            m = rand(Bool, siz...)
            XM = MaskedState(Xa, m, m)
            if !(p isa InterpolatingDiscreteFlow)
                Xt = Flowfusion.step(p, XM, Xa, 0.1, 0.1)
                @test isapprox(sum(tensor(Xt) .!= tensor(XM)), 0) #Because step size is zero
            else
                Xt = Flowfusion.step(p, XM, onehot(Xa), 0.1, 0.1)
                @test isapprox(sum(tensor(Xt) .!= tensor(XM)), 0) #Because step size is zero
            end
        end

    end

    @testset "PoissonIndelProcess" begin
        @testset "branch_prob_ud!: trivial cases with α=0 (identity substitutions)" begin
            rng = Flowfusion.MersenneTwister(0)
            K = 5
            λ = 0.7
            μ = 1.3
            α = 0.0             # Pdiag=1, Poff=0
        
            p = UniformDiscretePoissonIndelProcess(λ, μ, α, K)
            t = 0.25
            C = Flowfusion.make_precomp(p, t)
            B = Flowfusion.make_branch_ud(p, C)
        
            # helpers
            s2 = C.s2
            Iins = B.Iins
        
            # F matrices sized per problem
            F = zeros(Float64, 1, 1)
            @test Flowfusion.branch_prob_ud!(F, Int[], Int[], B) ≈ 1.0 atol=1e-14
        
            F = zeros(Float64, 2, 1) # (n+1)×(m+1) for n=1,m=0
            @test Flowfusion.branch_prob_ud!(F, [1], Int[], B) ≈ (1 - s2) atol=1e-12
        
            F = zeros(Float64, 1, 2) # n=0,m=1
            @test Flowfusion.branch_prob_ud!(F, Int[], [2], B) ≈ Iins atol=1e-12
        
            # root=[1], leaf=[2] (≠): only del+ins; two orders: delete-then-insert, insert-then-delete
            F = zeros(Float64, 2, 2)
            @test Flowfusion.branch_prob_ud!(F, [1], [2], B) ≈ 2*(1 - s2)*Iins atol=1e-12
        
            # root=[3], leaf=[3] (==): direct match + two del/ins orders
            F = zeros(Float64, 2, 2)
            @test Flowfusion.branch_prob_ud!(F, [3], [3], B) ≈ (s2 + 2*(1 - s2)*Iins) atol=1e-12
        end
        
        @testset "sample_alignment_ud: deterministic kernel" begin
            rng = Flowfusion.MersenneTwister(1)
            # Force only R moves when tokens match; no A/B, no off-diagonal R
            KUD = Flowfusion.KernelsUD{Float64}(0.0, 0.0, 1.0, 0.0)
        
            A = [1,2,3]
            B = [1,2,3]
            events, h = Flowfusion.sample_alignment_ud(rng, A, B, KUD)
            @test h ≈ 1.0 atol=1e-14
            @test all(ev.typ === :R for ev in events)
            @test length(events) == 3
            @test all(A[events[i].i] == B[events[i].j] == i for i in 1:3)
        
            # If sequences differ, forward mass should be zero
            A2 = [1,2,3]
            B2 = [1,2,4]
            events2, h2 = Flowfusion.sample_alignment_ud(rng, A2, B2, KUD)
            @test h2 == 0.0
        end
        
        @testset "Grouped vs full-tensor Doob rates agree" begin
            rng = Flowfusion.MersenneTwister(2)
            K = 7
            p = UniformDiscretePoissonIndelProcess(0.9, 0.8, 0.5, K)
            t = 0.4
        
            # small random case
            Xt = rand(rng, 1:K, 8)
            x1 = rand(rng, 1:K, 6)
        
            G = Flowfusion.doob_ud_grouped_fast(p, Xt, x1, t)
            F = Flowfusion.doob_ud_full_tensors_fast(p, Xt, x1, t)
        
            # Reconstruct full tensors from grouped
            K_, n = p.k, length(Xt)
            toks = G.sub_match_tokens
            sub_from_grouped = zeros(eltype(F.sub), K_, n)
            for i in 1:n
            per = (G.sub_other_count[i] > 0) ? G.sub_other[i] / G.sub_other_count[i] : zero(eltype(F.sub))
            @views sub_from_grouped[:, i] .= per
            @views sub_from_grouped[toks, i] .= G.sub_match[:, i]
            sub_from_grouped[Xt[i], i] = 0
            end
        
            ins_from_grouped = zeros(eltype(F.ins), K_, n+1)
            per_ins_other = (G.ins_other_count > 0) ? (G.ins_other ./ G.ins_other_count) : zero(eltype(F.ins))
            for s in 1:(n+1)
            @views ins_from_grouped[:, s] .= per_ins_other[s]
            @views ins_from_grouped[toks, s] .= G.ins_match[:, s]
            end
        
            @test isapprox(F.hcur, G.hcur; atol=1e-10, rtol=1e-10)
            @test maximum(abs.(F.del .- G.del)) ≤ 1e-8
            @test maximum(abs.(F.sub .- sub_from_grouped)) ≤ 1e-8
            @test maximum(abs.(F.ins .- ins_from_grouped)) ≤ 1e-8
        end
    end
end
