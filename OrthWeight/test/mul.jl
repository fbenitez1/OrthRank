@safetestset "Multiply tests" begin
  using LinearAlgebra
  using BandStruct
  using OrthWeight.GivensWeightMatrices
  using OrthWeight.BasicTypes
  using Random
  # using Test
  nb = 3

  lower_blocks = givens_block_sizes([
    1 3 5 7 9
    1 2 4 5 10
  ])

  upper_blocks = givens_block_sizes([
    1 3 5 7 9
    1 2 4 5 10
  ])

  upper_ranks = Consts(length(upper_blocks), 4)
  lower_ranks = Consts(length(lower_blocks), 4)
  rng = MersenneTwister(1234)

  for (m,n) in [(10, 12), (12, 10)]

    for E in [Float64, Complex{Float64}]
      for decomp in [LeadingDecomp(), TrailingDecomp()]
        gw = GivensWeight(
          E,
          decomp,
          rng,
          m,
          n;
          upper_rank_max = maximum(upper_ranks),
          lower_rank_max = maximum(lower_ranks),
          upper_ranks = upper_ranks,
          lower_ranks = lower_ranks,
          upper_blocks = upper_blocks,
          lower_blocks = lower_blocks,
          max_num_upper_rots=12, 
          max_num_lower_rots=12, 
        )
        A = Matrix(gw)
        B = randn(rng, E, n, nb)
        C0 = randn(rng, E, m, nb)
        alpha = rand(rng, E)
        beta = rand(rng, E)
        tol = 1e-13

        @testset "mul! 5 parameter test, $E, $decomp, $m × $n" begin
          C = copy(C0)
          mul!(C, gw, B, alpha, beta)
          @test opnorm(alpha * A*B + beta*C0 - C, Inf)/opnorm(C, Inf) <= tol
        end
        @testset "mul! 3 parameter test, $E, $decomp, $m × $n" begin
          C = copy(C0)
          mul!(C, gw, B)
          @test opnorm(A*B - C, Inf)/opnorm(C, Inf) <= tol
        end
        @testset "* test, $E, $decomp, $m × $n" begin
          @test opnorm(A*B - gw*B, Inf)/opnorm(A*B, Inf) <= tol
        end

        b = randn(rng, E, n)
        c0 = randn(rng, E, m)
        @testset "mul! 5 parameter vector test, $E, $decomp, $m × $n" begin
          c = copy(c0)
          mul!(c, gw, b, alpha, beta)
          @test norm(alpha * A*b + beta*c0 - c, Inf)/norm(c, Inf) <= tol
        end
        @testset "mul! 3 parameter vector test, $E, $decomp, $m × $n" begin
          c = copy(c0)
          mul!(c, gw, b)
          @test norm(A*b - c, Inf)/norm(c, Inf) <= tol
        end
        @testset "* test vector, $E, $decomp, $m × $n" begin
          @test norm(A*b - gw*b, Inf)/norm(A*b, Inf) <= tol
        end
      end
    end

  end
end
