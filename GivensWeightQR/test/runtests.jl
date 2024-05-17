using SafeTestsets

@safetestset "QR Factorization tests" begin

  using Random
  using GivensWeightQR.Precompile: run_QR
  using LinearAlgebra

  m = 100
  n = 100
  block_gap = 1
  upper_rank_max = 4
  lower_rank_max = 4
  tol = 1e-12
  rng = MersenneTwister(1234)
  @testset "||A - QR||, no overlap, square case" begin
    A, Q, R = run_QR(rng, m, n, block_gap, upper_rank_max, lower_rank_max)
    @test norm(A - Q * R, Inf) <= tol
  end

  m = 150
  n = 100
  @testset "||A - QR||, no overlap, tall case" begin
    A, Q, R = run_QR(rng, m, n, block_gap, upper_rank_max, lower_rank_max)
    @test norm(A - Q * R, Inf) <= tol
  end

  m = 100
  n = 150
  block_gap = 2
  @testset "||A - QR||, no overlap, wide case" begin
    A, Q, R = run_QR(rng, m, n, block_gap, upper_rank_max, lower_rank_max)
    @test norm(A - Q * R, Inf) <= tol
  end

end

@safetestset "Backslash operator tests" begin

  using Random
  using GivensWeightQR.Precompile:
    run_backslash_operator_vector, run_backslash_operator_matrix
  using LinearAlgebra

  m = 100
  n = 100
  block_gap = 1
  upper_rank_max = 4
  lower_rank_max = 4
  tol = 1e-12
  rng = MersenneTwister(1234)

  A, x_a, c = run_backslash_operator_vector(
    rng,
    m,
    n,
    block_gap,
    upper_rank_max,
    lower_rank_max,
  )
  @testset "backslash vector, ||A'(Ax - b)||" begin
    @test norm(A' * (A * x_a - c), Inf) / (norm(A, Inf) * norm(x_a, Inf)) <= tol
  end

  A, x_a, c = run_backslash_operator_matrix(
    rng,
    m,
    n,
    block_gap,
    upper_rank_max,
    lower_rank_max,
  )
  @testset "backslash matrix, ||A'(Ax - b)||" begin
    @test norm(A' * (A * x_a - c), Inf) / (norm(A, Inf) * norm(x_a, Inf)) <= tol
  end

  m=100
  n = 150
  A, x_a, c = run_backslash_operator_vector(
    rng,
    m,
    n,
    block_gap,
    upper_rank_max,
    lower_rank_max,
  )
  @testset "backslash vector, underdetermined, ||A'(Ax - b)||" begin
    @test norm(A' * (A * x_a - c), Inf) / (norm(A, Inf) * norm(x_a, Inf)) <= tol
  end

  m=150
  n = 100
  A, x_a, c = run_backslash_operator_vector(
    rng,
    m,
    n,
    block_gap,
    upper_rank_max,
    lower_rank_max,
  )
  @testset "backslash vector, overdetermined, ||A'(Ax - b)||" begin
    @test norm(A' * (A * x_a - c), Inf) / (norm(A, Inf) * norm(x_a, Inf)) <= tol
  end

end


using Random
using GivensWeightQR
using GivensWeightQR.Precompile:
  run_QR, run_backslash_operator_vector, run_backslash_operator_matrix
using LinearAlgebra
using JET: @test_opt, @test_call
using Test

# @test_opt and @test_call don't seem to work in a safetestset...
@testset "JET Tests" begin

  @testset "JET runQR opt" begin
    @test_opt target_modules = (GivensWeightQR,) run_QR(
      MersenneTwister(1234), 1, 1, 1, 1, 1)
  end

  @testset "JET runQR call" begin
    @test_call target_modules = (GivensWeightQR,) run_QR(
      MersenneTwister(1234), 1, 1, 1, 1, 1)
  end

  @testset "JET run backslash vector opt" begin
    @test_opt target_modules = (GivensWeightQR,) run_backslash_operator_vector(
      MersenneTwister(1234), 1, 1, 1, 1, 1)
  end

  @testset "JET run backslash vector call" begin
    @test_call target_modules = (GivensWeightQR,) run_backslash_operator_vector(
      MersenneTwister(1234), 1, 1, 1, 1, 1)
  end

  @testset "JET run backslash matrix opt" begin
    @test_opt target_modules = (GivensWeightQR,) run_backslash_operator_matrix(
      MersenneTwister(1234), 1, 1, 1, 1, 1)
  end

  @testset "JET run backslash matrix call" begin
    @test_call target_modules = (GivensWeightQR,) run_backslash_operator_matrix(
      MersenneTwister(1234), 1, 1, 1, 1, 1)
  end
end
