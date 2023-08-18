# Test that the sweeps in constructing a random WY-weight
# decomposition really are sweeps of orthogonal WY transformations.
using LinearAlgebra
using BandStruct
using Householder
using OrthWeight
using Random
using InPlace
using Test

function test_wy_construction()

  tol = 1e-14

  lower_blocks_sweep = wy_block_sizes([
    2 4 5 7
    2 3 4 6
  ])

  upper_blocks_sweep = wy_block_sizes([
    1 3 4 6
    3 4 6 7
  ])

  wyw_sweep = WYWeight(
    Float64,
    SpanStep(),
    LeadingDecomp(),
    Random.default_rng(),
    8,
    7,
    upper_ranks = [2 for j ∈ 1:length(upper_blocks_sweep)],
    lower_ranks = [2 for j ∈ 1:length(lower_blocks_sweep)],
    upper_blocks = upper_blocks_sweep,
    lower_blocks = lower_blocks_sweep,
  )

  sweep = SweepForward(wyw_sweep.lowerWY)
  A = Matrix{Float64}(I, 8, 8)
  A ⊛ sweep
  A ⊘ sweep
  @testset "Real Right Sweep Orthogonality" begin
    @test norm(Matrix{Float64}(I, 8, 8) - A, Inf) <= tol
  end
  sweep = SweepForward(wyw_sweep.upperWY)
  A = Matrix{Float64}(I, 7, 7)
  sweep ⊛ A
  sweep ⊘ A
  @testset "Real Left Sweep Orthogonality" begin
    @test norm(Matrix{Float64}(I, 7, 7) - A, Inf) <= tol
  end
end
