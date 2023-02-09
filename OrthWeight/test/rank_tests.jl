# Test that the sweeps in constructing a random WY-weight
# decomposition really are sweeps of orthogonal WY transformations.

using LinearAlgebra
using BandStruct
using Householder
using OrthWeight
using Random
using InPlace
using Test

function test_validate_ranks(E)

  tol_r = 1e-12

  # m = 60
  # n = 50

  # lower_blocks_r = block_sizes([
  #   10 20 30 40
  #   10 20 25 37
  # ])

  # upper_blocks_r = block_sizes([
  #   10 20 30 40
  #   10 20 25 37
  # ])

  m = 6
  n = 5

  lower_blocks_r = block_sizes([
    1 2 3
    1 2 4
  ])

  upper_blocks_r = block_sizes([
    1 2 3
    1 2 2
  ])

  r = 0

  wyw_r = WYWeight(
    E,
    SpanStep(),
    LeadingDecomp(),
    Random.default_rng(),
    m,
    n,
    upper_rank_max = r,
    lower_rank_max = r,
    upper_blocks = upper_blocks_r,
    lower_blocks = lower_blocks_r,
  )

  A = Matrix(wyw_r)

  @testset "Upper block ranks 60×50, rank 2, Leading." begin
    uflag = -1
    for l ∈ 1:length(upper_blocks_r)
      ranges = upper_block_ranges(wyw_r.b, l)
      ru = rank(view(A, ranges...), tol_r)
      ru == min(r, size_upper_block(wyw_r.b, l)...) || (uflag = l; break)
    end
    @test uflag == -1
  end

  @testset "Lower block ranks 60×50, rank 2, Leading." begin
    lflag = -1
    for l ∈ 1:length(lower_blocks_r)
      ranges = lower_block_ranges(wyw_r.b, l)
      rl = rank(view(A, ranges...), tol_r)
      rl == min(r, size_lower_block(wyw_r.b, l)...) || (lflag = l; break)
    end
    @test lflag == -1
  end
end
