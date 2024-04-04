using LinearAlgebra
using BandStruct
using BandStruct.BlockedBandColumnMatrices
using BandStruct.BandwidthInit
using OrthWeight.GivensWeightMatrices
using OrthWeight.BasicTypes
using Random
using Test

function test_validate_ranks(
  E;
  l_or_t,
  m,
  n,
  lower_blocks,
  upper_blocks,
  lower_ranks,
  upper_ranks,
  tol_r = 1e-12
)

  gw = GivensWeight(
    E,
    l_or_t,
    MersenneTwister(),
    m,
    n;
    upper_rank_max = maximum(upper_ranks),
    lower_rank_max = maximum(lower_ranks),
    upper_ranks = upper_ranks,
    lower_ranks = lower_ranks,
    upper_blocks = upper_blocks,
    max_num_upper_blocks = 2 * length(upper_blocks),
    lower_blocks = lower_blocks,
    max_num_lower_blocks = 2 * length(lower_blocks),
    max_num_upper_rots=4, 
    max_num_lower_rots=4, 
  )
  
  upper_ranks =
    constrain_upper_ranks(m, n, blocks = upper_blocks, ranks = upper_ranks)

  lower_ranks =
    constrain_lower_ranks(m, n, blocks = lower_blocks, ranks = lower_ranks)

  A = Matrix(gw)

  @testset "Upper block ranks $m×$n, $l_or_t." begin
    uflag = -1
    for l ∈ 1:length(upper_blocks)
      ranges = upper_block_ranges(gw.b, l)
      ru = rank(view(A, ranges...), tol_r)
      ru == upper_ranks[l] || begin 
        uflag = l
        println()
        println("Failure:")
        println("In upper block $l of size $(upper_blocks[l].mb)×$(n-upper_blocks[l].nb)")
        println("got rank $ru and expected $(upper_ranks[l]).")
        println("Leading block size was $(upper_blocks[l].mb)×$(upper_blocks[l].nb)")
        println()
        break
      end
    end
    @test uflag == -1
  end

  @testset "Lower block ranks $m×$n, $l_or_t." begin
    lflag = -1
    for l ∈ 1:length(lower_blocks)
      ranges = lower_block_ranges(gw.b, l)
      rl = rank(view(A, ranges...), tol_r)
      rl == lower_ranks[l] || (lflag = l; break)
    end
    @test lflag == -1
  end
end

function run_givens_weight_rank_tests()

  tol_r = 1e-12

  m = 60
  n = 50

  lower_blocks = givens_block_sizes([
    10 20 30 40
    10 20 25 37
  ])

  upper_blocks = givens_block_sizes([
    10 20 30 40
    10 20 25 37
  ])

  upper_ranks = Consts(length(upper_blocks), 5)
  lower_ranks = Consts(length(lower_blocks), 5)

  test_validate_ranks(
    Float64,
    l_or_t = TrailingDecomp(),
    m = m,
    n = n,
    upper_blocks = upper_blocks,
    lower_blocks = lower_blocks,
    upper_ranks = upper_ranks,
    lower_ranks = lower_ranks,
  )

  test_validate_ranks(
    Float64,
    l_or_t = LeadingDecomp(),
    m = m,
    n = n,
    upper_blocks = upper_blocks,
    lower_blocks = lower_blocks,
    upper_ranks = upper_ranks,
    lower_ranks = lower_ranks,
  )

  m = 60
  n = 50

  E = Float64

  lower_blocks = givens_block_sizes([
    10 20 30 40
    10 20 25 37
  ])

  upper_blocks = givens_block_sizes([
    10 20 30 40
    10 20 25 37
  ])

  upper_ranks = [10, 5, 7, 8]
  lower_ranks = [10, 5, 7, 8]
  
  m = 60; n = 50
  lower_blocks = givens_block_sizes([
    10 20 30 40
    10 20 25 37
  ])

  upper_blocks = givens_block_sizes([
    10 20 30 40
    10 20 25 37
  ])

  upper_ranks = [10, 5, 7, 8]
  lower_ranks = [10, 5, 7, 8]

  upper_ranks =
    constrain_upper_ranks(m, n, blocks = upper_blocks, ranks = upper_ranks)

  lower_ranks =
    constrain_lower_ranks(m, n, blocks = lower_blocks, ranks = lower_ranks)

  l_or_t = TrailingDecomp()

  gw = GivensWeight(
    E,
    l_or_t,
    MersenneTwister(),
    m,
    n;
    upper_rank_max = maximum(upper_ranks),
    lower_rank_max = maximum(lower_ranks),
    upper_ranks = upper_ranks,
    lower_ranks = lower_ranks,
    upper_blocks = upper_blocks,
    max_num_upper_blocks = 2 * length(upper_blocks),
    lower_blocks = lower_blocks,
    max_num_lower_blocks = 2 * length(lower_blocks),
    max_num_upper_rots=4, 
    max_num_lower_rots=4, 
  )


  test_validate_ranks(
    E,
    l_or_t = l_or_t,
    m = m,
    n = n,
    upper_blocks = upper_blocks,
    lower_blocks = lower_blocks,
    upper_ranks = upper_ranks,
    lower_ranks = lower_ranks,
  )

  l_or_t = LeadingDecomp()

  test_validate_ranks(
    E,
    l_or_t = l_or_t,
    m = m,
    n = n,
    upper_blocks = upper_blocks,
    lower_blocks = lower_blocks,
    upper_ranks = upper_ranks,
    lower_ranks = lower_ranks,
  )
  return nothing
end

run_givens_weight_rank_tests()
