using Random
using OrthWeight
using BandStruct
using Householder
using InPlace
using Rotations
using LinearAlgebra
using GivensWeightQR
using Test

function run_backward_substitution(
  rng::AbstractRNG,
  m::Int64,
  n::Int64,
  block_gap::Int64,
  upper_rank_max::Int64,
  lower_rank_max::Int64,
  )
  upper_blocks, lower_blocks = random_blocks_generator(rng,m, n, block_gap)
  num_blocks = length(upper_blocks)
  upper_ranks = Consts(length(upper_blocks), upper_rank_max)
  lower_ranks = Consts(length(upper_blocks), lower_rank_max)
  upper_ranks = constrain_upper_ranks(m, n, blocks = upper_blocks, ranks = upper_ranks)
  lower_ranks = constrain_lower_ranks(m, n, blocks = lower_blocks, ranks = lower_ranks)
  max_num_upper_rots = 
    div(block_gap + upper_rank_max,2,RoundUp)^2 +
    ((lower_rank_max)*(upper_rank_max + lower_rank_max - 1) + (block_gap - 1)*div(lower_rank_max*(lower_rank_max + 1),2))
    #Rotations needed by structure + rot needed to avoid fill-in (lrm^2 from previous block extended)
  max_num_lower_rots = (block_gap + lower_rank_max - 1) * lower_rank_max
  upper_rank_max = 2*block_gap + upper_rank_max + lower_rank_max 
  gw1 = GivensWeight(
    Float64,
    TrailingDecomp(),
    rng,
    m,
    n;
    upper_rank_max = upper_rank_max,
    lower_rank_max = lower_rank_max,
    upper_ranks = upper_ranks,
    lower_ranks = lower_ranks,
    upper_blocks = upper_blocks,
    max_num_upper_blocks = num_blocks,
    lower_blocks = lower_blocks,
    max_num_lower_blocks = num_blocks,
    max_num_upper_rots = max_num_upper_rots,
    max_num_lower_rots = max_num_lower_rots,
  )
  A = Matrix(gw1)
  x_a = zeros(Float64, n, 1)
  b = randn(m, 1)
  c = copy(b)
  Q = Matrix(1.0I, m, m)
  #x_a = solve(gw1, b)
  #solve!(x_a, gw1, b)
  solve!(x_a, gw1, b, Q)
  return Q, A, x_a, c
end

function test_backward_substitution(
  rng::AbstractRNG,
  m::Int64,
  n::Int64,
  block_gap::Int64,
  upper_rank_max::Int64,
  lower_rank_max::Int64,
  tol::Float64,
)

  Q, A, x_a, c =
    run_backward_substitution(rng, m, n, block_gap, upper_rank_max, lower_rank_max)
  @testset "||Ax - b||" begin
    @test norm((Q[1:m, 1:n])' * (A * x_a - c), Inf)/(norm(A,Inf)*norm(x_a,Inf)) <= tol
  end
end
