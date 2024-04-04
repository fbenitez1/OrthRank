using Random
using OrthWeight
using BandStruct
using Householder
using InPlace
using Rotations
using LinearAlgebra
using GivensWeightQR
using Test

function run_QR(
  m::Int64,
  n::Int64,
  num_blocks::Int64,
  upper_rank_max::Int64,
  lower_rank_max::Int64,
)
  upper_blocks, lower_blocks = random_blocks_generator(m, n, num_blocks)
  upper_ranks = Consts(num_blocks, upper_rank_max)
  lower_ranks = Consts(num_blocks, lower_rank_max)
  upper_ranks =
    constrain_upper_ranks(m, n, blocks = upper_blocks, ranks = upper_ranks)
  lower_ranks =
    constrain_lower_ranks(m, n, blocks = lower_blocks, ranks = lower_ranks)
  max_num_upper_rots =
    2 * lower_rank_max * (2 * (n ÷ num_blocks - 1) + upper_rank_max) +
    2 * (n ÷ num_blocks - 1) +
    upper_rank_max
  max_num_lower_rots = 2 * (n ÷ num_blocks - 1) + lower_rank_max
  upper_rank_max = 2 * (n ÷ num_blocks - 1) + upper_rank_max + lower_rank_max
  lower_rank_max = 2 * lower_rank_max
  max_num_upper_rots =
    2 * lower_rank_max * (2 * (n ÷ num_blocks - 1) + upper_rank_max) +
    2 * (n ÷ num_blocks - 1) +
    upper_rank_max
  max_num_lower_rots = 2 * (n ÷ num_blocks - 1) + lower_rank_max
  gw1 = GivensWeight(
    Float64,
    TrailingDecomp(),
    MersenneTwister(),
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
  Q = Matrix(1.0I, m, m)
  F = qr(gw1)
  create_Q!(Q, F)
  R = create_R(F)
  return A, Q, R
end

function test_QR(
  m::Int64,
  n::Int64,
  num_blocks::Int64,
  upper_rank_max::Int64,
  lower_rank_max::Int64,
  tol :: Float64
)
  A, Q, R = run_QR(m,n,num_blocks, upper_rank_max, lower_rank_max)
  @testset "||A - QR||" begin
    @test norm(A - Q * R, Inf) <= tol
  end
end
