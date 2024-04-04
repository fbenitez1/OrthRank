using Random
using OrthWeight
using BandStruct
using Householder
using InPlace
using Rotations
using LinearAlgebra
using GivensWeightQR
using Test

function test_backward_substitution(
  m::Int64,
  n::Int64,
  num_blocks::Int64,
  upper_rank_max::Int64,
  lower_rank_max::Int64,
  tol::Float64
  )
  upper_blocks, lower_blocks = random_blocks_generator(m,n,num_blocks)
  upper_ranks = Consts(num_blocks, upper_rank_max)
  lower_ranks = Consts(num_blocks, lower_rank_max)
  upper_ranks = constrain_upper_ranks(m, n, blocks = upper_blocks, ranks = upper_ranks)
  lower_ranks = constrain_lower_ranks(m, n, blocks = lower_blocks, ranks = lower_ranks)
  max_num_upper_rots = 2*lower_rank_max * (2*(n÷num_blocks - 1) + upper_rank_max ) + 2*(n÷num_blocks - 1) + upper_rank_max
  max_num_lower_rots = 2*(n÷num_blocks - 1) + lower_rank_max
  upper_rank_max = 2*(n÷num_blocks - 1) + upper_rank_max + lower_rank_max
  lower_rank_max = 2*lower_rank_max
  max_num_upper_rots = 2*lower_rank_max * (2*(n÷num_blocks - 1) + upper_rank_max ) + 2*(n÷num_blocks - 1) + upper_rank_max
  max_num_lower_rots = 2*(n÷num_blocks - 1) + lower_rank_max
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
  x_a = zeros(Float64, n , 1)
  b = randn(m, 1)
  c = copy(b)
  Q = Matrix(1.0I,m , m)
  #x_a = solve(gw1, b)
  #solve!(x_a, gw1, b)
  solve!(x_a, gw1, b, Q)
  @testset "||Ax - b||" begin
    @test norm((Q[1:m,1:n])'*(A*x_a - c), Inf) <= tol
  end
end
  