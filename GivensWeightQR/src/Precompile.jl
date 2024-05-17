module Precompile

using LinearAlgebra
using InPlace
using Random
using OrthWeight
using BandStruct

include("../test/test_QR.jl")
include("../test/test_backslash_operator_vector.jl")
include("../test/test_backslash_operator_matrix.jl")

function run_all()
  m = 100
  n = 100
  block_gap = 1
  upper_rank_max = 4
  lower_rank_max = 4
  rng = MersenneTwister(1234)
  run_QR(rng,m,n,block_gap,upper_rank_max,lower_rank_max)
  run_backslash_operator_vector(rng,m,n,block_gap,upper_rank_max,lower_rank_max)
  run_backslash_operator_matrix(rng,m,n,block_gap,upper_rank_max,lower_rank_max)
end

end
