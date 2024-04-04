module Precompile

using LinearAlgebra
using InPlace
using Random
using OrthWeight
using BandStruct

include("../test/test_QR.jl")
include("../test/test_backward_substitution.jl")
include("../test/test_ldiv_inplace.jl")
include("../test/test_qr_structure.jl")

function run_all()
  m = 100
  n = 100
  num_blocks = 10
  upper_rank_max = 4
  lower_rank_max = 4
  run_QR(m, n, num_blocks, upper_rank_max, lower_rank_max)
  # run_backward_substitution(m, n, num_blocks, upper_rank_max, lower_rank_max)
  # test_qr_structure(m,n,num_blocks,upper_rank_max,lower_rank_max,tol)
  # test_ldiv_inplace(m,n,num_blocks,upper_rank_max,lower_rank_max,tol)
end

end
