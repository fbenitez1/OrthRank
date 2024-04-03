using SafeTestsets
@safetestset "QR Factorization" begin
  include("test_QR.jl")
  m=100
  n=100
  num_blocks=10
  upper_rank_max=4
  lower_rank_max=4
  tol = 1e-12
  test_QR(m,n,num_blocks,upper_rank_max,lower_rank_max,tol)
end

@safetestset "Backward Substitution" begin
  include("test_backward_substitution.jl")
  m=100
  n = 100
  num_blocks=10
  upper_rank_max=4
  lower_rank_max=4
  tol = 1e-12
  test_backward_substitution(m,n,num_blocks,upper_rank_max,lower_rank_max,tol)
end