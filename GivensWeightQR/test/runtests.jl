using SafeTestsets

@safetestset "QR Factorization" begin
  include("test_QR.jl")
  m = 100
  n = 100
  num_blocks=10
  upper_rank_max=4
  lower_rank_max=4
  tol = 1e-12
  test_QR(m,n,num_blocks,upper_rank_max,lower_rank_max,tol)
end

@safetestset "QR Factorization structure" begin
  include("test_qr_structure.jl")
  m=100
  n = 100
  num_blocks = 10
  upper_rank_max = 4
  lower_rank_max = 4
  tol = 1e-12
  test_qr_structure(m,n,num_blocks,upper_rank_max,lower_rank_max,tol)
end

@safetestset "ldiv! in place" begin
  include("test_ldiv_inplace.jl")
  m = 100
  n = 100
  num_blocks = 10
  upper_rank_max = 4
  lower_rank_max = 4
  tol = 1e-12
  test_ldiv_inplace(m,n,num_blocks,upper_rank_max,lower_rank_max,tol)
end