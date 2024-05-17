using SafeTestsets

@safetestset "QR Factorization, no overlap: Square case" begin
  include("test_QR.jl")
  m = 100
  n = 100
  block_gap = 1
  upper_rank_max = 4
  lower_rank_max = 4
  tol = 1e-12
  rng = MersenneTwister(1234)
  test_QR(rng,m,n,block_gap,upper_rank_max,lower_rank_max,tol)
end

@safetestset "QR Factorization, no overlap: Tall case" begin
  include("test_QR.jl")
  m = 150
  n = 100
  block_gap = 1
  upper_rank_max = 4
  lower_rank_max = 4
  tol = 1e-12
  rng = MersenneTwister(1234)
  test_QR(rng,m,n,block_gap,upper_rank_max,lower_rank_max,tol)
end

@safetestset "QR Factorization, no overlap: Wide case" begin
  include("test_QR.jl")
  m = 100
  n = 150
  block_gap = 2
  upper_rank_max = 4
  lower_rank_max = 4
  tol = 1e-12
  rng = MersenneTwister(1234)
  test_QR(rng, m, n, block_gap, upper_rank_max, lower_rank_max, tol)
end

@safetestset "Backslash Operator: Vector case" begin
  include("test_backslash_operator_vector.jl")
  m = 100
  n = 100
  block_gap = 1
  upper_rank_max = 4
  lower_rank_max = 4
  tol = 1e-12
  rng = MersenneTwister(1234)
  test_backslash_operator_vector(
    rng,
    m,
    n,
    block_gap,
    upper_rank_max,
    lower_rank_max,
    tol,
  )
end

@safetestset "Backslash Operator: Matrix case" begin
  include("test_backslash_operator_matrix.jl")
  m=100
  n = 100
  block_gap = 1
  upper_rank_max = 4
  lower_rank_max = 4
  tol = 1e-12
  rng = MersenneTwister(1234)
  test_backslash_operator_matrix(
    rng,
    m,
    n,
    block_gap,
    upper_rank_max,
    lower_rank_max,
    tol,
  )
end

@safetestset "Backslash Operator: Underdetermined system" begin
  include("test_backslash_operator_vector.jl")
  m=100
  n = 150
  block_gap = 1
  upper_rank_max = 4
  lower_rank_max = 4
  tol = 1e-12
  rng = MersenneTwister(1234)
  test_backslash_operator_vector(
    rng,
    m,
    n,
    block_gap,
    upper_rank_max,
    lower_rank_max,
    tol,
  )
end

@safetestset "Backslash Operator: Overdetermined system" begin
  include("test_backslash_operator_vector.jl")
  m = 150
  n = 100
  block_gap = 1
  upper_rank_max = 4
  lower_rank_max = 4
  tol = 1e-12
  rng = MersenneTwister(1234)
  test_backslash_operator_vector(
    rng,
    m,
    n,
    block_gap,
    upper_rank_max,
    lower_rank_max,
    tol,
  )
end
