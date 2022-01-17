@safetestset "BandStruct Submatrix/View" begin
  using BandStruct.BandColumnMatrices
  using BandStruct.BlockedBandColumnMatrices
  using Random
  include("standard_test_case.jl")

  @testset "$E" for
    E ∈ [ Float64,
          Complex{Float64} ]

    (bc0, bbc0) = standard_test_case(E, upper_rank_max = 3, lower_rank_max = 2)

    mx_bc0 = Matrix(bc0)
    bc = copy(bc0)
    bbc = copy(bbc0)
    mx_bc = copy(mx_bc0)

    num_tests = 100

    function rand_range(bc :: AbstractBandColumn)
      (m,n)= size(bc)
      j1 = rand(1:m)
      j2 = rand(1:m)
      k1 = rand(1:n)
      k2 = rand(1:n)
      (UnitRange(j1,j2), UnitRange(k1,k2))
    end

    @testset "Submatrix Unblocked" for j ∈ 1:num_tests
      Random.seed!(j)
      (rows, cols) = rand_range(bc)
      @test Matrix(bc[rows, cols]) == mx_bc[rows, cols]
    end

    @testset "View Unblocked" for j ∈ 1:num_tests
      Random.seed!(j)
      (rows, cols) = rand_range(bc)
      @test Matrix(view(bc, rows, cols)) == mx_bc[rows, cols]
    end

    @testset "Submatrix Blocked" for j ∈ 1:num_tests
      Random.seed!(j)
      (rows, cols) = rand_range(bc)
      @test Matrix(bbc[rows, cols]) == mx_bc[rows, cols]
    end

    @testset "View Blocked" for j ∈ 1:num_tests
      Random.seed!(j)
      (rows, cols) = rand_range(bc)
      @test Matrix(view(bbc, rows, cols)) == mx_bc[rows, cols]
    end
  end
end
nothing
