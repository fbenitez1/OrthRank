@safetestset "BandStruct Multiply Tests" begin
  using BandStruct
  using Random
  using LinearAlgebra
  using Test

  @testset "$E" for
    E ∈ [ Float64,
          Complex{Float64} ]

    (bc0, bbc0) =
      BandStruct.standard_test_case(E, upper_rank_max = 3, lower_rank_max = 2)

    mx_bc0 = Matrix(bc0)
    bc = copy(bc0)
    bbc = copy(bbc0)
    mx_bc = copy(mx_bc0)
    m, n = size(bc)
    C = ones(E, m, 2)
    c = ones(E, m)
    B = ones(E, n, 2)
    b = ones(E, n)
    tol = 1e-13
    α = 2.1
    β = 3.3


    @testset "mul!, 5 parameter" begin
      mul!(C, bc, B, α, β)
      @test opnorm(C - (β*ones(E, m, 2) + α*mx_bc*B), Inf)/opnorm(C, Inf) <= tol
    end

    @testset "mul!, 3 parameter" begin
      mul!(C, bc, B)
      @test opnorm(C - mx_bc*B, Inf)/opnorm(C, Inf) <= tol
    end

    @testset "mul!, 5 parameter, vector" begin
      mul!(c, bc, b, α, β)
      @test norm(c - (β*ones(E, m) + α*mx_bc*b), Inf)/norm(c, Inf) <= tol
    end

    @testset "mul!, 3 parameter, vector" begin
      mul!(c, bc, b)
      @test norm(c - mx_bc*b, Inf)/norm(c, Inf) <= tol
    end

    @testset "*, vector" begin
      y = mx_bc*b
      @test norm(bc*b - y, Inf)/norm(y, Inf) <= tol
    end

    @testset "*, matrix" begin
      Y = mx_bc*B
      @test opnorm(bc*B - Y, Inf)/opnorm(Y, Inf) <= tol
    end
  end
end
nothing
