using SafeTestsets

@safetestset "Givens Rotations" begin
  using Rotations.Givens
  using LinearAlgebra
  using Random
  using InPlace
  @testset "Rotation Tests, $E" verbose = true for
    E ∈ [ Float64,
          Complex{Float64},
          ]
    
    tol = 1e-15

    @testset "Nonadjacent Left Zero and Inverse, $E" begin
      A = randn(E, 3, 3)
      r = lgivens(A[1, 1], A[3, 1], 1, 3)
      A1 = copy(A)
      r ⊘ A1
      @test abs(A1[3, 1]) <= tol
      r ⊛ A1
      err = norm(A1 - A, Inf) / norm(A, Inf)
      @test err <= tol
    end

    @testset "Nonadjacent Right Zero and Inverse, $E" begin
      A = randn(E, 3, 3)
      r = rgivens(A[1, 1], A[1, 3], 1, 3)
      A1 = copy(A)
      A1 ⊛ r
      @test abs(A1[1,3]) <= tol

      A1 ⊘ r
      err = norm(A1 - A, Inf) / norm(A, Inf)
      @test err <= tol
    end

    @testset "Nonadjacent Left Zero and Inverse, First Element, $E" begin
      A = randn(E, 3, 3)
      r = lgivens1(A[1, 1], A[3, 1], 1, 3)
      A1 = copy(A)
      r ⊘ A1
      @test abs(A1[1,1]) <= tol

      r ⊛ A1
      err = norm(A1 - A, Inf) / norm(A, Inf)
      @test err <= tol
    end

    @testset "Nonadjacent Right Zero and Inverse, First Element, $E" begin
      A_rrz1 = randn(E, 3, 3)
      r_rrz1 = rgivens1(A_rrz1[1, 1], A_rrz1[1, 3], 1, 3)
      A1_rrz1 = copy(A_rrz1)
      A1_rrz1 ⊛ r_rrz1
      @test abs(A1_rrz1[1,1]) <= tol

      A1_rrz1 ⊘ r_rrz1
      err_rrz1 = norm(A1_rrz1 - A_rrz1, Inf) / norm(A_rrz1, Inf)
      @test err_rrz1 <= tol
    end

    @testset "Adjacent Left Zero and Inverse, $E" begin
      A_ralz = randn(E, 3, 3)
      r_ralz = lgivens(A_ralz[1, 1], A_ralz[2, 1], 1)
      A1_ralz = copy(A_ralz)
      r_ralz ⊘ A1_ralz
      @test abs(A1_ralz[2,1]) <= tol
      
      r_ralz ⊛ A1_ralz
      err_ralz = norm(A1_ralz - A_ralz, Inf) / norm(A_ralz, Inf)
      @test abs(err_ralz) <= tol
    end

    @testset "Adjacent Right Zero and Inverse, $E" begin
      A_rarz = randn(E, 3, 3)
      r_rarz = rgivens(A_rarz[1, 1], A_rarz[1, 2], 1)
      A1_rarz = copy(A_rarz)
      A1_rarz ⊛ r_rarz
      @test abs(A1_rarz[1,2]) <= tol

      A1_rarz ⊘ r_rarz
      err_rarz = norm(A1_rarz - A_rarz, Inf) / norm(A_rarz, Inf)
      @test err_rarz <= tol
    end

    @testset "Adjacent Left Zero and Inverse, First Element, $E" begin
      A_ralz1 = randn(E, 3, 3)
      r_ralz1 = lgivens1(A_ralz1[1, 1], A_ralz1[2, 1], 1)
      A1_ralz1 = copy(A_ralz1)
      r_ralz1 ⊘ A1_ralz1
      @test abs(A1_ralz1[1,1]) <= tol

      r_ralz1 ⊛ A1_ralz1
      err_ralz1 = norm(A1_ralz1 - A_ralz1, Inf) / norm(A_ralz1, Inf)
      @test err_ralz1 <= tol
    end

    @testset "Nonadjacent Right Zero and Inverse, First Element, $E" begin
      A_rarz1 = randn(E, 3, 3)
      r_rarz1 = rgivens1(A_rarz1[1, 1], A_rarz1[1, 2], 1)
      A1_rarz1 = copy(A_rarz1)
      A1_rarz1 ⊛ r_rarz1
      @test abs(A1_rarz1[1,1]) <= tol

      A1_rarz1 ⊘ r_rarz1
      err_rarz1 = norm(A1_rarz1 - A_rarz1, Inf) / norm(A_rarz1, Inf)
      @test err_rarz1 <= tol
    end
  end
end
nothing
