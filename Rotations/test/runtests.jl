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

    @testset "Left Zero and Inverse, $E" begin
      A = randn(E, 3, 3)
      r = lgivens(A[1, 1], A[3, 1], (1, 3))
      A1 = copy(A)
      r ⊘ A1
      @test abs(A1[3, 1]) <= tol
      r ⊛ A1
      err = norm(A1 - A, Inf) / norm(A, Inf)
      @test err <= tol
    end

    @testset "Positive Real Left Zero and Inverse, $E" begin
      A = randn(E, 3, 3)
      r = lgivensPR(A[1, 1], A[3, 1], (1, 3))
      A1 = copy(A)
      r ⊘ A1
      @test abs(A1[3, 1]) <= tol
      @test real(A1[1,1]) >= 0.0
      @test abs(imag(A1[1,1])) <= tol
      r ⊛ A1
      err = norm(A1 - A, Inf) / norm(A, Inf)
      @test err <= tol
    end

    @testset "Right Zero and Inverse, $E" begin
      A = randn(E, 3, 3)
      r = rgivens(A[1, 1], A[1, 3], (1, 3))
      A1 = copy(A)
      A1 ⊛ r
      @test abs(A1[1,3]) <= tol

      A1 ⊘ r
      err = norm(A1 - A, Inf) / norm(A, Inf)
      @test err <= tol
    end

    @testset "Positive Real Right Zero and Inverse, $E" begin
      A = randn(E, 3, 3)
      r = rgivensPR(A[1, 1], A[1, 3], (1, 3))
      A1 = copy(A)
      A1 ⊛ r
      @test abs(A1[1,3]) <= tol
      @test real(A1[1,1]) >= 0.0
      @test abs(imag(A1[1,1])) <= tol
      A1 ⊘ r
      err = norm(A1 - A, Inf) / norm(A, Inf)
      @test err <= tol
    end

    @testset "Left Zero and Inverse, First Element, $E" begin
      A = randn(E, 3, 3)
      r = lgivens1(A[1, 1], A[3, 1], (1, 3))
      A1 = copy(A)
      r ⊘ A1
      @test abs(A1[1,1]) <= tol

      r ⊛ A1
      err = norm(A1 - A, Inf) / norm(A, Inf)
      @test err <= tol
    end

    @testset "Positive Real Left Zero and Inverse, First Element, $E" begin
      A = randn(E, 3, 3)
      r = lgivensPR1(A[1, 1], A[3, 1], (1, 3))
      A1 = copy(A)
      r ⊘ A1
      @test abs(A1[1,1]) <= tol
      @test real(A1[3,1]) >= 0.0
      @test abs(imag(A1[3,1])) <= tol
      r ⊛ A1
      err = norm(A1 - A, Inf) / norm(A, Inf)
      @test err <= tol
    end

    @testset "Right Zero and Inverse, First Element, $E" begin
      A = randn(E, 3, 3)
      r = rgivens1(A[1, 1], A[1, 3], (1, 3))
      A1 = copy(A)
      A1 ⊛ r
      @test abs(A1[1,1]) <= tol

      A1 ⊘ r
      err = norm(A1 - A, Inf) / norm(A, Inf)
      @test err <= tol
    end

    @testset "Positive Real Right Zero and Inverse, First Element, $E" begin
      A = randn(E, 3, 3)
      r = rgivensPR1(A[1, 1], A[1, 3], (1, 3))
      A1 = copy(A)
      A1 ⊛ r
      @test abs(A1[1,1]) <= tol
      @test real(A1[1,3]) >= 0.0
      @test abs(imag(A1[1,3])) <= tol
      A1 ⊘ r
      err = norm(A1 - A, Inf) / norm(A, Inf)
      @test err <= tol
    end

    @testset "Adjacent Left Zero and Inverse, $E" begin
      A = randn(E, 3, 3)
      r = lgivens(A[1, 1], A[2, 1], 1)
      A1 = copy(A)
      r ⊘ A1
      @test abs(A1[2,1]) <= tol
      
      r ⊛ A1
      err = norm(A1 - A, Inf) / norm(A, Inf)
      @test abs(err) <= tol
    end

    @testset "Adjacent Positive Real Left Zero and Inverse, $E" begin
      A = randn(E, 3, 3)
      r = lgivensPR(A[1, 1], A[2, 1], 1)
      A1 = copy(A)
      r ⊘ A1
      @test abs(A1[2,1]) <= tol
      @test real(A1[1,1]) >= 0.0
      @test abs(imag(A1[1,1])) <= tol
      r ⊛ A1
      err = norm(A1 - A, Inf) / norm(A, Inf)
      @test abs(err) <= tol
    end

    @testset "Adjacent Right Zero and Inverse, $E" begin
      A = randn(E, 3, 3)
      r = rgivens(A[1, 1], A[1, 2], 1)
      A1 = copy(A)
      A1 ⊛ r
      @test abs(A1[1,2]) <= tol

      A1 ⊘ r
      err = norm(A1 - A, Inf) / norm(A, Inf)
      @test err <= tol
    end

    @testset "Adjacent Positive Real Right Zero and Inverse, $E" begin
      A = randn(E, 3, 3)
      r = rgivensPR(A[1, 1], A[1, 2], 1)
      A1 = copy(A)
      A1 ⊛ r
      @test abs(A1[1,2]) <= tol
      @test real(A1[1,1]) >= 0.0
      @test abs(imag(A1[1,1])) <= tol

      A1 ⊘ r
      err = norm(A1 - A, Inf) / norm(A, Inf)
      @test err <= tol
    end

    @testset "Adjacent Left Zero and Inverse, First Element, $E" begin
      A = randn(E, 3, 3)
      r = lgivens1(A[1, 1], A[2, 1], 1)
      A1 = copy(A)
      r ⊘ A1
      @test abs(A1[1,1]) <= tol

      r ⊛ A1
      err = norm(A1 - A, Inf) / norm(A, Inf)
      @test err <= tol
    end

    @testset "Adjacent Pos. Real Left Zero and Inverse, First Element, $E" begin
      A = randn(E, 3, 3)
      r = lgivensPR1(A[1, 1], A[2, 1], 1)
      A1 = copy(A)
      r ⊘ A1
      @test abs(A1[1,1]) <= tol
      @test real(A1[2,1]) >= 0.0
      @test abs(imag(A1[2,1])) <= tol
      r ⊛ A1
      err = norm(A1 - A, Inf) / norm(A, Inf)
      @test err <= tol
    end

    @testset "Adjacent Right Zero and Inverse, First Element, $E" begin
      A = randn(E, 3, 3)
      r1 = rgivens1(A[1, 1], A[1, 2], 1)
      A1 = copy(A)
      A1 ⊛ r1
      @test abs(A1[1,1]) <= tol

      A1 ⊘ r1
      err1 = norm(A1 - A, Inf) / norm(A, Inf)
      @test err1 <= tol
    end

    @testset "Adjacent Pos. Real Right Zero and Inverse, First Element, $E" begin
      A = randn(E, 3, 3)
      r1 = rgivensPR1(A[1, 1], A[1, 2], 1)
      A1 = copy(A)
      A1 ⊛ r1
      @test abs(A1[1,1]) <= tol
      @test real(A1[1,2]) >= 0.0
      @test abs(imag(A1[1,2])) <= tol
      A1 ⊘ r1
      err1 = norm(A1 - A, Inf) / norm(A, Inf)
      @test err1 <= tol
    end
  end
end
nothing
