@safetestset "Band LQ Factorization" begin
  using BandStruct
  using Householder

  using Random
  using Rotations
  using InPlace
  using LinearAlgebra

  tol = 1e-12
  @testset "$E" for
    E ∈ [ Float64,
          Complex{Float64} ]

    # Tests
    m0=1000
    lbw=100
    ubw=100
    bs = 8

    B = makeBForLQ(
      E,
      m0,
      lbw,
      ubw,
      1
    )
    B0 = copy(B)

    lqBH(B, lbw, ubw)

    @test norm(svdvals(Matrix(B)) - svdvals(Matrix(B0))) <= tol
    @test norm(triu(Matrix(B),1)) <= tol

    # lqBWY
    B = copy(B0)
    lqBWY(B, lbw, ubw, block_size = bs)

    @test norm(svdvals(Matrix(B)) - svdvals(Matrix(B0))) <= tol
    @test norm(triu(Matrix(B),1)) <= tol

    B = copy(B0)
    (Qwy, B) = lqBWYSweep(B, lbw, ubw, block_size = bs)

    Q = Matrix{E}(I, m0, m0)
    Q ⊛ Qwy

    @test norm(svdvals(Matrix(B)) - svdvals(Matrix(B0))) <= tol
    @test norm(Matrix(B) - Matrix(B0) * Q) <= tol
    let
      B1 = Matrix(B0)
      B1 ⊛ Qwy
      @test norm(Matrix(B) - B1) <= tol
    end

    @test norm(triu(Matrix(B),1)) <= tol

    # Benchmarks

    # Tests
    m0=10000
    lbw=100
    ubw=100
    bs = 16

    B = makeBForLQ(
      E,
      m0,
      lbw,
      ubw,
      1
    )
    B0 = copy(B)

    # lqH
    println("Benchmarking lqH, $E:")
    @time lqBH(B, lbw, ubw)

    println("Benchmarking lqBWY, $E:")
    B=copy(B0);

    @time lqBWY(B, lbw, ubw, block_size=bs);
    B=copy(B0);
  end
end
