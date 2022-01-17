@safetestset "Band QR Factorization" begin
  using BandStruct.BandColumnMatrices
  using BandStruct.BlockedBandColumnMatrices
  using BandStruct.BandRotations
  using BandStruct.BandHouseholder
  using BandStruct.Factor
  using Householder

  using Random
  using Rotations
  using InPlace
  using LinearAlgebra
  using BenchmarkTools

  tol = 1e-13
  @testset "$E" for
    E ∈ [ Float64,
          Complex{Float64} ]

    m0=100
    m=10000
    lbw=100
    ubw=100

    B = makeBForQR(E, m0, lbw, ubw, 1)
    B0=copy(B)
    bandQRB(B, lbw, ubw)

    @test norm(svdvals(Matrix(B)) - svdvals(Matrix(B0))) <= tol
    @test norm(tril(Matrix(B),-1)) <= tol

    println("Timing bandQRB")
    B = makeBForQR(E, m, lbw, ubw, 1)
    @time bandQRB(B, lbw, ubw)

    A = makeA(E, m0, lbw, ubw);
    A0=copy(A)
    bandQRA(A, lbw, ubw)

    @test norm(svdvals(A) - svdvals(A0)) <= tol

    @test norm(tril(A,-1)) <= tol

    println("Timing bandQRA")
    A = makeA(E, m, lbw, ubw);
    @time bandQRA(A, lbw, ubw)
  end
end

