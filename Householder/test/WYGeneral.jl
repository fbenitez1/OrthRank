@safetestset "WY Times General" begin
  using InPlace
  using Householder
  using Random
  using LinearAlgebra

  @testset "$E" for
    E ∈ [ Float64,
          Complex{Float64},
          ]

    @testset "WY Sweeps and Inverse Sweeps" begin
      tol = 5e-14
      max_num_hs = 2
      m = 10
      n = 10
      num_blocks = 2
      bs = m ÷ num_blocks
      A = randn(E, m, n)
      A0 = copy(A)
      wy1 = WYTrans(
        E,
        max_num_WY = num_blocks,
        max_WY_size = bs,
        work_size = n * max_num_hs,
        max_num_hs = max_num_hs,
      )
      for l ∈ 1:num_blocks
        resetWYBlock!(wy1, block = l, offset = (l - 1) * bs, sizeWY = bs)
        wy1.num_hs[l] = max_num_hs
      end

      rand!(wy1)
      SweepForward(wy1) ⊛ A
      SweepForward(wy1) ⊘ A
      @test norm(A - A0) <= tol

    end

    @testset "Small QR Factorization, Left and Right Accumulation of Q" begin
      tol = 5e-14
      max_num_hs = 3
      m = 10
      n = 10
      A = randn(E, m, n)
      A0 = copy(A)
      wy1 = WYTrans(
        E,
        max_WY_size = m,
        work_size = n * max_num_hs,
        max_num_hs = max_num_hs,
      )
      resetWYBlock!(wy1, offset = 0, sizeWY = m)
      wy2 = WYTrans(
        E,
        max_WY_size = m,
        work_size = n * max_num_hs,
        max_num_hs = max_num_hs,
      )
      resetWYBlock!(wy2, offset = 0, sizeWY = m)
      Iₘ = Matrix{E}(I, m, m)
      work = zeros(E, m)

      for j = 1:3
        h = lhouseholder(A[j:m, j], 1, j - 1, work)
        h ⊘ A
        (wy1, 1) ⊛ h
        h ⊘ (wy2, 1)
      end

      Q1 = copy(Iₘ)
      Q1 ⊛ (wy1, 1)
      Q2 = copy(Iₘ)
      Q2 ⊘ (wy2, 1)
      @test norm(Q1 * A - A0) <= tol
      @test norm(Q2 * A - A0) <= tol
    end

    @testset "WY QR Factorization Test, return Q, 1000×900" begin
      m = 1000
      n = 900
      tol = 5e-13
      A = randn(E, m, n)
      A0 = copy(A)
      A[:, :] = A0
      (Q, R) = qrWY(A)
      @test norm(Q * R - A0, Inf) <= tol
    end

    @testset "WY QR Factorization Test, return WY Sweep, 1000×900" begin
      m = 1000
      n = 900
      tol = 5e-13
      A = randn(E, m, n)
      A0 = copy(A)
      A[:, :] = A0
      (Q, R) = qrWYSweep(A)
      Q ⊛ R
      @test norm(R - A0, Inf) <= tol
    end

  end
end
nothing
