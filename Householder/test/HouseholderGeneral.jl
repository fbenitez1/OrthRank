@safetestset "Householder Times General" begin
  using InPlace
  using Householder
  using Random
  using LinearAlgebra
  @testset "Householder Times General Tests, $E, zero=$l, col/row=$k" for
    E ∈ [ Float64,
          Complex{Float64},
          ],

    l ∈ [1,2,3],
    k ∈ [1,2,3]

    tol = 1e-14
    m=3

    @testset "Left Multiplication Zero and Inverse" begin
      a=randn(E,m,m)
      a0 = copy(a)
      h = lhouseholder(copy(a[:,k]),l,0,m)

      h ⊘ a
      column_nonzero!(a,l,k)
      @test norm(svdvals(a) - svdvals(a0)) <= tol
      @test norm(a[1:(l - 1), k]) + norm(a[(l + 1):m, k]) == 0.0
      h ⊛ a
      @test norm(a-a0) <= tol
    end

    @testset "Right Multiplication Zero and Inverse" begin
      a=randn(E,m,m)
      a0 = copy(a)
      h = rhouseholder(copy(a[k,:]),l,0,m)

      a ⊛ h
      row_nonzero!(a,k,l)
      @test norm(svdvals(a) - svdvals(a0)) <= tol
      @test norm(a[k,1:(l - 1)]) + norm(a[k, (l + 1):m]) == 0.0
      a ⊘ h
      @test norm(a-a0) <= tol
    end

    @testset "Left Multiplication Adjoint Zero and Inverse" begin
      a=rand(E,m,m)
      a0 = copy(a)
      a=a'
      a0=a0'
      h = lhouseholder(copy(a[:,k]),l,0,m)
      h ⊘ a
      column_nonzero!(a,l,k)
      @test norm(svdvals(a) - svdvals(a0)) <= tol
      @test norm(a[1:(l - 1), k]) + norm(a[(l + 1):m, k]) == 0.0

      h ⊛ a
      @test norm(a-a0) <= tol
    end

    @testset "Right Multiplication Adjoint Zero and Inverse" begin
      a=rand(Float64,m,m) .- 0.5
      a0 = copy(a)
      a=a'
      a0=a0'
      h = rhouseholder(copy(a[k,:]),l,0,m)
      a ⊛ h
      row_nonzero!(a,k,l)
      @test norm(svdvals(a) - svdvals(a0)) <= tol
      @test norm(a[k,1:(l - 1)]) + norm(a[k, (l + 1):m]) == 0.0
      a ⊘ h
      @test norm(a-a0) <= tol
    end
  end
end
nothing
