using SafeTestsets

@safetestset "General-to-General transformation tests" begin

  using InPlace
  using Random
  using LinearAlgebra
  using Test

  @testset "$E" for
    E ∈ [ Float64,
          Complex{Float64},
          ]

    a = randn(E, 3, 3)
    b = randn(E, 3, 3)

    @testset "Left apply!" begin
      b0 = copy(b)
      Linear(a) ⊛ b0
      @test iszero(norm(b0-a*b))
    end

    @testset "Right apply!" begin
      a0 = copy(a)
      a0 ⊛ Linear(b)
      @test iszero(norm(a0-a*b))
    end

    @testset "Left apply_inv!" begin
      b0 = copy(b)
      Linear(a) ⊘ b0
      @test iszero(norm(b0-a\b))
    end

    @testset "Right apply_inv!" begin
      a0 = copy(a)
      a0 ⊘ Linear(b)
      @test iszero(norm(a0-a/b))
    end

  end
end

using InPlace
using InPlace.Precompile: run_all
using JET: @test_opt, @test_call
using Test

@testset "JET Tests" begin

  @testset "JET run_all opt" begin
    @test_opt target_modules = (InPlace,) run_all()
  end

  @testset "JET run_all call" begin
    @test_call target_modules = (InPlace,) run_all()
  end
end

