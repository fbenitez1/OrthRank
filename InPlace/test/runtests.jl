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
      @test norm(b0-a*b) == 0.0
    end

    @testset "Right apply!" begin
      a0 = copy(a)
      a0 ⊛ Linear(b)
      @test norm(a0-a*b) == 0.0
    end

    @testset "Left apply_inv!" begin
      b0 = copy(b)
      Linear(a) ⊘ b0
      @test norm(b0-a\b) == 0.0
    end

    @testset "Right apply_inv!" begin
      a0 = copy(a)
      a0 ⊘ Linear(b)
      @test norm(a0-a/b) == 0.0
    end

  end
end
nothing
