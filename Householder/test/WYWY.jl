@safetestset "WY Times WY" begin
  using InPlace
  using Householder
  using Random
  using LinearAlgebra

  @testset "$E" verbose=true for
    E ∈ [ Float64,
          Complex{Float64},
          ]


    tol = 5e-14
    max_num_hs = 2
    m = 18

    range1 = 3:13
    size1 = length(range1)
    offset1 = first(range1) - 1
    num_hs1 = 3
    max_num_wy1 = 3
    l1 = 3

    range2 = 5:10
    offset2 = first(range2) - 1
    size2 = length(range2)
    num_hs2 = 2
    max_num_wy2 = 4
    l2 = 2

    max_num_hs1 = num_hs1 + 2 * num_hs2
    max_num_hs2 = num_hs2

    wy1 = WYTrans(
      E,
      max_num_WY = max_num_wy1,
      max_WY_size = m,
      work_size = m * max_num_hs1,
      max_num_hs = max_num_hs1,
    )

    wy2 = WYTrans(
      E,
      max_num_WY = max_num_wy2,
      max_WY_size = m,
      work_size = m * max_num_hs1,
      max_num_hs = max_num_hs2,
    )
    resetWYBlock!(wy1, block = l1, offset = offset1, sizeWY = size1)
    selectWY!(wy1, l1)
    wy1.num_hs[l1] = num_hs1
    rand!(wy1)

    resetWYBlock!(wy2, block = l2, offset = offset2, sizeWY = size2)
    selectWY!(wy2, l2)
    wy1.num_hs[l2] = num_hs2
    rand!(wy2)

    @testset "Identity Right WYWY Test" begin
      Iₘ = Matrix{E}(I, m, m)
      A = copy(Iₘ)
      B = copy(Iₘ)
      wy1c = deepcopy(wy1)

      wy1c ⊘ A
      wy2 ⊘ A

      wy1c ⊛ Linear(wy2)
      wy1c ⊘ B

      @test norm(A - B) ≈ 0.0 atol = tol
    end

    @testset "Random Matrix Right WYWY Test" begin
      A = randn(E, m, m)
      B = copy(A)
      wy1c = deepcopy(wy1)

      wy1c ⊘ A
      wy2 ⊘ A

      wy1c ⊛ Linear(wy2)
      wy1c ⊘ B

      @test norm(A - B) ≈ 0.0 atol = tol
    end

    @testset "Random Matrix WYWY Right Inverse Test" begin
      A = randn(E, m, m)
      B = copy(A)
      wy1c = deepcopy(wy1)

      wy1c ⊛ Linear(wy2)
      wy1c ⊘ Linear(wy2)
      wy1c ⊘ A
      wy1 ⊘ B

      @test norm(A - B) ≈ 0.0 atol = tol
    end

    @testset "Identity Left WYWY Test" begin
      Iₘ = Matrix{E}(I, m, m)
      A = copy(Iₘ)
      B = copy(Iₘ)
      wy1c = deepcopy(wy1)

      wy2 ⊘ A
      wy1c ⊘ A

      Linear(wy2) ⊛ wy1c
      wy1c ⊘ B

      @test norm(A - B) ≈ 0.0 atol = tol
    end

    @testset "Random Matrix Left WYWY Test" begin
      A = randn(E, m, m)
      B = copy(A)
      wy1c = deepcopy(wy1)

      wy2 ⊘ A
      wy1c ⊘ A

      Linear(wy2) ⊛ wy1c
      wy1c ⊘ B

      @test norm(A - B) ≈ 0.0 atol = tol
    end

    @testset "Random Matrix WYWY Left Inverse Test" begin
      A = randn(E, m, m)
      B = copy(A)
      wy1c = deepcopy(wy1)

      Linear(wy2) ⊛ wy1c
      Linear(wy2) ⊘ wy1c
      wy1c ⊘ A
      wy1 ⊘ B

      @test norm(A - B) ≈ 0.0 atol = tol
    end

  end
end
nothing
