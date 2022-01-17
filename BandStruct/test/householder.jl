@safetestset "Band Householder" begin
  using BandStruct.BandColumnMatrices
  using BandStruct.BlockedBandColumnMatrices
  using InPlace
  using Householder
  using Random
  using LinearAlgebra

  include("standard_test_case.jl")

  @testset "$E" for
    E ∈ [ Float64,
          Complex{Float64} ]

    # X   X   X | U | O   O | N | 
    #           + - + - - - + - + 
    # X   X   X   X | U   U | O | 
    # - - - +       |       |   | 
    # O   L | X   X | U   U | O | 
    #       |       + - - - + - + 
    # O   L | X   X   X   X | U | 
    # - - - + - +           + - + 
    # O   O | L | X   X   X   X | 
    # - - - + - + - +           | 
    # O   O | O | L | X   X   X | 
    #       |   |   |           + 
    # N   O | O | L | X   X   X   
    # - - - + - + - + - - - +     
    # N   N | O | O | O   L | X   

    tol = 2e-15
    (bc, bbc) = standard_test_case(E, upper_rank_max = 2, lower_rank_max = 2)

    @testset "$B Zero Elements" for
      (bc0, B) ∈ [ (bc, "BandColumn")
                   (bbc, "BlockedBandColumn") ]

      mx_bc0 = Matrix(bc0)

      work = zeros(E, maximum(size(bc0)))

      # Zero elements 3 to 4 in column 2 with a left Householder.
      bch_34_2 = copy(bc0)
      mx_bch_34_2 = Matrix(bch_34_2)
      v_34_2 = zeros(E, 3)

      h_34_2 = householder(bch_34_2, 2:4, 2, 1, 1, v_34_2, work)
      h_34_2 ⊘ bch_34_2

      mx_bch_34_2_a = copy(mx_bch_34_2)
      h_34_2 ⊘ mx_bch_34_2_a

      @test norm(svdvals(Matrix(bch_34_2)) - svdvals(mx_bch_34_2)) <= tol
      @test norm(Matrix(bch_34_2) - mx_bch_34_2_a) <= tol
      @test norm(Matrix(bch_34_2)[3:4,2]) <= tol

      h_34_2 ⊛ bch_34_2
      @test norm(Matrix(bch_34_2) - mx_bch_34_2) <= tol

      # Zero elements 5 to 6 in row 4 with a right Householder.
      bch_4_56 = copy(bc0)
      mx_bch_4_56 = Matrix(bch_4_56)
      v_4_56 = zeros(E, 3)

      h_4_56 = householder(bch_4_56, 4, 5:7, 3, 4, v_4_56, work)
      bch_4_56 ⊛ h_4_56

      mx_bch_4_56_a = copy(mx_bch_4_56)
      mx_bch_4_56_a ⊛ h_4_56

      @test norm(svdvals(Matrix(bch_4_56)) - svdvals(mx_bch_4_56)) <= tol
      @test norm(Matrix(bch_4_56) - mx_bch_4_56_a) <= tol
      @test norm(Matrix(bch_4_56)[4,5:6]) <= tol

      bch_4_56 ⊘ h_4_56
      @test norm(Matrix(bch_4_56) - mx_bch_4_56) <= tol
    end

    (bc, bbc) = standard_test_case(E, upper_rank_max = 2, lower_rank_max = 1)
    @testset "$B NoStorage Check" for
      (bc0, B) ∈ [ (bc, "BandColumn")
                   (bbc, "BlockedBandColumn") ]

      # X   X   X | U | O   O | N | 
      #           + - + - - - + - + 
      # X   X   X   X | U   U | O | 
      # - - - +       |       |   | 
      # O   L | X   X | U   U | O | 
      #       |       + - - - + - + 
      # N   L | X   X   X   X | U | 
      # - - - + - +           + - + 
      # N   O | L | X   X   X   X | 
      # - - - + - + - +           | 
      # N   N | O | L | X   X   X | 
      #       |   |   |           + 
      # N   N | N | L | X   X   X   
      # - - - + - + - + - - - +     
      # N   N | N | O | O   L | X   

      work = zeros(E, maximum(size(bc0)))

      bch_57_4 = copy(bc0)
      v_57_4 = zeros(E, 3)
      h_57_4 = householder(bch_57_4, 5:7, 4, 1, 4, v_57_4, work)
      @test_throws NoStorageForIndex h_57_4 ⊘ bch_57_4

      bch_14_4 = copy(bc0)
      v_14_4 = zeros(E, 4)
      h_14_4 = householder(bch_14_4, 1:4, 4, 4, 0, v_14_4, work)
      @test_throws NoStorageForIndex h_14_4 ⊘ bch_14_4

      bch_4_47 = copy(bc0)
      v_4_47 = zeros(E, 4)
      h_4_47 = householder(bch_4_47, 4, 4:7, 1, 3, v_4_47, work)
      @test_throws NoStorageForIndex bch_4_47 ⊛ h_4_47
    end
  end
end
