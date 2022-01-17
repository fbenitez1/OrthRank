@safetestset "Band Rotations" begin
  using BandStruct.BandColumnMatrices
  using BandStruct.BlockedBandColumnMatrices
  using InPlace
  using Rotations
  using Random
  using LinearAlgebra

  include("standard_test_case.jl")

  @testset "$E" for
    E ∈ [ Float64,
          Complex{Float64} ]

    tol = 1e-15
    (bc, bbc) = standard_test_case(E)

    @testset "$B" for
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

      mx_bc0 = Matrix(bc0)
      # Zero element 26 with a right rotation.
      bcr26 = copy(bc0)
      mx_bcr26 = Matrix(bcr26)
      rr26 = rgivens(bcr26[2,5], bcr26[2,6],5)
      bcr26 ⊛ rr26
      mx_bcr26 ⊛ rr26
      @test abs(bcr26[2, 6]) <= tol
      @test Matrix(bcr26) == mx_bcr26

      mx_bcr26[2,6]=zero(E)
      bcr26[2,6]=zero(E)
      notch_upper!(bcr26,2,6)
      @test Matrix(bcr26) == mx_bcr26
      bcr26 ⊘ rr26
      @test norm(Matrix(bcr26) - mx_bc0, Inf) <= tol

      # Zero element 1,4 with a right rotation (no storage error).
      bcr14 = copy(bc0)
      mx_bcr14 = Matrix(bcr14)
      rr14 = rgivens(bcr14[1,3], bcr14[1,4],3)
      @test_throws NoStorageForIndex bcr14 ⊛ rr14

      # Zero element 7,4 with a left rotation.
      bcl74 = copy(bc0)
      mx_bcl74 = Matrix(bcl74)
      rl74 = lgivens(bcl74[6,4], bcl74[7,4],6)
      rl74 ⊘ bcl74
      rl74 ⊘ mx_bcl74
      @test abs(bcl74[7, 4]) <= tol
      @test Matrix(bcl74) == mx_bcl74

      mx_bcl74[7,4] = zero(E)
      notch_lower!(bcl74,7,4)
      @test Matrix(bcl74) == mx_bcl74

      rl74 ⊛ bcl74
      @test norm(Matrix(bcl74) - mx_bc0, Inf) <= tol
    end
  end
end
nothing
