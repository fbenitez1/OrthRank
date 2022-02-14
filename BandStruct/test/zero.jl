@safetestset "Zero Elements" begin
  using BandStruct
  using Random
  using LinearAlgebra

  @testset "$E" for
    E ∈ [ Float64,
          Complex{Float64} ]

    (bc, bbc) =
      BandStruct.standard_test_case(E, upper_rank_max = 2, lower_rank_max = 1)

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

      bc0_za45 = copy(bc0)
      mx_za45 = Matrix(bc0)
      zero_above!(bc0_za45, 4, 5)
      mx_za45[1:4, 5] .= zero(E)

      @test Matrix(bc0_za45) == mx_za45
      
      bc0_zb45 = copy(bc0)
      mx_zb45 = Matrix(bc0)
      zero_below!(bc0_zb45, 4, 5)
      mx_zb45[4:8, 5] .= zero(E)

      @test Matrix(bc0_zb45) == mx_zb45

      bc0_zr45 = copy(bc0)
      mx_zr45 = Matrix(bc0)
      zero_right!(bc0_zr45, 4, 5)
      mx_zr45[4, 5:7] .= zero(E)

      @test Matrix(bc0_zr45) == mx_zr45

      bc0_zl45 = copy(bc0)
      mx_zl45 = Matrix(bc0)
      zero_left!(bc0_zl45, 4, 5)
      mx_zl45[4, 1:5] .= zero(E)
      
      @test Matrix(bc0_zl45) == mx_zl45

      bc0_za4_45 = copy(bc0)
      mx_za4_45 = Matrix(bc0)
      zero_above!(bc0_za4_45, 4, 4:5)
      mx_za4_45[1:4, 4:5] .= zero(E)

      @test Matrix(bc0_za4_45) == mx_za4_45

      bc0_zb4_45 = copy(bc0)
      mx_zb4_45 = Matrix(bc0)
      zero_below!(bc0_zb4_45, 4, 4:5)
      mx_zb4_45[4:7, 4:5] .= zero(E)

      @test Matrix(bc0_zb4_45) == mx_zb4_45

      bc0_zr45_5 = copy(bc0)
      mx_zr45_5 = Matrix(bc0)
      zero_right!(bc0_zr45_5, 4:5, 5)
      mx_zr45_5[4:5, 5:7] .= zero(E)

      @test Matrix(bc0_zr45_5) == mx_zr45_5

      bc0_zl45_5 = copy(bc0)
      mx_zl45_5 = Matrix(bc0)
      zero_left!(bc0_zl45_5, 4:5, 5)
      mx_zl45_5[4:5, 1:5] .= zero(E)

      @test Matrix(bc0_zl45_5) == mx_zl45_5

    end
  end
end
