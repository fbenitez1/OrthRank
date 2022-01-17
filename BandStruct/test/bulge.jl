@safetestset "Bulge Tests with Row Validation" begin
  using BandStruct.BandColumnMatrices
  using BandStruct.BlockedBandColumnMatrices
  using Random
  include("standard_test_case.jl")

  @testset "$E" for
    E ∈ [ Float64,
          Complex{Float64} ]

    (bc, bbc) = standard_test_case(E, upper_rank_max = 3, lower_rank_max = 2)

    @testset "$B" for
      (bc0, B) ∈ [ (bc, "BandColumn"),
                   (bbc, "BlockedBandColumn") ]

      bc0_bulge = copy(bc0)

      # wilk0 = wilk(toBandColumn(bc0_bulge))
      # print(wilk0.arr)
      # wilk0 =
      # [
      #   'X' 'X' 'X' 'U' 'O' 'O' 'O';
      #   'X' 'X' 'X' 'X' 'U' 'U' 'O';
      #   'O' 'L' 'X' 'X' 'U' 'U' 'O';
      #   'O' 'L' 'X' 'X' 'X' 'X' 'U';
      #   'O' 'O' 'L' 'X' 'X' 'X' 'X';
      #   'O' 'O' 'O' 'L' 'X' 'X' 'X';
      #   'N' 'O' 'O' 'L' 'X' 'X' 'X';
      #   'N' 'N' 'O' 'O' 'O' 'L' 'X'
      # ]

      bc1_bulge = copy(bc0_bulge)
      wilk1 = [
        'X'  'X'  'X'  'U'  'O'  'O'  'O'
        'X'  'X'  'X'  'X'  'U'  'U'  'O'
        'L'  'L'  'X'  'X'  'U'  'U'  'O'
        'L'  'L'  'X'  'X'  'X'  'X'  'U'
        'O'  'O'  'L'  'X'  'X'  'X'  'X'
        'O'  'O'  'O'  'L'  'X'  'X'  'X'
        'N'  'O'  'O'  'L'  'X'  'X'  'X'
        'N'  'N'  'O'  'O'  'O'  'L'  'X'
      ]

      bulge!(bc1_bulge, :, 1:2)
      @test validate_rows_first_last(bc1_bulge)
      @test wilk(toBandColumn(bc1_bulge)).arr == wilk1

      bc2_bulge = copy(bc0_bulge)
      wilk2 = [
        'X' 'X' 'X' 'U' 'O' 'O' 'O'
        'X' 'X' 'X' 'X' 'U' 'U' 'O'
        'O' 'L' 'X' 'X' 'U' 'U' 'O'
        'O' 'L' 'X' 'X' 'X' 'X' 'U'
        'O' 'L' 'L' 'X' 'X' 'X' 'X'
        'O' 'O' 'O' 'L' 'X' 'X' 'X'
        'N' 'O' 'O' 'L' 'X' 'X' 'X'
        'N' 'N' 'O' 'O' 'O' 'L' 'X'
      ]

      bulge!(bc2_bulge, :, 2:3)
      @test validate_rows_first_last(bc2_bulge)
      @test wilk(toBandColumn(bc2_bulge)).arr == wilk2

      bc4_bulge = copy(bc0_bulge)
      wilk4 = [
        'X' 'X' 'X' 'U' 'U' 'O' 'O'
        'X' 'X' 'X' 'X' 'U' 'U' 'O'
        'O' 'L' 'X' 'X' 'U' 'U' 'O'
        'O' 'L' 'X' 'X' 'X' 'X' 'U'
        'O' 'O' 'L' 'X' 'X' 'X' 'X'
        'O' 'O' 'O' 'L' 'X' 'X' 'X'
        'N' 'O' 'O' 'L' 'X' 'X' 'X'
        'N' 'N' 'O' 'O' 'O' 'L' 'X'
      ]

      bulge!(bc4_bulge,:, 4:5)
      @test validate_rows_first_last(bc4_bulge)
      @test wilk(toBandColumn(bc4_bulge)).arr == wilk4

      bc5_bulge = copy(bc0_bulge)
      wilk5 = [
        'X' 'X' 'X' 'U' 'O' 'O' 'O'
        'X' 'X' 'X' 'X' 'U' 'U' 'O'
        'O' 'L' 'X' 'X' 'U' 'U' 'O'
        'O' 'L' 'X' 'X' 'X' 'X' 'U'
        'O' 'O' 'L' 'X' 'X' 'X' 'X'
        'O' 'O' 'O' 'L' 'X' 'X' 'X'
        'N' 'O' 'O' 'L' 'X' 'X' 'X'
        'N' 'N' 'O' 'O' 'L' 'L' 'X'
      ]

      bulge!(bc5_bulge,:, 5:6)
      @test validate_rows_first_last(bc5_bulge)
      @test wilk(toBandColumn(bc5_bulge)).arr == wilk5

      bc6_bulge = copy(bc0_bulge)
      wilk6 = [
        'X' 'X' 'X' 'U' 'O' 'O' 'O'
        'X' 'X' 'X' 'X' 'U' 'U' 'U'
        'O' 'L' 'X' 'X' 'U' 'U' 'U'
        'O' 'L' 'X' 'X' 'X' 'X' 'U'
        'O' 'O' 'L' 'X' 'X' 'X' 'X'
        'O' 'O' 'O' 'L' 'X' 'X' 'X'
        'N' 'O' 'O' 'L' 'X' 'X' 'X'
        'N' 'N' 'O' 'O' 'O' 'L' 'X'
      ]

      bulge!(bc6_bulge,:, 6:7)
      @test validate_rows_first_last(bc6_bulge)
      @test wilk(toBandColumn(bc6_bulge)).arr == wilk6

      # Rows

      bcr1_bulge = copy(bc0_bulge)
      wilkr1 = [
        'X' 'X' 'X' 'U' 'U' 'U' 'O'
        'X' 'X' 'X' 'X' 'U' 'U' 'O'
        'O' 'L' 'X' 'X' 'U' 'U' 'O'
        'O' 'L' 'X' 'X' 'X' 'X' 'U'
        'O' 'O' 'L' 'X' 'X' 'X' 'X'
        'O' 'O' 'O' 'L' 'X' 'X' 'X'
        'N' 'O' 'O' 'L' 'X' 'X' 'X'
        'N' 'N' 'O' 'O' 'O' 'L' 'X'
      ]

      bulge!(bcr1_bulge, 1:2, :)
      @test validate_rows_first_last(bcr1_bulge)
      @test wilk(toBandColumn(bcr1_bulge)).arr == wilkr1

      bcr2_bulge = copy(bc0_bulge)
      wilkr2 = [
        'X' 'X' 'X' 'U' 'O' 'O' 'O'
        'X' 'X' 'X' 'X' 'U' 'U' 'O'
        'L' 'L' 'X' 'X' 'U' 'U' 'O'
        'O' 'L' 'X' 'X' 'X' 'X' 'U'
        'O' 'O' 'L' 'X' 'X' 'X' 'X'
        'O' 'O' 'O' 'L' 'X' 'X' 'X'
        'N' 'O' 'O' 'L' 'X' 'X' 'X'
        'N' 'N' 'O' 'O' 'O' 'L' 'X'
      ]

      bulge!(bcr2_bulge, 2:3, :)
      @test validate_rows_first_last(bcr2_bulge)
      @test wilk(toBandColumn(bcr2_bulge)).arr == wilkr2

      bcr3_bulge = copy(bc0_bulge)
      wilkr3 = [
        'X' 'X' 'X' 'U' 'O' 'O' 'O'
        'X' 'X' 'X' 'X' 'U' 'U' 'O'
        'O' 'L' 'X' 'X' 'U' 'U' 'U'
        'O' 'L' 'X' 'X' 'X' 'X' 'U'
        'O' 'O' 'L' 'X' 'X' 'X' 'X'
        'O' 'O' 'O' 'L' 'X' 'X' 'X'
        'N' 'O' 'O' 'L' 'X' 'X' 'X'
        'N' 'N' 'O' 'O' 'O' 'L' 'X'
      ]

      bulge!(bcr3_bulge, 3:4, :)
      @test validate_rows_first_last(bcr3_bulge)
      @test wilk(toBandColumn(bcr3_bulge)).arr == wilkr3

      bcr4_bulge = copy(bc0_bulge)
      wilkr4 = [
        'X' 'X' 'X' 'U' 'O' 'O' 'O'
        'X' 'X' 'X' 'X' 'U' 'U' 'O'
        'O' 'L' 'X' 'X' 'U' 'U' 'O'
        'O' 'L' 'X' 'X' 'X' 'X' 'U'
        'O' 'L' 'L' 'X' 'X' 'X' 'X'
        'O' 'O' 'O' 'L' 'X' 'X' 'X'
        'N' 'O' 'O' 'L' 'X' 'X' 'X'
        'N' 'N' 'O' 'O' 'O' 'L' 'X'
      ]

      bulge!(bcr4_bulge, 4:5, :)
      @test validate_rows_first_last(bcr4_bulge)
      @test wilk(toBandColumn(bcr4_bulge)).arr == wilkr4

      bcr5_bulge = copy(bc0_bulge)
      wilkr5 = [
        'X' 'X' 'X' 'U' 'O' 'O' 'O'
        'X' 'X' 'X' 'X' 'U' 'U' 'O'
        'O' 'L' 'X' 'X' 'U' 'U' 'O'
        'O' 'L' 'X' 'X' 'X' 'X' 'U'
        'O' 'O' 'L' 'X' 'X' 'X' 'X'
        'O' 'O' 'L' 'L' 'X' 'X' 'X'
        'N' 'O' 'O' 'L' 'X' 'X' 'X'
        'N' 'N' 'O' 'O' 'O' 'L' 'X'
      ]

      bulge!(bcr5_bulge, 5:6, :)
      @test validate_rows_first_last(bcr5_bulge)
      @test wilk(toBandColumn(bcr5_bulge)).arr == wilkr5

      bcr6_bulge = copy(bc0_bulge)
      wilkr6 = [
        'X' 'X' 'X' 'U' 'O' 'O' 'O'
        'X' 'X' 'X' 'X' 'U' 'U' 'O'
        'O' 'L' 'X' 'X' 'U' 'U' 'O'
        'O' 'L' 'X' 'X' 'X' 'X' 'U'
        'O' 'O' 'L' 'X' 'X' 'X' 'X'
        'O' 'O' 'O' 'L' 'X' 'X' 'X'
        'N' 'O' 'O' 'L' 'X' 'X' 'X'
        'N' 'N' 'O' 'O' 'O' 'L' 'X'
      ]

      bulge!(bcr6_bulge, 6:7, :)
      @test validate_rows_first_last(bcr6_bulge)
      @test wilk(toBandColumn(bcr6_bulge)).arr == wilkr6

      ####
      ## Index bulge
      ####

      bci16_bulge = copy(bc0_bulge)
      wilki16 = [
        'X' 'X' 'X' 'U' 'U' 'U' 'O'
        'X' 'X' 'X' 'X' 'U' 'U' 'O'
        'O' 'L' 'X' 'X' 'U' 'U' 'O'
        'O' 'L' 'X' 'X' 'X' 'X' 'U'
        'O' 'O' 'L' 'X' 'X' 'X' 'X'
        'O' 'O' 'O' 'L' 'X' 'X' 'X'
        'N' 'O' 'O' 'L' 'X' 'X' 'X'
        'N' 'N' 'O' 'O' 'O' 'L' 'X'
      ]

      bulge!(bci16_bulge, 1, 6)
      @test validate_rows_first_last(bci16_bulge)
      @test wilk(toBandColumn(bci16_bulge)).arr == wilki16

      bci61_bulge = copy(bc0_bulge)
      wilki61 = [
        'X' 'X' 'X' 'U' 'O' 'O' 'O'
        'X' 'X' 'X' 'X' 'U' 'U' 'O'
        'L' 'L' 'X' 'X' 'U' 'U' 'O'
        'L' 'L' 'X' 'X' 'X' 'X' 'U'
        'L' 'L' 'L' 'X' 'X' 'X' 'X'
        'L' 'L' 'L' 'L' 'X' 'X' 'X'
        'N' 'O' 'O' 'L' 'X' 'X' 'X'
        'N' 'N' 'O' 'O' 'O' 'L' 'X'
      ]

      bulge!(bci61_bulge, 6, 1)
      @test validate_rows_first_last(bci61_bulge)
      @test wilk(toBandColumn(bci61_bulge)).arr == wilki61

      bci62_bulge = copy(bc0_bulge)
      wilki62 = [
        'X' 'X' 'X' 'U' 'O' 'O' 'O'
        'X' 'X' 'X' 'X' 'U' 'U' 'O'
        'O' 'L' 'X' 'X' 'U' 'U' 'O'
        'O' 'L' 'X' 'X' 'X' 'X' 'U'
        'O' 'L' 'L' 'X' 'X' 'X' 'X'
        'O' 'L' 'L' 'L' 'X' 'X' 'X'
        'N' 'O' 'O' 'L' 'X' 'X' 'X'
        'N' 'N' 'O' 'O' 'O' 'L' 'X'
      ]

      bulge!(bci62_bulge, 6, 2)
      @test validate_rows_first_last(bci62_bulge)
      @test wilk(toBandColumn(bci62_bulge)).arr == wilki62
    end
  end
end
nothing
