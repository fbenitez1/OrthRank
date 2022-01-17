@safetestset "Notch Tests with Row Validation" begin
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

      bc0_notch = copy(bc0)
      # wilk0 = wilk(toBandColumn(bc0_notch))
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

      bc14_notch = copy(bc0_notch)
      wilk14_notch = [
        'X' 'X' 'X' 'O' 'O' 'O' 'O'
        'X' 'X' 'X' 'X' 'U' 'U' 'O'
        'O' 'L' 'X' 'X' 'U' 'U' 'O'
        'O' 'L' 'X' 'X' 'X' 'X' 'U'
        'O' 'O' 'L' 'X' 'X' 'X' 'X'
        'O' 'O' 'O' 'L' 'X' 'X' 'X'
        'N' 'O' 'O' 'L' 'X' 'X' 'X'
        'N' 'N' 'O' 'O' 'O' 'L' 'X'
      ]

      notch_upper!(bc14_notch,1, 4)
      @test validate_rows_first_last(bc14_notch)
      @test wilk(toBandColumn(bc14_notch)).arr == wilk14_notch

      bc26_notch = copy(bc0_notch)
      wilk26_notch = [
        'X' 'X' 'X' 'U' 'O' 'O' 'O'
        'X' 'X' 'X' 'X' 'U' 'O' 'O'
        'O' 'L' 'X' 'X' 'U' 'U' 'O'
        'O' 'L' 'X' 'X' 'X' 'X' 'U'
        'O' 'O' 'L' 'X' 'X' 'X' 'X'
        'O' 'O' 'O' 'L' 'X' 'X' 'X'
        'N' 'O' 'O' 'L' 'X' 'X' 'X'
        'N' 'N' 'O' 'O' 'O' 'L' 'X'
      ]

      notch_upper!(bc26_notch, 2, 6)
      @test validate_rows_first_last(bc26_notch)
      @test wilk(toBandColumn(bc26_notch)).arr == wilk26_notch

      bc32_notch = copy(bc0_notch)
      wilk32_notch = [
        'X' 'X' 'X' 'U' 'O' 'O' 'O'
        'X' 'X' 'X' 'X' 'U' 'U' 'O'
        'O' 'O' 'X' 'X' 'U' 'U' 'O'
        'O' 'O' 'X' 'X' 'X' 'X' 'U'
        'O' 'O' 'L' 'X' 'X' 'X' 'X'
        'O' 'O' 'O' 'L' 'X' 'X' 'X'
        'N' 'O' 'O' 'L' 'X' 'X' 'X'
        'N' 'N' 'O' 'O' 'O' 'L' 'X'
      ]

      notch_lower!(bc32_notch, 3, 2)
      @test validate_rows_first_last(bc32_notch)
      @test wilk(toBandColumn(bc32_notch)).arr == wilk32_notch

      bc42_notch = copy(bc0_notch)
      wilk42_notch = [
        'X' 'X' 'X' 'U' 'O' 'O' 'O'
        'X' 'X' 'X' 'X' 'U' 'U' 'O'
        'O' 'L' 'X' 'X' 'U' 'U' 'O'
        'O' 'O' 'X' 'X' 'X' 'X' 'U'
        'O' 'O' 'L' 'X' 'X' 'X' 'X'
        'O' 'O' 'O' 'L' 'X' 'X' 'X'
        'N' 'O' 'O' 'L' 'X' 'X' 'X'
        'N' 'N' 'O' 'O' 'O' 'L' 'X'
      ]

      notch_lower!(bc42_notch, 4, 2)
      @test validate_rows_first_last(bc42_notch)
      @test wilk(toBandColumn(bc42_notch)).arr == wilk42_notch

      bc64_notch = copy(bc0_notch)
      wilk64_notch = [
        'X' 'X' 'X' 'U' 'O' 'O' 'O'
        'X' 'X' 'X' 'X' 'U' 'U' 'O'
        'O' 'L' 'X' 'X' 'U' 'U' 'O'
        'O' 'L' 'X' 'X' 'X' 'X' 'U'
        'O' 'O' 'L' 'X' 'X' 'X' 'X'
        'O' 'O' 'O' 'O' 'X' 'X' 'X'
        'N' 'O' 'O' 'O' 'X' 'X' 'X'
        'N' 'N' 'O' 'O' 'O' 'L' 'X'
      ]

      notch_lower!(bc64_notch, 6, 4)
      @test validate_rows_first_last(bc64_notch)
      @test wilk(toBandColumn(bc64_notch)).arr == wilk64_notch

    end
  end
end
nothing
