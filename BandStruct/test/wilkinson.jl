@safetestset "Wilkinson Diagrams, Leading and Trailing Generation" begin
  using BandStruct
  using Random

  @testset "$E" for
    E ∈ [ Float64,
          Complex{Float64} ]
    

    bc0, bbc0 = BandStruct.standard_test_case(E)

    wilk_bbc0 = [
      'X' 'X' 'X' 'U' 'O' 'O' 'N'
      'X' 'X' 'X' 'X' 'U' 'U' 'O'
      'O' 'L' 'X' 'X' 'U' 'U' 'O'
      'N' 'L' 'X' 'X' 'X' 'X' 'U'
      'N' 'O' 'L' 'X' 'X' 'X' 'X'
      'N' 'N' 'O' 'L' 'X' 'X' 'X'
      'N' 'N' 'N' 'L' 'X' 'X' 'X'
      'N' 'N' 'N' 'O' 'O' 'L' 'X'
    ]

    @test wilk(toBandColumn(bbc0)).arr == wilk_bbc0
    @test wilk(toBandColumn(bc0)).arr == wilk_bbc0


    bc0T, bbc0T =
      BandStruct.standard_test_case(E, decomp_type = TrailingDecomp())

    wilk_bbc0T = [
      'X' 'X' 'X' 'U' 'O' 'O' 'N'
      'X' 'X' 'X' 'X' 'U' 'U' 'O'
      'L' 'L' 'X' 'X' 'U' 'U' 'O'
      'N' 'O' 'X' 'X' 'X' 'X' 'U'
      'N' 'O' 'L' 'X' 'X' 'X' 'X'
      'N' 'N' 'O' 'L' 'X' 'X' 'X'
      'N' 'N' 'N' 'O' 'X' 'X' 'X'
      'N' 'N' 'N' 'O' 'L' 'L' 'X'
    ]

    @test wilk(toBandColumn(bbc0T)).arr == wilk_bbc0T
    @test wilk(toBandColumn(bc0T)).arr == wilk_bbc0T
  end
end
