@safetestset "BandStruct Index Tests" begin
  using BandStruct.BandColumnMatrices
  using BandStruct.BlockedBandColumnMatrices
  using Random
  include("standard_test_case.jl")

  @testset "$E" for
    E ∈ [ Float64,
          Complex{Float64} ]

    (bc0, bbc0) = standard_test_case(E, upper_rank_max = 3, lower_rank_max = 2)

    mx_bc0 = Matrix(bc0)
    bc = copy(bc0)
    bbc = copy(bbc0)
    mx_bc = copy(mx_bc0)

    @testset "eachindex and get_elements.  (Blocked vs. unblocked)." begin
      @test collect(eachindex(bc)) == collect(eachindex(bbc))
      @test collect(get_elements(bc)) ==  collect(get_elements(bbc))
    end

    @testset "get and set_index!, with bandwidth validation checks." begin
      resbc = true
      resbbc = true
      for ix in (ix for ix ∈ eachindex(bc) if rand() > 0.75)
        x = rand(E)
        bc[ix] = x
        bbc[ix] = x
        mx_bc[ix] = x
        resbc = resbc && validate_rows_first_last(bc)
        resbbc = resbbc && validate_rows_first_last(bbc)
      end
      @test resbc
      @test resbbc
      @test Matrix(bc) == mx_bc
      @test Matrix(bbc) == mx_bc
    end
  end
end
nothing
