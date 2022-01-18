# Test that the sweeps in constructing a random WY-weight
# decomposition really are sweeps of orthogonal WY transformations.

@safetestset "Validate Ranks." begin
  using LinearAlgebra
  using BandStruct
  using Householder
  using OrthWeight
  using Random
  using InPlace

  @testset "$E" for
    E ∈ [ Float64,
          Complex{Float64} ]


    tol_r = 1e-14

    lower_blocks_r = [
      10 20 30 40
      10 20 25 37
    ]

    upper_blocks_r = [
      10 20 30 40
      10 20 25 37
    ]

    r = 2

    wyw_r = WYWeight(
      E,
      SpanStep,
      LeadingDecomp,
      Random.default_rng(),
      60,
      50,
      upper_rank_max = r,
      lower_rank_max = r,
      upper_blocks = upper_blocks_r,
      lower_blocks = lower_blocks_r,
    )

    A = Matrix(wyw_r)

    @testset "Upper block ranks" begin
      uflag = -1
      for l ∈ 1:size(upper_blocks_r, 2)
        ranges = upper_block_ranges(wyw_r.b, l)
        ru = rank(Matrix(view(wyw_r.b, ranges...)), tol_r)
        ru == r || (uflag = l; break)
      end
      @test uflag == -1
    end

    @testset "Lower block ranks" begin
      lflag = -1
      for l ∈ 1:size(upper_blocks_r, 2)
        ranges = lower_block_ranges(wyw_r.b, l)
        rl = rank(Matrix(view(wyw_r.b, ranges...)), tol_r)
        rl == r || (lflag = l; break)
      end
      @test lflag == -1
    end
  end
end
