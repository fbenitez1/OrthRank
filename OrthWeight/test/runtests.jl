using SafeTestsets
@safetestset "Validate Random WY-weight Construction." begin
  include("setup_sweeps_test.jl")
  test_wy_construction()
end

@safetestset "Validate WY Ranks." begin
  include("rank_tests.jl")
  test_validate_ranks(Float64)
  test_validate_ranks(Complex{Float64})
end

@safetestset "Validate Givens Weight Ranks." begin
  include("givens_weight_ranks.jl")
  run_givens_weight_rank_tests()
end
