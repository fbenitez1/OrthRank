using SafeTestsets
@safetestset "Validate Random WY-weight Construction." begin
  include("setup_sweeps_test.jl")
  test_wy_construction()
end

@safetestset "Validate Ranks." begin
  include("rank_tests.jl")
  test_validate_ranks(Float64)
  test_validate_ranks(Complex{Float64})
end
