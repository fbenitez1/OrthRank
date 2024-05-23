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

include("mul.jl")

using OrthWeight
using OrthWeight.Precompile: run_all
using JET: @test_opt, @test_call
using Test

@testset "JET Tests" begin

  @testset "JET run_all opt" begin
    @test_opt target_modules = (OrthWeight,) run_all()
  end

  @testset "JET run_all call" begin
    @test_call target_modules = (OrthWeight,) run_all()
  end
end
