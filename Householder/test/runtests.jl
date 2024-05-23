using SafeTestsets

include("HouseholderGeneral.jl")
include("WYGeneral.jl")
include("WYWY.jl")


using Householder
using Householder.Precompile: run_all
using JET: @test_opt, @test_call
using Test

@testset "JET Tests" begin

  @testset "JET run_all opt" begin
    @test_opt target_modules = (Householder,) run_all()
  end

  @testset "JET run_all call" begin
    @test_call target_modules = (Householder,) run_all()
  end
end

