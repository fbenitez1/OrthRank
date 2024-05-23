using SafeTestsets

include("index_lists.jl")
include("first_last_init.jl")
include("wilkinson.jl")
include("index.jl")
include("submatrix.jl")
include("range.jl")
include("bulge.jl")
include("notch.jl")
include("mul.jl")
include("rotations.jl")
include("householder.jl")
include("wy.jl")
include("zero.jl")
include("qr.jl")
include("lq.jl")

using BandStruct
using BandStruct.Precompile: run_all
using JET: @test_opt, @test_call
using Test

@testset "JET Tests" begin

  @testset "JET run_all opt" begin
    @test_opt target_modules = (BandStruct,) run_all()
  end

  @testset "JET run_all call" begin
    @test_call target_modules = (BandStruct,) run_all()
  end
end
