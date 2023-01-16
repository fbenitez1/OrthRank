using JET

module TestJET
using LinearAlgebra
using BandStruct
using Householder
using OrthWeight
using Random
using InPlace

macro testset(str, ex)
  esc(ex)
end

macro test(ex)
  esc(ex)
end

include("setup_sweeps_test.jl")

include("rank_tests.jl")
end

display(@report_opt ignored_modules = (Base,) TestJET.test_wy_construction())
display(@report_call ignored_modules = (Base,) TestJET.test_wy_construction())

display(@report_opt ignored_modules = (Base,) TestJET.test_validate_ranks(
  Float64,
))
display(@report_call ignored_modules = (Base,) TestJET.test_validate_ranks(
  Float64,
))

display(@report_opt ignored_modules = (Base,) TestJET.test_validate_ranks(Complex{
  Float64,
}))
display(@report_call ignored_modules = (Base,) TestJET.test_validate_ranks(Complex{
  Float64,
}))

