using Householder.Compute
using Householder.WY

using LinearAlgebra
using Random
using InPlace
using ShowTests
using Profile
using BenchmarkTools

tol = 5e-14
l=2
m=3

include("HouseholderGeneral.jl")
include("WYGeneral.jl")
include("WYWY.jl")
