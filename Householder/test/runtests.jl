if isdefined(@__MODULE__, :LanguageServer)
  include("src/Compute.jl")
  using .Compute
  include("src/WY.jl")
  using .WY
else
  using Householder.Compute
  using Householder.WY
end

using LinearAlgebra
using Random
using InPlace
using ShowTests
using Profile

tol = 1e-14
l=2
m=3

include("HouseholderGeneral.jl")
include("WYGeneral.jl")

