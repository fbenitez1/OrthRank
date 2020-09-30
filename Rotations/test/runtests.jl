if isdefined(@__MODULE__, :LanguageServer)
  include("src/Givens.jl")
  using .Givens
else
  using Rotations.Givens
end

using LinearAlgebra
using Random
using InPlace
using ShowTests

tol = 1e-15

include("general.jl")
