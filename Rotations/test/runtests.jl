if isdefined(@__MODULE__, :LanguageServer)
  include("src/Givens.jl")
  using .Givens
  using .Banded
else
  using Rotations.Givens
  using Rotations.Banded
end

using LinearAlgebra
using Random

using BandStruct
using BandStruct.BandColumnMatrices
using BandStruct.LeadingBandColumnMatrices

tol = 1e-15

include("show.jl")
include("general.jl")
include("bandstruct.jl")
