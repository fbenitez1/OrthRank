if isdefined(@__MODULE__, :LanguageServer)
  include("src/BandColumnMatrices.jl")
  using .BandColumnMatrices
  include("src/LeadingBandColumnMatrices.jl")
  using .LeadingBandColumnMatrices
  include("src/BandRotations.jl")
  using .BandRotations
else
  using BandStruct.BandColumnMatrices
  using BandStruct.LeadingBandColumnMatrices
  using BandStruct.BandRotations
end
using Random
using Rotations
using InPlace
using ShowTests
using LinearAlgebra

blocks = [
  1 3 4 6 7 9 9 11 12
  1 2 5 5 6 7 8 9 10
]

lbc0 = LeadingBandColumn(
  MersenneTwister(0),
  Float64,
  12,
  10,
  2,
  2,
  blocks,
  [1, 1, 1, 2, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1, 1, 1],
)


println("Testing BandStruct operations for a matrix with structure:")
println()


show(wilk(lbc0))
println()

bc0 = lbc0[1:end, 1:end]
mx_bc0 = Matrix(bc0)

include("index.jl")
include("submatrix.jl")
include("range.jl")
include("extend.jl")
include("trim.jl")
include("rotations.jl")
