using BandStruct.BandColumnMatrices
using BandStruct.LeadingBandColumnMatrices
using BandStruct.BandRotations
using BandStruct.BandHouseholder
using Householder

using Random
using Rotations
using InPlace
using ShowTests
using LinearAlgebra

lower_blocks = [
  2 4 5 7
  2 3 4 6
]

upper_blocks = [
  1 3 4 6
  3 4 6 7
]

lbc0 = LeadingBandColumn(
  MersenneTwister(0),
  Float64,
  8,
  7,
  3,
  2,
  upper_blocks,
  lower_blocks,
  [1, 2, 1, 0],
  [1, 1, 1, 1],
)

println("Testing BandStruct operations for a matrix with structure:")
println()


show(wilk(lbc0))

bc0 = copy(toBandColumn(lbc0))
mx_bc0 = Matrix(bc0)


include("index.jl")
include("submatrix.jl")
include("range.jl")
include("bulge.jl")
include("notch.jl")
include("rotations.jl")
include("householder.jl")
