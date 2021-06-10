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

include("first_last_init.jl")

lower_blocks = [
  2 4 5 7
  2 3 4 6
]

upper_blocks = [
  1 3 4 6
  3 4 6 7
]

lbc0 = LeadingBandColumn(
  Float64,
  MersenneTwister(0),
  8,
  7,
  upper_bw_max = 3,
  lower_bw_max = 2,
  upper_blocks = upper_blocks,
  lower_blocks = lower_blocks,
  upper_ranks = [1, 2, 1, 0],
  lower_ranks = [1, 1, 1, 1],
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
include("wy.jl")
include("bench_qr.jl")
include("bench_lq.jl")
include("zero.jl")
