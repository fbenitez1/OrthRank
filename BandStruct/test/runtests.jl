using BandStruct.BandColumnMatrices
using BandStruct.BlockedBandColumnMatrices
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

lbc0 = BlockedBandColumn(
  Float64,
  LeadingDecomp,
  MersenneTwister(0),
  8,
  7,
  upper_blocks = upper_blocks,
  lower_blocks = lower_blocks,
  upper_rank_max = 3,
  lower_rank_max = 2,
  upper_ranks = [1, 2, 1, 0],
  lower_ranks = [1, 1, 1, 1],
)

wilk_lbc0 = [
  'X' 'X' 'X' 'U' 'O' 'O' 'N'
  'X' 'X' 'X' 'X' 'U' 'U' 'O'
  'O' 'L' 'X' 'X' 'U' 'U' 'O'
  'O' 'L' 'X' 'X' 'X' 'X' 'U'
  'O' 'O' 'L' 'X' 'X' 'X' 'X'
  'O' 'O' 'O' 'L' 'X' 'X' 'X'
  'N' 'N' 'N' 'L' 'X' 'X' 'X'
  'N' 'N' 'N' 'N' 'O' 'L' 'X'
]

show_equality_result(
  "LBC LeadingDecomp Creation Test",
  ==,
  wilk(toBandColumn(lbc0)).arr,
  wilk_lbc0,
)

lbc0T = BlockedBandColumn(
  Float64,
  TrailingDecomp,
  8,
  7,
  upper_blocks = upper_blocks,
  lower_blocks = lower_blocks,
  upper_ranks = [1, 2, 1, 0],
  lower_ranks = [1, 1, 1, 1],
)

wilk_lbc0T = [
  'X' 'X' 'X' 'U' 'O' 'O' 'N'
  'X' 'X' 'X' 'X' 'U' 'U' 'O'
  'L' 'L' 'X' 'X' 'U' 'U' 'O'
  'O' 'O' 'X' 'X' 'X' 'X' 'U'
  'O' 'O' 'L' 'X' 'X' 'X' 'X'
  'O' 'O' 'O' 'L' 'X' 'X' 'X'
  'N' 'N' 'N' 'O' 'X' 'X' 'X'
  'N' 'N' 'N' 'N' 'L' 'L' 'X'
]

show_equality_result(
  "LBC TrailingDecomp Creation Test",
  ==,
  wilk(toBandColumn(lbc0T)).arr,
  wilk_lbc0T,
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
