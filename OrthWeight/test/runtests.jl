using LinearAlgebra
using BandStruct
using Householder
using OrthWeight
using Random

lower_blocks = [
  2 4 5 7
  2 3 4 6
]

upper_blocks = [
  1 3 4 6
  3 4 6 7
]

wyw = WYWeight(
  Float64,
  BigStep,
  8,
  7,
  upper_rank_max = 2,
  lower_rank_max = 2,
  upper_blocks = upper_blocks,
  lower_blocks = lower_blocks,
)

wywl = WYWeight(
  Float64,
  BigStep,
  LeadingDecomp,
  Random.default_rng(),
  8,
  7,
  upper_ranks = [2 for j∈1:size(upper_blocks,2)],
  lower_ranks = [2 for j∈1:size(lower_blocks,2)],
  upper_blocks = upper_blocks,
  lower_blocks = lower_blocks,
)

sweep = SweepForward(wywl.leftWY)
A = Matrix{Float64}(I, 8, 8)
sweep ⊛ A
sweep ⊘ A
