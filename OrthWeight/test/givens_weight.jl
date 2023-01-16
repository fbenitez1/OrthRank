using BandStruct
using BandStruct.BlockedBandColumnMatrices
using BandStruct.BandwidthInit
using OrthWeight.GivensWeightMatrices
using Random

m=8
n=7

lower_blocks = [
  2 4 5 7
  2 3 4 6
]

upper_blocks = [
  1 3 4 6
  3 4 6 7
]


bbc = BlockedBandColumn(
  Float64,
  LeadingDecomp,
  m,
  n,
  upper_rank_max = 4,
  lower_rank_max = 4,
  upper_blocks = upper_blocks,
  lower_blocks = lower_blocks,
  upper_ranks = [1, 2, 1, 0],
  lower_ranks = [1, 1, 1, 1],
)

C=Consts(5,3.14)
