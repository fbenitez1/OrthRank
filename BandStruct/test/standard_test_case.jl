function standard_test_case(
  ::Type{E};
  upper_rank_max = 2,
  lower_rank_max = 1,
  decomp_type = LeadingDecomp(),
) where {E <: Number}
  # standard upper rank max for extended is 3.  For lower is 2.
  lbl = [
    2 4 5 7
    2 3 4 6
  ]
  lower_blocks = [BlockSize(lbl[1,j], lbl[2,j]) for j∈1:4]

  ubl = [
    1 3 4 6
    3 4 6 7
  ]

  upper_blocks = [BlockSize(ubl[1,j], ubl[2,j]) for j∈1:4]
  
  bbc0 = BlockedBandColumn(
    E,
    decomp_type,
    MersenneTwister(0),
    8,
    7;
    upper_blocks = upper_blocks,
    lower_blocks = lower_blocks,
    upper_rank_max = upper_rank_max,
    lower_rank_max = lower_rank_max,
    max_num_blocks = 4,
    upper_ranks = [1, 2, 1, 0],
    lower_ranks = [1, 1, 1, 1],
  )
  bc0 = copy(toBandColumn(bbc0))
  (bc0, bbc0)
end
