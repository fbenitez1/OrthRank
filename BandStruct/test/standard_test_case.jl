function standard_test_case(
  E;
  upper_rank_max = 2,
  lower_rank_max = 1,
  decomp_type = LeadingDecomp,
)
  # standard upper rank max for extended is 3.  For lower is 2.
  lower_blocks = [
    2 4 5 7
    2 3 4 6
  ]

  upper_blocks = [
    1 3 4 6
    3 4 6 7
  ]
  bbc0 = BlockedBandColumn(
    E,
    decomp_type,
    MersenneTwister(0),
    8,
    7,
    upper_blocks = upper_blocks,
    lower_blocks = lower_blocks,
    upper_rank_max = upper_rank_max,
    lower_rank_max = lower_rank_max,
    upper_ranks = [1, 2, 1, 0],
    lower_ranks = [1, 1, 1, 1],
  )
  bc0 = copy(toBandColumn(bbc0))
  (bc0, bbc0)
end
