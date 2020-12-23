
    #                 1   2       3   4
    # A =   X   X   X | U | O   O | N |
    #                 +---+-----------+
    #       X   X   X   X | U   U | O |
    #     1 ------+       |       |   |
    #       O   L | X   X | U   U | O |
    #             |       +-------+---+
    #       O   L | X   X   X   X | U |
    #     2 ------+---+           +---+
    #       O   O | L | X   X   X   X |
    #     3 ------+---+---+           |
    #       O   O | O | L | X   X   X |
    #             |   |   |           +
    #       N   N | N | L | X   X   X 
    #     4 ------+---+---+-------+   
    #       N   N | N | N | O   L | X 

# X X U U O
# X X U U O
# X X X X U
# L X X X X
# O L X X X

bc1 = lbc[2:6, 3:7]
column_inband_index_ranges_bc1 = [1:4, 1:5, 1:5, 1:5, 3:5]



function compare_ranges(xs, ys)
  m = length(xs)
  res = m == length(ys)
  for j ∈ 1:m
    res = res && (xs[j] == ys[j] || isempty(xs[j]) && isempty(ys[j]))
  end
  res
end

println("""

**************
** Range tests
**************

""")

column_inband_index_ranges_lbc0 = [1:2, 1:4, 1:5, 1:7, 2:7, 2:8, 4:8]

show_equality_result(
  "LBC Matrix column_inband_index_range",
  compare_ranges,
  column_inband_index_ranges_lbc0,
  [inband_index_range(lbc0, :, k) for k = 1:lbc0.n],
)

show_equality_result(
  "BC Matrix column inband_index_range",
  compare_ranges,
  column_inband_index_ranges_lbc0,
  [inband_index_range(bc0, :, k) for k = 1:lbc0.n],
)

upper_column_inband_index_ranges_lbc0 =
  [1:0, 1:0, 1:0, 1:1, 2:3, 2:3, 4:4]

show_equality_result(
  "LBC Matrix column upper_inband_index_range",
  compare_ranges,
  upper_column_inband_index_ranges_lbc0,
  [upper_inband_index_range(lbc0, :, k) for k = 1:lbc0.n],
)

show_equality_result(
  "BC Matrix column upper_inband_index_range",
  compare_ranges,
  upper_column_inband_index_ranges_lbc0,
  [upper_inband_index_range(bc0, :, k) for k = 1:lbc0.n],
)

middle_column_inband_index_ranges_lbc0 =
  [1:2, 1:2, 1:4, 2:5, 4:7, 4:7, 5:8]

show_equality_result(
  "LBC Matrix column middle_inband_index_range",
  compare_ranges,
  middle_column_inband_index_ranges_lbc0,
  [middle_inband_index_range(lbc0, :, k) for k = 1:lbc0.n],
)

show_equality_result(
  "BC Matrix column middle_inband_index_range",
  compare_ranges,
  middle_column_inband_index_ranges_lbc0,
  [middle_inband_index_range(bc0, :, k) for k = 1:lbc0.n],
)

lower_column_inband_index_ranges_lbc0 =
  [1:0, 3:4, 5:5, 6:7, 1:0, 8:8, 1:0]

show_equality_result(
  "LBC Matrix column lower_inband_index_range",
  compare_ranges,
  lower_column_inband_index_ranges_lbc0,
  [lower_inband_index_range(lbc0, :, k) for k = 1:lbc0.n],
)

show_equality_result(
  "BC Matrix column lower_inband_index_range",
  compare_ranges,
  lower_column_inband_index_ranges_lbc0,
  [lower_inband_index_range(bc0, :, k) for k = 1:lbc0.n],
)

# X X U U O
# X X U U O
# X X X X U
# L X X X X
# O L X X X

bc1 = lbc[2:6, 3:7]

column_inband_index_ranges_bc1 = [1:4, 1:5, 1:5, 1:5, 3:5]
show_equality_result(
  "BC Submatrix column inband_index_range",
  compare_ranges,
  column_inband_index_ranges_bc1,
  [inband_index_range(bc1, :, k) for k = 1:bc1.n],
)

column_upper_inband_index_ranges_bc1 = [1:0, 1:0, 1:2, 1:2, 3:3]
show_equality_result(
  "BC Submatrix column upper_inband_index_range",
  compare_ranges,
  column_upper_inband_index_ranges_bc1,
  [upper_inband_index_range(bc1, :, k) for k = 1:bc1.n],
)

column_middle_inband_index_ranges_bc1 = [1:3, 1:4, 3:5, 3:5, 4:5]
show_equality_result(
  "BC Submatrix column middle_inband_index_range",
  compare_ranges,
  column_middle_inband_index_ranges_bc1,
  [middle_inband_index_range(bc1, :, k) for k = 1:bc1.n],
)

column_lower_inband_index_ranges_bc1 = [4:4, 5:5, 1:0, 1:0, 1:0]
show_equality_result(
  "BC Submatrix column lower_inband_index_range",
  compare_ranges,
  column_lower_inband_index_ranges_bc1,
  [lower_inband_index_range(bc1, :, k) for k = 1:bc1.n],
)

row_inband_index_ranges_bc1 = [1:4, 1:4, 1:5, 1:5, 2:5]
show_equality_result(
  "BC Submatrix row inband_index_range",
  compare_ranges,
  row_inband_index_ranges_bc1,
  [inband_index_range(bc1, j, : ) for j = 1:bc1.m],
)

row_upper_inband_index_ranges_bc1 = [3:4, 3:4, 5:5, 1:0, 1:0]
show_equality_result(
  "BC Submatrix row upper_inband_index_range",
  compare_ranges,
  row_upper_inband_index_ranges_bc1,
  [upper_inband_index_range(bc1, j, : ) for j = 1:bc1.m],
)

row_middle_inband_index_ranges_bc1 = [1:2, 1:2, 1:4, 2:5, 3:5]
show_equality_result(
  "BC Submatrix row middle_inband_index_range",
  compare_ranges,
  row_middle_inband_index_ranges_bc1,
  [middle_inband_index_range(bc1, j, : ) for j = 1:bc1.m],
)

row_lower_inband_index_ranges_bc1 = [1:0, 1:0, 1:0, 1:1, 2:2]
show_equality_result(
  "BC Submatrix row lower_inband_index_range",
  compare_ranges,
  row_lower_inband_index_ranges_bc1,
  [lower_inband_index_range(bc1, j, : ) for j = 1:bc1.m],
)
