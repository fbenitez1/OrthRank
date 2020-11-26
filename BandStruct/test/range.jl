
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

column_els_ranges_lbc0 = [1:3, 1:4, 3:4, 3:4, 3:7, 5:9, 7:9, 9:11, 9:12, 11:12]

show_equality_result(
  "LBC Matrix column inband_els_range",
  compare_ranges,
  column_els_ranges_lbc0,
  [inband_els_range(lbc0, :, k) for k = 1:10],
)

show_equality_result(
  "BC Matrix column els_range",
  compare_ranges,
  column_els_ranges_lbc0,
  [inband_els_range(bc0, :, k) for k = 1:10],
)

upper_column_els_ranges_lbc0 =
  [1:0, 1:1, 3:3, 3:3, 3:3, 5:6, 7:7, 9:9, 9:9, 11:11]

show_equality_result(
  "LBC Matrix column upper_els_range",
  compare_ranges,
  upper_column_els_ranges_lbc0,
  [upper_inband_els_range(lbc0, :, k) for k = 1:10],
)

show_equality_result(
  "BC Matrix column upper_els_range",
  compare_ranges,
  upper_column_els_ranges_lbc0,
  [upper_inband_els_range(bc0, :, k) for k = 1:10],
)

middle_column_els_ranges_lbc0 =
  [1:1, 2:3, 4:4, 4:4, 4:4, 7:7, 8:9, 1:0, 10:11, 12:12]

show_equality_result(
  "LBC Matrix column middle_els_range",
  compare_ranges,
  middle_column_els_ranges_lbc0,
  [middle_inband_els_range(lbc0, :, k) for k = 1:10],
)

show_equality_result(
  "BC Matrix column middle_els_range",
  compare_ranges,
  middle_column_els_ranges_lbc0,
  [middle_inband_els_range(bc0, :, k) for k = 1:10],
)

lower_column_els_ranges_lbc0 =
  [2:3, 4:4, 1:0, 1:0, 5:7, 8:9, 1:0, 10:11, 12:12, 1:0]

show_equality_result(
  "LBC Matrix column lower_els_range",
  compare_ranges,
  lower_column_els_ranges_lbc0,
  [lower_inband_els_range(lbc0, :, k) for k = 1:10],
)

show_equality_result(
  "BC Matrix column lower_els_range",
  compare_ranges,
  lower_column_els_ranges_lbc0,
  [lower_inband_els_range(bc0, :, k) for k = 1:10],
)

bc1 = bc[5:10, 2:9]

column_els_ranges_bc1 = [1:0, 1:0, 1:0, 1:3, 1:5, 3:5, 5:6, 5:6]
show_equality_result(
  "BC Submatrix column els_range",
  compare_ranges,
  column_els_ranges_bc1,
  [inband_els_range(bc1, :, k) for k = 1:8],
)

column_upper_els_ranges_bc1 = [1:0, 1:0, 1:0, 1:0, 1:2, 3:3, 5:5, 5:5]
show_equality_result(
  "BC Submatrix column upper_els_range",
  compare_ranges,
  column_upper_els_ranges_bc1,
  [upper_inband_els_range(bc1, :, k) for k = 1:8],
)

column_upper_els_ranges_bc1 = [1:0, 1:0, 1:0, 1:0, 1:2, 3:3, 5:5, 5:5]
show_equality_result(
  "BC Submatrix column upper_els_range",
  compare_ranges,
  column_upper_els_ranges_bc1,
  [upper_inband_els_range(bc1, :, k) for k = 1:8],
)

column_middle_els_ranges_bc1 = [1:0, 1:0, 1:0, 1:0, 3:3, 4:5, 1:0, 6:6]
show_equality_result(
  "BC Submatrix column middle_els_range",
  compare_ranges,
  column_middle_els_ranges_bc1,
  [middle_inband_els_range(bc1, :, k) for k = 1:8],
)

column_lower_els_ranges_bc1 = [1:0, 1:0, 1:0, 1:3, 4:5, 1:0, 6:6, 1:0]
show_equality_result(
  "BC Submatrix column lower_els_range",
  compare_ranges,
  column_lower_els_ranges_bc1,
  [lower_inband_els_range(bc1, :, k) for k = 1:8],
)

row_els_ranges_bc1 = [4:5, 4:5, 4:6, 5:6, 5:8, 7:8]
show_equality_result(
  "BC Submatrix row els_range",
  compare_ranges,
  row_els_ranges_bc1,
  [inband_els_range(bc1, j, : ) for j = 1:6],
)

row_upper_els_ranges_bc1 = [5:5, 5:5, 6:6, 1:0, 7:8, 1:0]
show_equality_result(
  "BC Submatrix row upper_els_range",
  compare_ranges,
  row_upper_els_ranges_bc1,
  [upper_inband_els_range(bc1, j, : ) for j = 1:6],
)

row_middle_els_ranges_bc1 = [1:0, 1:0, 5:5, 6:6, 6:6, 8:8]
show_equality_result(
  "BC Submatrix row middle_els_range",
  compare_ranges,
  row_middle_els_ranges_bc1,
  [middle_inband_els_range(bc1, j, : ) for j = 1:6],
)

row_lower_els_ranges_bc1 = [4:4, 4:4, 4:4, 5:5, 5:5, 7:7]
show_equality_result(
  "BC Submatrix row lower_els_range",
  compare_ranges,
  row_lower_els_ranges_bc1,
  [lower_inband_els_range(bc1, j, : ) for j = 1:6],
)
