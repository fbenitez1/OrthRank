bc = copy(bc0)
lbc = copy(lbc0)
mx_bc = copy(mx_bc0)

println("""

****************************
** Bandwidth notch tests
****************************

""")

# Notch the bandwidth of the example matrix from the documentation
# for LeadingBandColumn.

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


lbc0_notch = copy(lbc0)
wilk0_notch = wilk(toBandColumn(lbc0_notch))

# Index (1,4) notch upper.

lbc14_notch = copy(lbc0_notch)
wilk14_notch = [
  'X' 'X' 'X' 'O' 'O' 'O' 'N'
  'X' 'X' 'X' 'X' 'U' 'U' 'O'
  'O' 'L' 'X' 'X' 'U' 'U' 'O'
  'O' 'L' 'X' 'X' 'X' 'X' 'U'
  'O' 'O' 'L' 'X' 'X' 'X' 'X'
  'O' 'O' 'O' 'L' 'X' 'X' 'X'
  'N' 'N' 'N' 'L' 'X' 'X' 'X'
  'N' 'N' 'N' 'N' 'O' 'L' 'X'
]

notch_upper!(lbc14_notch,1, 4)
show_bool_result(
  "LBC notch_upper! column validate_rows_first_last, index (1,4)",
  validate_rows_first_last(lbc14_notch),
)
show_equality_result(
  "LBC notch_upper! column Wilkinson equality, index (1,4)",
  ==,
  wilk(toBandColumn(lbc14_notch)).arr,
  wilk14_notch,
)

# Original index (3,5) notch_upper! done in a submatrix (3:4,
# 4:6). (Creates a well).

bc35_notch = lbc0_notch[3:4,4:6]
res = false
try
  notch_upper!(bc35_notch, 1, 2)
catch e
  isa(e, WellError) && global res = true
end
show_bool_result(
  "notch_upper! Submatrix [3:4, 4:6] WellError test, original index (3,5)",
  res,
)

# Index (2,6)

lbc26_notch = copy(lbc0_notch)
wilk26_notch = [
  'X' 'X' 'X' 'U' 'O' 'O' 'N'
  'X' 'X' 'X' 'X' 'U' 'O' 'O'
  'O' 'L' 'X' 'X' 'U' 'U' 'O'
  'O' 'L' 'X' 'X' 'X' 'X' 'U'
  'O' 'O' 'L' 'X' 'X' 'X' 'X'
  'O' 'O' 'O' 'L' 'X' 'X' 'X'
  'N' 'N' 'N' 'L' 'X' 'X' 'X'
  'N' 'N' 'N' 'N' 'O' 'L' 'X'
]

notch_upper!(lbc26_notch, 2, 6)
show_bool_result(
  "LBC notch_upper! column validate_rows_first_last, index (2,6)",
  validate_rows_first_last(lbc26_notch),
)

show_equality_result(
  "LBC notch_upper! column Wilkinson equality, index (2,6)",
  ==,
  wilk(toBandColumn(lbc26_notch)).arr,
  wilk26_notch,
)

# notch_lower!, index (3,2)

lbc32_notch = copy(lbc0_notch)
wilk32_notch = [
  'X' 'X' 'X' 'U' 'O' 'O' 'N'
  'X' 'X' 'X' 'X' 'U' 'U' 'O'
  'O' 'O' 'X' 'X' 'U' 'U' 'O'
  'O' 'O' 'X' 'X' 'X' 'X' 'U'
  'O' 'O' 'L' 'X' 'X' 'X' 'X'
  'O' 'O' 'O' 'L' 'X' 'X' 'X'
  'N' 'N' 'N' 'L' 'X' 'X' 'X'
  'N' 'N' 'N' 'N' 'O' 'L' 'X'
]

notch_lower!(lbc32_notch, 3, 2)
show_bool_result(
  "LBC notch_lower! column validate_rows_first_last, index (3,2)",
  validate_rows_first_last(lbc32_notch),
)
show_equality_result(
  "LBC notch_lower! column Wilkinson equality, index (3,2)",
  ==,
  wilk(toBandColumn(lbc32_notch)).arr,
  wilk32_notch,
)

# notch_lower!, index (4,2)

lbc42_notch = copy(lbc0_notch)
wilk42_notch = [
  'X' 'X' 'X' 'U' 'O' 'O' 'N'
  'X' 'X' 'X' 'X' 'U' 'U' 'O'
  'O' 'L' 'X' 'X' 'U' 'U' 'O'
  'O' 'O' 'X' 'X' 'X' 'X' 'U'
  'O' 'O' 'L' 'X' 'X' 'X' 'X'
  'O' 'O' 'O' 'L' 'X' 'X' 'X'
  'N' 'N' 'N' 'L' 'X' 'X' 'X'
  'N' 'N' 'N' 'N' 'O' 'L' 'X'
]

notch_lower!(lbc42_notch, 4, 2)
show_bool_result(
  "LBC notch_lower! column validate_rows_first_last, index (4,2)",
  validate_rows_first_last(lbc42_notch),
)
show_equality_result(
  "LBC notch_lower! column Wilkinson equality, index (4,2)",
  ==,
  wilk(toBandColumn(lbc42_notch)).arr,
  wilk42_notch,
)

# notch_lower!, index (6,4).

lbcr64_notch = copy(lbc0_notch)
wilkr64_notch = [
  'X' 'X' 'X' 'U' 'O' 'O' 'N'
  'X' 'X' 'X' 'X' 'U' 'U' 'O'
  'O' 'L' 'X' 'X' 'U' 'U' 'O'
  'O' 'L' 'X' 'X' 'X' 'X' 'U'
  'O' 'O' 'L' 'X' 'X' 'X' 'X'
  'O' 'O' 'O' 'O' 'X' 'X' 'X'
  'N' 'N' 'N' 'O' 'X' 'X' 'X'
  'N' 'N' 'N' 'N' 'O' 'L' 'X'
]

notch_lower!(lbcr64_notch, 6, 4)
show_bool_result(
  "LBC notch_lower! row validate_rows_first_last, index (6,4)",
  validate_rows_first_last(lbcr64_notch),
)
show_equality_result(
  "LBC notch_lower! row Wilkinson equality, index (6,4)",
  ==,
  wilk(toBandColumn(lbcr64_notch)).arr,
  wilkr64_notch,
)

# Original index (6,4) notch_lower! done in submatrix (4:6,
# 4:6). (Creates a well).

bc64w_notch = lbc0_notch[4:6,4:6]
res = false
try
  notch_lower!(bc64w_notch, 3, 1)
catch e
  isa(e, WellError) && global res = true
end
show_bool_result(
  "notch_upper! Submatrix [4:6, 4:6] WellError test, original index (6,4)",
  res,
)
