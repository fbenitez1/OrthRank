bc = copy(bc0)
bbc = copy(bbc0)
mx_bc = copy(mx_bc0)

println("""

****************************
** Bandwidth notch tests
****************************

""")

# Notch the bandwidth of the example matrix from the documentation
# for BlockedBandColumn.

#                 1   2       3   4
# A =   X   X   X | U | O   O | O |
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
#       N   O | O | L | X   X   X 
#     4 ------+---+---+-------+   
#       N   N | O | O | O   L | X 

bbc0_notch = copy(bbc0)
wilk0_notch = wilk(toBandColumn(bbc0_notch))

# Index (1,4) notch upper.

bbc14_notch = copy(bbc0_notch)
wilk14_notch = [
  'X' 'X' 'X' 'O' 'O' 'O' 'O'
  'X' 'X' 'X' 'X' 'U' 'U' 'O'
  'O' 'L' 'X' 'X' 'U' 'U' 'O'
  'O' 'L' 'X' 'X' 'X' 'X' 'U'
  'O' 'O' 'L' 'X' 'X' 'X' 'X'
  'O' 'O' 'O' 'L' 'X' 'X' 'X'
  'N' 'O' 'O' 'L' 'X' 'X' 'X'
  'N' 'N' 'O' 'O' 'O' 'L' 'X'
]

notch_upper!(bbc14_notch,1, 4)
show_bool_result(
  "BBC notch_upper! column validate_rows_first_last, index (1,4)",
  validate_rows_first_last(bbc14_notch),
)
show_equality_result(
  "BBC notch_upper! column Wilkinson equality, index (1,4)",
  ==,
  wilk(toBandColumn(bbc14_notch)).arr,
  wilk14_notch,
)

# Index (2,6)

bbc26_notch = copy(bbc0_notch)
wilk26_notch = [
  'X' 'X' 'X' 'U' 'O' 'O' 'O'
  'X' 'X' 'X' 'X' 'U' 'O' 'O'
  'O' 'L' 'X' 'X' 'U' 'U' 'O'
  'O' 'L' 'X' 'X' 'X' 'X' 'U'
  'O' 'O' 'L' 'X' 'X' 'X' 'X'
  'O' 'O' 'O' 'L' 'X' 'X' 'X'
  'N' 'O' 'O' 'L' 'X' 'X' 'X'
  'N' 'N' 'O' 'O' 'O' 'L' 'X'
]

notch_upper!(bbc26_notch, 2, 6)
show_bool_result(
  "BBC notch_upper! column validate_rows_first_last, index (2,6)",
  validate_rows_first_last(bbc26_notch),
)

show_equality_result(
  "BBC notch_upper! column Wilkinson equality, index (2,6)",
  ==,
  wilk(toBandColumn(bbc26_notch)).arr,
  wilk26_notch,
)

# notch_lower!, index (3,2)

bbc32_notch = copy(bbc0_notch)
wilk32_notch = [
  'X' 'X' 'X' 'U' 'O' 'O' 'O'
  'X' 'X' 'X' 'X' 'U' 'U' 'O'
  'O' 'O' 'X' 'X' 'U' 'U' 'O'
  'O' 'O' 'X' 'X' 'X' 'X' 'U'
  'O' 'O' 'L' 'X' 'X' 'X' 'X'
  'O' 'O' 'O' 'L' 'X' 'X' 'X'
  'N' 'O' 'O' 'L' 'X' 'X' 'X'
  'N' 'N' 'O' 'O' 'O' 'L' 'X'
]

notch_lower!(bbc32_notch, 3, 2)
show_bool_result(
  "BBC notch_lower! column validate_rows_first_last, index (3,2)",
  validate_rows_first_last(bbc32_notch),
)
show_equality_result(
  "BBC notch_lower! column Wilkinson equality, index (3,2)",
  ==,
  wilk(toBandColumn(bbc32_notch)).arr,
  wilk32_notch,
)

# notch_lower!, index (4,2)

bbc42_notch = copy(bbc0_notch)
wilk42_notch = [
  'X' 'X' 'X' 'U' 'O' 'O' 'O'
  'X' 'X' 'X' 'X' 'U' 'U' 'O'
  'O' 'L' 'X' 'X' 'U' 'U' 'O'
  'O' 'O' 'X' 'X' 'X' 'X' 'U'
  'O' 'O' 'L' 'X' 'X' 'X' 'X'
  'O' 'O' 'O' 'L' 'X' 'X' 'X'
  'N' 'O' 'O' 'L' 'X' 'X' 'X'
  'N' 'N' 'O' 'O' 'O' 'L' 'X'
]

notch_lower!(bbc42_notch, 4, 2)
show_bool_result(
  "BBC notch_lower! column validate_rows_first_last, index (4,2)",
  validate_rows_first_last(bbc42_notch),
)
show_equality_result(
  "BBC notch_lower! column Wilkinson equality, index (4,2)",
  ==,
  wilk(toBandColumn(bbc42_notch)).arr,
  wilk42_notch,
)

# notch_lower!, index (6,4).

bbcr64_notch = copy(bbc0_notch)
wilkr64_notch = [
  'X' 'X' 'X' 'U' 'O' 'O' 'O'
  'X' 'X' 'X' 'X' 'U' 'U' 'O'
  'O' 'L' 'X' 'X' 'U' 'U' 'O'
  'O' 'L' 'X' 'X' 'X' 'X' 'U'
  'O' 'O' 'L' 'X' 'X' 'X' 'X'
  'O' 'O' 'O' 'O' 'X' 'X' 'X'
  'N' 'O' 'O' 'O' 'X' 'X' 'X'
  'N' 'N' 'O' 'O' 'O' 'L' 'X'
]

notch_lower!(bbcr64_notch, 6, 4)
show_bool_result(
  "BBC notch_lower! row validate_rows_first_last, index (6,4)",
  validate_rows_first_last(bbcr64_notch),
)
show_equality_result(
  "BBC notch_lower! row Wilkinson equality, index (6,4)",
  ==,
  wilk(toBandColumn(bbcr64_notch)).arr,
  wilkr64_notch,
)
