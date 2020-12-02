bc = copy(bc0)
lbc = copy(lbc0)
mx_bc = copy(mx_bc0)

println("""

****************************
** Bandwidth extension tests
****************************

""")

# Extend the bandwidth of the example matrix from the documentation
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

lbc0_ext = copy(lbc0)

# wilk0 = [
#   'X' 'X' 'X' 'U' 'O' 'O' 'N'
#   'X' 'X' 'X' 'X' 'U' 'U' 'O'
#   'O' 'L' 'X' 'X' 'U' 'U' 'O'
#   'O' 'L' 'X' 'X' 'X' 'X' 'U'
#   'O' 'O' 'L' 'X' 'X' 'X' 'X'
#   'O' 'O' 'O' 'L' 'X' 'X' 'X'
#   'N' 'N' 'N' 'L' 'X' 'X' 'X'
#   'N' 'N' 'N' 'N' 'O' 'L' 'X'
# ]

wilk0 = wilk(lbc0_ext[1:end,1:end])

lbc1_ext = copy(lbc0_ext)
wilk1 = [
  'X' 'X' 'X' 'U' 'O' 'O' 'N'
  'X' 'X' 'X' 'X' 'U' 'U' 'O'
  'L' 'L' 'X' 'X' 'U' 'U' 'O'
  'L' 'L' 'X' 'X' 'X' 'X' 'U'
  'O' 'O' 'L' 'X' 'X' 'X' 'X'
  'O' 'O' 'O' 'L' 'X' 'X' 'X'
  'N' 'N' 'N' 'L' 'X' 'X' 'X'
  'N' 'N' 'N' 'N' 'O' 'L' 'X'
]

extend_band!(lbc1_ext,:, 1)
show_bool_result(
  "LBC extend_band! validate_rbws (column 1)",
  validate_rbws(lbc1_ext),
)

lbc2_ext = copy(lbc0_ext)
wilkP = [
  'X' 'X' 'X' 'U' 'O' 'O' 'N'
  'X' 'X' 'X' 'X' 'U' 'U' 'O'
  'O' 'L' 'X' 'X' 'U' 'U' 'O'
  'O' 'L' 'X' 'X' 'X' 'X' 'U'
  'O' 'L' 'L' 'X' 'X' 'X' 'X'
  'O' 'O' 'O' 'L' 'X' 'X' 'X'
  'N' 'N' 'N' 'L' 'X' 'X' 'X'
  'N' 'N' 'N' 'N' 'O' 'L' 'X'
]

extend_band!(lbc2_ext,:, 2)
show_bool_result(
  "LBC extend_band! validate_rbws (column 2)",
  validate_rbws(lbc2_ext),
)

lbc4_ext = copy(lbc0_ext)
wilk4 = [
  'X' 'X' 'X' 'U' 'U' 'O' 'N'
  'X' 'X' 'X' 'X' 'U' 'U' 'O'
  'O' 'L' 'X' 'X' 'U' 'U' 'O'
  'O' 'L' 'X' 'X' 'X' 'X' 'U'
  'O' 'O' 'L' 'X' 'X' 'X' 'X'
  'O' 'O' 'O' 'L' 'X' 'X' 'X'
  'N' 'N' 'N' 'L' 'X' 'X' 'X'
  'N' 'N' 'N' 'N' 'O' 'L' 'X'
]

extend_band!(lbc4_ext,:, 4)
show_bool_result(
  "LBC extend_band! validate_rbws (column 4)",
  validate_rbws(lbc4_ext),
)

show_equality_result(
  "LBC extend_band! Wilkinson equality (column 4)",
  ==,
  wilk(lbc4_ext[1:end, 1:end]).arr,
  wilk4,
)

lbc5_ext = copy(lbc0_ext)
wilk5 = [
  'X' 'X' 'X' 'U' 'O' 'O' 'N'
  'X' 'X' 'X' 'X' 'U' 'U' 'O'
  'O' 'L' 'X' 'X' 'U' 'U' 'O'
  'O' 'L' 'X' 'X' 'X' 'X' 'U'
  'O' 'O' 'L' 'X' 'X' 'X' 'X'
  'O' 'O' 'O' 'L' 'X' 'X' 'X'
  'N' 'N' 'N' 'L' 'X' 'X' 'X'
  'N' 'N' 'N' 'N' 'L' 'L' 'X'
]

extend_band!(lbc5_ext,:, 5)
show_bool_result(
  "LBC extend_band! validate_rbws (column 5)",
  validate_rbws(lbc5_ext),
)

lbc6_ext = copy(lbc0_ext)
wilk6 = [
  'X' 'X' 'X' 'U' 'O' 'O' 'N'
  'X' 'X' 'X' 'X' 'U' 'U' 'U'
  'O' 'L' 'X' 'X' 'U' 'U' 'U'
  'O' 'L' 'X' 'X' 'X' 'X' 'U'
  'O' 'O' 'L' 'X' 'X' 'X' 'X'
  'O' 'O' 'O' 'L' 'X' 'X' 'X'
  'N' 'N' 'N' 'L' 'X' 'X' 'X'
  'N' 'N' 'N' 'N' 'O' 'L' 'X'
]

extend_band!(lbc6_ext,:, 6)
show_bool_result(
  "LBC extend_band! validate_rbws (column 6)",
  validate_rbws(lbc6_ext),
)

# Rows

lbcr1_ext = copy(lbc0_ext)
wilkr1 = [
  'X' 'X' 'X' 'U' 'U' 'U' 'N'
  'X' 'X' 'X' 'X' 'U' 'U' 'O'
  'O' 'L' 'X' 'X' 'U' 'U' 'O'
  'O' 'L' 'X' 'X' 'X' 'X' 'U'
  'O' 'O' 'L' 'X' 'X' 'X' 'X'
  'O' 'O' 'O' 'L' 'X' 'X' 'X'
  'N' 'N' 'N' 'L' 'X' 'X' 'X'
  'N' 'N' 'N' 'N' 'O' 'L' 'X'
]

extend_band!(lbcr1_ext, 1, :)
show_bool_result(
  "LBC extend_band! validate_rbws (row 1)",
  validate_rbws(lbcr1_ext),
)
show_equality_result(
  "LBC extend_band! Wilkinson equality (row 1)",
  ==,
  wilk(lbcr1_ext[1:end, 1:end]).arr,
  wilkr1,
)


lbcr2_ext = copy(lbc0_ext)
wilkr2 = [
  'X' 'X' 'X' 'U' 'O' 'O' 'N'
  'X' 'X' 'X' 'X' 'U' 'U' 'O'
  'L' 'L' 'X' 'X' 'U' 'U' 'O'
  'O' 'L' 'X' 'X' 'X' 'X' 'U'
  'O' 'O' 'L' 'X' 'X' 'X' 'X'
  'O' 'O' 'O' 'L' 'X' 'X' 'X'
  'N' 'N' 'N' 'L' 'X' 'X' 'X'
  'N' 'N' 'N' 'N' 'O' 'L' 'X'
]

extend_band!(lbcr2_ext, 2, :)
show_bool_result(
  "LBC extend_band! validate_rbws (row 2)",
  validate_rbws(lbcr2_ext),
)
show_equality_result(
  "LBC extend_band! Wilkinson equality (row 2)",
  ==,
  wilk(lbcr2_ext[1:end, 1:end]).arr,
  wilkr2,
)

lbcr3_ext = copy(lbc0_ext)
wilkr3 = [
  'X' 'X' 'X' 'U' 'O' 'O' 'N'
  'X' 'X' 'X' 'X' 'U' 'U' 'O'
  'O' 'L' 'X' 'X' 'U' 'U' 'U'
  'O' 'L' 'X' 'X' 'X' 'X' 'U'
  'O' 'O' 'L' 'X' 'X' 'X' 'X'
  'O' 'O' 'O' 'L' 'X' 'X' 'X'
  'N' 'N' 'N' 'L' 'X' 'X' 'X'
  'N' 'N' 'N' 'N' 'O' 'L' 'X'
]

extend_band!(lbcr3_ext, 3, :)
show_bool_result(
  "LBC extend_band! validate_rbws (row 3)",
  validate_rbws(lbcr3_ext),
)
show_equality_result(
  "LBC extend_band! Wilkinson equality (row 3)",
  ==,
  wilk(lbcr3_ext[1:end, 1:end]).arr,
  wilkr3,
)

lbcr4_ext = copy(lbc0_ext)
wilkr4 = [
  'X' 'X' 'X' 'U' 'O' 'O' 'N'
  'X' 'X' 'X' 'X' 'U' 'U' 'O'
  'O' 'L' 'X' 'X' 'U' 'U' 'O'
  'O' 'L' 'X' 'X' 'X' 'X' 'U'
  'O' 'L' 'L' 'X' 'X' 'X' 'X'
  'O' 'O' 'O' 'L' 'X' 'X' 'X'
  'N' 'N' 'N' 'L' 'X' 'X' 'X'
  'N' 'N' 'N' 'N' 'O' 'L' 'X'
]

extend_band!(lbcr4_ext, 4, :)
show_bool_result(
  "LBC extend_band! validate_rbws (row 4)",
  validate_rbws(lbcr4_ext),
)
show_equality_result(
  "LBC extend_band! Wilkinson equality (row 4)",
  ==,
  wilk(lbcr4_ext[1:end, 1:end]).arr,
  wilkr4,
)

lbcr5_ext = copy(lbc0_ext)
wilkr5 = [
  'X' 'X' 'X' 'U' 'O' 'O' 'N'
  'X' 'X' 'X' 'X' 'U' 'U' 'O'
  'O' 'L' 'X' 'X' 'U' 'U' 'O'
  'O' 'L' 'X' 'X' 'X' 'X' 'U'
  'O' 'O' 'L' 'X' 'X' 'X' 'X'
  'O' 'O' 'L' 'L' 'X' 'X' 'X'
  'N' 'N' 'N' 'L' 'X' 'X' 'X'
  'N' 'N' 'N' 'N' 'O' 'L' 'X'
]

extend_band!(lbcr5_ext, 5, :)
show_bool_result(
  "LBC extend_band! validate_rbws (row 5)",
  validate_rbws(lbcr5_ext),
)
show_equality_result(
  "LBC extend_band! Wilkinson equality (row 5)",
  ==,
  wilk(lbcr5_ext[1:end, 1:end]).arr,
  wilkr5,
)

lbcr6_ext = copy(lbc0_ext)
wilkr6 = [
  'X' 'X' 'X' 'U' 'O' 'O' 'N'
  'X' 'X' 'X' 'X' 'U' 'U' 'O'
  'O' 'L' 'X' 'X' 'U' 'U' 'O'
  'O' 'L' 'X' 'X' 'X' 'X' 'U'
  'O' 'O' 'L' 'X' 'X' 'X' 'X'
  'O' 'O' 'O' 'L' 'X' 'X' 'X'
  'N' 'N' 'N' 'L' 'X' 'X' 'X'
  'N' 'N' 'N' 'N' 'O' 'L' 'X'
]

extend_band!(lbcr6_ext, 6, :)
show_bool_result(
  "LBC extend_band! validate_rbws (row 6)",
  validate_rbws(lbcr6_ext),
)
show_equality_result(
  "LBC extend_band! Wilkinson equality (row 6)",
  ==,
  wilk(lbcr6_ext[1:end, 1:end]).arr,
  wilkr6,
)

####
## Scalar extend
####
# 16

lbce16_ext = copy(lbc0_ext)
wilke16 = [
  'X' 'X' 'X' 'U' 'U' 'U' 'N'
  'X' 'X' 'X' 'X' 'U' 'U' 'O'
  'O' 'L' 'X' 'X' 'U' 'U' 'O'
  'O' 'L' 'X' 'X' 'X' 'X' 'U'
  'O' 'O' 'L' 'X' 'X' 'X' 'X'
  'O' 'O' 'O' 'L' 'X' 'X' 'X'
  'N' 'N' 'N' 'L' 'X' 'X' 'X'
  'N' 'N' 'N' 'N' 'O' 'L' 'X'
]


extend_band!(lbce16_ext, 1, 6)
show_bool_result(
  "LBC extend_band! validate_rbws (element (1,6))",
  validate_rbws(lbce16_ext),
)
show_equality_result(
  "LBC extend_band! Wilkinson equality (element (1,6))",
  ==,
  wilk(lbce16_ext[1:end, 1:end]).arr,
  wilke16,
)

lbce61_ext = copy(lbc0_ext)
wilke61 = [
  'X' 'X' 'X' 'U' 'O' 'O' 'N'
  'X' 'X' 'X' 'X' 'U' 'U' 'O'
  'L' 'L' 'X' 'X' 'U' 'U' 'O'
  'L' 'L' 'X' 'X' 'X' 'X' 'U'
  'L' 'L' 'L' 'X' 'X' 'X' 'X'
  'L' 'L' 'L' 'L' 'X' 'X' 'X'
  'N' 'N' 'N' 'L' 'X' 'X' 'X'
  'N' 'N' 'N' 'N' 'O' 'L' 'X'
]

extend_band!(lbce61_ext, 6, 1)
show_bool_result(
  "LBC extend_band! validate_rbws (element (6,1))",
  validate_rbws(lbce61_ext),
)
show_equality_result(
  "LBC extend_band! Wilkinson equality (element (6,1))",
  ==,
  wilk(lbce61_ext[1:end, 1:end]).arr,
  wilke61,
)

# Scalar (6,2)

lbce62_ext = copy(lbc0_ext)
wilke62 = [
  'X' 'X' 'X' 'U' 'O' 'O' 'N'
  'X' 'X' 'X' 'X' 'U' 'U' 'O'
  'O' 'L' 'X' 'X' 'U' 'U' 'O'
  'O' 'L' 'X' 'X' 'X' 'X' 'U'
  'O' 'L' 'L' 'X' 'X' 'X' 'X'
  'O' 'L' 'L' 'L' 'X' 'X' 'X'
  'N' 'N' 'N' 'L' 'X' 'X' 'X'
  'N' 'N' 'N' 'N' 'O' 'L' 'X'
]

extend_band!(lbce62_ext, 6, 2)
show_bool_result(
  "LBC extend_band! validate_rbws (element (6,2))",
  validate_rbws(lbce62_ext),
)
show_equality_result(
  "LBC extend_band! Wilkinson equality (element (6,2))",
  ==,
  wilk(lbce62_ext[1:end, 1:end]).arr,
  wilke62,
)
