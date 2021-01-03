bc = copy(bc0)
lbc = copy(lbc0)
mx_bc = copy(mx_bc0)

println("""

****************************
** Bandwidth bulge tests
****************************

""")


#=

Bulge the bandwidth of the example matrix from the documentation for
LeadingBandColumn.

                1   2       3   4
A =   X   X   X | U | O   O | N |
                +---+-----------+
      X   X   X   X | U   U | O |
    1 ------+       |       |   |
      O   L | X   X | U   U | O |
            |       +-------+---+
      O   L | X   X   X   X | U |
    2 ------+---+           +---+
      O   O | L | X   X   X   X |
    3 ------+---+---+           |
      O   O | O | L | X   X   X |
            |   |   |           +
      N   N | N | L | X   X   X 
    4 ------+---+---+-------+   
      N   N | N | N | O   L | X 

=#

lbc0_bulge = copy(lbc0)

wilk0 = wilk(toBandColumn(lbc0_bulge))

lbc1_bulge = copy(lbc0_bulge)
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

bulge!(lbc1_bulge, :, 1:2)
show_bool_result(
  "LBC bulge! validate_rows_first_last (column 1)",
  validate_rows_first_last(lbc1_bulge),
)

show_equality_result(
  "LBC bulge! Wilkinson equality (column 1)",
  ==,
  wilk(toBandColumn(lbc1_bulge)).arr,
  wilk1,
)


lbc2_bulge = copy(lbc0_bulge)
wilk2 = [
  'X' 'X' 'X' 'U' 'O' 'O' 'N'
  'X' 'X' 'X' 'X' 'U' 'U' 'O'
  'O' 'L' 'X' 'X' 'U' 'U' 'O'
  'O' 'L' 'X' 'X' 'X' 'X' 'U'
  'O' 'L' 'L' 'X' 'X' 'X' 'X'
  'O' 'O' 'O' 'L' 'X' 'X' 'X'
  'N' 'N' 'N' 'L' 'X' 'X' 'X'
  'N' 'N' 'N' 'N' 'O' 'L' 'X'
]

bulge!(lbc2_bulge, :, 2:3)
show_bool_result(
  "LBC bulge! validate_rows_first_last (column 2)",
  validate_rows_first_last(lbc2_bulge),
)

show_equality_result(
  "LBC bulge! Wilkinson equality (column 2)",
  ==,
  wilk(toBandColumn(lbc2_bulge)).arr,
  wilk2,
)


lbc4_bulge = copy(lbc0_bulge)
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

bulge!(lbc4_bulge,:, 4:5)
show_bool_result(
  "LBC bulge! validate_rows_first_last (column 4)",
  validate_rows_first_last(lbc4_bulge),
)

show_equality_result(
  "LBC bulge! Wilkinson equality (column 4)",
  ==,
  wilk(toBandColumn(lbc4_bulge)).arr,
  wilk4,
)

lbc5_bulge = copy(lbc0_bulge)
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

bulge!(lbc5_bulge,:, 5:6)
show_bool_result(
  "LBC bulge! validate_rows_first_last (column 5)",
  validate_rows_first_last(lbc5_bulge),
)

show_equality_result(
  "LBC bulge! Wilkinson equality (column 5)",
  ==,
  wilk(toBandColumn(lbc5_bulge)).arr,
  wilk5,
)

lbc6_bulge = copy(lbc0_bulge)
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

bulge!(lbc6_bulge,:, 6:7)
show_bool_result(
  "LBC bulge! validate_rows_first_last (column 6)",
  validate_rows_first_last(lbc6_bulge),
)

show_equality_result(
  "LBC bulge! Wilkinson equality (column 6)",
  ==,
  wilk(toBandColumn(lbc6_bulge)).arr,
  wilk6,
)

# Rows

lbcr1_bulge = copy(lbc0_bulge)
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

bulge!(lbcr1_bulge, 1:2, :)
show_bool_result(
  "LBC bulge! validate_rows_first_last (row 1)",
  validate_rows_first_last(lbcr1_bulge),
)

show_equality_result(
  "LBC bulge! Wilkinson equality (row 1)",
  ==,
  wilk(lbcr1_bulge[1:end, 1:end]).arr,
  wilkr1,
)


lbcr2_bulge = copy(lbc0_bulge)
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

bulge!(lbcr2_bulge, 2:3, :)
show_bool_result(
  "LBC bulge! validate_rows_first_last (row 2)",
  validate_rows_first_last(lbcr2_bulge),
)

show_equality_result(
  "LBC bulge! Wilkinson equality (row 2)",
  ==,
  wilk(lbcr2_bulge[1:end, 1:end]).arr,
  wilkr2,
)

lbcr3_bulge = copy(lbc0_bulge)
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

bulge!(lbcr3_bulge, 3:4, :)
show_bool_result(
  "LBC bulge! validate_rows_first_last (row 3)",
  validate_rows_first_last(lbcr3_bulge),
)

show_equality_result(
  "LBC bulge! Wilkinson equality (row 3)",
  ==,
  wilk(lbcr3_bulge[1:end, 1:end]).arr,
  wilkr3,
)

lbcr4_bulge = copy(lbc0_bulge)
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

bulge!(lbcr4_bulge, 4:5, :)
show_bool_result(
  "LBC bulge! validate_rows_first_last (row 4)",
  validate_rows_first_last(lbcr4_bulge),
)

show_equality_result(
  "LBC bulge! Wilkinson equality (row 4)",
  ==,
  wilk(lbcr4_bulge[1:end, 1:end]).arr,
  wilkr4,
)

lbcr5_bulge = copy(lbc0_bulge)
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

bulge!(lbcr5_bulge, 5:6, :)
show_bool_result(
  "LBC bulge! validate_rows_first_last (row 5)",
  validate_rows_first_last(lbcr5_bulge),
)

show_equality_result(
  "LBC bulge! Wilkinson equality (row 5)",
  ==,
  wilk(lbcr5_bulge[1:end, 1:end]).arr,
  wilkr5,
)

lbcr6_bulge = copy(lbc0_bulge)
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

bulge!(lbcr6_bulge, 6:7, :)
show_bool_result(
  "LBC bulge! validate_rows_first_last (row 6)",
  validate_rows_first_last(lbcr6_bulge),
)

show_equality_result(
  "LBC bulge! Wilkinson equality (row 6)",
  ==,
  wilk(lbcr6_bulge[1:end, 1:end]).arr,
  wilkr6,
)

####
## Index bulge
####
# 16

lbci16_bulge = copy(lbc0_bulge)
wilki16 = [
  'X' 'X' 'X' 'U' 'U' 'U' 'N'
  'X' 'X' 'X' 'X' 'U' 'U' 'O'
  'O' 'L' 'X' 'X' 'U' 'U' 'O'
  'O' 'L' 'X' 'X' 'X' 'X' 'U'
  'O' 'O' 'L' 'X' 'X' 'X' 'X'
  'O' 'O' 'O' 'L' 'X' 'X' 'X'
  'N' 'N' 'N' 'L' 'X' 'X' 'X'
  'N' 'N' 'N' 'N' 'O' 'L' 'X'
]

bulge!(lbci16_bulge, 1, 6)
show_bool_result(
  "LBC bulge! validate_rows_first_last (index (1,6))",
  validate_rows_first_last(lbci16_bulge),
)

show_equality_result(
  "LBC bulge! Wilkinson equality (index (1,6))",
  ==,
  wilk(toBandColumn(lbci16_bulge)).arr,
  wilki16,
)

lbci61_bulge = copy(lbc0_bulge)
wilki61 = [
  'X' 'X' 'X' 'U' 'O' 'O' 'N'
  'X' 'X' 'X' 'X' 'U' 'U' 'O'
  'L' 'L' 'X' 'X' 'U' 'U' 'O'
  'L' 'L' 'X' 'X' 'X' 'X' 'U'
  'L' 'L' 'L' 'X' 'X' 'X' 'X'
  'L' 'L' 'L' 'L' 'X' 'X' 'X'
  'N' 'N' 'N' 'L' 'X' 'X' 'X'
  'N' 'N' 'N' 'N' 'O' 'L' 'X'
]

bulge!(lbci61_bulge, 6, 1)
show_bool_result(
  "LBC bulge! validate_rows_first_last (index (6,1))",
  validate_rows_first_last(lbci61_bulge),
)

show_equality_result(
  "LBC bulge! Wilkinson equality (index (6,1))",
  ==,
  wilk(toBandColumn(lbci61_bulge)).arr,
  wilki61,
)

# Scalar (6,2)

lbci62_bulge = copy(lbc0_bulge)
wilki62 = [
  'X' 'X' 'X' 'U' 'O' 'O' 'N'
  'X' 'X' 'X' 'X' 'U' 'U' 'O'
  'O' 'L' 'X' 'X' 'U' 'U' 'O'
  'O' 'L' 'X' 'X' 'X' 'X' 'U'
  'O' 'L' 'L' 'X' 'X' 'X' 'X'
  'O' 'L' 'L' 'L' 'X' 'X' 'X'
  'N' 'N' 'N' 'L' 'X' 'X' 'X'
  'N' 'N' 'N' 'N' 'O' 'L' 'X'
]

bulge!(lbci62_bulge, 6, 2)
show_bool_result(
  "LBC bulge! validate_rows_first_last (index (6,2))",
  validate_rows_first_last(lbci62_bulge),
)

show_equality_result(
  "LBC bulge! Wilkinson equality (index (6,2))",
  ==,
  wilk(toBandColumn(lbci62_bulge)).arr,
  wilki62,
)
