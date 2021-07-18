bc = copy(bc0)
bbc = copy(bbc0)
mx_bc = copy(mx_bc0)

println("""

****************************
** Bandwidth bulge tests
****************************

""")


#=

Bulge the bandwidth of the example matrix from the documentation for
BlockedBandColumn.

                1   2       3   4
A =   X   X   X | U | O   O | O |
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
      N   O | O | L | X   X   X 
    4 ------+---+---+-------+   
      N   N | O | O | O   L | X 

=#

bbc0_bulge = copy(bbc0)

wilk0 = wilk(toBandColumn(bbc0_bulge))

bbc1_bulge = copy(bbc0_bulge)
wilk1 = [
 'X'  'X'  'X'  'U'  'O'  'O'  'O'
 'X'  'X'  'X'  'X'  'U'  'U'  'O'
 'L'  'L'  'X'  'X'  'U'  'U'  'O'
 'L'  'L'  'X'  'X'  'X'  'X'  'U'
 'O'  'O'  'L'  'X'  'X'  'X'  'X'
 'O'  'O'  'O'  'L'  'X'  'X'  'X'
 'N'  'O'  'O'  'L'  'X'  'X'  'X'
 'N'  'N'  'O'  'O'  'O'  'L'  'X'
]

bulge!(bbc1_bulge, :, 1:2)
show_bool_result(
  "BBC bulge! validate_rows_first_last (column 1)",
  validate_rows_first_last(bbc1_bulge),
)

show_equality_result(
  "BBC bulge! Wilkinson equality (column 1)",
  ==,
  wilk(toBandColumn(bbc1_bulge)).arr,
  wilk1,
)


bbc2_bulge = copy(bbc0_bulge)
wilk2 = [
  'X' 'X' 'X' 'U' 'O' 'O' 'O'
  'X' 'X' 'X' 'X' 'U' 'U' 'O'
  'O' 'L' 'X' 'X' 'U' 'U' 'O'
  'O' 'L' 'X' 'X' 'X' 'X' 'U'
  'O' 'L' 'L' 'X' 'X' 'X' 'X'
  'O' 'O' 'O' 'L' 'X' 'X' 'X'
  'N' 'O' 'O' 'L' 'X' 'X' 'X'
  'N' 'N' 'O' 'O' 'O' 'L' 'X'
]

bulge!(bbc2_bulge, :, 2:3)
show_bool_result(
  "BBC bulge! validate_rows_first_last (column 2)",
  validate_rows_first_last(bbc2_bulge),
)

show_equality_result(
  "BBC bulge! Wilkinson equality (column 2)",
  ==,
  wilk(toBandColumn(bbc2_bulge)).arr,
  wilk2,
)


bbc4_bulge = copy(bbc0_bulge)
wilk4 = [
  'X' 'X' 'X' 'U' 'U' 'O' 'O'
  'X' 'X' 'X' 'X' 'U' 'U' 'O'
  'O' 'L' 'X' 'X' 'U' 'U' 'O'
  'O' 'L' 'X' 'X' 'X' 'X' 'U'
  'O' 'O' 'L' 'X' 'X' 'X' 'X'
  'O' 'O' 'O' 'L' 'X' 'X' 'X'
  'N' 'O' 'O' 'L' 'X' 'X' 'X'
  'N' 'N' 'O' 'O' 'O' 'L' 'X'
]

bulge!(bbc4_bulge,:, 4:5)
show_bool_result(
  "BBC bulge! validate_rows_first_last (column 4)",
  validate_rows_first_last(bbc4_bulge),
)

show_equality_result(
  "BBC bulge! Wilkinson equality (column 4)",
  ==,
  wilk(toBandColumn(bbc4_bulge)).arr,
  wilk4,
)

bbc5_bulge = copy(bbc0_bulge)
wilk5 = [
  'X' 'X' 'X' 'U' 'O' 'O' 'O'
  'X' 'X' 'X' 'X' 'U' 'U' 'O'
  'O' 'L' 'X' 'X' 'U' 'U' 'O'
  'O' 'L' 'X' 'X' 'X' 'X' 'U'
  'O' 'O' 'L' 'X' 'X' 'X' 'X'
  'O' 'O' 'O' 'L' 'X' 'X' 'X'
  'N' 'O' 'O' 'L' 'X' 'X' 'X'
  'N' 'N' 'O' 'O' 'L' 'L' 'X'
]

bulge!(bbc5_bulge,:, 5:6)
show_bool_result(
  "BBC bulge! validate_rows_first_last (column 5)",
  validate_rows_first_last(bbc5_bulge),
)

show_equality_result(
  "BBC bulge! Wilkinson equality (column 5)",
  ==,
  wilk(toBandColumn(bbc5_bulge)).arr,
  wilk5,
)

bbc6_bulge = copy(bbc0_bulge)
wilk6 = [
  'X' 'X' 'X' 'U' 'O' 'O' 'O'
  'X' 'X' 'X' 'X' 'U' 'U' 'U'
  'O' 'L' 'X' 'X' 'U' 'U' 'U'
  'O' 'L' 'X' 'X' 'X' 'X' 'U'
  'O' 'O' 'L' 'X' 'X' 'X' 'X'
  'O' 'O' 'O' 'L' 'X' 'X' 'X'
  'N' 'O' 'O' 'L' 'X' 'X' 'X'
  'N' 'N' 'O' 'O' 'O' 'L' 'X'
]

bulge!(bbc6_bulge,:, 6:7)
show_bool_result(
  "BBC bulge! validate_rows_first_last (column 6)",
  validate_rows_first_last(bbc6_bulge),
)

show_equality_result(
  "BBC bulge! Wilkinson equality (column 6)",
  ==,
  wilk(toBandColumn(bbc6_bulge)).arr,
  wilk6,
)

# Rows

bbcr1_bulge = copy(bbc0_bulge)
wilkr1 = [
  'X' 'X' 'X' 'U' 'U' 'U' 'O'
  'X' 'X' 'X' 'X' 'U' 'U' 'O'
  'O' 'L' 'X' 'X' 'U' 'U' 'O'
  'O' 'L' 'X' 'X' 'X' 'X' 'U'
  'O' 'O' 'L' 'X' 'X' 'X' 'X'
  'O' 'O' 'O' 'L' 'X' 'X' 'X'
  'N' 'O' 'O' 'L' 'X' 'X' 'X'
  'N' 'N' 'O' 'O' 'O' 'L' 'X'
]

bulge!(bbcr1_bulge, 1:2, :)
show_bool_result(
  "BBC bulge! validate_rows_first_last (row 1)",
  validate_rows_first_last(bbcr1_bulge),
)

show_equality_result(
  "BBC bulge! Wilkinson equality (row 1)",
  ==,
  wilk(bbcr1_bulge[1:end, 1:end]).arr,
  wilkr1,
)

bbcr2_bulge = copy(bbc0_bulge)
wilkr2 = [
  'X' 'X' 'X' 'U' 'O' 'O' 'O'
  'X' 'X' 'X' 'X' 'U' 'U' 'O'
  'L' 'L' 'X' 'X' 'U' 'U' 'O'
  'O' 'L' 'X' 'X' 'X' 'X' 'U'
  'O' 'O' 'L' 'X' 'X' 'X' 'X'
  'O' 'O' 'O' 'L' 'X' 'X' 'X'
  'N' 'O' 'O' 'L' 'X' 'X' 'X'
  'N' 'N' 'O' 'O' 'O' 'L' 'X'
]

bulge!(bbcr2_bulge, 2:3, :)
show_bool_result(
  "BBC bulge! validate_rows_first_last (row 2)",
  validate_rows_first_last(bbcr2_bulge),
)

show_equality_result(
  "BBC bulge! Wilkinson equality (row 2)",
  ==,
  wilk(bbcr2_bulge[1:end, 1:end]).arr,
  wilkr2,
)

bbcr3_bulge = copy(bbc0_bulge)
wilkr3 = [
  'X' 'X' 'X' 'U' 'O' 'O' 'O'
  'X' 'X' 'X' 'X' 'U' 'U' 'O'
  'O' 'L' 'X' 'X' 'U' 'U' 'U'
  'O' 'L' 'X' 'X' 'X' 'X' 'U'
  'O' 'O' 'L' 'X' 'X' 'X' 'X'
  'O' 'O' 'O' 'L' 'X' 'X' 'X'
  'N' 'O' 'O' 'L' 'X' 'X' 'X'
  'N' 'N' 'O' 'O' 'O' 'L' 'X'
]

bulge!(bbcr3_bulge, 3:4, :)
show_bool_result(
  "BBC bulge! validate_rows_first_last (row 3)",
  validate_rows_first_last(bbcr3_bulge),
)

show_equality_result(
  "BBC bulge! Wilkinson equality (row 3)",
  ==,
  wilk(bbcr3_bulge[1:end, 1:end]).arr,
  wilkr3,
)

bbcr4_bulge = copy(bbc0_bulge)
wilkr4 = [
  'X' 'X' 'X' 'U' 'O' 'O' 'O'
  'X' 'X' 'X' 'X' 'U' 'U' 'O'
  'O' 'L' 'X' 'X' 'U' 'U' 'O'
  'O' 'L' 'X' 'X' 'X' 'X' 'U'
  'O' 'L' 'L' 'X' 'X' 'X' 'X'
  'O' 'O' 'O' 'L' 'X' 'X' 'X'
  'N' 'O' 'O' 'L' 'X' 'X' 'X'
  'N' 'N' 'O' 'O' 'O' 'L' 'X'
]

bulge!(bbcr4_bulge, 4:5, :)
show_bool_result(
  "BBC bulge! validate_rows_first_last (row 4)",
  validate_rows_first_last(bbcr4_bulge),
)

show_equality_result(
  "BBC bulge! Wilkinson equality (row 4)",
  ==,
  wilk(bbcr4_bulge[1:end, 1:end]).arr,
  wilkr4,
)

bbcr5_bulge = copy(bbc0_bulge)
wilkr5 = [
  'X' 'X' 'X' 'U' 'O' 'O' 'O'
  'X' 'X' 'X' 'X' 'U' 'U' 'O'
  'O' 'L' 'X' 'X' 'U' 'U' 'O'
  'O' 'L' 'X' 'X' 'X' 'X' 'U'
  'O' 'O' 'L' 'X' 'X' 'X' 'X'
  'O' 'O' 'L' 'L' 'X' 'X' 'X'
  'N' 'O' 'O' 'L' 'X' 'X' 'X'
  'N' 'N' 'O' 'O' 'O' 'L' 'X'
]

bulge!(bbcr5_bulge, 5:6, :)
show_bool_result(
  "BBC bulge! validate_rows_first_last (row 5)",
  validate_rows_first_last(bbcr5_bulge),
)

show_equality_result(
  "BBC bulge! Wilkinson equality (row 5)",
  ==,
  wilk(bbcr5_bulge[1:end, 1:end]).arr,
  wilkr5,
)

bbcr6_bulge = copy(bbc0_bulge)
wilkr6 = [
  'X' 'X' 'X' 'U' 'O' 'O' 'O'
  'X' 'X' 'X' 'X' 'U' 'U' 'O'
  'O' 'L' 'X' 'X' 'U' 'U' 'O'
  'O' 'L' 'X' 'X' 'X' 'X' 'U'
  'O' 'O' 'L' 'X' 'X' 'X' 'X'
  'O' 'O' 'O' 'L' 'X' 'X' 'X'
  'N' 'O' 'O' 'L' 'X' 'X' 'X'
  'N' 'N' 'O' 'O' 'O' 'L' 'X'
]

bulge!(bbcr6_bulge, 6:7, :)
show_bool_result(
  "BBC bulge! validate_rows_first_last (row 6)",
  validate_rows_first_last(bbcr6_bulge),
)

show_equality_result(
  "BBC bulge! Wilkinson equality (row 6)",
  ==,
  wilk(bbcr6_bulge[1:end, 1:end]).arr,
  wilkr6,
)

####
## Index bulge
####
# 16

bbci16_bulge = copy(bbc0_bulge)
wilki16 = [
  'X' 'X' 'X' 'U' 'U' 'U' 'O'
  'X' 'X' 'X' 'X' 'U' 'U' 'O'
  'O' 'L' 'X' 'X' 'U' 'U' 'O'
  'O' 'L' 'X' 'X' 'X' 'X' 'U'
  'O' 'O' 'L' 'X' 'X' 'X' 'X'
  'O' 'O' 'O' 'L' 'X' 'X' 'X'
  'N' 'O' 'O' 'L' 'X' 'X' 'X'
  'N' 'N' 'O' 'O' 'O' 'L' 'X'
]

bulge!(bbci16_bulge, 1, 6)
show_bool_result(
  "BBC bulge! validate_rows_first_last (index (1,6))",
  validate_rows_first_last(bbci16_bulge),
)

show_equality_result(
  "BBC bulge! Wilkinson equality (index (1,6))",
  ==,
  wilk(toBandColumn(bbci16_bulge)).arr,
  wilki16,
)

bbci61_bulge = copy(bbc0_bulge)
wilki61 = [
  'X' 'X' 'X' 'U' 'O' 'O' 'O'
  'X' 'X' 'X' 'X' 'U' 'U' 'O'
  'L' 'L' 'X' 'X' 'U' 'U' 'O'
  'L' 'L' 'X' 'X' 'X' 'X' 'U'
  'L' 'L' 'L' 'X' 'X' 'X' 'X'
  'L' 'L' 'L' 'L' 'X' 'X' 'X'
  'N' 'O' 'O' 'L' 'X' 'X' 'X'
  'N' 'N' 'O' 'O' 'O' 'L' 'X'
]

bulge!(bbci61_bulge, 6, 1)
show_bool_result(
  "BBC bulge! validate_rows_first_last (index (6,1))",
  validate_rows_first_last(bbci61_bulge),
)

show_equality_result(
  "BBC bulge! Wilkinson equality (index (6,1))",
  ==,
  wilk(toBandColumn(bbci61_bulge)).arr,
  wilki61,
)

# Scalar (6,2)

bbci62_bulge = copy(bbc0_bulge)
wilki62 = [
  'X' 'X' 'X' 'U' 'O' 'O' 'O'
  'X' 'X' 'X' 'X' 'U' 'U' 'O'
  'O' 'L' 'X' 'X' 'U' 'U' 'O'
  'O' 'L' 'X' 'X' 'X' 'X' 'U'
  'O' 'L' 'L' 'X' 'X' 'X' 'X'
  'O' 'L' 'L' 'L' 'X' 'X' 'X'
  'N' 'O' 'O' 'L' 'X' 'X' 'X'
  'N' 'N' 'O' 'O' 'O' 'L' 'X'
]

bulge!(bbci62_bulge, 6, 2)
show_bool_result(
  "BBC bulge! validate_rows_first_last (index (6,2))",
  validate_rows_first_last(bbci62_bulge),
)

show_equality_result(
  "BBC bulge! Wilkinson equality (index (6,2))",
  ==,
  wilk(toBandColumn(bbci62_bulge)).arr,
  wilki62,
)
