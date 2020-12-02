bc = copy(bc0)
lbc = copy(lbc0)
mx_bc = copy(mx_bc0)

println("""

****************************
** Bandwidth trim tests
****************************

""")

# Trim the bandwidth of the example matrix from the documentation
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


lbc0_tr = copy(lbc0)
wilk0_tr = wilk(lbc0_tr[1:end,1:end])

# column 4 trim upper.

lbc4_tr = copy(lbc0_tr)
wilk4_tr = [
  'X' 'X' 'X' 'O' 'O' 'O' 'N'
  'X' 'X' 'X' 'X' 'U' 'U' 'O'
  'O' 'L' 'X' 'X' 'U' 'U' 'O'
  'O' 'L' 'X' 'X' 'X' 'X' 'U'
  'O' 'O' 'L' 'X' 'X' 'X' 'X'
  'O' 'O' 'O' 'L' 'X' 'X' 'X'
  'N' 'N' 'N' 'L' 'X' 'X' 'X'
  'N' 'N' 'N' 'N' 'O' 'L' 'X'
]

trim_upper!(lbc4_tr,:, 4)
show_bool_result(
  "LBC trim_upper! column validate_rbws (column 4)",
  validate_rbws(lbc4_tr),
)
show_equality_result(
  "LBC trim_upper! column Wilkinson equality (column 4)",
  ==,
  wilk(lbc4_tr[1:end, 1:end]).arr,
  wilk4_tr,
)

# column 5 trim upper (Creates a well).

lbc5_tr = copy(lbc0_tr)
res = false
try
  trim_upper!(lbc5_tr, :, 5)
catch e
  isa(e, WellError) && global res = true
end
show_bool_result("trim_upper! column WellError test", res)

# Column 6

lbc6_tr = copy(lbc0_tr)
wilk6_tr = [
  'X' 'X' 'X' 'U' 'O' 'O' 'N'
  'X' 'X' 'X' 'X' 'U' 'O' 'O'
  'O' 'L' 'X' 'X' 'U' 'U' 'O'
  'O' 'L' 'X' 'X' 'X' 'X' 'U'
  'O' 'O' 'L' 'X' 'X' 'X' 'X'
  'O' 'O' 'O' 'L' 'X' 'X' 'X'
  'N' 'N' 'N' 'L' 'X' 'X' 'X'
  'N' 'N' 'N' 'N' 'O' 'L' 'X'
]

trim_upper!(lbc6_tr,:, 6)
show_bool_result(
  "LBC trim_upper! column validate_rbws (column 6)",
  validate_rbws(lbc6_tr),
)
show_equality_result(
  "LBC trim_upper! column Wilkinson equality (column 6)",
  ==,
  wilk(lbc6_tr[1:end, 1:end]).arr,
  wilk6_tr,
)

# row 2 trim upper.

lbcr2_tr = copy(lbc0_tr)
wilkr2_tr = [
  'X' 'X' 'X' 'U' 'O' 'O' 'N'
  'X' 'X' 'X' 'X' 'U' 'O' 'O'
  'O' 'L' 'X' 'X' 'U' 'U' 'O'
  'O' 'L' 'X' 'X' 'X' 'X' 'U'
  'O' 'O' 'L' 'X' 'X' 'X' 'X'
  'O' 'O' 'O' 'L' 'X' 'X' 'X'
  'N' 'N' 'N' 'L' 'X' 'X' 'X'
  'N' 'N' 'N' 'N' 'O' 'L' 'X'
]


trim_upper!(lbcr2_tr, 2, :)
show_bool_result(
  "LBC trim_upper! row validate_rbws (row 2)",
  validate_rbws(lbcr2_tr),
)
show_equality_result(
  "LBC trim_upper! row Wilkinson equality (row 2)",
  ==,
  wilk(lbcr2_tr[1:end, 1:end]).arr,
  wilkr2_tr,
)

# Row 3 trim lower (WellError).

lbcr3l_tr = copy(lbc0_tr)
res=false
res = false
try
  trim_lower!(lbcr3l_tr, 3, :)
catch e
  isa(e, WellError) && global res = true
end
show_bool_result("trim_lower! row WellError test", res)

# Column 2 trim lower.

lbc2l_tr = copy(lbc0_tr)
wilk2l_tr = [
  'X' 'X' 'X' 'U' 'O' 'O' 'N'
  'X' 'X' 'X' 'X' 'U' 'U' 'O'
  'O' 'L' 'X' 'X' 'U' 'U' 'O'
  'O' 'O' 'X' 'X' 'X' 'X' 'U'
  'O' 'O' 'L' 'X' 'X' 'X' 'X'
  'O' 'O' 'O' 'L' 'X' 'X' 'X'
  'N' 'N' 'N' 'L' 'X' 'X' 'X'
  'N' 'N' 'N' 'N' 'O' 'L' 'X'
]

trim_lower!(lbc2l_tr, :, 2)
show_bool_result(
  "LBC trim_lower! column validate_rbws (column 2)",
  validate_rbws(lbc2l_tr),
)
show_equality_result(
  "LBC trim_lower! column Wilkinson equality (column 2)",
  ==,
  wilk(lbc2l_tr[1:end, 1:end]).arr,
  wilk2l_tr,
)

# Row 4 trim lower.

lbcr4l_tr = copy(lbc0_tr)
wilkr4l_tr = [
  'X' 'X' 'X' 'U' 'O' 'O' 'N'
  'X' 'X' 'X' 'X' 'U' 'U' 'O'
  'O' 'L' 'X' 'X' 'U' 'U' 'O'
  'O' 'O' 'X' 'X' 'X' 'X' 'U'
  'O' 'O' 'L' 'X' 'X' 'X' 'X'
  'O' 'O' 'O' 'L' 'X' 'X' 'X'
  'N' 'N' 'N' 'L' 'X' 'X' 'X'
  'N' 'N' 'N' 'N' 'O' 'L' 'X'
]

trim_lower!(lbcr4l_tr, 4, :)
show_bool_result(
  "LBC trim_lower! row validate_rbws (row 4)",
  validate_rbws(lbcr4l_tr),
)
show_equality_result(
  "LBC trim_lower! row Wilkinson equality (row 4)",
  ==,
  wilk(lbcr4l_tr[1:end, 1:end]).arr,
  wilkr4l_tr,
)
