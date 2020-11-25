bc = copy(bc0)
lbc = copy(lbc0)
mx_bc = copy(mx_bc0)

println("""

****************************
** Bandwidth trim tests
****************************

""")

blocks_tr = [
  1 3 4 9 10
  1 2 4 4 5
]

# X | U | O   O | N   
# _ ⌋   |       |     
# L   X | O   O | N   
#       |       |     
# L   X | U   U | O   
# _ _ _ ⌋       |     
# O   L   X   X | O   
# _ _ _ _ _ _ _ |     
# O   O   O   L | O   
#               |     
# O   O   O   L | O   
#               |     
# O   O   O   L | O   
#               |     
# N   O   O   L | O   
#               |     
# N   N   O   L | U   
# _ _ _ _ _ _ _ ⌋     
# N   N   O   L   X  

lbc0_tr = LeadingBandColumn(
  MersenneTwister(0),
  Float64,
  10,
  5,
  7,
  5,
  blocks_tr,
  [1, 1, 1, 1],
  [1, 1, 1, 1],
)
wilk0_tr = wilk(lbc0_tr[1:end,1:end])

# column 4 trim upper.

lbc4_tr = copy(lbc0_tr)
wilk4_tr = [
  'X' 'U' 'O' 'O' 'N'
  'L' 'X' 'O' 'O' 'N'
  'L' 'X' 'U' 'O' 'O'
  'O' 'L' 'X' 'X' 'O'
  'O' 'O' 'O' 'L' 'O'
  'O' 'O' 'O' 'L' 'O'
  'O' 'O' 'O' 'L' 'O'
  'N' 'O' 'O' 'L' 'O'
  'N' 'N' 'O' 'L' 'U'
  'N' 'N' 'O' 'L' 'X'
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

# column 3 trim upper (Creates a well).

lbc3_tr = copy(lbc0_tr)
res = false
try
  trim_upper!(lbc3_tr, :, 3)
catch e
  isa(e, WellError) && global res = true
end
show_bool_result("trim_upper! column WellError test", res)

# row 3 trim upper.

lbcr3_tr = copy(lbc0_tr)
wilkr3_tr = [
  'X' 'U' 'O' 'O' 'N'
  'L' 'X' 'O' 'O' 'N'
  'L' 'X' 'U' 'O' 'O'
  'O' 'L' 'X' 'X' 'O'
  'O' 'O' 'O' 'L' 'O'
  'O' 'O' 'O' 'L' 'O'
  'O' 'O' 'O' 'L' 'O'
  'N' 'O' 'O' 'L' 'O'
  'N' 'N' 'O' 'L' 'U'
  'N' 'N' 'O' 'L' 'X'
]
trim_upper!(lbcr3_tr, 3, :)
show_bool_result(
  "LBC trim_upper! row validate_rbws (row 3)",
  validate_rbws(lbcr3_tr),
)
show_equality_result(
  "LBC trim_upper! row Wilkinson equality (row 3)",
  ==,
  wilk(lbcr3_tr[1:end, 1:end]).arr,
  wilkr3_tr,
)

# Column 2 trim lower.

lbc2l_tr = copy(lbc0_tr)
wilk2l_tr = [
  'X' 'U' 'O' 'O' 'N'
  'L' 'X' 'O' 'O' 'N'
  'L' 'X' 'U' 'U' 'O'
  'O' 'O' 'X' 'X' 'O'
  'O' 'O' 'O' 'L' 'O'
  'O' 'O' 'O' 'L' 'O'
  'O' 'O' 'O' 'L' 'O'
  'N' 'O' 'O' 'L' 'O'
  'N' 'N' 'O' 'L' 'U'
  'N' 'N' 'O' 'L' 'X'
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
  'X' 'U' 'O' 'O' 'N'
  'L' 'X' 'O' 'O' 'N'
  'L' 'X' 'U' 'U' 'O'
  'O' 'O' 'X' 'X' 'O'
  'O' 'O' 'O' 'L' 'O'
  'O' 'O' 'O' 'L' 'O'
  'O' 'O' 'O' 'L' 'O'
  'N' 'O' 'O' 'L' 'O'
  'N' 'N' 'O' 'L' 'U'
  'N' 'N' 'O' 'L' 'X'
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

# Row 5 trim lower (WellError).

lbcr5l_tr = copy(lbc0_tr)
res=false
res = false
try
  trim_lower!(lbcr5l_tr, 5, :)
catch e
  isa(e, WellError) && global res = true
end
show_bool_result("trim_lower! row WellError test", res)
