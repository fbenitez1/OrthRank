println("""

****************************
** Bandwidth extension tests
****************************

""")

blocks_ext = [
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

lbc0_ext = LeadingBandColumn(
  MersenneTwister(0),
  Float64,
  10,
  5,
  7,
  5,
  blocks_ext,
  [1, 1, 1, 1],
  [1, 1, 1, 1],
)
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
# O   O   O   O | O   
#               |     
# N   O   O   O | O   
#               |     
# N   N   O   O | U   
# _ _ _ _ _ _ _ ⌋     
# N   N   O   O   X
lbc0_ext.cbws[3, 4] = 2
lbc0_ext.rbws[7:10, 1] .= 0
show_bool_result(
  "LBC extend_band! initial validate_rbws",
  validate_rbws(lbc0_ext),
)
wilk0 = wilk(lbc0_ext[1:10,1:5])

lbc4_ext = copy(lbc0_ext)
wilk4 =  
[ 'X'  'U'  'O'  'O'  'N'
  'L'  'X'  'O'  'O'  'N'
  'L'  'X'  'U'  'U'  'U'
  'O'  'L'  'X'  'X'  'U'
  'O'  'O'  'O'  'L'  'U'
  'O'  'O'  'O'  'L'  'U'
  'O'  'O'  'O'  'L'  'U'
  'N'  'O'  'O'  'L'  'U'
  'N'  'N'  'O'  'L'  'U'
  'N'  'N'  'O'  'L'  'X' ]
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

# Column 3

lbc3_ext = copy(lbc0_ext)
wilk3 = [
  'X' 'U' 'O' 'O' 'N'
  'L' 'X' 'O' 'O' 'N'
  'L' 'X' 'U' 'U' 'O'
  'O' 'L' 'X' 'X' 'O'
  'O' 'O' 'L' 'L' 'O'
  'O' 'O' 'L' 'L' 'O'
  'O' 'O' 'O' 'O' 'O'
  'N' 'O' 'O' 'O' 'O'
  'N' 'N' 'O' 'O' 'U'
  'N' 'N' 'O' 'O' 'X'
]
 
extend_band!(lbc3_ext,:, 3)
show_bool_result(
  "LBC extend_band! validate_rbws (column 3)",
  validate_rbws(lbc3_ext),
)
show_equality_result(
  "LBC extend_band! Wilkinson equality (column 3)",
  ==,
  wilk(lbc3_ext[1:end, 1:end]).arr,
  wilk3,
)

# Column 2

lbc2_ext = copy(lbc0_ext)
wilk2 = [
  'X' 'U' 'U' 'O' 'N'
  'L' 'X' 'U' 'O' 'N'
  'L' 'X' 'U' 'U' 'O'
  'O' 'L' 'X' 'X' 'O'
  'O' 'O' 'O' 'L' 'O'
  'O' 'O' 'O' 'L' 'O'
  'O' 'O' 'O' 'O' 'O'
  'N' 'O' 'O' 'O' 'O'
  'N' 'N' 'O' 'O' 'U'
  'N' 'N' 'O' 'O' 'X'
]
 
extend_band!(lbc2_ext,:, 2)
show_bool_result(
  "LBC extend_band! validate_rbws (column 2)",
  validate_rbws(lbc2_ext),
)
show_equality_result(
  "LBC extend_band! Wilkinson equality (column 2)",
  ==,
  wilk(lbc2_ext[1:end, 1:end]).arr,
  wilk2,
)

# Column 1

lbc1_ext = copy(lbc0_ext)
wilk1 = [
  'X' 'U' 'O' 'O' 'N'
  'L' 'X' 'O' 'O' 'N'
  'L' 'X' 'U' 'U' 'O'
  'L' 'L' 'X' 'X' 'O'
  'O' 'O' 'O' 'L' 'O'
  'O' 'O' 'O' 'L' 'O'
  'O' 'O' 'O' 'O' 'O'
  'N' 'O' 'O' 'O' 'O'
  'N' 'N' 'O' 'O' 'U'
  'N' 'N' 'O' 'O' 'X'
]
 
extend_band!(lbc1_ext,:, 1)
show_bool_result(
  "LBC extend_band! validate_rbws (column 1)",
  validate_rbws(lbc1_ext),
)
show_equality_result(
  "LBC extend_band! Wilkinson equality (column 1)",
  ==,
  wilk(lbc1_ext[1:end, 1:end]).arr,
  wilk1,
)

# Row 1 (Does nothing)

lbcr1_ext = copy(lbc0_ext)
wilkr1 = [
  'X' 'U' 'O' 'O' 'N'
  'L' 'X' 'O' 'O' 'N'
  'L' 'X' 'U' 'U' 'O'
  'O' 'L' 'X' 'X' 'O'
  'O' 'O' 'O' 'L' 'O'
  'O' 'O' 'O' 'L' 'O'
  'O' 'O' 'O' 'O' 'O'
  'N' 'O' 'O' 'O' 'O'
  'N' 'N' 'O' 'O' 'U'
  'N' 'N' 'O' 'O' 'X'
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

# Row 2

lbcr2_ext = copy(lbc0_ext)
wilkr2 = [
  'X' 'U' 'O' 'O' 'N'
  'L' 'X' 'U' 'U' 'N'
  'L' 'X' 'U' 'U' 'O'
  'O' 'L' 'X' 'X' 'O'
  'O' 'O' 'O' 'L' 'O'
  'O' 'O' 'O' 'L' 'O'
  'O' 'O' 'O' 'O' 'O'
  'N' 'O' 'O' 'O' 'O'
  'N' 'N' 'O' 'O' 'U'
  'N' 'N' 'O' 'O' 'X'
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

# Row 3

lbcr3_ext = copy(lbc0_ext)
wilkr3 = [
 'X'  'U'  'O'  'O'  'N'
 'L'  'X'  'O'  'O'  'N'
 'L'  'X'  'U'  'U'  'O'
 'L'  'L'  'X'  'X'  'O'
 'O'  'O'  'O'  'L'  'O'
 'O'  'O'  'O'  'L'  'O'
 'O'  'O'  'O'  'O'  'O'
 'N'  'O'  'O'  'O'  'O'
 'N'  'N'  'O'  'O'  'U'
 'N'  'N'  'O'  'O'  'X'
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

# Row 4

lbcr4_ext = copy(lbc0_ext)
wilkr4 = [
  'X' 'U' 'O' 'O' 'N'
  'L' 'X' 'O' 'O' 'N'
  'L' 'X' 'U' 'U' 'O'
  'O' 'L' 'X' 'X' 'O'
  'O' 'L' 'L' 'L' 'O'
  'O' 'O' 'O' 'L' 'O'
  'O' 'O' 'O' 'O' 'O'
  'N' 'O' 'O' 'O' 'O'
  'N' 'N' 'O' 'O' 'U'
  'N' 'N' 'O' 'O' 'X'
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

# Row 5 (Does nothing)

lbcr5_ext = copy(lbc0_ext)
wilkr5 = [
  'X' 'U' 'O' 'O' 'N'
  'L' 'X' 'O' 'O' 'N'
  'L' 'X' 'U' 'U' 'O'
  'O' 'L' 'X' 'X' 'O'
  'O' 'O' 'O' 'L' 'O'
  'O' 'O' 'O' 'L' 'O'
  'O' 'O' 'O' 'O' 'O'
  'N' 'O' 'O' 'O' 'O'
  'N' 'N' 'O' 'O' 'U'
  'N' 'N' 'O' 'O' 'X'
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

# Row 6

lbcr6_ext = copy(lbc0_ext)
wilkr6 = [
  'X' 'U' 'O' 'O' 'N'
  'L' 'X' 'O' 'O' 'N'
  'L' 'X' 'U' 'U' 'O'
  'O' 'L' 'X' 'X' 'O'
  'O' 'O' 'O' 'L' 'O'
  'O' 'O' 'O' 'L' 'O'
  'O' 'O' 'O' 'L' 'O'
  'N' 'O' 'O' 'O' 'O'
  'N' 'N' 'O' 'O' 'U'
  'N' 'N' 'O' 'O' 'X'
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

# Row 7 (Does nothing)

lbcr7_ext = copy(lbc0_ext)
wilkr7 = [
  'X' 'U' 'O' 'O' 'N'
  'L' 'X' 'O' 'O' 'N'
  'L' 'X' 'U' 'U' 'O'
  'O' 'L' 'X' 'X' 'O'
  'O' 'O' 'O' 'L' 'O'
  'O' 'O' 'O' 'L' 'O'
  'O' 'O' 'O' 'O' 'O'
  'N' 'O' 'O' 'O' 'O'
  'N' 'N' 'O' 'O' 'U'
  'N' 'N' 'O' 'O' 'X'
]
extend_band!(lbcr7_ext, 7, :)
show_bool_result(
  "LBC extend_band! validate_rbws (row 7)",
  validate_rbws(lbcr7_ext),
)
show_equality_result(
  "LBC extend_band! Wilkinson equality (row 7)",
  ==,
  wilk(lbcr7_ext[1:end, 1:end]).arr,
  wilkr7,
)

# Row 8

lbcr8_ext = copy(lbc0_ext)
wilkr8 = [
  'X' 'U' 'O' 'O' 'N'
  'L' 'X' 'O' 'O' 'N'
  'L' 'X' 'U' 'U' 'O'
  'O' 'L' 'X' 'X' 'O'
  'O' 'O' 'O' 'L' 'O'
  'O' 'O' 'O' 'L' 'O'
  'O' 'O' 'O' 'O' 'O'
  'N' 'O' 'O' 'O' 'U'
  'N' 'N' 'O' 'O' 'U'
  'N' 'N' 'O' 'O' 'X'
]
extend_band!(lbcr8_ext, 8, :)
show_bool_result(
  "LBC extend_band! validate_rbws (row 8)",
  validate_rbws(lbcr8_ext),
)
show_equality_result(
  "LBC extend_band! Wilkinson equality (row 8)",
  ==,
  wilk(lbcr8_ext[1:end, 1:end]).arr,
  wilkr8,
)

# Row 9 (does nothing)

lbcr9_ext = copy(lbc0_ext)
wilkr9 = [
  'X' 'U' 'O' 'O' 'N'
  'L' 'X' 'O' 'O' 'N'
  'L' 'X' 'U' 'U' 'O'
  'O' 'L' 'X' 'X' 'O'
  'O' 'O' 'O' 'L' 'O'
  'O' 'O' 'O' 'L' 'O'
  'O' 'O' 'O' 'O' 'O'
  'N' 'O' 'O' 'O' 'O'
  'N' 'N' 'O' 'O' 'U'
  'N' 'N' 'O' 'O' 'X'
]
extend_band!(lbcr9_ext, 9, :)
show_bool_result(
  "LBC extend_band! validate_rbws (row 9)",
  validate_rbws(lbcr9_ext),
)
show_equality_result(
  "LBC extend_band! Wilkinson equality (row 9)",
  ==,
  wilk(lbcr9_ext[1:end, 1:end]).arr,
  wilkr9,
)

####
## Scalar extend
####

# Scalar (6,2)

lbce62_ext = copy(lbc0_ext)
wilke62 = [
  'X' 'U' 'O' 'O' 'N'
  'L' 'X' 'O' 'O' 'N'
  'L' 'X' 'U' 'U' 'O'
  'O' 'L' 'X' 'X' 'O'
  'O' 'L' 'L' 'L' 'O'
  'O' 'L' 'L' 'L' 'O'
  'O' 'O' 'O' 'O' 'O'
  'N' 'O' 'O' 'O' 'O'
  'N' 'N' 'O' 'O' 'U'
  'N' 'N' 'O' 'O' 'X'
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

# Scalar (10,3)

lbce103_ext = copy(lbc0_ext)
wilke103 = [
  'X' 'U' 'O' 'O' 'N'
  'L' 'X' 'O' 'O' 'N'
  'L' 'X' 'U' 'U' 'O'
  'O' 'L' 'X' 'X' 'O'
  'O' 'O' 'L' 'L' 'O'
  'O' 'O' 'L' 'L' 'O'
  'O' 'O' 'L' 'L' 'O'
  'N' 'O' 'L' 'L' 'O'
  'N' 'N' 'L' 'L' 'U'
  'N' 'N' 'L' 'L' 'X'
]
extend_band!(lbce103_ext, 10, 3)
show_bool_result(
  "LBC extend_band! validate_rbws (element (10,3))",
  validate_rbws(lbce103_ext),
)
show_equality_result(
  "LBC extend_band! Wilkinson equality (element (10,3))",
  ==,
  wilk(lbce103_ext[1:end, 1:end]).arr,
  wilke103,
)

# Scalar (1,4)

lbce14_ext = copy(lbc0_ext)
wilke14 = [
  'X' 'U' 'U' 'U' 'N'
  'L' 'X' 'U' 'U' 'N'
  'L' 'X' 'U' 'U' 'O'
  'O' 'L' 'X' 'X' 'O'
  'O' 'O' 'O' 'L' 'O'
  'O' 'O' 'O' 'L' 'O'
  'O' 'O' 'O' 'O' 'O'
  'N' 'O' 'O' 'O' 'O'
  'N' 'N' 'O' 'O' 'U'
  'N' 'N' 'O' 'O' 'X'
]
extend_band!(lbce14_ext, 1, 4)
show_bool_result(
  "LBC extend_band! validate_rbws (element (1,4))",
  validate_rbws(lbce14_ext),
)
show_equality_result(
  "LBC extend_band! Wilkinson equality (element (1,4))",
  ==,
  wilk(lbce14_ext[1:end, 1:end]).arr,
  wilke14,
)
