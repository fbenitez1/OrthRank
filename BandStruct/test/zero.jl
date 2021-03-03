bc = copy(bc0)
lbc = copy(lbc0)
mx_bc = copy(mx_bc0)

println("""

****************************
** Zero tests
****************************

""")

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


lbc0_za45 = copy(lbc0)
mx_za45 = Matrix(lbc0)
zero_above!(lbc0_za45, 4, 5)
mx_za45[1:4, 5] .= 0.0

show_equality_result(
  "Real zero above test, index (4,5).",
  ==,
  Matrix(lbc0_za45),
  mx_za45
)

lbc0_zb45 = copy(lbc0)
mx_zb45 = Matrix(lbc0)
zero_below!(lbc0_zb45, 4, 5)
mx_zb45[4:8, 5] .= 0.0

show_equality_result(
  "Real zero below test, index (4,5).",
  ==,
  Matrix(lbc0_zb45),
  mx_zb45
)

lbc0_zr45 = copy(lbc0)
mx_zr45 = Matrix(lbc0)
zero_right!(lbc0_zr45, 4, 5)
mx_zr45[4, 5:7] .= 0.0

show_equality_result(
  "Real zero right test, index (4,5).",
  ==,
  Matrix(lbc0_zr45),
  mx_zr45
)

lbc0_zl45 = copy(lbc0)
mx_zl45 = Matrix(lbc0)
zero_left!(lbc0_zl45, 4, 5)
mx_zl45[4, 1:5] .= 0.0

show_equality_result(
  "Real zero left test, index (4,5).",
  ==,
  Matrix(lbc0_zl45),
  mx_zl45
)

lbc0_za4_45 = copy(lbc0)
mx_za4_45 = Matrix(lbc0)
zero_above!(lbc0_za4_45, 4, 4:5)
mx_za4_45[1:4, 4:5] .= 0.0

show_equality_result(
  "Real zero above test, indices (4,4:5).",
  ==,
  Matrix(lbc0_za4_45),
  mx_za4_45
)

lbc0_zb4_45 = copy(lbc0)
mx_zb4_45 = Matrix(lbc0)
zero_below!(lbc0_zb4_45, 4, 4:5)
mx_zb4_45[4:7, 4:5] .= 0.0

show_equality_result(
  "Real zero below test, indices (4,4:5).",
  ==,
  Matrix(lbc0_zb4_45),
  mx_zb4_45
)

lbc0_zr45_5 = copy(lbc0)
mx_zr45_5 = Matrix(lbc0)
zero_right!(lbc0_zr45_5, 4:5, 5)
mx_zr45_5[4:5, 5:7] .= 0.0

show_equality_result(
  "Real zero right test, indices (4:5,5).",
  ==,
  Matrix(lbc0_zr45_5),
  mx_zr45_5
)

lbc0_zl45_5 = copy(lbc0)
mx_zl45_5 = Matrix(lbc0)
zero_left!(lbc0_zl45_5, 4:5, 5)
mx_zl45_5[4:5, 1:5] .= 0.0

show_equality_result(
  "Real zero left test, indices (4:5,5).",
  ==,
  Matrix(lbc0_zl45_5),
  mx_zl45_5
)

