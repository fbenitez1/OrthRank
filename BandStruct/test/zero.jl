bc = copy(bc0)
bbc = copy(bbc0)
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


bbc0_za45 = copy(bbc0)
mx_za45 = Matrix(bbc0)
zero_above!(bbc0_za45, 4, 5)
mx_za45[1:4, 5] .= 0.0

show_equality_result(
  "Real zero above test, index (4,5).",
  ==,
  Matrix(bbc0_za45),
  mx_za45
)

bbc0_zb45 = copy(bbc0)
mx_zb45 = Matrix(bbc0)
zero_below!(bbc0_zb45, 4, 5)
mx_zb45[4:8, 5] .= 0.0

show_equality_result(
  "Real zero below test, index (4,5).",
  ==,
  Matrix(bbc0_zb45),
  mx_zb45
)

bbc0_zr45 = copy(bbc0)
mx_zr45 = Matrix(bbc0)
zero_right!(bbc0_zr45, 4, 5)
mx_zr45[4, 5:7] .= 0.0

show_equality_result(
  "Real zero right test, index (4,5).",
  ==,
  Matrix(bbc0_zr45),
  mx_zr45
)

bbc0_zl45 = copy(bbc0)
mx_zl45 = Matrix(bbc0)
zero_left!(bbc0_zl45, 4, 5)
mx_zl45[4, 1:5] .= 0.0

show_equality_result(
  "Real zero left test, index (4,5).",
  ==,
  Matrix(bbc0_zl45),
  mx_zl45
)

bbc0_za4_45 = copy(bbc0)
mx_za4_45 = Matrix(bbc0)
zero_above!(bbc0_za4_45, 4, 4:5)
mx_za4_45[1:4, 4:5] .= 0.0

show_equality_result(
  "Real zero above test, indices (4,4:5).",
  ==,
  Matrix(bbc0_za4_45),
  mx_za4_45
)

bbc0_zb4_45 = copy(bbc0)
mx_zb4_45 = Matrix(bbc0)
zero_below!(bbc0_zb4_45, 4, 4:5)
mx_zb4_45[4:7, 4:5] .= 0.0

show_equality_result(
  "Real zero below test, indices (4,4:5).",
  ==,
  Matrix(bbc0_zb4_45),
  mx_zb4_45
)

bbc0_zr45_5 = copy(bbc0)
mx_zr45_5 = Matrix(bbc0)
zero_right!(bbc0_zr45_5, 4:5, 5)
mx_zr45_5[4:5, 5:7] .= 0.0

show_equality_result(
  "Real zero right test, indices (4:5,5).",
  ==,
  Matrix(bbc0_zr45_5),
  mx_zr45_5
)

bbc0_zl45_5 = copy(bbc0)
mx_zl45_5 = Matrix(bbc0)
zero_left!(bbc0_zl45_5, 4:5, 5)
mx_zl45_5[4:5, 1:5] .= 0.0

show_equality_result(
  "Real zero left test, indices (4:5,5).",
  ==,
  Matrix(bbc0_zl45_5),
  mx_zl45_5
)

