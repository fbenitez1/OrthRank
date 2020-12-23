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

tol = 1e-15

println("""

*******************************
** Testing BandStruct Rotations
*******************************

Matrix structure:
""")

show(wilk(lbc0))
println()

bc0 = lbc0[1:end,1:end]
mx_bc0 = Matrix(bc0)

# Zero element 26 with a right rotation.
bcr26 = copy(bc0)
mx_bcr26 = Matrix(bcr26)
rr26 = rgivens(bcr26[2,5], bcr26[2,6],5)
bcr26 ⊛ rr26
mx_bcr26 ⊛ rr26
show_error_result(
  "Real BandStruct Givens zero test, right zero element (2,6)",
  abs(bcr26[2, 6]),
  tol,
)
show_equality_result(
  "Real BandStruct Givens transformation test",
  Matrix(bcr26),
  mx_bcr26,
)

mx_bcr26[2,6]=0.0
bcr26[2,6]=0.0
notch_upper!(bcr26,2,6)
show_equality_result(
  "notch_upper! equality test, element (2,6)",
  Matrix(bcr26),
  mx_bcr26,
)

bcr26 ⊘ rr26
show_error_result(
  "Real BandStruct Givens right inverse",
  norm(Matrix(bcr26) - mx_bc0, Inf),
  tol,
)

# Zero element 1,4 with a right rotation (no storage error).
bcr14 = copy(bc0)
mx_bcr14 = Matrix(bcr14)
rr14 = rgivens(bcr14[1,3], bcr14[1,4],3)
res=false
try 
  bcr14 ⊛ rr14
catch e
  if isa(e, NoStorageForIndex)
    global res=true
  end
end
show_bool_result("NoStorageForIndex rotation test, right zero element (1,4)", res)

# Zero element 7,4 with a left rotation.
bcl74 = copy(bc0)
mx_bcl74 = Matrix(bcl74)
rl74 = lgivens(bcl74[6,4], bcl74[7,4],6)
rl74 ⊘ bcl74
rl74 ⊘ mx_bcl74
show_error_result(
  "Real BandStruct Givens zero test, left zero element (7,4)",
  abs(bcl74[7, 4]),
  tol,
)
show_equality_result(
  "Real BandStruct Givens transformation test",
  Matrix(bcl74),
  mx_bcl74,
)

# mx_bcl74[7,4]=0.0
notch_lower!(bcl74,7,4)
show_equality_result(
  "notch_lower! equality test, element (7,4)",
  Matrix(bcl74),
  mx_bcl74,
)

rl74 ⊛ bcl74
show_error_result(
  "Real BandStruct Givens left inverse",
  norm(Matrix(bcl74) - mx_bc0, Inf),
  tol,
)
