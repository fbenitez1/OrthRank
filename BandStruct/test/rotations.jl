# X   X   X | U | O   O | N | 
#           + - + - - - + - + 
# X   X   X   X | U   U | O | 
# - - - +       |       |   | 
# O   L | X   X | U   U | O | 
#       |       + - - - + - + 
# N   L | X   X   X   X | U | 
# - - - + - +           + - + 
# N   O | L | X   X   X   X | 
# - - - + - + - +           | 
# N   N | O | L | X   X   X | 
#       |   |   |           + 
# N   N | N | L | X   X   X   
# - - - + - + - + - - - +     
# N   N | N | O | O   L | X   

tol = 1e-15

println("""

*******************************
** Testing BandStruct Rotations
*******************************

Matrix structure:
""")

bbc0_r = BlockedBandColumn(
  Float64,
  LeadingDecomp,
  MersenneTwister(0),
  8,
  7,
  upper_blocks = upper_blocks,
  lower_blocks = lower_blocks,
  upper_rank_max = 2,
  lower_rank_max = 1,
  upper_ranks = [1, 2, 1, 0],
  lower_ranks = [1, 1, 1, 1],
)

show(wilk(bbc0_r))
println()

bc0 = toBandColumn(bbc0_r)
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
