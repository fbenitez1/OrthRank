blocks = [
  1 3 4 6 7 9 9 11 12
  1 2 5 5 6 7 8 9 10
]

lbc0 = LeadingBandColumn(
  MersenneTwister(0),
  Float64,
  12,
  10,
  2,
  2,
  blocks,
  [1, 1, 1, 2, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1, 1, 1],
)

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

# Zero element 35 with a right rotation.
bcr35 = copy(bc0)
mx_bcr35 = Matrix(bcr35)
rr35 = rgivens(bcr35[3,4], bcr35[3,5],4)
bcr35 ⊛ rr35
mx_bcr35 ⊛ rr35
show_error_result(
  "Real BandStruct Givens zero test, right zero element (3,5)",
  abs(bcr35[3, 5]),
  tol,
)
show_equality_result(
  "Real BandStruct Givens transformation test",
  Matrix(bcr35),
  mx_bcr35,
)

trim_upper!(bcr35,:,5)
show_equality_result(
  "trim_upper! equality test, element (3,5)",
  Matrix(bcr35),
  mx_bcr35,
)

bcr35 ⊘ rr35
show_error_result(
  "Real BandStruct Givens right inverse",
  norm(Matrix(bcr35) - mx_bc0, Inf),
  tol,
)

# Zero element 5,6 with a right rotation (no storage error).
bcr56 = copy(bc0)
mx_bcr56 = Matrix(bcr56)
rr56 = rgivens(bcr56[5,5], bcr56[5,6],5)
res=false
try 
  bcr56 ⊛ rr56
catch e
  if isa(e, NoStorageForIndex)
    global res=true
  end
end
show_bool_result("NoStorageForIndex rotation test, right zero element (5,6)", res)

# Zero element 11,8 with a left rotation.
bcl118 = copy(bc0)
mx_bcl118 = Matrix(bcl118)
rl118 = lgivens(bcl118[10,8], bcl118[11,8],10)
rl118 ⊘ bcl118
rl118 ⊘ mx_bcl118
show_error_result(
  "Real BandStruct Givens zero test, left zero element (11,8)",
  abs(bcl118[11, 8]),
  tol,
)
show_equality_result(
  "Real BandStruct Givens transformation test",
  Matrix(bcl118),
  mx_bcl118,
)

trim_lower!(bcl118,11,:)
show_equality_result(
  "trim_upper! equality test, element (11,8)",
  Matrix(bcl118),
  mx_bcl118,
)

rl118 ⊛ bcl118
show_error_result(
  "Real BandStruct Givens left inverse",
  norm(Matrix(bcl118) - mx_bc0, Inf),
  tol,
)
