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

tol = 2e-15

println("""

*******************************
** Testing BandStruct Householders
*******************************

Matrix structure:
""")

show(wilk(bbc0))
println()

bc0 = bbc0[:,:]
mx_bc0 = Matrix(bc0)

work = zeros(Float64, maximum(size(bc)))

# Zero elements 3 to 4 in column 2 with a left Householder.
bch_34_2 = copy(bc0)
mx_bch_34_2 = Matrix(bch_34_2)
v_34_2 = zeros(Float64, 3)

h_34_2 = householder(bch_34_2, 2:4, 2, 1, 1, v_34_2, work)
h_34_2 ⊘ bch_34_2

mx_bch_34_2_a = copy(mx_bch_34_2)
h_34_2 ⊘ mx_bch_34_2_a

show_error_result(
  "Real left Householder singular value test",
  norm(svdvals(Matrix(bch_34_2)) - svdvals(mx_bch_34_2)),
  tol)

show_error_result(
  "Real left Householder unstructured transformation test",
  norm(Matrix(bch_34_2) - mx_bch_34_2_a),
  tol
)

show_error_result(
  "Real left Householder zero elements test",
  norm(Matrix(bch_34_2)[3:4,2]),
  tol
)

h_34_2 ⊛ bch_34_2
show_error_result(
  "Real left Householder inverse test",
  norm(Matrix(bch_34_2) - mx_bch_34_2),
  tol)

# Zero elements 5 to 6 in row 4 with a right Householder.
bch_4_56 = copy(bc0)
mx_bch_4_56 = Matrix(bch_4_56)
v_4_56 = zeros(Float64, 3)

h_4_56 = householder(bch_4_56, 4, 5:7, 3, 4, v_4_56, work)
bch_4_56 ⊛ h_4_56

mx_bch_4_56_a = copy(mx_bch_4_56)
mx_bch_4_56_a ⊛ h_4_56

show_error_result(
  "Real right Householder singular value test",
  norm(svdvals(Matrix(bch_4_56)) - svdvals(mx_bch_4_56)),
  tol)

show_error_result(
  "Real right Householder unstructured transformation test",
  norm(Matrix(bch_4_56) - mx_bch_4_56_a),
  tol
)

show_error_result(
  "Real right Householder zero elements test",
  norm(Matrix(bch_4_56)[4,5:6]),
  tol
)

bch_4_56 ⊘ h_4_56
show_error_result(
  "Real right Householder inverse test",
  norm(Matrix(bch_4_56) - mx_bch_4_56),
  tol)
