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
** Testing BandStruct Householders
*******************************

Matrix structure:
""")

show(wilk(lbc0))
println()

bc0 = lbc0[:,:]
mx_bc0 = Matrix(bc0)

work = zeros(Float64, maximum(size(bc)))

# Zero elements 3 to 4 in column 2 with a left Householder.
bch_34_2 = copy(bc0)
mx_bch_34_2 = Matrix(bch_34_2)
v_34_2 = zeros(Float64, 3)

h_34_2 = householder(bch_34_2, 2:4, 2, 1, 1, v_34_2, work)
h_34_2 ⊘ bch_34_2

println(svdvals(Matrix(bch_34_2)))
println(svdvals(mx_bch_34_2))
