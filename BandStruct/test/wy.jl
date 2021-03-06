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
** Testing BandStruct WY
*******************************

Matrix structure:
""")

show(wilk(lbc0))
println()

m=8
n=7

bc0 = lbc0[:,:]
mx_bc0 = Matrix(bc0)

max_num_hs=2
lw = maximum(size(bc0))
work = zeros(Float64, lw)

wyl=WYTrans(Float64,3,m,n,max_num_hs)
# set WY transformation 2 with offset 2 and size 3.
selectWY!(wyl, 2)
resetWYBlock!(wyl, block = 2, offset = 2, sizeWY = 3)

# Do a QR factorization of bc[3:5,2:3]

bcwy_35_23 = copy(bc0)
bcwy_35_23h = copy(bc0)
mx_bcwy_35_23 = Matrix(bcwy_35_23)

bulge_lower!(bcwy_35_23, 5 , 2)
h_35_2 = householder(
  bcwy_35_23,
  3:5,
  2,
  lw,
)
h_35_2 ⊘ bcwy_35_23h
wyl ⊛ h_35_2
wyl ⊘ bcwy_35_23

show_error_result(
  "Real left WY/Householder single housholder comparison test",
  norm(Matrix(bcwy_35_23) - Matrix(bcwy_35_23h)),
  tol
)

show_error_result(
  "Real left WY single Householder zero test",
  norm(Matrix(bcwy_35_23)[4:5,2]),
  tol
)

show_error_result(
  "Real left WY single Householder SVD test",
  norm(svdvals(Matrix(bcwy_35_23)) - svdvals(mx_bc0)),
  tol
)

h_45_3 = householder(
  bcwy_35_23,
  4:5,
  3,
  lw,
)

h_45_3 ⊘ bcwy_35_23h
wyl ⊛ h_45_3

bcwy_35_23w = copy(bc0)
wyl ⊘ bcwy_35_23w

show_error_result(
  "Real left WY/Householder two housholder comparison test",
  norm(Matrix(bcwy_35_23w) - Matrix(bcwy_35_23h)),
  tol
)

show_error_result(
  "Real left WY two Householder zero test",
  norm(Matrix(bcwy_35_23w)[4:5,2]) +
  abs(bcwy_35_23w[5,3]),
  tol
)

show_error_result(
  "Real left WY two Householder SVD test",
  norm(svdvals(Matrix(bcwy_35_23w)) - svdvals(mx_bc0)),
  tol
)

# Right multiplication: Do an LQ factorization of bc[2:3,5:7]

bcwy_23_57 = copy(bc0)
bcwy_23_57[2,7]=1.0
bcwy_23_57[3,7]=2.0

bcwy_23_57h = copy(bcwy_23_57)
bcwy_23_57w = copy(bcwy_23_57)
mx_bcwy_23_57 = Matrix(bcwy_23_57)

wyr=WYTrans(Float64,3,m,n,max_num_hs)
# set WY transformation 2 with offset 2 and size 3.
selectWY!(wyr, 2)
resetWYBlock!(wyr, block = 2, offset = 4, sizeWY = 3)

h_2_57 = householder(
  bcwy_23_57,
  2,
  5:7,
  lw,
)

bcwy_23_57h ⊛ h_2_57

wyr ⊛ h_2_57

bcwy_23_57 ⊛ wyr

show_error_result(
  "Real right WY/Householder single housholder comparison test",
  norm(Matrix(bcwy_23_57) - Matrix(bcwy_23_57h)),
  tol
)

show_error_result(
  "Real right WY single Householder zero test",
  norm(Matrix(bcwy_23_57)[2,6:7]),
  tol
)

show_error_result(
  "Real left WY single Householder SVD test",
  norm(svdvals(Matrix(bcwy_23_57)) - svdvals(mx_bcwy_23_57)),
  tol
)

h_3_67 = householder(
  bcwy_23_57,
  3,
  6:7,
  lw,
)

bcwy_23_57h ⊛ h_3_67

wyr ⊛ h_3_67

bcwy_23_57w ⊛ wyr

show_error_result(
  "Real right WY/Householder two housholder comparison test",
  norm(Matrix(bcwy_23_57w) - Matrix(bcwy_23_57h)),
  tol
)

show_error_result(
  "Real right WY two Householder zero test",
  norm(Matrix(bcwy_23_57w)[2,6:7]) + norm(Matrix(bcwy_23_57w)[3,7:7]),
  tol
)

show_error_result(
  "Real left WY two Householder SVD test",
  norm(svdvals(Matrix(bcwy_23_57w)) - svdvals(mx_bcwy_23_57)),
  tol
)
