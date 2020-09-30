using InPlace
using Random
using ShowTests

a = randn(Float64, 3, 3)
b = randn(Float64, 3, 3)
a0 = copy(a)
b0 = copy(b)

Linear(a) ⊛ b
show_equality_result("InPlace General Left Product", a0*b0, b)

b=copy(b0)
b ⊛ Linear(a)
show_equality_result("InPlace General Right Product", b0*a0, b)

b=copy(b0)
Linear(a) ⊘ b
show_equality_result("InPlace General Left Inverse", a0\b0, b)

b=copy(b0)
b ⊘ Linear(a)
show_equality_result("InPlace General Right Inverse", b0/a0, b)



