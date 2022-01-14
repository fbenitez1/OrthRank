using InPlace
using Random
using ShowTests
using LinearAlgebra

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

b=copy(b0)
apply!(Linear(a), b)
show_equality_result("InPlace General Left apply!", a0*b0, b)

b=copy(b0)
apply!(b, Linear(a))
show_equality_result("InPlace General Right apply!", b0*a0, b)

b=copy(b0)
apply_inv!(Linear(a), b)
show_equality_result("InPlace General Left apply_inv!", a0\b0, b)

b=copy(b0)
apply_inv!(b, Linear(a))
show_equality_result("InPlace General Right apply_inv!", b0/a0, b)

