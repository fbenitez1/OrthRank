using InPlace
using Random
using LinearAlgebra

function run_givens(::Type{E}) where {E}
  A = randn(E, 3, 3)
  r = lgivens(A[1, 1], A[3, 1], (1, 3))
  r ⊘ A
  r ⊛ A
  r = lgivensPR(A[1, 1], A[3, 1], (1, 3))
  r ⊘ A
  r ⊛ A
  r = rgivens(A[1, 1], A[1, 3], (1, 3))
  A ⊘ r
  A ⊛ r
  r = rgivensPR(A[1, 1], A[1, 3], (1, 3))
  A ⊘ r
  A ⊛ r

  A = view(randn(E, 4, 4), 1:3, 1:3)
  r = lgivens(A[1, 1], A[3, 1], (1, 3))
  r ⊘ A
  r ⊛ A
  r = lgivensPR(A[1, 1], A[3, 1], (1, 3))
  r ⊘ A
  r ⊛ A
  r = rgivens(A[1, 1], A[1, 3], (1, 3))
  A ⊘ r
  A ⊛ r
  r = rgivensPR(A[1, 1], A[1, 3], (1, 3))
  A ⊘ r
  A ⊛ r

  A = randn(E, 4, 4)[1:3, 1:3]
  r = lgivens(A[1, 1], A[3, 1], (1, 3))
  r ⊘ A
  r ⊛ A
  r = lgivensPR(A[1, 1], A[3, 1], (1, 3))
  r ⊘ A
  r ⊛ A
  r = rgivens(A[1, 1], A[1, 3], (1, 3))
  A ⊘ r
  A ⊛ r
  r = rgivensPR(A[1, 1], A[1, 3], (1, 3))
  A ⊘ r
  A ⊛ r
end

function run_cases()
  run_givens(Float64)
  run_givens(Complex{Float64})
end

run_cases()
