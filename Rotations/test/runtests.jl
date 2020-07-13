if isdefined(@__MODULE__, :LanguageServer)
  include("src/Givens.jl")
  using .Givens
else
  using Rotations.Givens
end

using LinearAlgebra

function show_result(testname, err, tol)
  abserr = abs(err)
  if (abserr < tol)
    println("Success: ", testname, ", error: ", abserr)
  else
    println("Failure: ", testname, ", error: ", abserr)
  end
end

tol = 1e-15
A = randn(Float64, 3, 3)
r = lgivens(A[1, 1], A[3, 1], 1, 3)
A1 = copy(A)
r ⊘ A1
show_result("Real left zero test", A1[3, 1], tol)

r ⊛ A1
err = norm(A1 - A, Inf) / norm(A, Inf)
show_result("Real left inverse", err, tol)

A = randn(Float64, 3, 3)
r = rgivens(A[1, 1], A[1, 3], 1, 3)
A1 = copy(A)
A1 ⊛ r
show_result("Real right zero test", A1[1, 3], tol)

A1 ⊘ r
err = norm(A1 - A, Inf) / norm(A, Inf)
show_result("Real right inverse", err, tol)

# Complex tests 

A = randn(Complex{Float64}, 3, 3)
r = lgivens(A[1, 1], A[3, 1], 1, 3)
A1 = copy(A)
r ⊘ A1
show_result("Complex left zero test", A1[3, 1], tol)

r ⊛ A1
err = norm(A1 - A, Inf) / norm(A, Inf)
show_result("Complex left inverse", err, tol)

A = randn(Complex{Float64}, 3, 3)
r = rgivens(A[1, 1], A[1, 3], 1, 3)
A1 = copy(A)
A1 ⊛ r
show_result("Complex right zero test", A1[1, 3], tol)

A1 ⊘ r
err = norm(A1 - A, Inf) / norm(A, Inf)
show_result("Complex right inverse", err, tol)
