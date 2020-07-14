if isdefined(@__MODULE__, :LanguageServer)
  include("src/Givens.jl")
  using .Givens
else
  using Rotations.Givens
end

using LinearAlgebra

function show_result(testname, err, tol)
  abserr = abs(err)
  if abserr < tol
    println("Success: ", testname, ", error: ", abserr)
  else
    println()
    println("****  Failure: ", testname, ", error: ", abserr)
    println()
  end
end

tol = 1e-15
A_rlz = randn(Float64, 3, 3)
r_rlz = lgivens(A_rlz[1, 1], A_rlz[3, 1], 1, 3)
A1_rlz = copy(A_rlz)
r_rlz ⊘ A1_rlz
show_result("Real left zero test", A1_rlz[3, 1], tol)

r_rlz ⊛ A1_rlz
err_rlz = norm(A1_rlz - A_rlz, Inf) / norm(A_rlz, Inf)
show_result("Real left inverse", err_rlz, tol)

A_rrz = randn(Float64, 3, 3)
r_rrz = rgivens(A_rrz[1, 1], A_rrz[1, 3], 1, 3)
A1_rrz = copy(A_rrz)
A1_rrz ⊛ r_rrz
show_result("Real right zero test", A1_rrz[1, 3], tol)

A1_rrz ⊘ r_rrz
err_rrz = norm(A1_rrz - A_rrz, Inf) / norm(A_rrz, Inf)
show_result("Real right inverse", err_rrz, tol)

A_rlz1 = randn(Float64, 3, 3)
r_rlz1 = lgivens1(A_rlz1[1, 1], A_rlz1[3, 1], 1, 3)
A1_rlz1 = copy(A_rlz1)
r_rlz1 ⊘ A1_rlz1
show_result("Real left zero test (first element)", A1_rlz1[1, 1], tol)

r_rlz1 ⊛ A1_rlz1
err_rlz1 = norm(A1_rlz1 - A_rlz1, Inf) / norm(A_rlz1, Inf)
show_result("Real left inverse (first element)", err_rlz1, tol)

A_rrz1 = randn(Float64, 3, 3)
r_rrz1 = rgivens1(A_rrz1[1, 1], A_rrz1[1, 3], 1, 3)
A1_rrz1 = copy(A_rrz1)
A1_rrz1 ⊛ r_rrz1
show_result("Real right zero test (first element)", A1_rrz1[1, 1], tol)

A1_rrz1 ⊘ r_rrz1
err_rrz1 = norm(A1_rrz1 - A_rrz1, Inf) / norm(A_rrz1, Inf)
show_result("Real right inverse (first element)", err_rrz1, tol)


# Complex tests 

A_clz = randn(Complex{Float64}, 3, 3)
r_clz = lgivens(A_clz[1, 1], A_clz[3, 1], 1, 3)
A1_clz = copy(A_clz)
r_clz ⊘ A1_clz
show_result("Complex left zero test", A1_clz[3, 1], tol)

r_clz ⊛ A1_clz
err_clz = norm(A1_clz - A_clz, Inf) / norm(A_clz, Inf)
show_result("Complex left inverse", err_clz, tol)

A_crz = randn(Complex{Float64}, 3, 3)
r_crz = rgivens(A_crz[1, 1], A_crz[1, 3], 1, 3)
A1_crz = copy(A_crz)
A1_crz ⊛ r_crz
show_result("Complex right zero test", A1_crz[1, 3], tol)

A1_crz ⊘ r_crz
err_crz = norm(A1_crz - A_crz, Inf) / norm(A_crz, Inf)
show_result("Complex right inverse", err_crz, tol)

A_clz1 = randn(Complex{Float64}, 3, 3)
r_clz1 = lgivens1(A_clz1[1, 1], A_clz1[3, 1], 1, 3)
A1_clz1 = copy(A_clz1)
r_clz1 ⊘ A1_clz1
show_result("Complex left zero test (first element)", A1_clz1[1, 1], tol)

r_clz1 ⊛ A1_clz1
err_clz1 = norm(A1_clz1 - A_clz1, Inf) / norm(A_clz1, Inf)
show_result("Complex left inverse (first element)", err_clz1, tol)

A_crz1 = randn(Complex{Float64}, 3, 3)
r_crz1 = rgivens1(A_crz1[1, 1], A_crz1[1, 3], 1, 3)
A1_crz1 = copy(A_crz1)
A1_crz1 ⊛ r_crz1
show_result("Complex right zero test (first element)", A1_crz1[1, 1], tol)

A1_crz1 ⊘ r_crz1
err_crz1 = norm(A1_crz1 - A_crz1, Inf) / norm(A_crz1, Inf)
show_result("Complex right inverse (first element)", err_crz1, tol)
