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
