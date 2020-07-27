println("""

******************************************************
** Tests of nonadjacent rotations on general matrices.
******************************************************
""")

A_rlz = randn(Float64, 3, 3)
r_rlz = lgivens(A_rlz[1, 1], A_rlz[3, 1], 1, 3)
A1_rlz = copy(A_rlz)
r_rlz ⊘ A1_rlz
show_error_result("Real left zero test", A1_rlz[3, 1], tol)

r_rlz ⊛ A1_rlz
err_rlz = norm(A1_rlz - A_rlz, Inf) / norm(A_rlz, Inf)
show_error_result("Real left inverse", err_rlz, tol)

A_rrz = randn(Float64, 3, 3)
r_rrz = rgivens(A_rrz[1, 1], A_rrz[1, 3], 1, 3)
A1_rrz = copy(A_rrz)
A1_rrz ⊛ r_rrz
show_error_result("Real right zero test", A1_rrz[1, 3], tol)

A1_rrz ⊘ r_rrz
err_rrz = norm(A1_rrz - A_rrz, Inf) / norm(A_rrz, Inf)
show_error_result("Real right inverse", err_rrz, tol)

A_rlz1 = randn(Float64, 3, 3)
r_rlz1 = lgivens1(A_rlz1[1, 1], A_rlz1[3, 1], 1, 3)
A1_rlz1 = copy(A_rlz1)
r_rlz1 ⊘ A1_rlz1
show_error_result("Real left zero test (first element)", A1_rlz1[1, 1], tol)

r_rlz1 ⊛ A1_rlz1
err_rlz1 = norm(A1_rlz1 - A_rlz1, Inf) / norm(A_rlz1, Inf)
show_error_result("Real left inverse (first element)", err_rlz1, tol)

A_rrz1 = randn(Float64, 3, 3)
r_rrz1 = rgivens1(A_rrz1[1, 1], A_rrz1[1, 3], 1, 3)
A1_rrz1 = copy(A_rrz1)
A1_rrz1 ⊛ r_rrz1
show_error_result("Real right zero test (first element)", A1_rrz1[1, 1], tol)

A1_rrz1 ⊘ r_rrz1
err_rrz1 = norm(A1_rrz1 - A_rrz1, Inf) / norm(A_rrz1, Inf)
show_error_result("Real right inverse (first element)", err_rrz1, tol)


# Complex tests 

A_clz = randn(Complex{Float64}, 3, 3)
r_clz = lgivens(A_clz[1, 1], A_clz[3, 1], 1, 3)
A1_clz = copy(A_clz)
r_clz ⊘ A1_clz
show_error_result("Complex left zero test", A1_clz[3, 1], tol)

r_clz ⊛ A1_clz
err_clz = norm(A1_clz - A_clz, Inf) / norm(A_clz, Inf)
show_error_result("Complex left inverse", err_clz, tol)

A_crz = randn(Complex{Float64}, 3, 3)
r_crz = rgivens(A_crz[1, 1], A_crz[1, 3], 1, 3)
A1_crz = copy(A_crz)
A1_crz ⊛ r_crz
show_error_result("Complex right zero test", A1_crz[1, 3], tol)

A1_crz ⊘ r_crz
err_crz = norm(A1_crz - A_crz, Inf) / norm(A_crz, Inf)
show_error_result("Complex right inverse", err_crz, tol)

A_clz1 = randn(Complex{Float64}, 3, 3)
r_clz1 = lgivens1(A_clz1[1, 1], A_clz1[3, 1], 1, 3)
A1_clz1 = copy(A_clz1)
r_clz1 ⊘ A1_clz1
show_error_result("Complex left zero test (first element)", A1_clz1[1, 1], tol)

r_clz1 ⊛ A1_clz1
err_clz1 = norm(A1_clz1 - A_clz1, Inf) / norm(A_clz1, Inf)
show_error_result("Complex left inverse (first element)", err_clz1, tol)

A_crz1 = randn(Complex{Float64}, 3, 3)
r_crz1 = rgivens1(A_crz1[1, 1], A_crz1[1, 3], 1, 3)
A1_crz1 = copy(A_crz1)
A1_crz1 ⊛ r_crz1
show_error_result("Complex right zero test (first element)", A1_crz1[1, 1], tol)

A1_crz1 ⊘ r_crz1
err_crz1 = norm(A1_crz1 - A_crz1, Inf) / norm(A_crz1, Inf)
show_error_result("Complex right inverse (first element)", err_crz1, tol)

println("""

***************************************************
** Tests of adjacent rotations on general matrices.
***************************************************
""")

tol = 1e-15
A_ralz = randn(Float64, 3, 3)
r_ralz = lgivens(A_ralz[1, 1], A_ralz[2, 1], 1)
A1_ralz = copy(A_ralz)
r_ralz ⊘ A1_ralz
show_error_result("Real left zero test", A1_ralz[2, 1], tol)

r_ralz ⊛ A1_ralz
err_ralz = norm(A1_ralz - A_ralz, Inf) / norm(A_ralz, Inf)
show_error_result("Real left inverse", err_ralz, tol)

A_rarz = randn(Float64, 3, 3)
r_rarz = rgivens(A_rarz[1, 1], A_rarz[1, 2], 1)
A1_rarz = copy(A_rarz)
A1_rarz ⊛ r_rarz
show_error_result("Real right zero test", A1_rarz[1, 2], tol)

A1_rarz ⊘ r_rarz
err_rarz = norm(A1_rarz - A_rarz, Inf) / norm(A_rarz, Inf)
show_error_result("Real right inverse", err_rarz, tol)

A_ralz1 = randn(Float64, 3, 3)
r_ralz1 = lgivens1(A_ralz1[1, 1], A_ralz1[2, 1], 1)
A1_ralz1 = copy(A_ralz1)
r_ralz1 ⊘ A1_ralz1
show_error_result("Real left zero test (first element)", A1_ralz1[1, 1], tol)

r_ralz1 ⊛ A1_ralz1
err_ralz1 = norm(A1_ralz1 - A_ralz1, Inf) / norm(A_ralz1, Inf)
show_error_result("Real left inverse (first element)", err_ralz1, tol)

A_rarz1 = randn(Float64, 3, 3)
r_rarz1 = rgivens1(A_rarz1[1, 1], A_rarz1[1, 2], 1)
A1_rarz1 = copy(A_rarz1)
A1_rarz1 ⊛ r_rarz1
show_error_result("Real right zero test (first element)", A1_rarz1[1, 1], tol)

A1_rarz1 ⊘ r_rarz1
err_rarz1 = norm(A1_rarz1 - A_rarz1, Inf) / norm(A_rarz1, Inf)
show_error_result("Real right inverse (first element)", err_rarz1, tol)


# Complex tests 

A_calz = randn(Complex{Float64}, 3, 3)
r_calz = lgivens(A_calz[1, 1], A_calz[2, 1], 1)
A1_calz = copy(A_calz)
r_calz ⊘ A1_calz
show_error_result("Complex left zero test", A1_calz[2, 1], tol)

r_calz ⊛ A1_calz
err_calz = norm(A1_calz - A_calz, Inf) / norm(A_calz, Inf)
show_error_result("Complex left inverse", err_calz, tol)

A_carz = randn(Complex{Float64}, 3, 3)
r_carz = rgivens(A_carz[1, 1], A_carz[1, 2], 1)
A1_carz = copy(A_carz)
A1_carz ⊛ r_carz
show_error_result("Complex right zero test", A1_carz[1, 2], tol)

A1_carz ⊘ r_carz
err_carz = norm(A1_carz - A_carz, Inf) / norm(A_carz, Inf)
show_error_result("Complex right inverse", err_carz, tol)

A_calz1 = randn(Complex{Float64}, 3, 3)
r_calz1 = lgivens1(A_calz1[1, 1], A_calz1[2, 1], 1)
A1_calz1 = copy(A_calz1)
r_calz1 ⊘ A1_calz1
show_error_result("Complex left zero test (first element)", A1_calz1[1, 1], tol)

r_calz1 ⊛ A1_calz1
err_calz1 = norm(A1_calz1 - A_calz1, Inf) / norm(A_calz1, Inf)
show_error_result("Complex left inverse (first element)", err_calz1, tol)

A_carz1 = randn(Complex{Float64}, 3, 3)
r_carz1 = rgivens1(A_carz1[1, 1], A_carz1[1, 2], 1)
A1_carz1 = copy(A_carz1)
A1_carz1 ⊛ r_carz1
show_error_result(
  "Complex right zero test (first element)",
  A1_carz1[1, 1],
  tol,
)

A1_carz1 ⊘ r_carz1
err_carz1 = norm(A1_carz1 - A_carz1, Inf) / norm(A_carz1, Inf)
show_error_result("Complex right inverse (first element)", err_carz1, tol)
