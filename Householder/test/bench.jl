using Householder

using Random
using InPlace
using LinearAlgebra
using BenchmarkTools

m=1000
n=900
E=Float64
A=randn(E,m,n)
A0=copy(A)

# With MKL.
#
# Benchmarking basic Householder QR:
# Backward error: 2.1143700848757227e-12
#   801.568 ms (5 allocations: 7.64 MiB)

# Benchmarking LAPACK QR:
# Backward error: 9.363518135803047e-13
#   64.239 ms (17 allocations: 42.53 MiB)

# Benchmarking WY QR:
# Backward error: 1.4562074846772347e-12
#   63.025 ms (16 allocations: 8.42 MiB)



print("""

Benchmarking basic Householder QR:
""")
A .= A0
(Q,R) = qrH(A)
println("Backward error: ", norm(Q*R-A0))
@btime begin
  A .= A0
  (Q,R) = qrH(A)
end

print("""

Benchmarking LAPACK QR:
""")
A .= A0
(Q,R) = qrLA(A)
println("Backward error: ", norm(Q*R-A0))
@btime begin
  A .= A0
  (Q,R) = qrLA(A)
end

print("""

Benchmarking WY QR:
""")
A .= A0
(Q,R) = qrWY(A)
println("Backward error: ", norm(Q*R-A0))
@btime begin
  A .= A0
  (Q,R) = qrWY(A)
  nothing
end

print("""

Testing WY QR with Q Returned as a SweepForward{WYTrans}:
""")
A .= A0
(Q,R) = qrWYSweep(A)
Q⊛R
println("Backward error: ", norm(R-A0))
