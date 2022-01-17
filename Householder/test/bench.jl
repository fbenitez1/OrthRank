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


print("""

Benchmarking basic Householder QR:
""")
A[:,:]=A0
(Q,R) = qrH(A)
println("Backward error: ", norm(Q*R-A0))
@btime begin
  A[:,:]=A0
  (Q,R) = qrH(A)
end

print("""

Benchmarking LAPACK QR:
""")
A[:,:]=A0
(Q,R) = qrLA(A)
println("Backward error: ", norm(Q*R-A0))
@btime begin
  A[:,:]=A0
  (Q,R) = qrLA(A)
end

print("""

Benchmarking WY QR:
""")
A[:,:]=A0
(Q,R) = qrWY(A)
println("Backward error: ", norm(Q*R-A0))
@btime begin
  A[:,:]=A0
  (Q,R) = qrWY(A)
  nothing
end

print("""

Testing WY QR with Q Returned as a SweepForward{WYTrans}:
""")
A[:,:]=A0
(Q,R) = qrWYSweep(A)
Q⊛R
println("Backward error: ", norm(R-A0))
