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

# Benchmarking basic Householder QR:
# Backward error: 2.0948689662281884e-12
#   806.030 ms (7 allocations: 7.65 MiB)

# Benchmarking LAPACK QR:
# Backward error: 8.985424305429105e-13
#   74.940 ms (19 allocations: 42.53 MiB)

# Benchmarking WY QR:
# Backward error: 1.3541753069454746e-12
#   213.774 ms (18 allocations: 8.42 MiB)

# Testing WY QR with Q Returned as a SweepForward{WYTrans}:
# Backward error: 1.2331889234282412e-12


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
