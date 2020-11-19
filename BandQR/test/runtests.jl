using BandQR.Factorization

# Generate a random banded type containing floats.
function makeA(
  n::Integer,
  lbw::Integer,
  ubw::Integer,
  ::Type{T} = Float64,
) where {R<:AbstractFloat,T<:Union{R,Complex{R}}}
  a = zeros(T, n, n)
  for j = 1:n
    for k = max(1, j - lbw):min(n, j + ubw)
      a[j, k] = randn(T)
    end
  end
  a::Array{T,2}
end

# Test problem size and lower and upper bandwidths.
n = 10000
lbw = 5
ubw = 5

# Real and complex test cases, (possibly) storing the original matrix.
a = makeA(n, lbw, ubw)
a0 = copy(a)
a_small = makeA(5, 1, 1)

# ac = makeA(n,lbw,ubw,Complex{Float64})
# ac0 = copy(ac)

# The benchmarks.
precompile(bandQR, (Array{Float64,2}, Int, Int))
@time bandQR(a, lbw, ubw)
# @time bandQR(ac,lbw,ubw)
# Results:
# 0.003723 seconds (4 allocations: 160 bytes)
# 0.009409 seconds (4 allocations: 160 bytes)

# A similar C benchmark on the same machine ran in about 0.004
# seconds.

GC.gc()
