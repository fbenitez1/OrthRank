using BandStruct.BandColumnMatrices
using BandStruct.LeadingBandColumnMatrices
using BandStruct.BandRotations
using BandStruct.BandHouseholder
using Householder

using Random
using Rotations
using InPlace
using ShowTests
using LinearAlgebra

# Generate a random banded matrix in unstructured form.
function makeA(
  T::Type{E},
  m::Integer,
  lbw::Integer,
  ubw::Integer,
) where {R<:AbstractFloat,E<:Union{R,Complex{R}}}
  a = zeros(T, m, m)
  for j = 1:m
    @views randn!(a[max(1, j - lbw):min(m, j + ubw),j])
  end
  a
end

function makeB(
  T::Type{E},
  m::Int,
  lbw::Int,
  ubw::Int,
) where {R<:AbstractFloat,E<:Union{R,Complex{R}}}

  blocks = [k for j ∈ 1:2, k ∈ 1:m]
  lbc = LeadingBandColumn(
    T,
    MersenneTwister(0),
    m,
    m,
    upper_bw_max = lbw + ubw,
    lower_bw_max = lbw,
    upper_blocks = blocks,
    lower_blocks = blocks,
    upper_ranks = [ubw for j ∈ 1:m],
    lower_ranks = [lbw for j ∈ 1:m],
  )

  toBandColumn(lbc)
end

function bandQRB(
  B::AbstractBandColumn{S,E},
  lbw::Int,
  ubw::Int,
) where {S,E<:Number}
  (m, n) = size(B)
  work = zeros(E, lbw + ubw + 1)
  v = zeros(E, lbw + 1)
  @inbounds for k ∈ 1:(n - 1)
    j_end = min(k + lbw, m)
    bulge_lower!(B, j_end, k)
    h = householder(B, k:min(k + lbw, m), k, v, work)
    h ⊘ B
    notch_lower!(B, k + 1, k)
  end
  nothing
end

function bandQRA(A::AbstractArray{E,2}, lbw::Int, ubw::Int) where {E<:Number}
  (ma, _) = size(A)
  work = zeros(E, lbw + ubw + 1)
  v = zeros(E, lbw + 1)
  @inbounds for k = 1:(ma - 1)
    j1 = min(k + lbw, ma)
    h = householder(A, k:j1, k, v, work)
    @views h ⊘ A[:, k:min(ma, k + lbw + ubw)]
    A[(k + 1):j1, k] .= zero(E)
  end
  nothing
end

tol = 1e-13


m0=100
m=10000
lbw=100
ubw=100

# The benchmarks.
println()
println("Testing and benchmarking bandQRB:")
B = makeB(Float64, m0, lbw, ubw)
B0=copy(B)
bandQRB(B, lbw, ubw)

show_error_result(
  "Structured Band QR Singular Values Test",
  norm(svdvals(Matrix(B)) - svdvals(Matrix(B0))),
  tol
)

show_error_result(
  "Structured Band QR Lower Triangular Test",
  norm(tril(Matrix(B),-1)),
  tol
)

B = makeB(Float64, m, lbw, ubw)
@time bandQRB(B, lbw, ubw)

println()
println("Testing and benchmarking bandQRA:")
A = makeA(Float64, m0, lbw, ubw);
A0=copy(A)
bandQRA(A, lbw, ubw)

show_error_result(
  "Unstructured Band QR Singular Values Test",
  norm(svdvals(A) - svdvals(A0)),
  tol
)

show_error_result(
  "Unstructured Band QR Lower Triangular Test",
  norm(tril(A,-1)),
  tol
)

A = makeA(Float64, m, lbw, ubw);
@time bandQRA(A, lbw, ubw)
