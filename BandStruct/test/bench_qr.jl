using BandStruct.BandColumnMatrices
using BandStruct.BlockedBandColumnMatrices
using BandStruct.BandRotations
using BandStruct.BandHouseholder
using Householder

using Random
using Rotations
using InPlace
using ShowTests
using LinearAlgebra
using BenchmarkTools

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

# make a band matrix with space for a QR factorization.
function makeB(
  T::Type{E},
  m::Int,
  lower_rank::Int,
  upper_rank::Int,
  block_size::Int
) where {R<:AbstractFloat,E<:Union{R,Complex{R}}}

  blocks = [k for _ ∈ 1:2, k ∈ 1:m]
  bbc = BlockedBandColumn(
    T,
    LeadingDecomp,
    MersenneTwister(0),
    m,
    m,
    upper_rank_max = lower_rank + upper_rank + block_size,
    lower_rank_max = lower_rank,
    upper_blocks = blocks,
    lower_blocks = blocks,
    upper_ranks = [upper_rank for j ∈ 1:m],
    lower_ranks = [lower_rank for j ∈ 1:m],
  )

  toBandColumn(bbc)
end

# Simple Householder band QR.
function bandQRB(
  B::AbstractBandColumn{S,E},
  lbw::Int,
  ubw::Int,
) where {S,E<:Number}
  (m, n) = size(B)
  work = zeros(E, lbw + ubw + 1)
  v = zeros(E, lbw + 1)
  @inbounds for k ∈ 1:(n - 1)
    h = householder(B, k:min(k + lbw, m), k, v, work)
    h ⊘ B
    notch_lower!(B, k + 1, k)
  end
  nothing
end

# Unstructured band QR
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

# get an appropriately sized WYTrans for a blocked band QR factorization.
function get_WY(
  B::AbstractBandColumn{S,E},
  lbw::Int,
  ubw::Int;
  block_size::Int=16
) where {S,E<:Number}
  m, n = size(B)
  blocks, rem = divrem(n, block_size)
  blocks = rem > 0 ? blocks + 1 : blocks
  WYTrans(
    E,
    max_num_WY = blocks,
    max_WY_size = m,
    work_size = m * (block_size + 2) + m * (block_size + lbw),
    max_num_hs = block_size,
  )
end

function qrBWYSweep(
  B::AbstractBandColumn{S,E},
  lbw::Int,
  ubw::Int;
  block_size::Int=16
) where {S,E<:Number}
  wy = get_WY(B, lbw, ubw, block_size = block_size)
  qrBWYSweep(wy, B, lbw, ubw, block_size = block_size)
end

function qrBWYSweep(
  wy, # Orthogonal Q stored as overlapping WY transformations.
  B::AbstractBandColumn{S,E},
  lbw::Int, # Lower Bandwidth
  ubw::Int; # Upper Bandwidth
  block_size::Int=32
) where {S,E<:Number}
  m, n = size(B)
  blocks, rem = divrem(n, block_size)
  blocks = rem > 0 ? blocks + 1 : blocks
  v = zeros(E, lbw + 1)
  workh = zeros(E, m)
  @views for b ∈ 1:blocks
    selectWY!(wy, b)
    offs = (b - 1) * block_size
    resetWYBlock!(
      wy,
      block = b,
      offset = offs,
      sizeWY = min(block_size + lbw, m - offs),
    )
    block_end = min(b * block_size, n)
    for k ∈ ((b - 1) * block_size + 1):block_end
      j_end = min(k + lbw, m)
      h = householder(B, k:j_end, k, v, workh)
      h ⊘ B[:, k: block_end] 
      k < m && notch_lower!(B, k+1, k)
      wy ⊛ h
    end
    wy ⊘ B[:, (block_end + 1):n]
  end
  (SweepForward(wy), B)
end


tol = 1e-13


m0=100
m=10000
lbw=100
ubw=100

# The benchmarks.
println()
println("Testing and benchmarking bandQRB:")
B = makeB(Float64, m0, lbw, ubw, 1)
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

B = makeB(Float64, m, lbw, ubw, 1)
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


m=10000
lbw=400
ubw=400
bs = 16
B = makeB(Float64, m, lbw, ubw, 32)
B0=copy(B)
wy = get_WY(B, lbw, ubw, block_size=bs)
q,r = qrBWYSweep(wy, B, lbw, ubw, block_size=bs)
display(
  @benchmark qrBWYSweep(wy1, B1, $lbw, $ubw, block_size = $bs) evals = 1 setup =
  begin
    B1 = copy(B0)
    wy1 = get_WY(B1, $lbw, $ubw, block_size = $bs)
  end)
GC.gc()
