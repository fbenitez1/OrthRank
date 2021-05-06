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
  ubw::Int;
  lower_bw_max = lbw+ubw,
  upper_bw_max = ubw,
) where {R<:AbstractFloat,E<:Union{R,Complex{R}}}

  blocks = [k for j ∈ 1:2, k ∈ 1:m]
  lbc = LeadingBandColumn(
    T,
    MersenneTwister(0),
    m,
    m,
    upper_bw_max = upper_bw_max,
    lower_bw_max = lower_bw_max,
    upper_blocks = blocks,
    lower_blocks = blocks,
    upper_ranks = [ubw for j ∈ 1:m],
    lower_ranks = [lbw for j ∈ 1:m],
  )
  toBandColumn(lbc)
end

function lqBH(
  B::AbstractBandColumn{S,E},
  lbw::Int,
  ubw::Int
) where {S,E<:Number}
  (m, n) = size(B)
  work = zeros(E, lbw + ubw + 1)
  v = zeros(E,ubw + 1)
  @inbounds for j ∈ 1:(m - 1)
    k_end = min(j + ubw, n)
    bulge_upper!(B, j, k_end)
    h = householder(B, j, j:k_end, v, work)
    B ⊛ h
    notch_upper!(B, j, j + 1)
  end
  nothing
end


function lqBWY(
  B::AbstractBandColumn{S,E},
  lbw::Int,
  ubw::Int;
  block_size::Int = 3,
) where {S,E<:Number}
  m, n = size(B)
  blocks, rem = divrem(m, block_size)
  blocks = rem > 0 ? blocks + 1 : blocks
  v = zeros(E, ubw + 1)
  wy = WYTrans(
    E,
    max_WY_size = n,
    work_size = n * (block_size + 2) + n * (block_size + ubw),
    max_num_hs = block_size + 2,
  )
  workh = zeros(E, n)
  selectWY!(wy, 1)
  @views for b ∈ 1:blocks
    j_dense = last_inband_index(B, :, ((b - 1) * block_size + 1))
    offs = (b - 1) * block_size
    resetWYBlock!(wy, offset = offs, sizeWY = min(block_size + ubw, n - offs))
    block_end = min(b * block_size, m)
    for j ∈ ((b - 1) * block_size + 1):block_end
      k_end = min(j + ubw, n)
      h = householder(B, j, j:k_end, v, workh)
      B[j:block_end, :] ⊛ h
      B[j_dense+1:m, :] ⊛ h
      j < n && notch_upper!(B, j, j + 1)
      wy ⊛ h
    end
    B[(block_end + 1):j_dense, :] ⊛ wy
  end
  nothing
end

function lqBWYSweep(
  B::AbstractBandColumn{S,E},
  lbw::Int,
  ubw::Int;
  block_size::Int=3
) where {S,E<:Number}
  m, n = size(B)
  blocks, rem = divrem(m, block_size)
  blocks = rem > 0 ? blocks + 1 : blocks
  v = zeros(E, ubw + 1)
  wy = WYTrans(
    E,
    max_num_WY = blocks,
    max_WY_size = n,
    work_size = n * (block_size + 2) + n * (block_size + ubw),
    max_num_hs = block_size + 2,
  )
  workh = zeros(E, n)
  @views for b ∈ 1:blocks
    selectWY!(wy, b)
    offs = (b - 1) * block_size
    resetWYBlock!(
      wy,
      block = b,
      offset = offs,
      sizeWY = min(block_size + ubw, n - offs),
    )
    block_end = min(b * block_size, m)
    for j ∈ ((b - 1) * block_size + 1):block_end
      k_end = min(j + ubw, n)
      h = householder(B, j, j:k_end, v, workh)
      B[j: block_end,:] ⊛ h
      j < n && notch_upper!(B, j, j + 1)
      wy ⊛ h
    end
    B[(block_end + 1):m,:] ⊛ wy
  end
  (SweepForward(wy), B)
end

tol = 1e-12

# Tests
m0=1000
lbw=100
ubw=100
bs = 8

B = makeB(
  Float64,
  m0,
  lbw,
  ubw,
  upper_bw_max = ubw + bs,
  lower_bw_max = ubw + lbw + bs,
)
B0 = copy(B)

# lqH
println()
println("Testing lqH:")
lqBH(B, lbw, ubw)

show_error_result(
  "Structured Band LQ Singular Values Test",
  norm(svdvals(Matrix(B)) - svdvals(Matrix(B0))),
  tol
)

show_error_result(
  "Structured Band LQ Lower Triangular Test",
  norm(triu(Matrix(B),1)),
  tol
)

# lqBWY
println()
println("Testing lqBWY:")
B = copy(B0)
lqBWY(B, lbw, ubw, block_size = bs)

show_error_result(
  "Structured Band LQ Singular Values Test",
  norm(svdvals(Matrix(B)) - svdvals(Matrix(B0))),
  tol
)

show_error_result(
  "Structured Band LQ Lower Triangular Test",
  norm(triu(Matrix(B),1)),
  tol
)

println()
println("Testing lqBWYSweep:")
B = copy(B0)
(Qwy, B) = lqBWYSweep(B, lbw, ubw, block_size = bs)

Q = Matrix{Float64}(I, m0, m0)
Q ⊛ Qwy

show_error_result(
  "Structured Band LQ Singular Values Test",
  norm(svdvals(Matrix(B)) - svdvals(Matrix(B0))),
  tol
)

show_error_result(
  "Structured Band LQ Backward Error Test 1",
  norm(Matrix(B) - Matrix(B0) * Q),
  tol
)

show_error_result(
  "Structured Band LQ Backward Error Test 2",
  let
    B1 = Matrix(B0)
    B1 ⊛ Qwy
    norm(Matrix(B) - B1)
  end,
  tol,
)

show_error_result(
  "Structured Band LQ Lower Triangular Test",
  norm(triu(Matrix(B),1)),
  tol
)

# Benchmarks

# Tests
m0=10000
lbw=100
ubw=100
bs = 16

B = makeB(
  Float64,
  m0,
  lbw,
  ubw,
  upper_bw_max = ubw + bs,
  lower_bw_max = ubw + lbw + bs,
)
B0 = copy(B)

# lqH
println()
println("Benchmarking lqH:")
@time lqBH(B, lbw, ubw)


println()
println("Benchmarking lqBWY:")
B=copy(B0);

using Profile
Profile.clear();
Profile.init(n=10^7, delay=0.0001)
@time lqBWY(B, lbw, ubw, block_size=bs);
B=copy(B0);
@profile lqBWY(B, lbw, ubw, block_size=bs);
