module BandFactor

export bandQRB,
  bandQRA, get_WY, qrBWYSweep, lqBH, lqBWY, lqBWYSweep, makeBForQR, makeBForLQ, makeA

using LinearAlgebra
using Random

using Householder
using InPlace

using ..BandColumnMatrices
using ..BlockedBandColumnMatrices
using ..BandRotations
using ..BandHouseholder

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

# make a Fixed BW band matrix with space for a QR factorization.
function makeBForQR(
  T::Type{E},
  m::Int,
  lower_rank::Int,
  upper_rank::Int,
  block_size::Int
) where {R<:AbstractFloat,E<:Union{R,Complex{R}}}

  blocks = [k for _ ∈ 1:2, k ∈ 1:m]
  bbc = BlockedBandColumn(
    T,
    LeadingDecomp(),
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

function makeBForLQ(
  T::Type{E},
  m::Int,
  lower_rank::Int,
  upper_rank::Int,
  block_size::Int
) where {R<:AbstractFloat,E<:Union{R,Complex{R}}}

  blocks = [k for j ∈ 1:2, k ∈ 1:m]
  bbc = BlockedBandColumn(
    T,
    LeadingDecomp(),
    MersenneTwister(0),
    m,
    m,
    upper_rank_max = upper_rank,
    lower_rank_max = lower_rank + upper_rank + block_size,
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
  work = similar_zeros(band_elements(B), lbw + ubw + 1)
  v = similar_zeros(band_elements(B), lbw + 1)
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
  work = similar_zeros(A, lbw + ubw + 1)
  v = similar_zeros(A, lbw + 1)
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
  v = similar_zeros(band_elements(B), lbw + 1)
  workh = similar_zeros(band_elements(B), m)
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

function lqBH(
  B::AbstractBandColumn{S,E},
  lbw::Int,
  ubw::Int
) where {S,E<:Number}
  (m, n) = size(B)
  work = similar_zeros(band_elements(B), lbw + ubw + 1)
  v = similar_zeros(band_elements(B),ubw + 1)
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
  v = similar_zeros(band_elements(B), ubw + 1)
  wy = WYTrans(
    E,
    max_WY_size = n,
    work_size = n * (block_size + 2) + n * (block_size + ubw),
    max_num_hs = block_size + 2,
  )
  workh = similar_zeros(band_elements(B), n)
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
  v = similar_zeros(band_elements(B), ubw + 1)
  wy = WYTrans(
    E,
    max_num_WY = blocks,
    max_WY_size = n,
    work_size = n * (block_size + 2) + n * (block_size + ubw),
    max_num_hs = block_size + 2,
  )
  workh = similar_zeros(band_elements(B), n)
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


end # module
