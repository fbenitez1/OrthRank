module Factor

using LinearAlgebra
using InPlace
using ..Compute
using ..WY

export qrH, qrWY, qrWYSweep, qrLA

function qrH(A::AbstractArray{E,2}) where {E<:Number}
  m, n = size(A)
  Q = similar_leftI(A)
  v = similar_zeros(A, m)
  work = similar_zeros(A, m)
  @inbounds @views for k ∈ axes(A,2)
    vk = v[1:(m - k + 1)]
    vk[:] = A[k:m, k]
    h = lhouseholder(vk, 1, k - 1, work)
    h ⊘ A[:, k:n]
    A[(k + 1):m, k] .= zero(E)
    Q ⊛ h
  end
  (Q, A)
end

function qrWY(A::Array{E,2}; block_size::Int=32) where {E<:Number}

  Base.require_one_based_indexing(A)
  m, n = size(A)
  blocks, rem = divrem(n, block_size)
  blocks = rem > 0 ? blocks + 1 : blocks
  Q = similar_leftI(A)
  v = similar_zeros(A, m)
  wy = WYTrans(
    E,
    max_WY_size = m,
    work_size = m * (block_size + 2),
    max_num_hs = block_size + 2,
  )
  workh = similar_zeros(A, m)
  selectWY!(wy, 1)
  @inbounds @views for b ∈ 1:blocks
    offs = (b - 1) * block_size
    resetWYBlock!(wy, offset = offs, sizeWY = m - offs)
    block_end = min(b * block_size, n)
    for k ∈ ((b - 1) * block_size + 1):block_end
      vk = v[1:(m - k + 1)]
      h = householder(A, k:m, k, vk, workh)
      h ⊘ A[:, k:block_end]
      A[(k + 1):m, k] .= zero(E)
      wy ⊛ h
    end
    Q ⊛ wy
    wy ⊘ A[:, (block_end + 1):n]
  end
  (Q, A)
end

function qrWYSweep(A::Array{E,2}; block_size::Int=32) where {E<:Number}

  Base.require_one_based_indexing(A)
  m, n = size(A)
  blocks, rem = divrem(n, block_size)
  blocks = rem > 0 ? blocks + 1 : blocks
  v = similar_zeros(A, m)
  wy = WYTrans(
    E,
    max_num_WY = blocks,
    max_WY_size = m,
    work_size = m * (block_size + 2),
    max_num_hs = block_size + 2,
  )
  workh = similar_zeros(A, m)
  @inbounds @views for b ∈ 1:blocks
    offs = (b - 1) * block_size
    selectWY!(wy, b)
    resetWYBlock!(wy, block = b, offset = offs, sizeWY = m - offs)
    block_end = min(b * block_size, n)
    for k ∈ ((b - 1) * block_size + 1):block_end
      vk = v[1:(m - k + 1)]
      h = householder(A, k:m, k, vk, workh)
      h ⊘ A[:, k:block_end]
      A[(k + 1):m, k] .= zero(E)
      wy ⊛ h
    end
    wy ⊘ A[:, (block_end + 1):n]
  end
  (SweepForward(wy), A)
end

function qrLA(A::AbstractArray{E,2}) where {E<:Number}
  m, n = size(A)
  qrA=qr(A)
  Q = similar_leftI(A)
  Q = qrA.Q * Q
  R = similar_zeros(A, m, n)
  R[1:n,1:n] = qrA.R
  (Q,R)
end
end
