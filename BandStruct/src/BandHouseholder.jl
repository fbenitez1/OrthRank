module BandHouseholder

using LinearAlgebra
using Base: @propagate_inbounds
using Printf

using Householder.Compute
using Householder.WY

using ..BandColumnMatrices
using ..LeadingBandColumnMatrices
using InPlace

# vector and work.
@propagate_inbounds function Compute.householder(
  bc::AbstractBandColumn{S,E},
  js::UnitRange{Int},
  k::Int,
  l::Int64,
  offs::Int64,
  v::AbstractArray{E,1},
  work::AbstractArray{E,1},
) where {E<:Number,S}
  lv = length(v)
  ljs = length(js)
  @boundscheck begin
    lv == ljs || throw(DimensionMismatch(@sprintf(
      """
      In lhouseholder(bc::AbstractBandColumn{S,E}, js, k, l, offs, v, work),
      length(v) is %d and length(js) is %d.  These should be equal.
      """,
      lv,
      ljs
    )))
    ((1:lv) .+ offs) ⊆ inband_index_range(bc, :, k) ||
      throw(SubcolumnIndicesNotInband(js, k))
  end

  storage_offs = storage_offset(bc, k)
  bc_els = band_elements(bc)

  @inbounds for j ∈ 1:lv
    v[j] = bc_els[j + offs - storage_offs, k]
  end

  lhouseholder(v, l, offs, work)
end

@propagate_inbounds function Compute.householder(
  bc::AbstractBandColumn{S,E},
  j::Int,
  ks::UnitRange{Int},
  l::Int64,
  offs::Int64,
  v::AbstractArray{E,1},
  work::AbstractArray{E,1},
) where {E<:Number,S}
  lv = length(v)
  lks = length(ks)
  @boundscheck begin
    lv == lks || throw(DimensionMismatch(@sprintf(
      """
      In lhouseholder(bc::AbstractBandColumn{S,E}, j, ks, l, offs, v, work),
      length(v) is %d and length(ks) is %d.  These should be equal.
      """,
      lv,
      lks
    )))
    ((1:lv) .+ offs) ⊆ inband_index_range(bc, j, :) ||
      throw(SubrowIndicesNotInband(j, ks))
  end

  bc_els = band_elements(bc)

  @inbounds for k ∈ 1:lv
    storage_offs = storage_offset(bc, k + offs)
    v[k] = bc_els[j - storage_offs, k + offs]
  end
  rhouseholder(v, l, offs, work)
end


@inline function InPlace.:⊛(
  bc::AbstractBandColumn{S,E},
  h::HouseholderTrans{E},
) where {E<:Number,S}

  m = h.size
  v = reshape(h.v, m, 1)
  offs = h.offs
  β = h.β

  work = h.work
  lw = length(work)

  k_first = offs + 1
  k_last = offs + h.size
  n_bc = k_last - k_first + 1

  j_first = first_inband_index(bc, :, k_first)
  j_last = last_inband_index(bc, :, k_last)
  m_bc = j_last - j_first + 1
  bc_els = band_elements(bc)

  @boundscheck begin
    check_bc_storage_bounds(bc, j_first, k_last)
    check_bc_storage_bounds(bc, j_last, k_first)

    lw >= m_bc || throw(DimensionMismatch(@sprintf(
      """
      In bc ⊛ h, h has work array of length %d. The active block
      of bc is %d×%d and requires a work array of length %d
      (the number of rows in the block).
      """,
      length(work),
      m_bc,
      n_bc,
      m_bc
    )))
  end

  @inbounds begin
    bulge_upper!(bc, j_first, k_last)
    bulge_lower!(bc, j_last, k_first)
    work[1:m_bc] .= zero(E)
    # Accumulate w = bc * v in work array by a linear combination of
    # columns of bc.
    for k ∈ 1:m
      j0 = first_inband_index(bc, :, k + offs)
      j1 = last_inband_index(bc, :, k + offs)
      storage_offs = storage_offset(bc, k + offs)
      @simd for j ∈ j0:j1
        work[j - j_first + 1] += bc_els[j - storage_offs, k + offs] * v[k]
      end
    end
    # Subtract β * w * vᴴ from bc.
    for k ∈ 1:m
      storage_offs = storage_offset(bc, k + offs)
      @simd for j ∈ j_first:j_last
        bc_els[j - storage_offs, k + offs] -=
          β * work[j - j_first + 1] * conj(v[k])
      end
    end
  end
  nothing
end

@inline function InPlace.:⊛(
  h::HouseholderTrans{E},
  bc::AbstractBandColumn{S,E},
) where {E<:Number,S}

  m = h.size
  v = reshape(h.v, m, 1)
  offs = h.offs
  β = h.β

  j_first = offs + 1
  j_last = offs + h.size
  m_bc = j_last - j_first + 1

  k_first = first_inband_index(bc, j_first, :)
  k_last = last_inband_index(bc, j_last, :)
  n_bc = k_last - k_first + 1
  bc_els = band_elements(bc)

  @boundscheck begin
    check_bc_storage_bounds(bc, j_first, k_last)
    check_bc_storage_bounds(bc, j_last, k_first)
  end

  @inbounds begin
    bulge_upper!(bc, j_first, k_last)
    bulge_lower!(bc, j_last, k_first)
    for k ∈ k_first:k_last
      x = zero(E)
      storage_offs = storage_offset(bc, k)
      jrange = (offs .+ (1:m)) ∩ inband_index_range(bc, :, k)
      # Form x = vᴴ * bc[:,k].
      @simd for j ∈ jrange
        x = x + conj(v[j - offs]) * bc_els[j - storage_offs, k]
      end
      # Subtract v * x from bc[:,k].
      x = β * x
      @simd for j ∈ 1:m
        bc_els[offs + j - storage_offs, k] -= v[j] * x
      end
    end
  end
  nothing
end

@inline function InPlace.:⊘(
  bc::AbstractBandColumn{S,E},
  h::HouseholderTrans{E},
) where {E<:Number,S}

  m = h.size
  v = reshape(h.v, m, 1)
  offs = h.offs
  β̄ = conj(h.β)

  work = h.work
  lw = length(work)

  k_first = offs + 1
  k_last = offs + h.size
  n_bc = k_last - k_first + 1

  j_first = first_inband_index(bc, :, k_first)
  j_last = last_inband_index(bc, :, k_last)
  m_bc = j_last - j_first + 1
  bc_els = band_elements(bc)

  @boundscheck begin
    check_bc_storage_bounds(bc, j_first, k_last)
    check_bc_storage_bounds(bc, j_last, k_first)

    lw >= m_bc || throw(DimensionMismatch(@sprintf(
      """
      In bc ⊛ h, h has work array of length %d. The active block
      of bc is %d×%d and requires a work array of length %d
      (the number of rows in the block).
      """,
      length(work),
      m_bc,
      n_bc,
      m_bc
    )))
  end

  @inbounds begin
    bulge_upper!(bc, j_first, k_last)
    bulge_lower!(bc, j_last, k_first)
    work[1:m_bc] .= zero(E)
    # Accumulate w = bc * v in work array by a linear combination of
    # columns of bc.
    for k ∈ 1:m
      j0 = first_inband_index(bc, :, k + offs)
      j1 = last_inband_index(bc, :, k + offs)
      storage_offs = storage_offset(bc, k + offs)
      @simd for j ∈ j0:j1
        work[j - j_first + 1] += bc_els[j - storage_offs, k + offs] * v[k]
      end
    end
    # Subtract β̄ * w * vᴴ from bc.
    for k ∈ 1:m
      storage_offs = storage_offset(bc, k + offs)
      @simd for j ∈ j_first:j_last
        bc_els[j - storage_offs, k + offs] -=
          β̄ * work[j - j_first + 1] * conj(v[k])
      end
    end
  end
  nothing
end

@inline function InPlace.:⊘(
  h::HouseholderTrans{E},
  bc::AbstractBandColumn{S,E},
) where {E<:Number,S}

  m = h.size
  v = reshape(h.v, m, 1)
  offs = h.offs
  β̄ = conj(h.β)

  j_first = offs + 1
  j_last = offs + h.size
  m_bc = j_last - j_first + 1

  k_first = first_inband_index(bc, j_first, :)
  k_last = last_inband_index(bc, j_last, :)
  n_bc = k_last - k_first + 1
  bc_els = band_elements(bc)

  @boundscheck begin
    check_bc_storage_bounds(bc, j_first, k_last)
    check_bc_storage_bounds(bc, j_last, k_first)
  end
  
  @inbounds begin
    bulge_upper!(bc, j_first, k_last)
    bulge_lower!(bc, j_last, k_first)
    for k ∈ k_first:k_last
      x = zero(E)
      storage_offs = storage_offset(bc, k)
      jrange = (offs .+ (1:m)) ∩ inband_index_range(bc, :, k)
      # Form x = vᴴ * bc[:,k].
      @simd for j ∈ jrange
        x = x + conj(v[j-offs]) * bc_els[j - storage_offs, k]
      end
      # Subtract v * x from bc[:,k].
      x = β̄ * x
      @simd for j ∈ 1:m
        bc_els[offs + j - storage_offs, k] -= v[j] * x
      end
    end
  end
  nothing
end

@inline function InPlace.apply!(
  bc::AbstractBandColumn{S,E},
  wy::WYTrans{E},
  k::Int
) where {E<:Number,S}

  num_WY = wy.num_WY[]
  @boundscheck k ∈ 1:num_WY || throw_WYBlockNotAvailable(k, num_WY)
  @inbounds begin
    inds = 1:wy.sizes[k]
    offset = wy.offsets[k]
    (mbc,nbc) = size(bc)
    num_hs= wy.num_hs[k]
  end

  @boundscheck begin
    inds .+ offset ⊆ 1:nbc ||
      throw_ColumnRange_DimensionMismatch(mbc, nbc, inds)
  end

  lw = length(wy.work)

  n_bc0 = wy.sizes[k]
  k_first = offset + 1
  k_last = offset + n_bc0

  j_first = first_inband_index(bc, :, k_first)
  j_last = last_inband_index(bc, :, k_last)
  m_bc0 = j_last - j_first + 1
  bc_els = band_elements(bc)

  @boundscheck begin
    lw >= m_bc0 * num_hs ||
      throw_WorkSizeError(mbc, nbc, m_bc0 * num_hs, length(wy.work))
    check_bc_storage_bounds(bc, j_first, k_last)
    check_bc_storage_bounds(bc, j_last, k_first)
  end

  @inbounds begin
    bulge_upper!(bc, j_first, k_last)
    bulge_lower!(bc, j_last, k_first)
    @views begin
      work = reshape(wy.work[1:mbc*num_hs], mbc, num_hs)
      W = wy.W[inds, 1:num_hs, k]
      Y = wy.Y[inds, 1:num_hs, k]
    end
    work .= zero(E)

    # Accumulate work = bc * W.
    for l ∈ 1:num_hs
      for kk ∈ 1:n_bc0
        j0 = first_inband_index(bc, :, kk + offset)
        j1 = last_inband_index(bc, :, kk + offset)
        storage_offs = storage_offset(bc, kk + offset)
        @simd for j ∈ j0:j1
          work[j - j_first + 1, l] += bc_els[j - storage_offs, kk + offset] *
            W[kk, l]
        end
      end
    end

    # Subtract bc * W * Yᴴ = work * Yᴴ from bc.
    for l ∈ 1:num_hs
      for kk ∈ 1:n_bc0
        storage_offs = storage_offset(bc, kk + offset)
        @simd for j ∈ j_first:j_last
          bc_els[j - storage_offs, kk + offset] -=
            work[j - j_first + 1, l] * conj(Y[kk,l])
        end
      end
    end
  end
  nothing
end

@inline function InPlace.apply_inv!(
  bc::AbstractBandColumn{S,E},
  wy::WYTrans{E},
  k::Int
) where {E<:Number,S}

  num_WY = wy.num_WY[]
  @boundscheck k ∈ 1:num_WY || throw_WYBlockNotAvailable(k, num_WY)
  @inbounds begin
    inds = 1:wy.sizes[k]
    offset = wy.offsets[k]
    (mbc,nbc) = size(bc)
    num_hs= wy.num_hs[k]
  end

  @boundscheck begin
    inds .+ offset ⊆ 1:nbc ||
      throw_ColumnRange_DimensionMismatch(mbc, nbc, inds)
  end

  lw = length(wy.work)
  n_bc0 = wy.sizes[k]
  k_first = offset + 1
  k_last = offset + n_bc0

  j_first = first_inband_index(bc, :, k_first)
  j_last = last_inband_index(bc, :, k_last)
  m_bc0 = j_last - j_first + 1
  bc_els = band_elements(bc)

  @boundscheck begin
    lw >= m_bc0 * num_hs ||
      throw_WorkSizeError(mbc, nbc, m_bc0 * num_hs, length(wy.work))
    check_bc_storage_bounds(bc, j_first, k_last)
    check_bc_storage_bounds(bc, j_last, k_first)
  end

  @inbounds begin
    bulge_upper!(bc, j_first, k_last)
    bulge_lower!(bc, j_last, k_first)
    @views begin
      work = reshape(wy.work[1:mbc*num_hs], mbc, num_hs)
      W = wy.W[inds, 1:num_hs, k]
      Y = wy.Y[inds, 1:num_hs, k]
    end
    work .= zero(E)

    # Accumulate work = bc * Y.
    for l ∈ 1:num_hs
      for kk ∈ 1:n_bc0
        j0 = first_inband_index(bc, :, kk + offset)
        j1 = last_inband_index(bc, :, kk + offset)
        storage_offs = storage_offset(bc, kk + offset)
        @simd for j ∈ j0:j1
          work[j - j_first + 1, l] += bc_els[j - storage_offs, kk + offset] *
            Y[kk, l]
        end
      end
    end

    # Subtract bc * Y * Wᴴ = work * Wᴴ from bc.
    for l ∈ 1:num_hs
      for kk ∈ 1:n_bc0
        storage_offs = storage_offset(bc, kk + offset)
        @simd for j ∈ j_first:j_last
          bc_els[j - storage_offs, kk + offset] -=
            work[j - j_first + 1, l] * conj(W[kk,l])
        end
      end
    end
  end
  nothing
end


@inline function InPlace.apply!(
  wy::WYTrans{E},
  k::Int,
  bc::AbstractBandColumn{S,E}
) where {E<:Number,S}

  num_WY = wy.num_WY[]
  @boundscheck k ∈ 1:num_WY || throw_WYBlockNotAvailable(k, num_WY)
  @inbounds begin
    inds = 1:wy.sizes[k]
    offset = wy.offsets[k]
    (mbc,nbc) = size(bc)
    m_bc0 = wy.sizes[k]
    num_hs= wy.num_hs[k]
  end

  @boundscheck begin
    inds .+ offset ⊆ 1:mbc ||
      throw_RowRange_DimensionMismatch(mbc, nbc, inds)
  end

  lw = length(wy.work)

  j_first = offset + 1
  j_last = offset + m_bc0

  k_first = first_inband_index(bc, j_first, :)
  k_last = last_inband_index(bc, j_last, :)
  n_bc0 = k_last - k_first + 1
  bc_els = band_elements(bc)

  @boundscheck begin
    lw >= n_bc0 * num_hs ||
      throw_WorkSizeError(mbc, nbc, n_bc0 * num_hs, length(wy.work))
    check_bc_storage_bounds(bc, j_first, k_last)
    check_bc_storage_bounds(bc, j_last, k_first)
  end

  @inbounds begin
    bulge_upper!(bc, j_first, k_last)
    bulge_lower!(bc, j_last, k_first)
    @views begin
      work = reshape(wy.work[1:n_bc0*num_hs], num_hs, n_bc0)
      W = wy.W[inds, 1:num_hs, k]
      Y = wy.Y[inds, 1:num_hs, k]
    end
    work .= zero(E)
    # Form work[l,kk] = Y[:,l]ᴴ * bc[:,kk].
    for l∈1:num_hs
      for kk ∈ k_first:k_last
        x = zero(E)
        storage_offs = storage_offset(bc, kk)
        jrange = (offset .+ (1:m_bc0)) ∩ inband_index_range(bc, :, kk)
        @simd for j ∈ jrange
          work[l,kk-k_first+1] += conj(Y[j - offset,l]) * bc_els[j - storage_offs, kk]
        end
      end
    end
    # Subtract  W * work from bc[j_first:j_last,k_first:k_last].
    for l∈1:num_hs
      for kk∈k_first:k_last
        storage_offs = storage_offset(bc, kk)
        jrange = (offset .+ (1:m_bc0)) ∩ inband_index_range(bc, :, kk)
        @simd for j ∈ jrange
          bc_els[j - storage_offs, kk] -= W[j-offset,l] * work[l,kk-k_first+1]
        end
      end
    end
  end
  nothing
end

@inline function InPlace.apply_inv!(
  wy::WYTrans{E},
  k::Int,
  bc::AbstractBandColumn{S,E}
) where {E<:Number,S}

  num_WY = wy.num_WY[]
  @boundscheck k ∈ 1:num_WY || throw_WYBlockNotAvailable(k, num_WY)
  @inbounds begin
    inds = 1:wy.sizes[k]
    offset = wy.offsets[k]
    (mbc,nbc) = size(bc)
    m_bc0 = wy.sizes[k]
    num_hs= wy.num_hs[k]
  end

  @boundscheck begin
    inds .+ offset ⊆ 1:mbc ||
      throw_RowRange_DimensionMismatch(mbc, nbc, inds)
  end

  lw = length(wy.work)

  j_first = offset + 1
  j_last = offset + m_bc0

  k_first = first_inband_index(bc, j_first, :)
  k_last = last_inband_index(bc, j_last, :)
  n_bc0 = k_last - k_first + 1
  bc_els = band_elements(bc)

  @boundscheck begin
    lw >= n_bc0 * num_hs ||
      throw_WorkSizeError(mbc, nbc, n_bc0 * num_hs, length(wy.work))
    check_bc_storage_bounds(bc, j_first, k_last)
    check_bc_storage_bounds(bc, j_last, k_first)
  end

  @inbounds begin
    bulge_upper!(bc, j_first, k_last)
    bulge_lower!(bc, j_last, k_first)
    @views begin
      work = reshape(wy.work[1:n_bc0*num_hs], num_hs, n_bc0)
      W = wy.W[inds, 1:num_hs, k]
      Y = wy.Y[inds, 1:num_hs, k]
    end
    work .= zero(E)
    # Form work[l,kk] = W[:,l]ᴴ * bc[:,kk].
    for l∈1:num_hs
      for kk ∈ k_first:k_last
        x = zero(E)
        storage_offs = storage_offset(bc, kk)
        jrange = (offset .+ (1:m_bc0)) ∩ inband_index_range(bc, :, kk)
        @simd for j ∈ jrange
          work[l,kk-k_first+1] += conj(W[j - offset,l]) * bc_els[j - storage_offs, kk]
        end
      end
    end
    # Subtract  Y * work from bc[j_first:j_last,k_first:k_last].
    for l∈1:num_hs
      for kk∈k_first:k_last
        storage_offs = storage_offset(bc, kk)
        jrange = (offset .+ (1:m_bc0)) ∩ inband_index_range(bc, :, kk)
        @simd for j ∈ jrange
          bc_els[j - storage_offs, kk] -= Y[j-offset,l] * work[l,kk-k_first+1]
        end
      end
    end
  end
  nothing
end

end # module
