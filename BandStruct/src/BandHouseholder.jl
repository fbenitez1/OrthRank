module BandHouseholder

using LinearAlgebra
using Base: @propagate_inbounds
using Printf

using Householder.Compute

using ..BandColumnMatrices
using ..LeadingBandColumnMatrices
using InPlace

@inline function Compute.householder(
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

@inline function Compute.householder(
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
    ((1:lv) .+ offs) ⊆ inband_index_range(bc, j, ks) ||
      throw(SubrowIndicesNotInband(j, ks))
  end

  bc_els = band_elements(bc)

  @inbounds for k ∈ 1:lv
    storage_offs = storage_offset(bc, k + offs)
    v[j] = bc_els[j - storage_offs, k + offs]
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
    work[1:m_bc] .= zero(E)
    # Accumulate w = bc * v in work array by a linear combination of
    # columns of bc.
    for k ∈ 1:m
      j0 = first_inband_index(bc, :, k)
      j1 = last_inband_index(bc, :, k)
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
    bulge_upper!(bc, j_first, k_last)
    bulge_lower!(bc, j_last, k_first)
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

  @inbounds for k ∈ k_first:k_last
    x = zero(E)
    storage_offs = storage_offset(bc, k)
    jrange = (offs .+ (1:m)) ∩ inband_index_range(bc, :, k)
    # Form x = vᴴ * bc[:,k].
    @simd for j ∈ jrange
      x = x + conj(v[j]) * bc_els[j - storage_offs, k]
    end
    # Subtract v * x from bc[:,k].
    x = β * x
    @simd for j ∈ 1:m
      bc_els[offs + j - storage_offs, k] -= v[j] * x
    end
    bulge_upper!(bc, j_first, k_last)
    bulge_lower!(bc, j_last, k_first)
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
    work[1:m_bc] .= zero(E)
    # Accumulate w = bc * v in work array by a linear combination of
    # columns of bc.
    for k ∈ 1:m
      j0 = first_inband_index(bc, :, k)
      j1 = last_inband_index(bc, :, k)
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
    bulge_upper!(bc, j_first, k_last)
    bulge_lower!(bc, j_last, k_first)
  end
  nothing
end

@inline function InPlace.:⊘(
  h::HouseholderTrans{E},
  bc::AbstractBandColumn{E,S},
) where {E<:Number,S}

  m = h.size
  v = reshape(h.v, m, 1)
  offs = h.offs
  β̄ = h.β

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

  @inbounds for k ∈ k_first:k_last
    x = zero(E)
    storage_offs = storage_offset(bc, k)
    jrange = (offs .+ (1:m)) ∩ inband_index_range(bc, :, k)
    # Form x = vᴴ * bc[:,k].
    @simd for j ∈ jrange
      x = x + conj(v[j]) * bc_els[j - storage_offs, k]
    end
    # Subtract v * x from bc[:,k].
    x = β̄ * x
    @simd for j ∈ 1:m
      bc_els[offs + j - storage_offs, k] -= v[j] * x
    end
    bulge_upper!(bc, j_first, k_last)
    bulge_lower!(bc, j_last, k_first)
  end
  nothing
end

end # module

