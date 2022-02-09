module BandRotations

using LinearAlgebra
using Base: @propagate_inbounds
using LoopVectorization

using Rotations.Givens

using ..BandColumnMatrices
using ..BlockedBandColumnMatrices
using InPlace

macro real_tturbo(t, ex)
  return esc(quote
               if $t <: Real
                 @tturbo $ex
               else
                 @inbounds @fastmath $ex
               end
             end)
end

@propagate_inbounds function InPlace.apply_right!(
  ::Type{Band{E}},
  bc::AbstractBandColumn{S,E},
  r::AdjRot{TS,TC};
  offset = 0
) where {S,TS<:Number,TC<:Number,E<:Number}

  check_inplace_rotation_types(TS, TC, E)
  j, _ = get_inds(r)
  k = j + offset
  c = r.c
  s = r.s
  jrange = hull(inband_index_range(bc, :, k), inband_index_range(bc, :, k + 1))
  bulge!(bc, :, k:(k+1))
  bc_els = band_elements(bc)
  col_offs = col_offset(bc)
  st_offs_k = storage_offset(bc,k)
  st_offs_k1 = storage_offset(bc,k+1)
  @real_tturbo E for j ∈ jrange
    b0 = bc_els[j - st_offs_k, k + col_offs]
    b1 = bc_els[j - st_offs_k1, k + 1 + col_offs]
    bc_els[j - st_offs_k, k + col_offs] = b0 * c - b1 * conj(s)
    bc_els[j - st_offs_k1, k + 1 + col_offs] = b0 * s + b1 * conj(c)
  end
  nothing
end

@propagate_inbounds function InPlace.apply_right_inv!(
  ::Type{Band{E}},
  bc::AbstractBandColumn{S,E},
  r::AdjRot{TS,TC};
  offset = 0
) where {S,TS<:Number,TC<:Number,E<:Number}
  
  check_inplace_rotation_types(TS, TC, E)
  j, _ = get_inds(r)
  k = j + offset
  c = r.c
  s = r.s
  jrange = hull(inband_index_range(bc, :, k), inband_index_range(bc, :, k + 1))
  bulge!(bc, :, k:(k+1))
  bc_els = band_elements(bc)
  col_offs = col_offset(bc)
  st_offs_k = storage_offset(bc,k)
  st_offs_k1 = storage_offset(bc,k+1)
  @real_tturbo E for j ∈ jrange
    b0 = bc_els[j - st_offs_k, k + col_offs]
    b1 = bc_els[j - st_offs_k1, k + 1 + col_offs]
    bc_els[j - st_offs_k, k + col_offs] = b0 * conj(c) + b1 * conj(s)
    bc_els[j - st_offs_k1, k + 1 + col_offs] = -b0 * s + b1 * c
  end
  nothing
end

@propagate_inbounds function InPlace.apply_left!(
  ::Type{Band{E}},
  r::AdjRot{TS,TC},
  bc::AbstractBandColumn{S,E};
  offset = 0
) where {S,TS<:Number,TC<:Number,E<:Number}

  check_inplace_rotation_types(TS, TC, E)
  j, _ = get_inds(r)
  j = j + offset
  c = r.c
  s = r.s
  krange = hull(inband_index_range(bc, j, :), inband_index_range(bc, j + 1, :))
  bulge!(bc, j:(j+1), :)

  bc_els = band_elements(bc)
  col_offs = col_offset(bc)
  row_offs = row_offset(bc)
  upper_bw_max = bc.upper_bw_max
  cols_first_last = bc.cols_first_last
  # @real_tturbo E for k ∈ krange
  @inbounds @fastmath @simd for k ∈ krange
    st_offs_k = cols_first_last[3, k + col_offs] - row_offs - upper_bw_max
    b0 = bc_els[j - st_offs_k, k + col_offs]
    b1 = bc_els[j + 1 - st_offs_k, k + col_offs]
    bc_els[j - st_offs_k, k + col_offs] = c*b0 + s*b1
    bc_els[j + 1 - st_offs_k, k + col_offs] = -conj(s)*b0 + conj(c)*b1
  end
  nothing
end

@propagate_inbounds function InPlace.apply_left_inv!(
  ::Type{Band{E}},
  r::AdjRot{TS,TC},
  bc::AbstractBandColumn{S, E};
  offset = 0
) where {S,TS<:Number,TC<:Number,E<:Number}

  check_inplace_rotation_types(TS, TC, E)
  j, _ = get_inds(r)
  j = j + offset
  c = r.c
  s = r.s
  krange = hull(inband_index_range(bc, j, :), inband_index_range(bc, j+1, :))
  bulge!(bc, j:(j+1), :)

  bc_els = band_elements(bc)
  col_offs = col_offset(bc)
  row_offs = row_offset(bc)
  upper_bw_max = bc.upper_bw_max
  cols_first_last = bc.cols_first_last
  # @real_tturbo E for k ∈ krange
  @inbounds @fastmath @simd for k ∈ krange
    st_offs_k = cols_first_last[3, k + col_offs] - row_offs - upper_bw_max
    b0 = bc_els[j - st_offs_k, k + col_offs]
    b1 = bc_els[j + 1 - st_offs_k, k + col_offs]
    bc_els[j - st_offs_k, k + col_offs] = conj(c)*b0 - s*b1
    bc_els[j + 1 - st_offs_k, k + col_offs] = conj(s)*b0 + c*b1
  end
  nothing
end

end # module
