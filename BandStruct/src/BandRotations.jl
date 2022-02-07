module BandRotations

using LinearAlgebra
using Base: @propagate_inbounds

using Rotations.Givens

using ..BandColumnMatrices
using ..BlockedBandColumnMatrices
using InPlace

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
  for j ∈ jrange
    tmp = bc[j, k]
    setindex_no_bulge!(bc, tmp * c - bc[j, k + 1] * conj(s), j, k)
    setindex_no_bulge!(bc, tmp * s + bc[j, k + 1] * c, j, k + 1)
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
  for j ∈ jrange
    tmp = bc[j, k]
    setindex_no_bulge!(bc, tmp * c + bc[j, k + 1] * conj(s), j, k)
    setindex_no_bulge!(bc, -tmp * s + bc[j, k + 1] * c, j, k + 1)
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
  for k ∈ krange
    tmp = bc[j, k]
    setindex_no_bulge!(bc, c * tmp + bc[j + 1, k] * s, j, k)
    setindex_no_bulge!(bc, -conj(s) * tmp + c * bc[j + 1, k], j+1, k)
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
  for k ∈ krange
    tmp = bc[j, k]
    setindex_no_bulge!(bc, c * tmp - s * bc[j + 1, k], j, k)
    setindex_no_bulge!(bc, conj(s) * tmp + c * bc[j + 1, k], j + 1, k)
  end
  nothing
end

end # module
