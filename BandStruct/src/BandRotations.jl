module BandRotations

using LinearAlgebra
using Base: @propagate_inbounds

using Rotations.Givens

using ..BandColumnMatrices
using ..BlockedBandColumnMatrices
using InPlace

@propagate_inbounds function InPlace.apply_right!(
  ::Type{Band{E}},
  bc::AbstractBandColumn{S, E},
  r::AdjRot{R,E};
  offset = 0
) where {S,R<:AbstractFloat,E<:Union{R,Complex{R}}}
  c = r.c
  s = r.s
  k = r.j + offset
  jrange = hull(inband_index_range(bc, :, k), inband_index_range(bc, :, k + 1))
  bulge!(bc, :, k:(k+1))
  for j ∈ jrange
    tmp = bc[j, k]
    setindex_no_bulge!(bc, tmp * c - bc[j, k + 1] * s, j, k)
    setindex_no_bulge!(bc, tmp * conj(s) + bc[j, k + 1] * c, j, k + 1)
  end
  nothing
end

@propagate_inbounds function InPlace.apply_right_inv!(
  ::Type{Band{E}},
  bc::AbstractBandColumn{S, E},
  r::AdjRot{R,E};
  offset = 0
) where {S, R<:AbstractFloat,E<:Union{R,Complex{R}}}
  c = r.c
  s = r.s
  k = r.j + offset
  jrange = hull(inband_index_range(bc, :, k), inband_index_range(bc, :, k + 1))
  bulge!(bc, :, k:(k+1))
  for j ∈ jrange
    tmp = bc[j, k]
    setindex_no_bulge!(bc, tmp * c + bc[j, k + 1] * s, j, k)
    setindex_no_bulge!(bc, -tmp * conj(s) + bc[j, k + 1] * c, j, k + 1)
  end
  nothing
end

@propagate_inbounds function InPlace.apply_left!(
  ::Type{Band{E}},
  r::AdjRot{R,E},
  bc::AbstractBandColumn{S, E};
  offset = 0
) where {S, R<:AbstractFloat,E<:Union{R,Complex{R}}}
  c = r.c
  s = r.s
  j = r.j + offset
  krange = hull(inband_index_range(bc, j, :), inband_index_range(bc, j + 1, :))
  bulge!(bc, j:(j+1), :)
  for k ∈ krange
    tmp = bc[j, k]
    setindex_no_bulge!(bc, c * tmp + bc[j + 1, k] * conj(s), j, k)
    setindex_no_bulge!(bc, -s * tmp + c * bc[j + 1, k], j+1, k)
  end
  nothing
end

@propagate_inbounds function InPlace.apply_left_inv!(
  ::Type{Band{E}},
  r::AdjRot{R,E},
  bc::AbstractBandColumn{S, E};
  offset = 0
) where {S, R<:AbstractFloat,E<:Union{R,Complex{R}}}

  c = r.c
  s = r.s
  j = r.j + offset
  krange = hull(inband_index_range(bc, j, :), inband_index_range(bc, j+1, :))
  bulge!(bc, j:(j+1), :)
  for k ∈ krange
    tmp = bc[j, k]
    setindex_no_bulge!(bc, c * tmp - conj(s) * bc[j + 1, k], j, k)
    setindex_no_bulge!(bc, s * tmp + c * bc[j + 1, k], j + 1, k)
  end
  nothing
end

end # module
