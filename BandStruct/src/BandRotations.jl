module BandRotations

using LinearAlgebra
using Base: @propagate_inbounds

using Rotations.Givens

using ..BandColumnMatrices
using ..LeadingBandColumnMatrices
using InPlace

@propagate_inbounds @inline function InPlace.:⊛(
  bc::AbstractBandColumn{S, E},
  r::AdjRot{R,E},
) where {S,R<:AbstractFloat,E<:Union{R,Complex{R}}}
  c = r.c
  s = r.s
  k = r.j
  jrange = hull(inband_index_range(bc, :, k), inband_index_range(bc, :, k + 1))
  bulge!(bc, :, k:(k+1))
  for j ∈ jrange
    tmp = bc[j, k]
    setindex_no_bulge!(bc, tmp * c - bc[j, k + 1] * s, j, k)
    setindex_no_bulge!(bc, tmp * conj(s) + bc[j, k + 1] * c, j, k + 1)
  end
  nothing
end

@propagate_inbounds @inline function InPlace.:⊘(
  bc::AbstractBandColumn{S, E},
  r::AdjRot{R,E},
) where {S, R<:AbstractFloat,E<:Union{R,Complex{R}}}
  c = r.c
  s = r.s
  k = r.j
  jrange = hull(inband_index_range(bc, :, k), inband_index_range(bc, :, k + 1))
  bulge!(bc, :, k:(k+1))
  for j ∈ jrange
    tmp = bc[j, k]
    setindex_no_bulge!(bc, tmp * c + bc[j, k + 1] * s, j, k)
    setindex_no_bulge!(bc, -tmp * conj(s) + bc[j, k + 1] * c, j, k + 1)
  end
  nothing
end

@propagate_inbounds @inline function InPlace.:⊛(
  r::AdjRot{R,E},
  bc::AbstractBandColumn{S, E},
) where {S, R<:AbstractFloat,E<:Union{R,Complex{R}}}
  c = r.c
  s = r.s
  j = r.j
  krange = hull(inband_index_range(bc, j, :), inband_index_range(bc, j + 1, :))
  bulge!(bc, j:(j+1), :)
  for k ∈ krange
    tmp = bc[j, k]
    setindex_no_bulge!(bc, c * tmp + bc[j + 1, k] * conj(s), j, k)
    setindex_no_bulge!(bc, -s * tmp + c * bc[j + 1, k], j+1, k)
  end
  nothing
end

@propagate_inbounds @inline function InPlace.:⊘(
  r::AdjRot{R,E},
  bc::AbstractBandColumn{S, E},
) where {S, R<:AbstractFloat,E<:Union{R,Complex{R}}}
  c = r.c
  s = r.s
  j = r.j
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
