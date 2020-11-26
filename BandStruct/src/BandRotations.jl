module BandRotations

using LinearAlgebra
using Base: @propagate_inbounds

using Rotations.Givens

using BandStruct
using BandStruct.BandColumnMatrices
using BandStruct.LeadingBandColumnMatrices
using InPlace



@propagate_inbounds @inline function InPlace.:⊛(
  bc::AbstractBandColumn{E},
  r::AdjRot{R,E},
) where {R<:AbstractFloat,E<:Union{R,Complex{R}}}
  c = r.c
  s = r.s
  k = r.j
  jrange = hull(inband_els_range(bc, :, k), inband_els_range(bc, :, k + 1))
  extend_band!(bc, :, k)
  for j ∈ jrange
    tmp = bc[j, k]
    setindex_noext!(bc, tmp * c - bc[j, k + 1] * s, j, k)
    setindex_noext!(bc, tmp * conj(s) + bc[j, k + 1] * c, j, k + 1)
  end
  nothing
end

@propagate_inbounds @inline function InPlace.:⊘(
  bc::AbstractBandColumn{E},
  r::AdjRot{R,E},
) where {R<:AbstractFloat,E<:Union{R,Complex{R}}}
  c = r.c
  s = r.s
  k = r.j
  jrange = hull(inband_els_range(bc, :, k), inband_els_range(bc, :, k + 1))
  extend_band!(bc, :, k)
  for j ∈ jrange
    tmp = bc[j, k]
    setindex_noext!(bc, tmp * c + bc[j, k + 1] * s, j, k)
    setindex_noext!(bc, -tmp * conj(s) + bc[j, k + 1] * c, j, k + 1)
  end
  nothing
end

@propagate_inbounds @inline function InPlace.:⊛(
  r::AdjRot{R,E},
  bc::AbstractBandColumn{E},
) where {R<:AbstractFloat,E<:Union{R,Complex{R}}}
  c = r.c
  s = r.s
  j = r.j
  krange = hull(inband_els_range(bc, j, :), inband_els_range(bc, j + 1, :))
  extend_band!(bc, j, :)
  for k ∈ krange
    tmp = bc[j, k]
    setindex_noext!(bc, c * tmp + bc[j + 1, k] * conj(s), j, k)
    setindex_noext!(bc, -s * tmp + c * bc[j + 1, k], j+1, k)
  end
  nothing
end

@propagate_inbounds @inline function InPlace.:⊘(
  r::AdjRot{R,E},
  bc::AbstractBandColumn{E},
) where {R<:AbstractFloat,E<:Union{R,Complex{R}}}
  c = r.c
  s = r.s
  j = r.j
  krange = hull(inband_els_range(bc, j, :), inband_els_range(bc, j+1, :))
  extend_band!(bc, j, :)
  for k ∈ krange
    tmp = bc[j, k]
    setindex_noext!(bc, c * tmp - conj(s) * bc[j + 1, k], j, k)
    setindex_noext!(bc, s * tmp + c * bc[j + 1, k], j + 1, k)
  end
  nothing
end

end # module
