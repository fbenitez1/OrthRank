module Banded
if isdefined(@__MODULE__, :LanguageServer)
  include("Givens.jl")
  using .Givens
else
  using Rotations.Givens
end

using LinearAlgebra
using BandStruct
using BandStruct.BandColumnMatrices
using BandStruct.LeadingBandColumnMatrices
import InPlace

import Base: @propagate_inbounds

@propagate_inbounds @inline function InPlace.:⊛(
  bc::AbstractBandColumn{E,AE,AI},
  r::AdjRot{R,E},
) where {R<:AbstractFloat,E<:Union{R,Complex{R}},AE,AI}
  c = r.c
  s = r.s
  k = r.j
  jrange = hull(els_range(bc, :, k), els_range(bc, :, k + 1))
  extend_band!(bc, :, k)
  for j ∈ jrange
    tmp = bc[j, k]
    setindex_noext!(bc, tmp * c - bc[j, k + 1] * s, j, k)
    setindex_noext!(bc, tmp * conj(s) + bc[j, k + 1] * c, j, k + 1)
  end
  nothing
end

@propagate_inbounds @inline function InPlace.:⊘(
  bc::AbstractBandColumn{E,AE,AI},
  r::AdjRot{R,E},
) where {R<:AbstractFloat,E<:Union{R,Complex{R}},AE,AI}
  c = r.c
  s = r.s
  k = r.j
  jrange = hull(els_range(bc, :, k), els_range(bc, :, k + 1))
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
  bc::AbstractBandColumn{E,AE,AI},
) where {R<:AbstractFloat,E<:Union{R,Complex{R}},AE,AI}
  c = r.c
  s = r.s
  j = r.j
  krange = hull(els_range(bc, j, :), els_range(bc, j + 1, :))
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
  bc::AbstractBandColumn{E,AE,AI},
) where {R<:AbstractFloat,E<:Union{R,Complex{R}},AE,AI}
  c = r.c
  s = r.s
  j = r.j
  krange = hull(els_range(bc, j, :), els_range(bc, j+1, :))
  extend_band!(bc, j, :)
  for k ∈ krange
    tmp = bc[j, k]
    setindex_noext!(bc, c * tmp - conj(s) * bc[j + 1, k], j, k)
    setindex_noext!(bc, s * tmp + c * bc[j + 1, k], j + 1, k)
  end
  nothing
end

end # module
