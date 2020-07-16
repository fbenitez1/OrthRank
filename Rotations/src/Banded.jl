module Banded
if isdefined(@__MODULE__, :LanguageServer)
  include("Givens.jl")
  using .Givens
  import .Givens: ⊛
else
  using Rotations.Givens
  import Rotations.Givens: ⊛
end

using LinearAlgebra
using BandStruct
using BandStruct.BandColumnMatrices
using BandStruct.LeadingBandColumnMatrices

import Base: @propagate_inbounds

export ⊛, ⊘, rgivens, rgivens1, lgivens, lgivens1, Rot, AdjRot, unsafe_mult


unsafe_mult(a,b) = @inbounds a ⊛ b

MultBCRotReal = (BandColumn{Float64, Array{Float64,2}, Array{Int,2}}, AdjRot{Float64, Float64})
@propagate_inbounds @inline function ⊛(
  bc::AbstractBandColumn{E,AE,AI},
  r::AdjRot{R,E},
) where {R<:AbstractFloat,E<:Union{R,Complex{R}},AE, AI}
  begin
  c = r.c
  s = r.s
  (_, n) = size(bc)
  k = r.j
  jrange=hull(els_range(bc,k), els_range(bc,k+1))
  for j ∈ jrange
    tmp = bc[j, k]
    bc[j, k] = tmp * c - bc[j, k + 1] * s
    bc[j, k + 1] = tmp * conj(s) + bc[j, k + 1] * c
  end
  nothing
  end
end

end # module
