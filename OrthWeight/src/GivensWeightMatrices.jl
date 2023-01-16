module GivensWeightMatrices

using BandStruct
using BandStruct.BlockedBandColumnMatrices
using BandStruct.BandwidthInit
using OrthWeight.BasicTypes
using InPlace: apply!, apply_inv!
using Rotations
using LinearAlgebra
using Random

export AbstractGivensWeight,
  GivensWeight,
  get_max_Δn,
  get_max_Δm,
  max_num_rots,
  Consts

struct Consts{T} <: AbstractVector{T}
  length::Int
  value::T
end

Base.size(C::Consts) = (C.length,)

function Base.getindex(C::Consts, i::Int)
  if i ∈ axes(C, 1)
    C.value
  else
    throw(BoundsError(C, i))
  end
end

function Base.setindex!(C::Consts{T}, i::Int, x::T) where {T}
  if i ∈ axes(C, 1)
    C.value = x
  else
    throw(BoundsError(C, i))
  end
end

Base.IndexStyle(::Type{Consts{T}}) where T = IndexLinear()


abstract type AbstractGivensWeight <: OrthWeightDecomp end

struct GivensWeight{LR,B,RR} <: AbstractGivensWeight
  upper_decomp::Base.RefValue{Decomp}
  lower_decomp::Base.RefValue{Decomp}
  leftRots::LR
  b::B
  rightRots::RR
  lower_ranks::Vector{Int}
  lower_compressed::Vector{Bool}
  upper_ranks::Vector{Int}
  upper_compressed::Vector{Bool}
end

function GivensWeight(
  ::Type{E},
  m::Int,
  n::Int;
  upper_decomp::Decomp,
  lower_decomp::Decomp,
  upper_ranks::AbstractVector{Int},
  lower_ranks::AbstractVector{Int},
  upper_blocks::AbstractVector{Int},
  lower_blocks::AbstractVector{Int},
) where E

end

function get_max_Δm(m::Int, blocks::AbstractMatrix{<:Int})
  max_Δm = 0
  for k ∈ axes(blocks, 2)[2:end]
    Δm = blocks[1, k] - blocks[1, k - 1]
    max_Δm = Δm > max_Δm ? Δm : max_Δm
  end
  return max(max_Δm, blocks[1, begin], m - blocks[1, end])
end

function get_max_Δn(n::Int, blocks::AbstractMatrix{<:Int})
  max_Δn = 0
  for k ∈ axes(blocks, 2)[2:end]
    Δn = blocks[2, k] - blocks[2, k - 1]
    max_Δn = Δn > max_Δn ? Δn : max_Δn
  end
  return max(max_Δn, blocks[2, begin], n - blocks[2, end])
end


function max_num_rots(
  m::Int,
  n::Int,
  rmax::Int,
  blocks::AbstractMatrix{<:Int},
)

  Δm = get_max_Δm(m, blocks)
  Δn = get_max_Δn(n, blocks)
  
  return max(
    rmax * (Δm + 1) + ((rmax - 1) * rmax) ÷ 2,
    rmax * (Δn + 1) + ((rmax - 1) * rmax) ÷ 2,
  )
end

end
