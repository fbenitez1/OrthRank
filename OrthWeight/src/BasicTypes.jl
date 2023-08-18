module BasicTypes

using BandStruct
using Base: something

export OrthWeightDecomp,
  Consts,
  AbstractCompressibleData,
  Step,
  SpanStep,
  NullStep,
  UpLow,
  Upper,
  Lower,
  TransformSizes,
  Num_hs,
  NumRots,
  Offsets,
  filter_compressed,
  LowerCompressed,
  UpperCompressed,
  getindex_or_scalar,
  maybe_zero,
  expand_or_set!,
  maybe_set!,
  to_block_data_index_list

function to_block_data_index_list(
  blocks::Union{
    AbstractVector{<:AbstractBlockData},
    IndexList{<:AbstractBlockData},
  },
  B;
  max_length::Union{Int,Nothing} = nothing,
)
  if blocks isa IndexList
    blocks_B = IndexList([
      let (; mb, nb) = blocks[li]
        B(mb = mb, nb = nb)
      end for li in blocks
        ], max_length = something(max_length, blocks.max_length))
  else
    blocks_B = IndexList([
      let (; mb, nb) = bd
        B(mb = mb, nb = nb)
      end for bd in blocks
        ], max_length = something(max_length, length(blocks)))
  end
  return blocks_B
end


# Constant array of given size.
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

abstract type OrthWeightDecomp end

abstract type AbstractCompressibleData <: AbstractBlockData end


abstract type Step end

"""
      struct SpanStep end

  A type marking a WYWeight with steps that are large relative to
  the off-diagonal ranks so that transformations are computed to introduce
  zeros directly into a basis for the span of block columns or rows.
"""
struct SpanStep <: Step end
Base.iterate(t::SpanStep) = (t, nothing)
Base.iterate(::SpanStep, ::Any) = nothing

"""
        struct NullStep end

  A type marking a WYWeight with steps that are small relative to
  the off-diagonal ranks so that transformations are computed to
  introduce zeros directly into a basis for the null space of a block.
"""
struct NullStep <: Step end
Base.iterate(t::NullStep) = (t, nothing)
Base.iterate(::NullStep, ::Any) = nothing

abstract type UpLow end

struct Upper <: UpLow end
Base.iterate(t::Upper) = (t, nothing)
Base.iterate(::Upper, ::Any) = nothing

struct Lower <: UpLow end
Base.iterate(t::Lower) = (t, nothing)
Base.iterate(::Lower, ::Any) = nothing

struct TransformSizes end
Base.iterate(t::TransformSizes) = (t, nothing)
Base.iterate(::TransformSizes, ::Any) = nothing

struct Num_hs end
Base.iterate(t::Num_hs) = (t, nothing)
Base.iterate(::Num_hs, ::Any) = nothing

struct NumRots end
Base.iterate(t::NumRots) = (t, nothing)
Base.iterate(::NumRots, ::Any) = nothing


struct Offsets end
Base.iterate(t::Offsets) = (t, nothing)
Base.iterate(::Offsets, ::Any) = nothing


# Iterators over blocks that are compressed.

function filter_compressed(
  l::IndexList{B},
) where {B<:AbstractCompressibleData}
  Iterators.filter(ind -> l[ind].compressed, l)
end

function filter_compressed(
  l::Iterators.Reverse{IndexList{B}},
) where {B<:AbstractCompressibleData}
  Iterators.filter(ind -> l.itr[ind].compressed, l)
end

struct LowerCompressed{D}
  blocks::D
end

struct UpperCompressed{D}
  blocks::D
end

function Base.length(lc::LowerCompressed{<:OrthWeightDecomp})
  len = 0
  for _ ∈ lc
    len += 1
  end
  len
end

Base.length(rlc::Iterators.Reverse{<:LowerCompressed{<:OrthWeightDecomp}}) =
  length(rlc.itr)

function Base.iterate(lc::LowerCompressed{<:OrthWeightDecomp}, l::Int)
  while (l <= lc.blocks.b.num_blocks) && (!lc.blocks.lower_compressed[l])
    l += 1
  end
  l > lc.blocks.b.num_blocks ? nothing : (l, l + 1)
end

Base.iterate(lc::LowerCompressed{<:OrthWeightDecomp}) = Base.iterate(lc, 1)

function Base.iterate(
  rlc::Iterators.Reverse{<:LowerCompressed{<:OrthWeightDecomp}},
  l::Int,
)
  while (l >= 1) && (!rlc.itr.decomp.lower_compressed[l])
    l -= 1
  end
  l < 1 ? nothing : (l, l - 1)
end

Base.iterate(rlc::Iterators.Reverse{<:LowerCompressed{<:OrthWeightDecomp}}) =
  Base.iterate(rlc, rlc.itr.decomp.b.num_blocks)

function Base.length(uc::UpperCompressed{<:OrthWeightDecomp})
  len = 0
  for i ∈ uc
    len += 1
  end
  len
end

function Base.length(
  ruc::Iterators.Reverse{<:UpperCompressed{<:OrthWeightDecomp}},
)
  length(ruc.itr)
end

function Base.iterate(uc::UpperCompressed{<:OrthWeightDecomp}, l::Int)
  while (l <= uc.blocks.b.num_blocks) && (!uc.blocks.upper_compressed[l])
    l += 1
  end
  l > uc.blocks.b.num_blocks ? nothing : (l, l + 1)
end

Base.iterate(uc::UpperCompressed{<:OrthWeightDecomp}) = Base.iterate(uc, 1)

function Base.iterate(
  ruc::Iterators.Reverse{<:UpperCompressed{<:OrthWeightDecomp}},
  l::Int,
)
  while (l >= 1) && (!ruc.itr.decomp.upper_compressed[l])
    l -= 1
  end
  l < 1 ? nothing : (l, l - 1)
end

Base.iterate(ruc::Iterators.Reverse{<:UpperCompressed{<:OrthWeightDecomp}}) =
  Base.iterate(ruc, ruc.itr.decomp.b.num_blocks)

# Set or increase xref[k] to y.
expand_or_set!(b, xref, k, y) = xref[k] = b ? max(xref[k], y) : y

# Increase xref[] to y as needed, to get a running maximum.
expand_or_set!(_, xref::Ref{Int}, _, y) = xref[] = max(xref[], y)

expand_or_set!(_, ::Nothing, _, y) = return y

# Set values if given a vector.
maybe_set!(a::AbstractVector, k, y) = a[k] = y
maybe_set!(::Nothing, _, y) = return y


# Initialize with zero if not expanding.
maybe_zero(expand::Bool, r::Ref{Int}) = expand || (r[] = 0)
maybe_zero(expand::Bool, r::Vector{Int}) = expand || (r .= 0)
maybe_zero(::Bool, ::Nothing) = nothing
maybe_zero(exp::Bool, args...) = (x -> maybe_zero(exp, x)).(args)

# Pretend to index into a scalar, which is treated as a constant that
# doesn't depend on the index.
getindex_or_scalar(a, _) = a
getindex_or_scalar(a::AbstractArray, k) = a[k]


end
