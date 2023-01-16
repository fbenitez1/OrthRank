module BasicTypes

using BandStruct

export OrthWeightDecomp,
  Step,
  SpanStep,
  NullStep,
  Left,
  Right,
  Sizes,
  Num_hs,
  Offsets,
  LowerCompressed,
  UpperCompressed

abstract type OrthWeightDecomp end

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

abstract type Side end

struct Left <: Side end
Base.iterate(t::Left) = (t, nothing)
Base.iterate(::Left, ::Any) = nothing

struct Right <: Side end
Base.iterate(t::Right) = (t, nothing)
Base.iterate(::Right, ::Any) = nothing

struct Sizes end
Base.iterate(t::Sizes) = (t, nothing)
Base.iterate(::Sizes, ::Any) = nothing

struct Num_hs end
Base.iterate(t::Num_hs) = (t, nothing)
Base.iterate(::Num_hs, ::Any) = nothing


struct Offsets end
Base.iterate(t::Offsets) = (t, nothing)
Base.iterate(::Offsets, ::Any) = nothing


# Iterators over blocks.

struct LowerCompressed{D}
  decomp::D
end

struct UpperCompressed{D}
  decomp::D
end

@inline function Base.length(
  rlc::Iterators.Reverse{<:LowerCompressed{<:OrthWeightDecomp}},
)
  length(rlc.itr)
end

@inline function Base.iterate(lc::LowerCompressed{<:OrthWeightDecomp})
  l = 1
  while (l <= lc.decomp.b.num_blocks) && (!lc.decomp.lower_compressed[l])
    l += 1
  end
  l > lc.decomp.b.num_blocks ? nothing : (l, l + 1)
end

@inline function Base.iterate(
  rlc::Iterators.Reverse{<:LowerCompressed{<:OrthWeightDecomp}},
)
  lc = rlc.itr
  l = lc.decomp.b.num_blocks
  while (l >= 1) && (!lc.decomp.lower_compressed[l])
    l -= 1
  end
  l < 1 ? nothing : (l, l - 1)
end

@inline function Base.iterate(lc::LowerCompressed{<:OrthWeightDecomp}, l::Int)
  while (l <= lc.decomp.b.num_blocks) && (!lc.decomp.lower_compressed[l])
    l += 1
  end
  l > lc.decomp.b.num_blocks ? nothing : (l, l + 1)
end

@inline function Base.iterate(
  rlc::Iterators.Reverse{<:LowerCompressed{<:OrthWeightDecomp}},
  l::Int,
)
  while (l >= 1) && (!rlc.itr.decomp.lower_compressed[l])
    l -= 1
  end
  l < 1 ? nothing : (l, l - 1)
end

@inline function Base.length(lc::UpperCompressed{<:OrthWeightDecomp})
  l = 1
  len = 0
  while l <= lc.decomp.b.num_blocks
    lc.decomp.upper_compressed[l] && (len += 1)
    l += 1
  end
  len
end

@inline function Base.length(
  rlc::Iterators.Reverse{<:UpperCompressed{<:OrthWeightDecomp}},
)
  length(rlc.itr)
end

@inline function Base.iterate(lc::UpperCompressed{<:OrthWeightDecomp})
  l = 1
  while (l <= lc.decomp.b.num_blocks) && (!lc.decomp.upper_compressed[l])
    l += 1
  end
  l > lc.decomp.b.num_blocks ? nothing : (l, l + 1)
end

@inline function Base.iterate(
  rlc::Iterators.Reverse{<:UpperCompressed{<:OrthWeightDecomp}},
)
  lc = rlc.itr
  l = lc.decomp.b.num_blocks
  while (l >= 1) && (!lc.decomp.upper_compressed[l])
    l -= 1
  end
  l < 1 ? nothing : (l, l - 1)
end

@inline function Base.iterate(lc::UpperCompressed{<:OrthWeightDecomp}, l::Int)
  while (l <= lc.decomp.b.num_blocks) && (!lc.decomp.upper_compressed[l])
    l += 1
  end
  l > lc.decomp.b.num_blocks ? nothing : (l, l + 1)
end

@inline function Base.iterate(
  rlc::Iterators.Reverse{<:UpperCompressed{<:OrthWeightDecomp}},
  l::Int,
)
  while (l >= 1) && (!rlc.itr.decomp.upper_compressed[l])
    l -= 1
  end
  l < 1 ? nothing : (l, l - 1)
end

end
