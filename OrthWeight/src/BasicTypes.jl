module BasicTypes

using BandStruct

export OrthWeightDecomp,
  Step,
  SpanStep,
  NullStep,
  UpLow,
  Upper,
  Lower,
  Sizes,
  Num_hs,
  Offsets,
  LowerCompressed,
  UpperCompressed,
  getindex_or_scalar,
  maybe_zero,
  expand_or_set!

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

abstract type UpLow end

struct Upper <: UpLow end
Base.iterate(t::Upper) = (t, nothing)
Base.iterate(::Upper, ::Any) = nothing

struct Lower <: UpLow end
Base.iterate(t::Lower) = (t, nothing)
Base.iterate(::Lower, ::Any) = nothing

struct Sizes end
Base.iterate(t::Sizes) = (t, nothing)
Base.iterate(::Sizes, ::Any) = nothing

struct Num_hs end
Base.iterate(t::Num_hs) = (t, nothing)
Base.iterate(::Num_hs, ::Any) = nothing


struct Offsets end
Base.iterate(t::Offsets) = (t, nothing)
Base.iterate(::Offsets, ::Any) = nothing


# Iterators over blocks that are compressed.

struct LowerCompressed{D}
  decomp::D
end

struct UpperCompressed{D}
  decomp::D
end

function Base.length(lc::LowerCompressed{<:OrthWeightDecomp})
  len = 0
  for i ∈ lc
    len += 1
  end
  len
end

Base.length(rlc::Iterators.Reverse{<:LowerCompressed{<:OrthWeightDecomp}}) =
  length(rlc.itr)

function Base.iterate(lc::LowerCompressed{<:OrthWeightDecomp}, l::Int)
  while (l <= lc.decomp.b.num_blocks) && (!lc.decomp.lower_compressed[l])
    l += 1
  end
  l > lc.decomp.b.num_blocks ? nothing : (l, l + 1)
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
  while (l <= uc.decomp.b.num_blocks) && (!uc.decomp.upper_compressed[l])
    l += 1
  end
  l > uc.decomp.b.num_blocks ? nothing : (l, l + 1)
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
