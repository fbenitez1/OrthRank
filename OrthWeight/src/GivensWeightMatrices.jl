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
  Consts,
  set_givens_weight_transform_params!,
  get_givens_weight_transform_params,
  get_givens_weight_max_transform_params


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

# Givens-weight represenation.

abstract type AbstractGivensWeight <: OrthWeightDecomp end

"""
    GivensWeight{LR,B,UR}

A GivensWeight matrix with weight matrix of type `B` and lower and upper
rotations of type `LR` and `UR` respectively.

# Fields

  - `lower_decomp::Base.RefValue{Union{Nothing,Decomp}}`:
    Type of decomposition in the lower part.

  - `upper_decomp::Base.RefValue{Union{Nothing,Decomp}}`:
    Type of decomposition in the upper part.


  - `step::::Base.RefValue{Union{Nothing,NullStep,SpanStep}}`: Flag
    whether zeros are introduced from a span of columns/rows or from a
    null space.

  - `lowerRot::LR`: Lower rotations for a banded Givens weight decomposition.

  - `b::B`: Weight matrix.

  - `upperRot::UR`: Upper rotations for a banded Givens weight decomposition.

  - `upper_ranks::Vector{Int}`: Upper ranks

  - `lower_ranks::Vector{Int}`: Lower ranks
"""
struct GivensWeight{LR,B,RR} <: AbstractGivensWeight
  lower_decomp::Base.RefValue{Decomp}
  upper_decomp::Base.RefValue{Decomp}
  lowerRots::LR
  b::B
  upperRots::RR
  lower_rank_max::Int
  lower_ranks::Vector{Int}
  lower_compressed::Vector{Bool}
  upper_rank_max::Int
  upper_ranks::Vector{Int}
  upper_compressed::Vector{Bool}
end

"""
    GivensWeight(
      ::Type{E},
      m::Int,
      n::Int;
      upper_rank_max::Int,
      lower_rank_max::Int,
      upper_blocks::Array{Int,2},
      lower_blocks::Array{Int,2},
    ) where {E<:Number}

Generic `GivensWeight` with zero ranks but room for either a leading or
trailing decomposition with ranks up to `upper_rank_max` and
`lower_rank_max`.

"""
function GivensWeight(
  ::Type{E},
  m::Int,
  n::Int;
  lower_blocks::AbstractMatrix{Int},
  lower_rank_max::Int,
  upper_blocks::AbstractMatrix{Int},
  upper_rank_max::Int,
) where {E}

  num_blocks = size(upper_blocks, 2)

  lower_blocks = Matrix(lower_blocks)
  upper_blocks = Matrix(upper_blocks)

  bbc = BlockedBandColumn(
    E,
    m,
    n;
    lower_blocks = lower_blocks,
    lower_rank_max = lower_rank_max,
    upper_blocks = upper_blocks,
    upper_rank_max = upper_rank_max,
  )
  
end


"""
    get_max_Δm(m::Int, blocks::AbstractMatrix{<:Int})

Get maximum increment in block row size.
"""
function get_max_Δm(m::Int, blocks::AbstractMatrix{<:Int})
  max_Δm = 0
  for k ∈ axes(blocks, 2)[2:end]
    Δm = blocks[1, k] - blocks[1, k - 1]
    max_Δm = Δm > max_Δm ? Δm : max_Δm
  end
  return max(max_Δm, blocks[1, begin], m - blocks[1, end])
end

"""
    get_max_Δn(n::Int, blocks::AbstractMatrix{<:Int})

Get maximum increment in block column size.
"""
function get_max_Δn(n::Int, blocks::AbstractMatrix{<:Int})
  max_Δn = 0
  for k ∈ axes(blocks, 2)[2:end]
    Δn = blocks[2, k] - blocks[2, k - 1]
    max_Δn = Δn > max_Δn ? Δn : max_Δn
  end
  return max(max_Δn, blocks[2, begin], n - blocks[2, end])
end

"""
    function max_num_rots(
      m::Int,
      n::Int,
      rmax::Int,
      blocks::AbstractMatrix{<:Int},
    )

Get maximum number of rotations required for any block.
"""
function max_num_rots(m::Int, n::Int, rmax::Int, blocks::AbstractMatrix{<:Int})

  Δm = get_max_Δm(m, blocks)
  Δn = get_max_Δn(n, blocks)

  return max(
    rmax * (Δm + 1) + ((rmax - 1) * rmax) ÷ 2,
    rmax * (Δn + 1) + ((rmax - 1) * rmax) ÷ 2,
  )
end

function _forward_get_offset(range, trange)
  # assuming:
  # dᵣ = setdiffᵣ(range, old_range)
  # trange = dᵣ ∪ᵣ last(old_range, old_rank)
  if !isempty(trange)
    # offset to the columns/rows to be compressed in block lb
    first(trange) - 1
  elseif !isempty(range)
    # Since trange is empty, no new columns/rows in range b
    # AND (b-1 has rank bound zero OR b-1 has no columns/rows).  This
    # handles the case in which b-1 has columns/rows and b has the same
    # columns/rows, but no compression is required because the rank bound
    # of b-1 was zero.
    last(range) # skip past all the columns/rows in block b.
  else
    # block b and b-1 have no columns/rows.
    0
  end
end

function _backward_get_offset(n, range, trange)
  # assuming:
  # dᵣ = setdiffᵣ(range, old_range)
  # trange = dᵣ ∪ᵣ first(old_range, old_rank)
  if !isempty(trange)
    # offset to the columns to be compressed in block ub
    first(trange) - 1
  elseif !isempty(range)
    # Since trange is empty, no new columns/rows in range b
    # AND (b+1 has rank bound zero OR b+1 has no columns/rows).  This
    # handles the case in which b+1 has columns/rows and b has the same
    # columns/rows, but no compression is required because the rank bound
    # of b+1 was zero.
    # last(cols_ub)
    first(range) - 2 # first column/row after offset is before first(cols_ub)
  else
    # block b and b+1 have no columns/rows.  Offset through entire matrix.
    n
  end
end

"""
    set_givens_weight_transform_params!(
      side::Union{Left, Right},
      decomp::Union{Decomp, Tuple{Vararg{Decomp}}},
      step::Union{Step, Tuple{Vararg{Step}}},
      m::Int,
      n::Int;
      lower_blocks::Union{AbstractArray{Int,2}, Nothing} = nothing,
      lower_ranks::Union{AbstractVector{Int}, Nothing} = nothing,
      upper_blocks::Union{AbstractArray{Int,2}, Nothing} = nothing,
      upper_ranks::Union{AbstractVector{Int}, Nothing} = nothing,
      sizes::Union{AbstractVector{Int}, Ref{Int}, Nothing}=nothing,
      num_rots::Union{AbstractVector{Int}, Ref{Int}, Nothing}=nothing,
      offsets::Union{AbstractVector{Int}, Nothing}=nothing,
      expand_values::Bool=false,
    )

Compute transform `sizes`, `num_rots`, `offsets` values in place,
depending on which is provided.  If a `Ref{Int}` is given, compute the
maximum value over the range of indices.  If `expand_values` is
`true` compute maximum of all types of decompositions specified so
that the structure can hold multiple types of decompositions.

"""
function set_givens_weight_transform_params!(
  ::Lower,
  ::LeadingDecomp,
  m::Int,
  n::Int;
  lower_blocks::Union{AbstractArray{Int,2},Nothing} = nothing,
  lower_ranks::Union{AbstractVector{Int},Nothing} = nothing,
  upper_blocks::Union{AbstractArray{Int,2},Nothing} = nothing,
  upper_ranks::Union{AbstractVector{Int},Nothing} = nothing,
  sizes::Union{AbstractVector{Int},Ref{Int},Nothing} = nothing,
  num_rots::Union{AbstractVector{Int},Ref{Int},Nothing} = nothing,
  offsets::Union{AbstractVector{Int},Nothing} = nothing,
  expand_values::Bool = false,
)

  # start with zeros if not expanding other values to find a maximum.
  maybe_zero(expand_values, sizes, num_rots)

  # leading lower
  old_cols_lb = 1:0
  old_rank = 0
  for lb ∈ axes(lower_blocks, 2)[begin:end]
    _, cols_lb = lower_block_ranges(lower_blocks, m, n, lb)
    # new columns in block lb.
    dᵣ = setdiffᵣ(cols_lb, old_cols_lb)

    # columns acted on by the rotations associated with block lb,
    # assuming block lb-1 is column compressed.
    trange = dᵣ ∪ᵣ last(old_cols_lb, old_rank)
    tsize = length(trange)

    expand_or_set!(expand_values, sizes, lb, tsize)

    expand_or_set!(
      expand_values,
      num_rots,
      lb,
      (tsize - lower_ranks[lb]) * lower_ranks[lb],
    )

    maybe_set!(offsets, lb, _forward_get_offset(cols_lb, trange))

    old_cols_lb = cols_lb
    old_rank = lower_ranks[lb]
  end
  return nothing
end

function set_givens_weight_transform_params!(
  ::Upper,
  ::LeadingDecomp,
  m::Int,
  n::Int;
  lower_blocks::Union{AbstractArray{Int,2},Nothing} = nothing,
  lower_ranks::Union{AbstractVector{Int},Nothing} = nothing,
  upper_blocks::Union{AbstractArray{Int,2},Nothing} = nothing,
  upper_ranks::Union{AbstractVector{Int},Nothing} = nothing,
  sizes::Union{AbstractVector{Int},Ref{Int},Nothing} = nothing,
  num_rots::Union{AbstractVector{Int},Ref{Int},Nothing} = nothing,
  offsets::Union{AbstractVector{Int},Nothing} = nothing,
  expand_values::Bool = false,
)

  # start with zeros if not expanding other values to find a maximum.
  maybe_zero(expand_values, sizes, num_rots)

  # leading upper
  old_rows_ub = 1:0
  old_rank = 0
  for ub ∈ axes(upper_blocks, 2)[begin:end]
    rows_ub, _ = upper_block_ranges(upper_blocks, m, n, ub)
    # new rows in block ub.
    dᵣ = setdiffᵣ(rows_ub, old_rows_ub)
    trange = dᵣ ∪ᵣ last(old_rows_ub, old_rank)
    tsize = length(trange)

    expand_or_set!(expand_values, sizes, ub, tsize)

    expand_or_set!(
      expand_values,
      num_rots,
      ub,
      (tsize - upper_ranks(ub)) * upper_ranks(ub),
    )

    maybe_set!(offsets, ub, _forward_get_offset(rows_ub, trange))

    old_rows_ub = rows_ub
    old_rank = upper_ranks(ub)
  end
  return nothing
end

function set_givens_weight_transform_params!(
  ::Upper,
  ::TrailingDecomp,
  m::Int,
  n::Int;
  lower_blocks::Union{AbstractArray{Int,2},Nothing} = nothing,
  lower_ranks::Union{AbstractVector{Int},Nothing} = nothing,
  upper_blocks::Union{AbstractArray{Int,2},Nothing} = nothing,
  upper_ranks::Union{AbstractVector{Int},Nothing} = nothing,
  sizes::Union{AbstractVector{Int},Ref{Int},Nothing} = nothing,
  num_rots::Union{AbstractVector{Int},Ref{Int},Nothing} = nothing,
  offsets::Union{AbstractVector{Int},Nothing} = nothing,
  expand_values::Bool = false,
)

  maybe_zero(expand_values, sizes, num_rots)

  # trailing upper
  old_cols_ub = 1:0
  old_rank = 0
  for ub ∈ axes(upper_blocks, 2)[end:-1:begin]
    (_, cols_ub) = upper_block_ranges(upper_blocks, m, n, ub)
    # new columns in block ub relative to ub+1.
    dᵣ = setdiffᵣ(cols_ub, old_cols_ub)
    trange = dᵣ ∪ᵣ first(old_cols_ub, old_rank)
    tsize = length(trange)

    # columns acted on by the rotations associated with block ub,
    # assuming block ub+1 is column compressed.
    expand_or_set!(expand_values, sizes, ub, tsize)

    expand_or_set!(
      expand_values,
      num_rots,
      ub,
      (tsize - upper_ranks[ub]) * upper_ranks[ub],
    )

    maybe_set!(offsets, ub, _backward_get_offset(n, cols_ub, trange))

    old_cols_ub = cols_ub
    old_rank = upper_ranks(ub)
  end

  return nothing
end

function set_givens_weight_transform_params!(
  ::Lower,
  ::TrailingDecomp,
  m::Int,
  n::Int;
  lower_blocks::Union{AbstractArray{Int,2},Nothing} = nothing,
  lower_ranks::Union{AbstractVector{Int},Nothing} = nothing,
  upper_blocks::Union{AbstractArray{Int,2},Nothing} = nothing,
  upper_ranks::Union{AbstractVector{Int},Nothing} = nothing,
  sizes::Union{AbstractVector{Int},Ref{Int},Nothing} = nothing,
  num_hs::Union{AbstractVector{Int},Ref{Int},Nothing} = nothing,
  offsets::Union{AbstractVector{Int},Nothing} = nothing,
  expand_values::Bool = false,
)
  num_blocks = size(lower_blocks, 2)

  maybe_zero(expand_values, sizes, num_hs)

  # trailing lower
  old_rows_lb = 1:0
  old_rank = 0
  for lb ∈ num_blocks:-1:1
    rows_lb, _ = lower_block_ranges(lower_blocks, m, n, lb)
    dᵣ = setdiffᵣ(rows_lb, old_rows_lb)
    trange = dᵣ ∪ᵣ first(old_rows_lb, old_rank)
    tsize = length(trange)

    expand_or_set!(expand_values, sizes, lb, tsize)

    expand_or_set!(
      expand_values,
      num_hs,
      lb,
      (tsize - lower_ranks[lb]) * lower_ranks[lb],
    )

    maybe_set!(offsets, lb, _backward_get_offset(m, rows_lb, trange))

    old_rows_lb = rows_lb
    old_rank = lower_ranks[lb]
  end
  return nothing
end

function set_givens_weight_transform_params!(
  lower_upper::Union{Lower,Upper},
  decomp::Union{Decomp,Tuple{Vararg{Decomp}}},
  m::Int,
  n::Int;
  lower_blocks::Union{AbstractArray{Int,2},Nothing} = nothing,
  lower_ranks::Union{AbstractVector{Int},Nothing} = nothing,
  upper_blocks::Union{AbstractArray{Int,2},Nothing} = nothing,
  upper_ranks::Union{AbstractVector{Int},Nothing} = nothing,
  sizes::Union{AbstractVector{Int},Ref{Int},Nothing} = nothing,
  num_rots::Union{AbstractVector{Int},Ref{Int},Nothing} = nothing,
  offsets::Union{AbstractVector{Int},Nothing} = nothing,
  expand_values::Bool = false,
)

  expand = expand_values

  for s ∈ step
    for d ∈ decomp
      set_givens_weight_transform_params!(
        lower_upper,
        d,
        m,
        n;
        lower_blocks = lower_blocks,
        lower_ranks = lower_ranks,
        upper_blocks = upper_blocks,
        upper_ranks = upper_ranks,
        sizes = sizes,
        num_rots = num_rots,
        offsets = offsets,
        expand_values = expand,
      )
      # If set is called more than once, always expand after the first
      # call.
      expand = true
    end
  end

  return nothing
end


"""
    get_givens_transform_params(
      lower_upper::Union{Lower,Upper},
      decomp::Union{Decomp, Tuple{Vararg{Decomp}}},
      m::Int,
      n::Int,
      params::Vararg{Union{Sizes,NumRots,Offsets}};
      lower_blocks::Union{AbstractArray{Int,2},Nothing} = nothing,
      lower_ranks::Union{AbstractVector{Int},Nothing} = nothing,
      upper_blocks::Union{AbstractArray{Int,2},Nothing} = nothing,
      upper_ranks::Union{AbstractVector{Int},Nothing} = nothing,
    )

Compute `sizes`, `num_rots`, `offsets` values, depending on which
selectors are provided as `Vararg` parameters.
"""
function get_givens_weight_transform_params(
  lower_upper::Union{Lower,Upper},
  decomp::Union{Decomp,Tuple{Vararg{Decomp}}},
  m::Int,
  n::Int,
  params::Vararg{Union{Sizes,NumRots,Offsets}};
  lower_blocks::Union{AbstractArray{Int,2},Nothing} = nothing,
  lower_ranks::Union{AbstractVector{Int},Nothing} = nothing,
  upper_blocks::Union{AbstractArray{Int,2},Nothing} = nothing,
  upper_ranks::Union{AbstractVector{Int},Nothing} = nothing,
)

  num_blocks = size(something(lower_blocks, upper_blocks), 2)

  sizes = nothing
  offsets = nothing
  num_rots = nothing

  for p ∈ params
    p == Sizes() && (sizes = zeros(Int, num_blocks))
    p == NumRots() && (num_rots = zeros(Int, num_blocks))
    p == Offsets() && (offsets = zeros(Int, num_blocks))
  end

  set_givens_weight_transform_params!(
    lower_upper,
    decomp,
    m,
    n,
    lower_blocks = lower_blocks,
    lower_ranks = lower_ranks,
    upper_blocks = upper_blocks,
    upper_ranks = upper_ranks,
    sizes = sizes,
    num_rots = num_rots,
    offsets = offsets,
    expand_values = false,
  )

  result::Vector{Vector{Int}} = []

  for p ∈ params
    p == Sizes() && push!(result, sizes)
    p == NumRots() && push!(result, num_rots)
    p == Offsets() && push!(result, offsets)
  end
  length(result) == 1 ? result[1] : tuple(result...)

end

"""
    get_givens_weight_max_transform_params(
      lower_upper::Union{Lower,Upper},
      decomp::Union{Decomp, Tuple{Vararg{Decomp}}},
      step::Union{Step, Tuple{Vararg{Step}}},
      m::Int,
      n::Int,
      params::Vararg{Union{Sizes,NumRots}};
      lower_blocks::Union{AbstractArray{Int,2},Nothing} = nothing,
      lower_ranks::Union{AbstractVector{Int},Nothing} = nothing,
      upper_blocks::Union{AbstractArray{Int,2},Nothing} = nothing,
      upper_ranks::Union{AbstractVector{Int},Nothing} = nothing,
    )

Compute requested maxima for `sizes` and `num_hs` for the type(s) of
decompositions/transforms specified.
"""
function get_givens_weight_max_transform_params(
  lower_upper::Union{Lower,Upper},
  decomp::Union{Decomp,Tuple{Vararg{Decomp}}},
  m::Int,
  n::Int,
  params::Vararg{Union{Sizes,NumRots}};
  lower_blocks::Union{AbstractArray{Int,2},Nothing} = nothing,
  lower_ranks::Union{AbstractVector{Int},Nothing} = nothing,
  upper_blocks::Union{AbstractArray{Int,2},Nothing} = nothing,
  upper_ranks::Union{AbstractVector{Int},Nothing} = nothing,
)

  sizes = nothing
  num_rots = nothing

  for p ∈ params
    p == Sizes() && (sizes = Ref(0))
    p == NumRots() && (num_rots = Ref(0))
  end

  set_givens_weight_transform_params!(
    lower_upper,
    decomp,
    m,
    n;
    lower_blocks = lower_blocks,
    lower_ranks = lower_ranks,
    upper_blocks = upper_blocks,
    upper_ranks = upper_ranks,
    sizes = sizes,
    num_rots = num_rots,
    expand_values = false,
  )

  result::Vector{Int} = []

  for p ∈ params
    p == Sizes() && push!(result, sizes[])
    p == NumRots() && push!(result, num_rots[])
  end
  length(result) == 1 ? result[1] : tuple(result...)

end

end
