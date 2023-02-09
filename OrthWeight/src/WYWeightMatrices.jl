module WYWeightMatrices

export WYWeight,
  WYBlockData,
  block_sizes,
  set_WYWeight_transform_params!,
  get_WYWeight_transform_params,
  get_WYWeight_max_transform_params

using BandStruct
using BandStruct.BlockedBandColumnMatrices
using BandStruct.BandwidthInit
using OrthWeight.BasicTypes
using Householder: WYTrans
using Householder.WY
using InPlace: apply!, apply_inv!
using LinearAlgebra

using Random

mutable struct WYBlockData <: AbstractCompressibleData
  mb::Int
  nb::Int
  block_rank::Int
  compressed::Bool
  wy_index::Int
end

# const empty_WY_Float64 = WYTrans(Float64, 0, 0, 0, 0)
# const empty_WY_Complex64 = WYTrans(Complex{Float64}, 0, 0, 0, 0)

# empty_WY(E) = WYTrans(E, 0, 0, 0, 0)
# empty_WY(::Type{Float64}) = empty_WY_Float64
# empty_WY(::Type{Complex{Float64}}) = empty_WY_Complex64

WYBlockData(
  mb,
  nb;
  block_rank = 0,
  compressed = true,
  wy_index = 0
) = WYBlockData(
  mb,
  nb,
  block_rank,
  compressed,
  wy_index,
)


WYBlockData(;
  mb,
  nb,
  block_rank = 0,
  compressed = true,
  wy_index = 0
) = WYBlockData(
  mb,
  nb,
  block_rank,
  compressed,
  wy_index,
)


"""
    WYWeight{LWY,B,RWY}

A WYWeight matrix with weight matrix of type `B` and lower and upper
WY transformations of type `LWY` and `UWY` respectively.

# Fields

  - `decomp::Base.RefValue{Union{Nothing,Decomp}}`:
    Type of decomposition.

  - `step::::Base.RefValue{Union{Nothing,NullStep,SpanStep}}`: Flag
    whether zeros are introduced from a span of columns/rows or from a
    null space.

  - `lowerWY::LWY`: Lower WY for a banded WY weight decomposition.

  - `b::B`: Weight matrix.

  - `upperWY::UWY`: Upper WY for a banded WY weight decomposition.

  - `upper_ranks::Vector{Int}`: Upper ranks

  - `lower_ranks::Vector{Int}`: Lower ranks
"""
struct WYWeight{E,WY1,WY2} # <: OrthWeightDecomp
  decomp::Base.RefValue{Union{Nothing,LeadingDecomp,TrailingDecomp}}
  step::Base.RefValue{Union{Nothing,NullStep,SpanStep}}
  b::BlockedBandColumn{E,WYBlockData}
  lowerWY::WY1
  upperWY::WY2
end

"""

    WYWeight(
      ::Type{E},
      step::Union{SpanStep,NullStep},
      m::Int,
      n::Int;
      upper_rank_max::Int,
      lower_rank_max::Int,
      upper_blocks::Union{
        AbstractVector{<:AbstractBlockData},
        IndexList{<:AbstractBlockData},
      },
      max_num_upper_blocks = length(upper_blocks),
      lower_blocks::Union{
        AbstractVector{<:AbstractBlockData},
        IndexList{<:AbstractBlockData},
      },
      max_num_lower_blocks = length(lower_blocks),
    ) where {E<:Number}

Generic `WYWeight` with zero ranks but room for either a leading or
trailing decomposition with ranks up to `upper_rank_max` and
`lower_rank_max`.

"""
function WYWeight(
  ::Type{E},
  step::Union{SpanStep,NullStep},
  m::Int,
  n::Int;
  upper_rank_max::Int,
  lower_rank_max::Int,
  upper_blocks::Union{
    AbstractVector{<:AbstractBlockData},
    IndexList{<:AbstractBlockData},
  },
  max_num_upper_blocks = length(upper_blocks),
  lower_blocks::Union{
    AbstractVector{<:AbstractBlockData},
    IndexList{<:AbstractBlockData},
  },
  max_num_lower_blocks = length(lower_blocks),
) where {E<:Number}

  upper_blocks_wy = [
    let (; mb, nb) = bd
      WYBlockData(mb = mb, nb = nb)
    end for bd in upper_blocks
  ]

  lower_blocks_wy = [
    let (; mb, nb) = bd
      WYBlockData(mb = mb, nb = nb)
    end for bd in lower_blocks
  ]

  bbc = BlockedBandColumn(
    E,
    m,
    n,
    upper_rank_max = upper_rank_max,
    lower_rank_max = lower_rank_max,
    upper_blocks = upper_blocks_wy,
    lower_blocks = lower_blocks_wy,
  )

  lower_max_sizes, lower_max_num_hs = get_WYWeight_max_transform_params(
    Lower(),
    (LeadingDecomp(), TrailingDecomp()),
    step,
    m,
    n,
    Sizes(),
    Num_hs(),
    upper_blocks = upper_blocks,
    upper_ranks = upper_rank_max,
    lower_blocks = lower_blocks,
    lower_ranks = lower_rank_max,
  )

  # n * (max_WY_size + max_num_hs) where max_num_hs is potentially as
  # large as max_WY_size.
  lower_work_size = n * (2 * lower_max_sizes)
  lowerWY = WYTrans(
    E,
    max_num_WY = max_num_lower_blocks,
    max_WY_size = lower_max_sizes,
    max_num_hs = lower_max_num_hs,
    work_size = lower_work_size,
  )

  wy_index = 0
  for lb ∈ lower_blocks_wy
    wy_index += 1
    lower_blocks_wy[lb].wy_index = wy_index
  end


  upper_max_sizes, upper_max_num_hs = get_WYWeight_max_transform_params(
    Upper(),
    (LeadingDecomp(), TrailingDecomp()),
    step,
    m,
    n,
    Sizes(),
    Num_hs(),
    upper_blocks = upper_blocks,
    upper_ranks = upper_rank_max,
    lower_blocks = lower_blocks,
    lower_ranks = lower_rank_max,
  )

  upper_work_size = m * (2 * upper_max_sizes)
  upperWY = WYTrans(
    E,
    max_num_WY = max_num_upper_blocks,
    max_WY_size = upper_max_sizes,
    max_num_hs = upper_max_num_hs,
    work_size = upper_work_size,
  )

  wy_index = 0
  for ub ∈ upper_blocks_wy
    wy_index += 1
    upper_blocks_wy[ub].wy_index = wy_index
  end

  decomp_ref =
    Base.RefValue{Union{Nothing,LeadingDecomp,TrailingDecomp}}(nothing)
  step_ref = Base.RefValue{Union{Nothing,NullStep,SpanStep}}(step)

  return WYWeight(decomp_ref, step_ref, bbc, lowerWY, upperWY)
end

"""
    WYWeight(
      ::Type{E},
      step::Union{SpanStep,NullStep},
      decomp::Union{Nothing,Decomp},
      rng::AbstractRNG,
      m::Int,
      n::Int;
      upper_ranks::Union{Vector{Int},Nothing} = nothing,
      lower_ranks::Union{Vector{Int},Nothing} = nothing,
      upper_rank_max::Int = maximum(upper_ranks),
      lower_rank_max::Int = maximum(lower_ranks),
      upper_blocks::Union{
        AbstractVector{<:AbstractBlockData},
        IndexList{<:AbstractBlockData},
      },
      max_num_upper_blocks::Int = length(upper_blocks),
      lower_blocks::Union{
        AbstractVector{<:AbstractBlockData},
        IndexList{<:AbstractBlockData},
      },
      max_num_lower_blocks::Int = length(lower_blocks),
    ) where {E<:Number}

A random WYWeight with specified upper and lower ranks.  The structure provides
enough bandwidth for conversion.  If no upper or lower ranks are given, use `upper_rank_max`
and `lower_rank_max`.
"""
function WYWeight(
  ::Type{E},
  step::Union{SpanStep,NullStep},
  decomp::Union{Nothing,Decomp},
  rng::AbstractRNG,
  m::Int,
  n::Int;
  upper_ranks::Union{Vector{Int},Nothing} = nothing,
  lower_ranks::Union{Vector{Int},Nothing} = nothing,
  upper_rank_max::Int = maximum(upper_ranks),
  lower_rank_max::Int = maximum(lower_ranks),
  upper_blocks::Union{
    AbstractVector{<:AbstractBlockData},
    IndexList{<:AbstractBlockData},
  },
  max_num_upper_blocks::Int = length(upper_blocks),
  lower_blocks::Union{
    AbstractVector{<:AbstractBlockData},
    IndexList{<:AbstractBlockData},
  },
  max_num_lower_blocks::Int = length(lower_blocks),
) where {E<:Number}

  upper_blocks_wy = IndexList([
    let (; mb, nb) = bd
      WYBlockData(mb = mb, nb = nb)
    end for bd in upper_blocks
  ], max_length = max_num_upper_blocks)

  lower_blocks_wy = IndexList([
    let (; mb, nb) = bd
      WYBlockData(mb = mb, nb = nb)
    end for bd in lower_blocks
  ], max_length = max_num_lower_blocks)

  num_upper_blocks = length(upper_blocks)
  num_lower_blocks = length(lower_blocks)

  upper_ranks =
    isnothing(upper_ranks) ? fill(upper_rank_max, num_upper_blocks) : upper_ranks
  lower_ranks =
    isnothing(lower_ranks) ? fill(lower_rank_max, num_lower_blocks) : lower_ranks

  bbc = BlockedBandColumn(
    E,
    decomp,
    rng,
    m,
    n,
    upper_rank_max = upper_rank_max,
    lower_rank_max = lower_rank_max,
    upper_blocks = upper_blocks_wy,
    upper_ranks = upper_ranks,
    lower_blocks = lower_blocks_wy,
    lower_ranks = lower_ranks,
  )

  lower_max_sizes, lower_max_num_hs = get_WYWeight_max_transform_params(
    Lower(),
    (LeadingDecomp(), TrailingDecomp()),
    step,
    m,
    n,
    Sizes(),
    Num_hs(),
    upper_blocks = bbc.upper_blocks,
    upper_ranks = upper_rank_max,
    lower_blocks = bbc.lower_blocks,
    lower_ranks = lower_rank_max,
  )

  lower_work_size = n * (2 * lower_max_sizes)

  lowerWY = WYTrans(
    E,
    max_num_WY = max_num_lower_blocks,
    max_WY_size = lower_max_sizes,
    max_num_hs = lower_max_num_hs,
    work_size = lower_work_size,
  )

  lb_ind = 0
  for lb ∈ bbc.lower_blocks
    lb_ind += 1
    bbc.lower_blocks[lb].wy_index = lb_ind
    bbc.lower_blocks[lb].compressed = true
    bbc.lower_blocks[lb].block_rank = lower_ranks[lb_ind]
  end

  # Set the offsets, sizes, num_hs in lowerWY.
  set_WYWeight_transform_params!(
    Lower(),
    decomp,
    step,
    m,
    n,
    lower_blocks = bbc.lower_blocks,
    lower_ranks = lower_rank_max,
    num_hs = lowerWY.num_hs,
    offsets = lowerWY.offsets,
    sizes = lowerWY.sizes,
  )

  rand!(rng, lowerWY)

  upper_max_sizes, upper_max_num_hs = get_WYWeight_max_transform_params(
    Upper(),
    (LeadingDecomp(), TrailingDecomp()),
    step,
    m,
    n,
    Sizes(),
    Num_hs();
    upper_blocks = bbc.upper_blocks,
    upper_ranks = upper_rank_max,
    lower_blocks = bbc.lower_blocks,
    lower_ranks = lower_rank_max,
  )

  upper_work_size = m * (2 * upper_max_sizes)

  upperWY = WYTrans(
    E,
    max_num_WY = max_num_upper_blocks,
    max_WY_size = upper_max_sizes,
    max_num_hs = upper_max_num_hs,
    work_size = upper_work_size,
  )

  ub_ind = 0
  for lb ∈ bbc.upper_blocks
    ub_ind += 1
    bbc.upper_blocks[lb].wy_index = ub_ind
    bbc.upper_blocks[lb].compressed = true
    bbc.upper_blocks[lb].block_rank = upper_ranks[ub_ind]
  end

  set_WYWeight_transform_params!(
    Upper(),
    decomp,
    step,
    m,
    n,
    upper_blocks = bbc.upper_blocks,
    upper_ranks = upper_rank_max,
    num_hs = upperWY.num_hs,
    offsets = upperWY.offsets,
    sizes = upperWY.sizes,
  )

  rand!(rng, upperWY)

  decomp_ref = Base.RefValue{Union{Nothing,LeadingDecomp,TrailingDecomp}}(decomp)
  step_ref = Base.RefValue{Union{Nothing,NullStep,SpanStep}}(step)

  return WYWeight(decomp_ref, step_ref, bbc, lowerWY, upperWY)

end

"""
    set_WYWeight_transform_params!(
      side::Union{Lower,Upper},
      decomp::Union{Decomp,Tuple{Vararg{Decomp}}},
      step::Union{Step,Tuple{Vararg{Step}}},
      m::Int,
      n::Int;
      lower_blocks::Union{IndexList{<:AbstractBlockData},Nothing} = nothing,
      lower_ranks::Union{AbstractVector{Int},Int,Nothing} = nothing,
      upper_blocks::Union{IndexList{<:AbstractBlockData},Nothing} = nothing,
      upper_ranks::Union{AbstractVector{Int},Int,Nothing} = nothing,
      sizes::Union{AbstractVector{Int},Ref{Int},Nothing} = nothing,
      num_hs::Union{AbstractVector{Int},Ref{Int},Nothing} = nothing,
      offsets::Union{AbstractVector{Int},Nothing} = nothing,
      find_maximum::Bool = false,
    )

Compute `sizes`, `num_hs`, `offsets` values in place, depending on
which is provided.  If a `Ref{Int}` is given, compute the maximum
value over the range of indices.  This can optionally be done for a WY
Sweep for a leading or trailing decomposition where the
transformations are on the left or right and the Householders are
computed for a SpanStep (from column/row spaces, which have lower
dimension) or for a NullStep (from null spaces).  If `find_maximum` is
`true` compute maximum of all types of decompositions specified so
that the structure can hold multiple types of decompositions.
"""
function set_WYWeight_transform_params!(
  side::Union{Lower,Upper},
  decomp::Union{Decomp,Tuple{Vararg{Decomp}}},
  step::Union{Step,Tuple{Vararg{Step}}},
  m::Int,
  n::Int;
  lower_blocks::Union{IndexList{<:AbstractBlockData},Nothing} = nothing,
  lower_ranks::Union{AbstractVector{Int},Int,Nothing} = nothing,
  upper_blocks::Union{IndexList{<:AbstractBlockData},Nothing} = nothing,
  upper_ranks::Union{AbstractVector{Int},Int,Nothing} = nothing,
  sizes::Union{AbstractVector{Int},Ref{Int},Nothing} = nothing,
  num_hs::Union{AbstractVector{Int},Ref{Int},Nothing} = nothing,
  offsets::Union{AbstractVector{Int},Nothing} = nothing,
  find_maximum::Bool = false,
)

  expand = find_maximum

  for s ∈ step
    for d ∈ decomp
      set_WYWeight_transform_params!(
        side,
        d,
        s,
        m,
        n,
        lower_blocks = lower_blocks,
        lower_ranks = lower_ranks,
        upper_blocks = upper_blocks,
        upper_ranks = upper_ranks,
        sizes = sizes,
        num_hs = num_hs,
        offsets = offsets,
        find_maximum = expand,
      )
      # If set is called more than once, always expand after the first
      # call.
      expand = true
    end
  end

  nothing
end

function set_WYWeight_transform_params!(
  ::Lower,
  ::LeadingDecomp,
  step::Union{SpanStep,NullStep},
  m::Int,
  n::Int;
  lower_blocks::IndexList{<:AbstractBlockData},
  lower_ranks::Union{AbstractVector{Int},Int,Nothing} = nothing,
  upper_blocks::Union{IndexList{<:AbstractBlockData},Nothing} = nothing,
  upper_ranks::Union{AbstractVector{Int},Int,Nothing} = nothing,
  sizes::Union{AbstractVector{Int},Ref{Int},Nothing} = nothing,
  num_hs::Union{AbstractVector{Int},Ref{Int},Nothing} = nothing,
  offsets::Union{AbstractVector{Int},Nothing} = nothing,
  find_maximum::Bool = false,
)

  lrank(k) = getindex_or_scalar(lower_ranks, k)
  
  # start with zeros if not expanding other values to find a maximum.
  maybe_zero(find_maximum, sizes, num_hs)

  # leading lower
  old_cols_lb = 1:0
  old_rank = 0
  lb_count = 0
  for lb ∈ lower_blocks
    lb_count += 1
    _, cols_lb = lower_block_ranges(lower_blocks, m, n, lb)
    dᵣ = setdiffᵣ(cols_lb, old_cols_lb)
    trange = dᵣ ∪ᵣ last(old_cols_lb, old_rank)
  
    tsize = length(trange)
  
    if !isa(sizes, Nothing)
      expand_or_set!(find_maximum, sizes, lb_count, tsize)
    end
  
    if !isa(num_hs, Nothing)
      expand_or_set!(
        find_maximum,
        num_hs,
        lb_count,
        step == SpanStep ? lrank(lb_count) : tsize - lrank(lb_count),
      )
    end
  
    if isa(offsets, AbstractArray{Int})
      offsets[lb_count] = if !isempty(trange)
        first(trange) - 1
      elseif !isempty(cols_lb)
        last(cols_lb)
      else
        0
      end
    end

    old_cols_lb = cols_lb
    old_rank = lrank(lb_count)
  end
  nothing
end

function set_WYWeight_transform_params!(
  ::Upper,
  ::LeadingDecomp,
  step::Union{SpanStep,NullStep},
  m::Int,
  n::Int;
  lower_blocks::Union{IndexList{<:AbstractBlockData},Nothing} = nothing,
  lower_ranks::Union{AbstractVector{Int},Int,Nothing} = nothing,
  upper_blocks::IndexList{<:AbstractBlockData},
  upper_ranks::Union{AbstractVector{Int},Int} = nothing,
  sizes::Union{AbstractVector{Int},Ref{Int},Nothing} = nothing,
  num_hs::Union{AbstractVector{Int},Ref{Int},Nothing} = nothing,
  offsets::Union{AbstractVector{Int},Nothing} = nothing,
  find_maximum::Bool = false,
)

  urank(k) = getindex_or_scalar(upper_ranks, k)

  maybe_zero(find_maximum, sizes, num_hs)

  # leading upper
  old_rows_ub = 1:0
  old_rank = 0
  ub_count = 0
  for ub ∈ upper_blocks
    ub_count += 1
    rows_ub, _ = upper_block_ranges(upper_blocks, m, n, ub)
    dᵣ = setdiffᵣ(rows_ub, old_rows_ub)
    trange = dᵣ ∪ᵣ last(old_rows_ub, old_rank)
    tsize = length(trange)

    if !isa(sizes, Nothing)
      expand_or_set!(find_maximum, sizes, ub_count, tsize)
    end

    if !isa(num_hs, Nothing)
      expand_or_set!(
        find_maximum,
        num_hs,
        ub_count,
        step == SpanStep ? urank(ub_count) : tsize - urank(ub_count),
      )
    end

    if isa(offsets, AbstractArray{Int})
      offsets[ub_count] = if !isempty(trange)
        first(trange) - 1
      elseif !isempty(rows_ub)
        last(rows_ub)
      else
        0
      end
    end

    old_rows_ub = rows_ub
    old_rank = urank(ub_count)
  end

  nothing
end

function set_WYWeight_transform_params!(
  ::Upper,
  ::TrailingDecomp,
  step::Union{SpanStep,NullStep},
  m::Int,
  n::Int;
  lower_blocks::Union{IndexList{<:AbstractBlockData},Nothing} = nothing,
  lower_ranks::Union{AbstractVector{Int},Int,Nothing} = nothing,
  upper_blocks::IndexList{<:AbstractBlockData} = nothing,
  upper_ranks::Union{AbstractVector{Int},Int} = nothing,
  sizes::Union{AbstractVector{Int},Ref{Int},Nothing} = nothing,
  num_hs::Union{AbstractVector{Int},Ref{Int},Nothing} = nothing,
  offsets::Union{AbstractVector{Int},Nothing} = nothing,
  find_maximum::Bool = false,
)

  urank(k) = getindex_or_scalar(upper_ranks, k)

  maybe_zero(find_maximum, sizes, num_hs)

  # trailing upper
  old_cols_ub = 1:0
  old_rank = 0
  ub_count = length(upper_blocks)
  for ub ∈ Iterators.Reverse(upper_blocks)
    (_, cols_ub) = upper_block_ranges(upper_blocks, m, n, ub)
    dᵣ = setdiffᵣ(cols_ub, old_cols_ub)
    trange = dᵣ ∪ᵣ first(old_cols_ub, old_rank)
    tsize = length(trange)
  
    if !isa(sizes, Nothing)
      expand_or_set!(find_maximum, sizes, ub_count, tsize)
    end
  
    if !isa(num_hs, Nothing)
      expand_or_set!(
        find_maximum,
        num_hs,
        ub_count,
        step == SpanStep ? urank(ub_count) : tsize - urank(ub_count),
      )
    end
  
    if isa(offsets, AbstractArray{Int})
      offsets[ub_count] = if !isempty(trange)
        first(trange) - 1
      elseif !isempty(cols_ub)
        last(cols_ub)
      else
        0
      end
    end
  
    old_cols_ub = cols_ub
    old_rank = urank(ub_count)
    ub_count -= 1
  end

  nothing
end

function set_WYWeight_transform_params!(
  ::Lower,
  ::TrailingDecomp,
  step::Union{SpanStep,NullStep},
  m::Int,
  n::Int;
  lower_blocks::IndexList{<:AbstractBlockData} = nothing,
  lower_ranks::Union{AbstractVector{Int},Int} = nothing,
  upper_blocks::Union{IndexList{<:AbstractBlockData},Nothing} = nothing,
  upper_ranks::Union{AbstractVector{Int},Int,Nothing} = nothing,
  sizes::Union{AbstractVector{Int},Ref{Int},Nothing} = nothing,
  num_hs::Union{AbstractVector{Int},Ref{Int},Nothing} = nothing,
  offsets::Union{AbstractVector{Int},Nothing} = nothing,
  find_maximum::Bool = false,
)

  lrank(k) = getindex_or_scalar(lower_ranks, k)

  maybe_zero(find_maximum, sizes, num_hs)

  # trailing lower
  old_rows_lb = 1:0
  old_rank = 0
  lb_count = length(lower_blocks)
  for lb ∈ Iterators.Reverse(lower_blocks)
    rows_lb, _ = lower_block_ranges(lower_blocks, m, n, lb)
    dᵣ = setdiffᵣ(rows_lb, old_rows_lb)
    trange = dᵣ ∪ᵣ first(old_rows_lb, old_rank)
    tsize = length(trange)
  
    if !isa(sizes, Nothing)
      expand_or_set!(find_maximum, sizes, lb_count, tsize)
    end
  
    if !isa(num_hs, Nothing)
      expand_or_set!(
        find_maximum,
        num_hs,
        lb_count,
        step == SpanStep ? lrank(lb_count) : tsize - lrank(lb_count),
      )
    end
  
    if isa(offsets, AbstractArray{Int})
      offsets[lb_count] = if !isempty(trange)
        first(trange) - 1
      elseif !isempty(rows_lb)
        last(rows_lb)
      else
        0
      end
    end
  
    old_rows_lb = rows_lb
    old_rank = lrank(lb_count)
    lb_count -= 1
  end
  nothing
end

"""
    get_WYWeight_transform_params(
      side::Union{Left,Right},
      decomp::Union{Decomp, Tuple{Vararg{Decomp}}},
      step::Union{Step, Tuple{Vararg{Step}}},
      m::Int,
      n::Int,
      params::Vararg{Union{Sizes,Num_hs,Offsets}};
      lower_blocks::Union{IndexList{<:AbstractBlockData},Nothing} = nothing,
      lower_ranks::Union{AbstractVector{Int},Int,Nothing} = nothing,
      upper_blocks::Union{IndexList{<:AbstractBlockData},Nothing} = nothing,
      upper_ranks::Union{AbstractVector{Int},Int,Nothing} = nothing,
    )

Compute `sizes`, `num_hs`, `offsets` values, depending on which
selectors are provided as `Vararg` parameters.
"""
function get_WYWeight_transform_params(
  side::Union{Lower,Upper},
  decomp::Union{Decomp,Tuple{Vararg{Decomp}}},
  step::Union{Step,Tuple{Vararg{Step}}},
  m::Int,
  n::Int,
  params::Vararg{Union{Sizes,Num_hs,Offsets}};
  lower_blocks::Union{IndexList{<:AbstractBlockData},Nothing} = nothing,
  lower_ranks::Union{AbstractVector{Int},Int,Nothing} = nothing,
  upper_blocks::Union{IndexList{<:AbstractBlockData},Nothing} = nothing,
  upper_ranks::Union{AbstractVector{Int},Int,Nothing} = nothing,
)

  num_blocks = length(something(lower_blocks, upper_blocks))

  sizes = nothing
  offsets = nothing
  num_hs = nothing

  for p ∈ params
    p == Sizes() && (sizes = zeros(Int, num_blocks))
    p == Num_hs() && (num_hs = zeros(Int, num_blocks))
    p == Offsets() && (offsets = zeros(Int, num_blocks))
  end

  set_WYWeight_transform_params!(
    side,
    decomp,
    step,
    m,
    n,
    lower_blocks = lower_blocks,
    lower_ranks = lower_ranks,
    upper_blocks = upper_blocks,
    upper_ranks = upper_ranks,
    sizes = sizes,
    num_hs = num_hs,
    offsets = offsets,
    find_maximum = false,
  )

  result::Vector{Vector{Int}} = []

  for p ∈ params
    p == Sizes() && push!(result, sizes)
    p == Num_hs() && push!(result, num_hs)
    p == Offsets() && push!(result, offsets)
  end
  length(result) == 1 ? result[1] : tuple(result...)

end

"""
    get_WYWeight_max_transform_params(
      side::Union{Left,Right},
      decomp::Union{Decomp, Tuple{Vararg{Decomp}}},
      step::Union{Step, Tuple{Vararg{Step}}},
      m::Int,
      n::Int,
      params::Vararg{Union{Sizes,Num_hs}};
      lower_blocks::Union{IndexList{<:AbstractBlockData},Nothing} = nothing,
      lower_ranks::Union{AbstractVector{Int},Int,Nothing} = nothing,
      upper_blocks::Union{IndexList{<:AbstractBlockData},Nothing} = nothing,
      upper_ranks::Union{AbstractVector{Int},Int,Nothing} = nothing,
    )

Compute requested maxima for `sizes` and `num_hs` for the type(s) of
decompositions/transforms specified.
"""
function get_WYWeight_max_transform_params(
  side::Union{Lower,Upper},
  decomp::Union{Decomp,Tuple{Vararg{Decomp}}},
  step::Union{Step,Tuple{Vararg{Step}}},
  m::Int,
  n::Int,
  params::Vararg{Union{Sizes,Num_hs}};
  lower_blocks::Union{IndexList{<:AbstractBlockData},Nothing} = nothing,
  lower_ranks::Union{AbstractVector{Int},Int,Nothing} = nothing,
  upper_blocks::Union{IndexList{<:AbstractBlockData},Nothing} = nothing,
  upper_ranks::Union{AbstractVector{Int},Int,Nothing} = nothing,
)

  sizes = nothing
  num_hs = nothing

  for p ∈ params
    p == Sizes() && (sizes = Ref(0))
    p == Num_hs() && (num_hs = Ref(0))
  end

  set_WYWeight_transform_params!(
    side,
    decomp,
    step,
    m,
    n,
    lower_blocks = lower_blocks,
    lower_ranks = lower_ranks,
    upper_blocks = upper_blocks,
    upper_ranks = upper_ranks,
    sizes = sizes,
    num_hs = num_hs,
    find_maximum = false,
  )

  result::Vector{Int} = []

  for p ∈ params
    p == Sizes() && push!(result, sizes[])
    p == Num_hs() && push!(result, num_hs[])
  end
  length(result) == 1 ? result[1] : tuple(result...)

end

Base.size(wyw::WYWeight) = size(wyw.b)


# Matrix construction.

function LinearAlgebra.Matrix(wyw::WYWeight)
  Matrix(typeof(wyw.decomp[]), wyw)
end

function LinearAlgebra.Matrix(::Type{LeadingDecomp}, wyw::WYWeight)
  bbc = wyw.b
  a = Matrix(bbc)
  lwy = wyw.lowerWY
  for l ∈ filter_compressed(Iterators.Reverse(wyw.b.lower_blocks))
    rows, _ = lower_block_ranges(bbc, l)
    wy_ind = (wyw.b.lower_blocks[l]).wy_index
    # Main.@infiltrate
    apply_inv!(view(a, rows, :), (lwy, wy_ind))
  end
  uwy = wyw.upperWY
  for l ∈ filter_compressed(Iterators.reverse(wyw.b.upper_blocks))
    _, cols = upper_block_ranges(bbc, l)
    wy_ind = (wyw.b.upper_blocks[l]).wy_index
    apply!((uwy, wy_ind), view(a, :, cols))
  end
  return a
end

function LinearAlgebra.Matrix(::Type{TrailingDecomp}, wyw::WYWeight)
  bbc = wyw.b
  a = Matrix(bbc)
  lwy = wyw.lowerWY
  for l ∈ filter_compressed(wyw.b.lower_blocks)
    _, cols = lower_block_ranges(bbc, l)
    wy_ind = (wyw.b.lower_blocks[l]).wy_index
    apply!((lwy, wy_ind), view(a, :, cols))
  end
  uwy = wyw.upperWY
  for l ∈ filter_compressed(wyw.b.upper_blocks)
    rows, _ = upper_block_ranges(bbc, l)
    wy_ind = (wyw.b.upper_blocks[l]).wy_index
    apply_inv!(view(a, rows, :), (uwy, wy_ind))
  end
  return a
end

end
