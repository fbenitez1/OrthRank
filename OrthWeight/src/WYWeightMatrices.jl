module WYWeightMatrices

export WYWeight,
  set_WYWeight_transform_params!,
  get_WYWeight_transform_params,
  get_WYWeight_max_transform_params

using BandStruct
using BandStruct.BlockedBandColumnMatrices
using BandStruct.BandwidthInit
using OrthWeight.BasicTypes
using Householder: WYTrans
using InPlace: apply!, apply_inv!
using LinearAlgebra

using Random

"""
    WYWeight{LWY,B,RWY}

A WYWeight matrix with weight matrix of type `B` and left and weight
WY transformations of type `LWY` and `RWY` respectively.

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
struct WYWeight{LWY,B,UWY} <: OrthWeightDecomp
  decomp::Base.RefValue{Union{Nothing,Decomp}}
  step::Base.RefValue{Union{Nothing,NullStep,SpanStep}}
  lowerWY::LWY
  b::B
  upperWY::UWY
  upper_ranks::Vector{Int}
  upper_compressed::Vector{Bool}
  lower_ranks::Vector{Int}
  lower_compressed::Vector{Bool}
end

"""

    WYWeight(
      ::Type{E},
      step::Union{SpanStep, NullStep},
      m::Int,
      n::Int;
      upper_rank_max::Int,
      lower_rank_max::Int,
      upper_blocks::Array{Int,2},
      lower_blocks::Array{Int,2},
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
  upper_blocks::Array{Int,2},
  lower_blocks::Array{Int,2},
) where {E<:Number}

  bbc = BlockedBandColumn(
    E,
    m,
    n,
    upper_rank_max = upper_rank_max,
    lower_rank_max = lower_rank_max,
    upper_blocks = upper_blocks,
    lower_blocks = lower_blocks,
  )

  num_blocks = size(upper_blocks, 2)
  size(lower_blocks, 2) == num_blocks ||
    error("""In a WYWeight, the number of upper blocks should equal the number
             of lower blocks""")

  rank_max = max(upper_rank_max, lower_rank_max)

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
    max_num_WY = num_blocks,
    max_WY_size = lower_max_sizes,
    max_num_hs = lower_max_num_hs,
    work_size = lower_work_size,
  )

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
    max_num_WY = num_blocks,
    max_WY_size = upper_max_sizes,
    max_num_hs = upper_max_num_hs,
    work_size = upper_work_size,
  )

  upper_ranks = zeros(Int, num_blocks)
  lower_ranks = zeros(Int, num_blocks)
  decomp_ref = Base.RefValue{Union{Nothing,Decomp}}(nothing)
  step_ref = Base.RefValue{Union{Nothing,NullStep,SpanStep}}(step)
  upper_compressed = fill(true, num_blocks)
  lower_compressed = fill(true, num_blocks)
  WYWeight(
    decomp_ref,
    step_ref,
    lowerWY,
    bbc,
    upperWY,
    upper_ranks,
    upper_compressed,
    lower_ranks,
    lower_compressed,
  )
end

"""
    WYWeight(
      ::Type{E},
      step::Union{SpanStep, NullStep},
      decomp::Union{Nothing, Decomp},
      rng::AbstractRNG,
      m::Int,
      n::Int;
      upper_ranks::Vector{Int},
      lower_ranks::Vector{Int},
      upper_rank_max::Int = maximum(upper_ranks),
      lower_rank_max::Int = maximum(lower_ranks),
      upper_blocks::Array{Int,2},
      lower_blocks::Array{Int,2},
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
  upper_blocks::Array{Int,2},
  lower_blocks::Array{Int,2},
) where {E<:Number}

  num_blocks = size(upper_blocks, 2)

  upper_ranks =
    isnothing(upper_ranks) ? fill(upper_rank_max, num_blocks) : upper_ranks
  lower_ranks =
    isnothing(lower_ranks) ? fill(lower_rank_max, num_blocks) : lower_ranks

  bbc = BlockedBandColumn(
    E,
    decomp,
    rng,
    m,
    n,
    upper_rank_max = upper_rank_max,
    lower_rank_max = lower_rank_max,
    upper_blocks = upper_blocks,
    upper_ranks = upper_ranks,
    lower_blocks = lower_blocks,
    lower_ranks = lower_ranks,
  )

  num_blocks = size(upper_blocks, 2)
  size(lower_blocks, 2) == num_blocks ||
    error("""In a WYWeight, the number of upper blocks should equal the number
             of lower blocks""")

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

  lower_work_size = n * (2 * lower_max_sizes)

  lowerWY = WYTrans(
    E,
    max_num_WY = num_blocks,
    max_WY_size = lower_max_sizes,
    max_num_hs = lower_max_num_hs,
    work_size = lower_work_size,
  )

  set_WYWeight_transform_params!(
    Lower(),
    decomp,
    step,
    m,
    n,
    lower_blocks = lower_blocks,
    lower_ranks = lower_rank_max,
    upper_blocks = upper_blocks,
    upper_ranks = upper_rank_max,
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
    Num_hs(),
    upper_blocks = upper_blocks,
    upper_ranks = upper_rank_max,
    lower_blocks = lower_blocks,
    lower_ranks = lower_rank_max,
  )

  upper_work_size = m * (2 * upper_max_sizes)

  upperWY = WYTrans(
    E,
    max_num_WY = num_blocks,
    max_WY_size = upper_max_sizes,
    max_num_hs = upper_max_num_hs,
    work_size = upper_work_size,
  )

  set_WYWeight_transform_params!(
    Upper(),
    decomp,
    step,
    m,
    n,
    lower_blocks = lower_blocks,
    lower_ranks = lower_ranks,
    upper_blocks = upper_blocks,
    upper_ranks = upper_rank_max,
    num_hs = upperWY.num_hs,
    offsets = upperWY.offsets,
    sizes = upperWY.sizes,
  )

  rand!(rng, upperWY)

  decomp_ref = Base.RefValue{Union{Nothing,Decomp}}(decomp)
  step_ref = Base.RefValue{Union{Nothing,NullStep,SpanStep}}(step)
  upper_compressed = fill(true, num_blocks)
  lower_compressed = fill(true, num_blocks)

  WYWeight(
    decomp_ref,
    step_ref,
    lowerWY,
    bbc,
    upperWY,
    upper_ranks,
    upper_compressed,
    lower_ranks,
    lower_compressed,
  )
end

function Base.length(lc::LowerCompressed{<:OrthWeightDecomp})
  l = 1
  len = 0
  while l <= lc.decomp.b.num_blocks
    lc.decomp.lower_compressed[l] && (len += 1)
    l += 1
  end
  len
end

"""
    set_WYWeight_transform_params!(
      side::Union{Left, Right},
      decomp::Union{Decomp, Tuple{Vararg{Decomp}}},
      step::Union{Step, Tuple{Vararg{Step}}},
      m::Int,
      n::Int;
      lower_blocks::Union{AbstractArray{Int,2}, Nothing} = nothing,
      lower_ranks::Union{AbstractVector{Int}, Int, Nothing} = nothing,
      upper_blocks::Union{AbstractArray{Int,2}, Nothing} = nothing,
      upper_ranks::Union{AbstractVector{Int}, Int, Nothing} = nothing,
      sizes::Union{AbstractVector{Int}, Ref{Int}, Nothing}=nothing,
      num_hs::Union{AbstractVector{Int}, Ref{Int}, Nothing}=nothing,
      offsets::Union{AbstractVector{Int}, Nothing}=nothing,
      find_maximum::Bool=false,
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
  lower_blocks::Union{AbstractArray{Int,2},Nothing} = nothing,
  lower_ranks::Union{AbstractVector{Int},Int,Nothing} = nothing,
  upper_blocks::Union{AbstractArray{Int,2},Nothing} = nothing,
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
  lower_blocks::Union{AbstractArray{Int,2},Nothing} = nothing,
  lower_ranks::Union{AbstractVector{Int},Int,Nothing} = nothing,
  upper_blocks::Union{AbstractArray{Int,2},Nothing} = nothing,
  upper_ranks::Union{AbstractVector{Int},Int,Nothing} = nothing,
  sizes::Union{AbstractVector{Int},Ref{Int},Nothing} = nothing,
  num_hs::Union{AbstractVector{Int},Ref{Int},Nothing} = nothing,
  offsets::Union{AbstractVector{Int},Nothing} = nothing,
  find_maximum::Bool = false,
)
  num_blocks = size(lower_blocks, 2)
  lrank(k) = getindex_or_scalar(lower_ranks, k)
  
  # start with zeros if not expanding other values to find a maximum.
  maybe_zero(find_maximum, sizes, num_hs)

  # leading lower
  old_cols_lb = 1:0
  old_rank = 0
  for lb ∈ 1:num_blocks
    _, cols_lb = lower_block_ranges(lower_blocks, m, n, lb)
    dᵣ = setdiffᵣ(cols_lb, old_cols_lb)
    trange = dᵣ ∪ᵣ last(old_cols_lb, old_rank)
  
    tsize = length(trange)
  
    if !isa(sizes, Nothing)
      expand_or_set!(find_maximum, sizes, lb, tsize)
    end
  
    if !isa(num_hs, Nothing)
      expand_or_set!(
        find_maximum,
        num_hs,
        lb,
        step == SpanStep ? lrank(lb) : tsize - lrank(lb),
      )
    end
  
    if isa(offsets, AbstractArray{Int})
      offsets[lb] = if !isempty(trange)
        first(trange) - 1
      elseif !isempty(cols_lb)
        last(cols_lb)
      else
        0
      end
    end

    old_cols_lb = cols_lb
    old_rank = lrank(lb)
  end
  nothing
end

function set_WYWeight_transform_params!(
  ::Upper,
  ::LeadingDecomp,
  step::Union{SpanStep,NullStep},
  m::Int,
  n::Int;
  lower_blocks::Union{AbstractArray{Int,2},Nothing} = nothing,
  lower_ranks::Union{AbstractVector{Int},Int,Nothing} = nothing,
  upper_blocks::Union{AbstractArray{Int,2},Nothing} = nothing,
  upper_ranks::Union{AbstractVector{Int},Int,Nothing} = nothing,
  sizes::Union{AbstractVector{Int},Ref{Int},Nothing} = nothing,
  num_hs::Union{AbstractVector{Int},Ref{Int},Nothing} = nothing,
  offsets::Union{AbstractVector{Int},Nothing} = nothing,
  find_maximum::Bool = false,
)

  num_blocks = size(upper_blocks, 2)

  urank(k) = getindex_or_scalar(upper_ranks, k)

  maybe_zero(find_maximum, sizes, num_hs)

  # leading upper
  old_rows_ub = 1:0
  old_rank = 0
  for ub ∈ 1:num_blocks
    rows_ub, _ = upper_block_ranges(upper_blocks, m, n, ub)
    dᵣ = setdiffᵣ(rows_ub, old_rows_ub)
    trange = dᵣ ∪ᵣ last(old_rows_ub, old_rank)
    tsize = length(trange)

    if !isa(sizes, Nothing)
      expand_or_set!(find_maximum, sizes, ub, tsize)
    end

    if !isa(num_hs, Nothing)
      expand_or_set!(
        find_maximum,
        num_hs,
        ub,
        step == SpanStep ? urank(ub) : tsize - urank(ub),
      )
    end

    if isa(offsets, AbstractArray{Int})
      offsets[ub] = if !isempty(trange)
        first(trange) - 1
      elseif !isempty(rows_ub)
        last(rows_ub)
      else
        0
      end
    end

    old_rows_ub = rows_ub
    old_rank = urank(ub)
  end

  nothing
end

function set_WYWeight_transform_params!(
  ::Upper,
  ::TrailingDecomp,
  step::Union{SpanStep,NullStep},
  m::Int,
  n::Int;
  lower_blocks::Union{AbstractArray{Int,2},Nothing} = nothing,
  lower_ranks::Union{AbstractVector{Int},Int,Nothing} = nothing,
  upper_blocks::Union{AbstractArray{Int,2},Nothing} = nothing,
  upper_ranks::Union{AbstractVector{Int},Int,Nothing} = nothing,
  sizes::Union{AbstractVector{Int},Ref{Int},Nothing} = nothing,
  num_hs::Union{AbstractVector{Int},Ref{Int},Nothing} = nothing,
  offsets::Union{AbstractVector{Int},Nothing} = nothing,
  find_maximum::Bool = false,
)
  num_blocks = size(upper_blocks, 2)

  urank(k) = getindex_or_scalar(upper_ranks, k)

  maybe_zero(find_maximum, sizes, num_hs)

  # trailing upper
  old_cols_ub = 1:0
  old_rank = 0
  for ub ∈ num_blocks:-1:1
    (_, cols_ub) = upper_block_ranges(upper_blocks, m, n, ub)
    dᵣ = setdiffᵣ(cols_ub, old_cols_ub)
    trange = dᵣ ∪ᵣ first(old_cols_ub, old_rank)
    tsize = length(trange)
  
    if !isa(sizes, Nothing)
      expand_or_set!(find_maximum, sizes, ub, tsize)
    end
  
    if !isa(num_hs, Nothing)
      expand_or_set!(
        find_maximum,
        num_hs,
        ub,
        step == SpanStep ? urank(ub) : tsize - urank(ub),
      )
    end
  
    if isa(offsets, AbstractArray{Int})
      offsets[ub] = if !isempty(trange)
        first(trange) - 1
      elseif !isempty(cols_ub)
        last(cols_ub)
      else
        0
      end
    end
  
    old_cols_ub = cols_ub
    old_rank = urank(ub)
  end

  nothing
end

function set_WYWeight_transform_params!(
  ::Lower,
  ::TrailingDecomp,
  step::Union{SpanStep,NullStep},
  m::Int,
  n::Int;
  lower_blocks::Union{AbstractArray{Int,2},Nothing} = nothing,
  lower_ranks::Union{AbstractVector{Int},Int,Nothing} = nothing,
  upper_blocks::Union{AbstractArray{Int,2},Nothing} = nothing,
  upper_ranks::Union{AbstractVector{Int},Int,Nothing} = nothing,
  sizes::Union{AbstractVector{Int},Ref{Int},Nothing} = nothing,
  num_hs::Union{AbstractVector{Int},Ref{Int},Nothing} = nothing,
  offsets::Union{AbstractVector{Int},Nothing} = nothing,
  find_maximum::Bool = false,
)
  num_blocks = size(lower_blocks, 2)

  lrank(k) = getindex_or_scalar(lower_ranks, k)

  maybe_zero(find_maximum, sizes, num_hs)

  # trailing lower
  old_rows_lb = 1:0
  old_rank = 0
  for lb ∈ num_blocks:-1:1
    rows_lb, _ = lower_block_ranges(lower_blocks, m, n, lb)
    dᵣ = setdiffᵣ(rows_lb, old_rows_lb)
    trange = dᵣ ∪ᵣ first(old_rows_lb, old_rank)
    tsize = length(trange)
  
    if !isa(sizes, Nothing)
      expand_or_set!(find_maximum, sizes, lb, tsize)
    end
  
    if !isa(num_hs, Nothing)
      expand_or_set!(
        find_maximum,
        num_hs,
        lb,
        step == SpanStep ? lrank(lb) : tsize - lrank(lb),
      )
    end
  
    if isa(offsets, AbstractArray{Int})
      offsets[lb] = if !isempty(trange)
        first(trange) - 1
      elseif !isempty(rows_lb)
        last(rows_lb)
      else
        0
      end
    end
  
    old_rows_lb = rows_lb
    old_rank = lrank(lb)
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
      lower_blocks::Union{AbstractArray{Int,2},Nothing} = nothing,
      lower_ranks::Union{AbstractVector{Int},Int,Nothing} = nothing,
      upper_blocks::Union{AbstractArray{Int,2},Nothing} = nothing,
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
  lower_blocks::Union{AbstractArray{Int,2},Nothing} = nothing,
  lower_ranks::Union{AbstractVector{Int},Int,Nothing} = nothing,
  upper_blocks::Union{AbstractArray{Int,2},Nothing} = nothing,
  upper_ranks::Union{AbstractVector{Int},Int,Nothing} = nothing,
)

  num_blocks = size(something(lower_blocks, upper_blocks), 2)

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
      lower_blocks::Union{AbstractArray{Int,2},Nothing} = nothing,
      lower_ranks::Union{AbstractVector{Int},Int,Nothing} = nothing,
      upper_blocks::Union{AbstractArray{Int,2},Nothing} = nothing,
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
  lower_blocks::Union{AbstractArray{Int,2},Nothing} = nothing,
  lower_ranks::Union{AbstractVector{Int},Int,Nothing} = nothing,
  upper_blocks::Union{AbstractArray{Int,2},Nothing} = nothing,
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
  uwy = wyw.upperWY
  @views for l ∈ Iterators.reverse(LowerCompressed(wyw))
    rows, _ = lower_block_ranges(bbc, l)
    apply_inv!(a[rows, :], (uwy, l))
  end
  @views for l ∈ Iterators.reverse(UpperCompressed(wyw))
    _, cols = upper_block_ranges(bbc, l)
    apply!((lwy, l), a[:, cols])
  end
  a
end

function LinearAlgebra.Matrix(::Type{TrailingDecomp}, wyw::WYWeight)
  bbc = wyw.b
  a = Matrix(bbc)
  lwy = wyw.lowerWY
  uwy = wyw.upperWY
  @views for l ∈ LowerCompressed(wyw)
    _, cols = lower_block_ranges(bbc, l)
    apply!((lwy, l), a[:, cols])
  end
  @views for l ∈ UpperCompressed(wyw)
    rows, _ = upper_block_ranges(bbc, l)
    apply_inv!(a[rows, :], (uwy, l))
  end
  a
end


end
