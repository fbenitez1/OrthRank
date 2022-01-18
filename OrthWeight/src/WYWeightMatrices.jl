module WYWeightMatrices

export WYWeight,
  SpanStep,
  NullStep,
  LowerCompressed,
  UpperCompressed,
  Left,
  Right,
  set_WYWeight_transform_params!,
  get_WYWeight_transform_params,
  get_WYWeight_max_transform_params

using BandStruct
using Householder: WYTrans
using InPlace: apply!, apply_inv!
using LinearAlgebra

using Random

"""
      struct SpanStep end

  A type marking a WYWeight with steps that are large relative to
  the off-diagonal ranks so that transformations are computed to introduce
  zeros directly into a basis for the span of block columns or rows.
"""
struct SpanStep end
Base.iterate(t::Type{SpanStep}) = (t, nothing)
Base.iterate(::Type{SpanStep}, ::Any) = nothing
BandStruct.singleton(::Type{SpanStep}) = SpanStep()

"""
        struct NullStep end

  A type marking a WYWeight with steps that are small relative to
  the off-diagonal ranks so that transformations are computed to
  introduce zeros directly into a basis for the null space of a block.
"""
struct NullStep end
Base.iterate(t::Type{NullStep}) = (t, nothing)
Base.iterate(::Type{NullStep}, ::Any) = nothing
BandStruct.singleton(::Type{NullStep}) = NullStep()


struct Left end
Base.iterate(t::Type{Left}) = (t, nothing)
Base.iterate(::Type{Left}, ::Any) = nothing
BandStruct.singleton(::Type{Left}) = Left()


struct Right end
Base.iterate(t::Type{Right}) = (t, nothing)
Base.iterate(::Type{Right}, ::Any) = nothing
BandStruct.singleton(::Type{Right}) = Right()


struct Sizes end
Base.iterate(t::Type{Sizes}) = (t, nothing)
Base.iterate(::Type{Sizes}, ::Any) = nothing
BandStruct.singleton(::Type{Sizes}) = Sizes()

struct Num_hs end
Base.iterate(t::Type{Num_hs}) = (t, nothing)
Base.iterate(::Type{Num_hs}, ::Any) = nothing
BandStruct.singleton(::Type{Num_hs}) = Num_hs()


struct Offsets end
Base.iterate(t::Type{Offsets}) = (t, nothing)
Base.iterate(::Type{Offsets}, ::Any) = nothing
BandStruct.singleton(::Type{Offsets}) = Offsets()


"""
    UnionOrTuple{A,B} = Union{A, B, Tuple{DataType,DataType}}
"""
UnionOrTuple{A,B} = Union{A, B, Tuple{DataType,DataType}}

"""
    WYWeight{LWY,B,RWY}

A WYWeight matrix with weight matrix of type `B` and left and weight
WY transformations of type `LWY` and `RWY` respectively.

# Fields

  - `decomp::Base.RefValue{Union{Nothing,LeadingDecomp,TrailingDecomp}}`:
    Type of decomposition.

  - `step::::Base.RefValue{Union{Nothing,NullStep,SpanStep}}`: Flag
    whether zeros are introduced from a span of columns/rows or from a
    null space.

  - `leftWY::LWY`: Left WY for a banded WY weight decomposition.

  - `b::B`: Weight matrix.

  - `rightWY::RWY`: Right WY for a banded WY weight decomposition.

  - `upper_ranks::Vector{Int}`: Upper ranks

  - `lower_ranks::Vector{Int}`: Lower ranks
"""
struct WYWeight{LWY,B,RWY}
  decomp::Base.RefValue{Union{Nothing,LeadingDecomp,TrailingDecomp}}
  step::Base.RefValue{Union{Nothing,NullStep,SpanStep}}
  leftWY::LWY
  b::B
  rightWY::RWY
  upper_ranks::Vector{Int}
  upper_compressed::Vector{Bool}
  lower_ranks::Vector{Int}
  lower_compressed::Vector{Bool}
end

"""

    WYWeight(
      ::Type{E},
      step::Union{Type{SpanStep}, Type{NullStep}},
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
  step::Union{Type{SpanStep}, Type{NullStep}},
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
  
  num_blocks = size(upper_blocks,2)
  size(lower_blocks, 2) == num_blocks ||
    error("""In a WYWeight, the number of upper blocks should equal the number
             of lower blocks""")

  rank_max = max(upper_rank_max, lower_rank_max)

  left_max_sizes, left_max_num_hs = get_WYWeight_max_transform_params(
    Left,
    (LeadingDecomp, TrailingDecomp),
    step,
    m,
    n,
    Sizes,
    Num_hs,
    upper_blocks = upper_blocks,
    upper_ranks = upper_rank_max,
    lower_blocks = lower_blocks,
    lower_ranks = lower_rank_max,
  )

  # n * (max_WY_size + max_num_hs) where max_num_hs is potentially as
  # large as max_WY_size.
  left_work_size = n * (2*left_max_sizes)

  leftWY = WYTrans(
    E,
    max_num_WY = num_blocks,
    max_WY_size = left_max_sizes,
    max_num_hs = left_max_num_hs,
    work_size = left_work_size
  )

  right_max_sizes, right_max_num_hs = get_WYWeight_max_transform_params(
    Right,
    (LeadingDecomp, TrailingDecomp),
    step,
    m,
    n,
    Sizes,
    Num_hs,
    upper_blocks=upper_blocks,
    upper_ranks=upper_rank_max,
    lower_blocks=lower_blocks,
    lower_ranks=lower_rank_max,
  )

  right_work_size = m * (2*right_max_sizes)

  rightWY = WYTrans(
    E,
    max_num_WY = num_blocks,
    max_WY_size = right_max_sizes,
    max_num_hs = right_max_num_hs,
    work_size = right_work_size
  )

  upper_ranks = zeros(Int, num_blocks)
  lower_ranks = zeros(Int, num_blocks)
  decomp_ref = Base.RefValue{Union{Nothing, LeadingDecomp, TrailingDecomp}}(nothing)
  step_ref = Base.RefValue{Union{Nothing, NullStep, SpanStep}}(singleton(step))
  upper_compressed = fill(true, num_blocks)
  lower_compressed = fill(true, num_blocks)
  WYWeight(
    decomp_ref,
    step_ref,
    leftWY,
    bbc,
    rightWY,
    upper_ranks,
    upper_compressed,
    lower_ranks,
    lower_compressed,
  )
end

"""
    WYWeight(
      ::Type{E},
      step::Union{Type{SpanStep}, Type{NullStep}},
      decomp::Union{Type{Nothing}, Type{LeadingDecomp}, Type{TrailingDecomp}},
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
  step::Union{Type{SpanStep}, Type{NullStep}},
  decomp::Union{Type{Nothing}, Type{LeadingDecomp}, Type{TrailingDecomp}},
  rng::AbstractRNG,
  m::Int,
  n::Int;
  upper_ranks::Union{Vector{Int}, Nothing}=nothing,
  lower_ranks::Union{Vector{Int}, Nothing}=nothing,
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
  
  num_blocks = size(upper_blocks,2)
  size(lower_blocks, 2) == num_blocks ||
    error("""In a WYWeight, the number of upper blocks should equal the number
             of lower blocks""")

  left_max_sizes, left_max_num_hs = get_WYWeight_max_transform_params(
    Left,
    (LeadingDecomp, TrailingDecomp),
    step,
    m,
    n,
    Sizes,
    Num_hs,
    upper_blocks = upper_blocks,
    upper_ranks = upper_rank_max,
    lower_blocks = lower_blocks,
    lower_ranks = lower_rank_max,
  )

  left_work_size = n * (2*left_max_sizes)

  leftWY = WYTrans(
    E,
    max_num_WY = num_blocks,
    max_WY_size = left_max_sizes,
    max_num_hs = left_max_num_hs,
    work_size = left_work_size
  )

  set_WYWeight_transform_params!(
    Left,
    decomp,
    step,
    m,
    n,
    lower_blocks = lower_blocks,
    lower_ranks = lower_rank_max,
    upper_blocks = upper_blocks,
    upper_ranks = upper_rank_max,
    num_hs = leftWY.num_hs,
    offsets = leftWY.offsets,
    sizes = leftWY.sizes,
  )

  rand!(rng, leftWY)

  right_max_sizes, right_max_num_hs = get_WYWeight_max_transform_params(
    Right,
    (LeadingDecomp, TrailingDecomp),
    step,
    m,
    n,
    Sizes,
    Num_hs,
    upper_blocks=upper_blocks,
    upper_ranks=upper_rank_max,
    lower_blocks=lower_blocks,
    lower_ranks=lower_rank_max,
  )

  right_work_size = m * (2*right_max_sizes)

  rightWY = WYTrans(
    E,
    max_num_WY = num_blocks,
    max_WY_size = right_max_sizes,
    max_num_hs = right_max_num_hs,
    work_size = right_work_size
  )

  set_WYWeight_transform_params!(
    Right,
    decomp,
    step,
    m,
    n,
    lower_blocks = lower_blocks,
    lower_ranks = lower_ranks,
    upper_blocks = upper_blocks,
    upper_ranks = upper_rank_max,
    num_hs = rightWY.num_hs,
    offsets = rightWY.offsets,
    sizes = rightWY.sizes
  )

  rand!(rng, rightWY)

  decomp_ref =
    Base.RefValue{Union{Nothing,LeadingDecomp,TrailingDecomp}}(singleton(decomp))
  step_ref = Base.RefValue{Union{Nothing,NullStep,SpanStep}}(singleton(step))
  upper_compressed = fill(true, num_blocks)
  lower_compressed = fill(true, num_blocks)

  WYWeight(
    decomp_ref,
    step_ref,
    leftWY,
    bbc,
    rightWY,
    upper_ranks,
    upper_compressed,
    lower_ranks,
    lower_compressed,
  )
end

# Iterators over blocks.

struct LowerCompressed{WY}
  wy::WY
end

function Base.length(lc::LowerCompressed{<:WYWeight})
  l = 1
  len = 0
  while l <= lc.wy.b.num_blocks
    lc.wy.lower_compressed[l] && (len += 1)
    l += 1
  end
  len
end

@inline function Base.length(
  rlc::Iterators.Reverse{<:LowerCompressed{<:WYWeight}},
)
  length(rlc.itr)
end

@inline function Base.iterate(lc::LowerCompressed{<:WYWeight})
  l = 1
  while (l <= lc.wy.b.num_blocks) && (!lc.wy.lower_compressed[l])
    l += 1
  end
  l > lc.wy.b.num_blocks ? nothing : (l, l + 1)
end

@inline function Base.iterate(
  rlc::Iterators.Reverse{<:LowerCompressed{<:WYWeight}},
)
  lc = rlc.itr
  l = lc.wy.b.num_blocks
  while (l >= 1) && (!lc.wy.lower_compressed[l])
    l -= 1
  end
  l < 1 ? nothing : (l, l - 1)
end

@inline function Base.iterate(lc::LowerCompressed{<:WYWeight}, l::Int)
  while (l <= lc.wy.b.num_blocks) && (!lc.wy.lower_compressed[l])
    l += 1
  end
  l > lc.wy.b.num_blocks ? nothing : (l, l + 1)
end

@inline function Base.iterate(
  rlc::Iterators.Reverse{<:LowerCompressed{<:WYWeight}},
  l::Int,
)
  while (l >= 1) && (!rlc.itr.wy.lower_compressed[l])
    l -= 1
  end
  l < 1 ? nothing : (l, l - 1)
end

struct UpperCompressed{WY}
  wy::WY
end

@inline function Base.length(lc::UpperCompressed{<:WYWeight})
  l = 1
  len = 0
  while l <= lc.wy.b.num_blocks
    lc.wy.upper_compressed[l] && (len += 1)
    l += 1
  end
  len
end

@inline function Base.length(
  rlc::Iterators.Reverse{<:UpperCompressed{<:WYWeight}},
)
  length(rlc.itr)
end

@inline function Base.iterate(lc::UpperCompressed{<:WYWeight})
  l = 1
  while (l <= lc.wy.b.num_blocks) && (!lc.wy.upper_compressed[l])
    l += 1
  end
  l > lc.wy.b.num_blocks ? nothing : (l, l + 1)
end

@inline function Base.iterate(
  rlc::Iterators.Reverse{<:UpperCompressed{<:WYWeight}},
)
  lc = rlc.itr
  l = lc.wy.b.num_blocks
  while (l >= 1) && (!lc.wy.upper_compressed[l])
    l -= 1
  end
  l < 1 ? nothing : (l, l - 1)
end

@inline function Base.iterate(lc::UpperCompressed{<:WYWeight}, l::Int)
  while (l <= lc.wy.b.num_blocks) && (!lc.wy.upper_compressed[l])
    l += 1
  end
  l > lc.wy.b.num_blocks ? nothing : (l, l + 1)
end

@inline function Base.iterate(
  rlc::Iterators.Reverse{<:UpperCompressed{<:WYWeight}},
  l::Int,
)
  while (l >= 1) && (!rlc.itr.wy.upper_compressed[l])
    l -= 1
  end
  l < 1 ? nothing : (l, l - 1)
end

# Set or increase xref[k] to y.
expand_or_set!(b, xref, k, y) = xref[k] = b ? max(xref[k], y) : y

# Increase xref[] to y as needed, to get a running maximum.
expand_or_set!(_, xref::Ref{Int}, _, y) = xref[] = max(xref[], y)

# Initialize with zero if not expanding.
maybe_zero(expand::Bool, r::Ref{Int}) = expand || (r[]=0)
maybe_zero(expand::Bool, r::Vector{Int}) = expand || (r.=0)
maybe_zero(::Bool, ::Nothing) = nothing
maybe_zero(exp::Bool, args...) = (x -> maybe_zero(exp, x)).(args)

# Pretend to index into a scalar, which is treated as a constant that
# doesn't depend on the index.
getindex_or_scalar(a, _) = a
getindex_or_scalar(a::AbstractArray, k) = a[k]

"""
    set_WYWeight_transform_params!(
      side::Union{Type{Left}, Type{Right}},
      decomp::UnionOrTuple{Type{LeadingDecomp},Type{TrailingDecomp}},
      step::UnionOrTuple{Type{SpanStep},Type{NullStep}},
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
  side::Union{Type{Left}, Type{Right}},
  decomp::UnionOrTuple{Type{LeadingDecomp},Type{TrailingDecomp}},
  step::UnionOrTuple{Type{SpanStep},Type{NullStep}},
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
  ::Type{Right},
  ::Type{LeadingDecomp},
  step::Union{Type{SpanStep},Type{NullStep}},
  m::Int,
  n::Int;
  lower_blocks::Union{AbstractArray{Int,2}, Nothing} = nothing,
  lower_ranks::Union{AbstractVector{Int}, Int, Nothing} = nothing,
  upper_blocks::Union{AbstractArray{Int,2}, Nothing} = nothing,
  upper_ranks::Union{AbstractVector{Int}, Int, Nothing} = nothing,
  sizes::Union{AbstractVector{Int}, Ref{Int}, Nothing}=nothing,
  num_hs::Union{AbstractVector{Int}, Ref{Int}, Nothing}=nothing,
  offsets::Union{AbstractVector{Int}, Nothing}=nothing,
  find_maximum::Bool=false
)
  num_blocks = size(lower_blocks, 2)
  lrank(k) = getindex_or_scalar(lower_ranks, k)

  maybe_zero(find_maximum, sizes, num_hs)

  # leading lower
  old_cols_lb = 1:0
  old_rank = 0
  for lb ∈ 1:num_blocks
    _, cols_lb = lower_block_ranges(lower_blocks, m, n, lb)
    dᵣ = setdiffᵣ(cols_lb, old_cols_lb)
    trange = dᵣ ∪ᵣ last(old_cols_lb, old_rank)

    tsize = length(trange)
    
    !isa(sizes, Nothing) &&
      expand_or_set!(find_maximum, sizes, lb, tsize)
    
    !isa(num_hs, Nothing) && expand_or_set!(
      find_maximum,
      num_hs,
      lb,
      step == SpanStep ? lrank(lb) : tsize - lrank(lb),
    )

    isa(offsets, AbstractArray{Int}) && (offsets[lb] = if !isempty(trange)
      first(trange) - 1
    elseif !isempty(cols_lb)
      last(cols_lb)
    else
      0
    end)

    old_cols_lb = cols_lb
    old_rank = lrank(lb)
  end
  nothing
end

function set_WYWeight_transform_params!(
  ::Type{Left},
  ::Type{LeadingDecomp},
  step::Union{Type{SpanStep},Type{NullStep}},
  m::Int,
  n::Int;
  lower_blocks::Union{AbstractArray{Int,2}, Nothing} = nothing,
  lower_ranks::Union{AbstractVector{Int}, Int, Nothing} = nothing,
  upper_blocks::Union{AbstractArray{Int,2}, Nothing} = nothing,
  upper_ranks::Union{AbstractVector{Int}, Int, Nothing} = nothing,
  sizes::Union{AbstractVector{Int}, Ref{Int}, Nothing}=nothing,
  num_hs::Union{AbstractVector{Int}, Ref{Int}, Nothing}=nothing,
  offsets::Union{AbstractVector{Int}, Nothing}=nothing,
  find_maximum::Bool=false
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

    !isa(sizes, Nothing) &&
      expand_or_set!(find_maximum, sizes, ub, tsize)

    !isa(num_hs, Nothing) && expand_or_set!(
      find_maximum,
      num_hs,
      ub,
      step == SpanStep ? urank(ub) : tsize - urank(ub)
    )

    isa(offsets, AbstractArray{Int}) && (offsets[ub] = if !isempty(trange)
      first(trange) - 1
    elseif !isempty(rows_ub)
      last(rows_ub)
    else
      0
    end)

    old_rows_ub = rows_ub
    old_rank = urank(ub)
  end

  nothing
end

function set_WYWeight_transform_params!(
  ::Type{Right},
  ::Type{TrailingDecomp},
  step::Union{Type{SpanStep},Type{NullStep}},
  m::Int,
  n::Int;
  lower_blocks::Union{AbstractArray{Int,2}, Nothing} = nothing,
  lower_ranks::Union{AbstractVector{Int}, Int, Nothing} = nothing,
  upper_blocks::Union{AbstractArray{Int,2}, Nothing} = nothing,
  upper_ranks::Union{AbstractVector{Int}, Int, Nothing} = nothing,
  sizes::Union{AbstractVector{Int}, Ref{Int}, Nothing}=nothing,
  num_hs::Union{AbstractVector{Int}, Ref{Int}, Nothing}=nothing,
  offsets::Union{AbstractVector{Int}, Nothing}=nothing,
  find_maximum::Bool=false
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

    !isa(sizes, Nothing) &&
      expand_or_set!(find_maximum, sizes, ub, tsize)
    
    !isa(num_hs, Nothing) && expand_or_set!(
      find_maximum,
      num_hs,
      ub,
      step == SpanStep ? urank(ub) : tsize - urank(ub)
    )

    isa(offsets, AbstractArray{Int}) && (offsets[ub] = if !isempty(trange)
      first(trange) - 1
    elseif !isempty(cols_ub)
      last(cols_ub)
    else
      0
    end)

    old_cols_ub = cols_ub
    old_rank = urank(ub)
  end

  nothing
end

function set_WYWeight_transform_params!(
  ::Type{Left},
  ::Type{TrailingDecomp},
  step::Union{Type{SpanStep},Type{NullStep}},
  m::Int,
  n::Int;
  lower_blocks::Union{AbstractArray{Int,2}, Nothing} = nothing,
  lower_ranks::Union{AbstractVector{Int}, Int, Nothing} = nothing,
  upper_blocks::Union{AbstractArray{Int,2}, Nothing} = nothing,
  upper_ranks::Union{AbstractVector{Int}, Int, Nothing} = nothing,
  sizes::Union{AbstractVector{Int}, Ref{Int}, Nothing}=nothing,
  num_hs::Union{AbstractVector{Int}, Ref{Int}, Nothing}=nothing,
  offsets::Union{AbstractVector{Int}, Nothing}=nothing,
  find_maximum::Bool=false
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

    !isa(sizes, Nothing) &&
      expand_or_set!(find_maximum, sizes, lb, tsize)
    
    !isa(num_hs, Nothing) && expand_or_set!(
      find_maximum,
      num_hs,
      lb,
      step == SpanStep ? lrank(lb) : tsize - lrank(lb)
    )

    isa(offsets, AbstractArray{Int}) && (offsets[lb] = if !isempty(trange)
      first(trange) - 1
    elseif !isempty(rows_lb)
      last(rows_lb)
    else
      0
    end)

    old_rows_lb = rows_lb
    old_rank = lrank(lb)
  end
  nothing
end

"""
    get_WYWeight_transform_params(
      side::Union{Type{Left},Type{Right}},
      decomp::UnionOrTuple{<:Type{TrailingDecomp},<:Type{TrailingDecomp}},
      step::UnionOrTuple{<:Type{SpanStep},<:Type{NullStep}},
      m::Int,
      n::Int,
      params::Vararg{Union{Type{Sizes},Type{Num_hs},Type{Offsets}}};
      lower_blocks::Union{AbstractArray{Int,2},Nothing} = nothing,
      lower_ranks::Union{AbstractVector{Int},Int,Nothing} = nothing,
      upper_blocks::Union{AbstractArray{Int,2},Nothing} = nothing,
      upper_ranks::Union{AbstractVector{Int},Int,Nothing} = nothing,
    )

Compute `sizes`, `num_hs`, `offsets` values, depending on which
selectors are provided as `Vararg` parameters.
"""
function get_WYWeight_transform_params(
  side::Union{Type{Left},Type{Right}},
  decomp::UnionOrTuple{<:Type{TrailingDecomp},<:Type{TrailingDecomp}},
  step::UnionOrTuple{<:Type{SpanStep},<:Type{NullStep}},
  m::Int,
  n::Int,
  params::Vararg{Union{Type{Sizes},Type{Num_hs},Type{Offsets}}};
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
    p == Sizes && (sizes = zeros(Int, num_blocks))
    p == Num_hs && (num_hs = zeros(Int, num_blocks))
    p == Offsets && (offsets = zeros(Int, num_blocks))
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
    p == Sizes && push!(result, sizes)
    p == Num_hs && push!(result, num_hs)
    p == Offsets && push!(result, offsets)
  end
  length(result) == 1 ? result[1] : tuple(result...)

end
"""
    get_WYWeight_max_transform_params(
      side::Union{Type{Left},Type{Right}},
      decomp::UnionOrTuple{<:Type{TrailingDecomp},<:Type{TrailingDecomp}},
      step::UnionOrTuple{<:Type{SpanStep},<:Type{NullStep}},
      m::Int,
      n::Int,
      params::Vararg{Union{Type{Sizes},Type{Num_hs}}};
      lower_blocks::Union{AbstractArray{Int,2},Nothing} = nothing,
      lower_ranks::Union{AbstractVector{Int},Int,Nothing} = nothing,
      upper_blocks::Union{AbstractArray{Int,2},Nothing} = nothing,
      upper_ranks::Union{AbstractVector{Int},Int,Nothing} = nothing,
    )

Compute requested maxima for `sizes` and `num_hs` for the type(s) of
decompositions/transforms specified.
"""
function get_WYWeight_max_transform_params(
  side::Union{Type{Left},Type{Right}},
  decomp::UnionOrTuple{<:Type{TrailingDecomp},<:Type{TrailingDecomp}},
  step::UnionOrTuple{<:Type{SpanStep},<:Type{NullStep}},
  m::Int,
  n::Int,
  params::Vararg{Union{Type{Sizes},Type{Num_hs}}};
  lower_blocks::Union{AbstractArray{Int,2},Nothing} = nothing,
  lower_ranks::Union{AbstractVector{Int},Int,Nothing} = nothing,
  upper_blocks::Union{AbstractArray{Int,2},Nothing} = nothing,
  upper_ranks::Union{AbstractVector{Int},Int,Nothing} = nothing,
)

  sizes = nothing
  num_hs = nothing

  for p ∈ params
    p == Sizes && (sizes = Ref(0))
    p == Num_hs && (num_hs = Ref(0))
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
    p == Sizes && push!(result, sizes[])
    p == Num_hs && push!(result, num_hs[])
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
  lwy = wyw.leftWY
  rwy = wyw.rightWY
  @views for l ∈ Iterators.reverse(LowerCompressed(wyw))
    rows, _ = lower_block_ranges(bbc, l)
    apply_inv!(a[rows, :], (rwy, l))
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
  lwy = wyw.leftWY
  rwy = wyw.rightWY
  @views for l ∈ LowerCompressed(wyw)
    _, cols = lower_block_ranges(bbc, l)
    apply!((lwy, l), a[:, cols])
  end
  @views for l ∈ UpperCompressed(wyw)
    rows, _ = upper_block_ranges(bbc, l)
    apply_inv!(a[rows, :], (rwy, l))
  end
  a
end

end
