module BandwidthInit
using Printf
using Random
using Match
using ErrorTypes

using ..BandColumnMatrices
using ..IndexLists
using InPlace

export AbstractBlockData,
  BlockSize,
  block_sizes,
  leading_lower_ranks_to_cols_first_last!,
  leading_upper_ranks_to_cols_first_last!,
  trailing_lower_ranks_to_cols_first_last!,
  trailing_upper_ranks_to_cols_first_last!,
  constrain_upper_ranks!,
  constrain_upper_ranks,
  constrain_lower_ranks!,
  constrain_lower_ranks,
  lower_block_ranges,
  upper_block_ranges,
  size_lower_block,
  size_upper_block,
  NonUnitRangeError,
  setdiffᵣ,
  ∪ᵣ,
  get_cols_first_last,
  get_cols_first_last!,
  get_rows_first_last,
  get_rows_first_last!,
  leading_block_ranges,
  trailing_block_ranges,
  forward_last_trailing_including_index,
  forward_first_leading_including_index,
  backward_last_trailing_including_index,
  backward_first_leading_including_index

abstract type AbstractBlockData end

struct BlockSize <: AbstractBlockData
  mb::Int
  nb::Int
end

BlockSize(; mb=0, nb=0) = BlockSize(mb, nb)

function block_sizes(a::AbstractMatrix; bd=BlockSize)
  [bd(;mb=a[1, j], nb=a[2, j]) for j ∈ axes(a, 2)]
end


Base.iterate(bd::BlockSize) = (bd.mb, 1)
Base.iterate(bd::BlockSize, i) =
  @match i begin
    1 => (bd.nb, 2)
    2 => nothing
  end

"""

# AbstractBlockedBandColumn

    AbstractBlockedBandColumn{E,AE,AI} <:
      AbstractBandColumn{NonSub,E,AE,AI}

An AbstractBlockedBandColumn should implement the following:

  get_lower_blocks

"""
abstract type AbstractBlockedBandColumn{E,AE,AI} <:
              AbstractBandColumn{NonSub,E,AE,AI} end

struct NonUnitRangeError <: Exception end

"""
    function setdiffᵣ(xs::AbstractUnitRange{Int}, ys::AbstractUnitRange{Int})

A set difference function that returns unit ranges or an error if the
result cannot be represented as a unit range.
"""
function setdiffᵣ(xs::AbstractUnitRange{Int}, ys::AbstractUnitRange{Int})
  x0 = first(xs)
  x1 = last(xs)
  y0 = first(ys)
  y1 = last(ys)
  if isempty(ys)
    xs
  elseif y0 == x0 
    y1+1:x1
  elseif y1 == x1
    x0:y0-1
  else
    throw(NonUnitRangeError)
  end
end

"""
    function ∪ᵣ(xs::AbstractUnitRange{Int}, ys::AbstractUnitRange{Int})

A union that returns unit ranges or an error if the result cannot be
represented as a unit range.
"""
function ∪ᵣ(xs::AbstractUnitRange{Int}, ys::AbstractUnitRange{Int})
  x0 = first(xs)
  x1 = last(xs)
  y0 = first(ys)
  y1 = last(ys)
  if isempty(xs)
    ys
  elseif isempty(ys)
    xs
  elseif x1 ∈ ys || x1 == y0 - 1
    x0:y1
  elseif x0 ∈ ys || x0 == y1 + 1
    y0:x1
  else
    throw(NonUnitRangeError)
  end
end

"""
    get_cols_first_last(
      m::Int,
      n::Int,
      upper_blocks::IndexList{BD},
      lower_blocks::IndexList{BD},
      r_upper::Int,
      r_lower::Int
    ) where {BD<:AbstractBlockData}

Compute cols_first_last.  This works for either a leading or trailing
decomposition and provides enough extra bandwidth for conversion
between them.

"""
function get_cols_first_last(
  m::Int,
  n::Int,
  upper_blocks::IndexList{BD},
  lower_blocks::IndexList{BD},
  r_upper::Int,
  r_lower::Int
) where {BD<:AbstractBlockData}
  cols_first_last = Array{Int,2}(undef, 6, n)
  get_cols_first_last!(
    m=m,
    n=n,
    upper_blocks=upper_blocks,
    lower_blocks=lower_blocks,
    max_ru = r_upper,
    max_rl = r_lower,
    cols_first_last = cols_first_last,
  )
  return cols_first_last
end

# Compute rows_first_last.  This works for either a leading or
# trailing decomposition and provides enough extra bandwidth for
# conversion between them.
function get_rows_first_last(
  m::Int,
  n::Int,
  upper_blocks::IndexList{BD},
  lower_blocks::IndexList{BD},
  r_upper::Int,
  r_lower::Int
) where {BD<:AbstractBlockData}
  rows_first_last = zeros(Int, m, 6)
  get_rows_first_last!(
    m=m,
    n=n,
    upper_blocks=upper_blocks,
    lower_blocks=lower_blocks,
    max_ru=r_upper,
    max_rl=r_lower,
    rows_first_last=rows_first_last,
  )
  return rows_first_last
end

function leading_block_ranges(
  blocks,
  m::Int,
  n::Int,
  l::Union{Before,After,ListIndex},
)
  @match l begin
    Before() => (1:0, 1:0)
    After() => (1:m, 1:n)
    _ => (1:(blocks[l].mb), 1:(blocks[l].nb))
  end
end


function trailing_block_ranges(
  blocks,
  m::Int,
  n::Int,
  l::Union{Before,After,ListIndex},
)
  @match l begin
    Before() => (1:m, 1:n)
    After() => ((m + 1):m, (n + 1):n)
    _ => ((blocks[l].mb + 1):m, (blocks[l].nb + 1):n)
  end
end

# for upper in get_cols_first_last.
function forward_last_trailing_including_index(blocks, i, axis, m, n, b)
  isempty(blocks) && return b
  while true
    # An error in next_list_index means already on the last block.
    next_b = b == Before() ? first_list_index(blocks) :
      (@unwrap_or next_list_index(blocks, b) (return b))
    trange = trailing_block_ranges(blocks, m, n, next_b)[axis]
    i ∈ trange || return b
    # Can advance one or more blocks.  Don't need to worry about
    # going past the last block and returning After() since all
    # columns should be in the trailing column set for the last
    # block.
    b = next_b
  end
end

# for lower in get_cols_first_last; starts with After()
function backward_first_leading_including_index(blocks, i, axis, m, n, b)
  isempty(blocks) && return b
  while true
    prev_b = b == After() ? last_list_index(blocks) :
      (@unwrap_or prev_list_index(blocks, b) (return b))
    lrange = leading_block_ranges(blocks, m, n, prev_b)[axis]
    i ∈ lrange || return b
    b = prev_b
  end
end

# for lower in get_cols_first_last.
function forward_first_leading_including_index(blocks, i, axis, m, n, b)
  isempty(blocks) && return b
  while true
    lrange_b = leading_block_ranges(blocks, m, n, b)[axis]
    i ∈ lrange_b && return b # this b is still the one providing the
                             # constraint.  
    # error means already on the last block.
    next_b = b == Before() ? first_list_index(blocks) :
      @unwrap_or next_list_index(blocks, b) (return After())
    lrange_next_b = leading_block_ranges(blocks, m, n, next_b)[axis]
    i ∈ lrange_next_b && return next_b
    b = next_b
  end
end

# for upper in get_cols_first_last; Starts with After()
function backward_last_trailing_including_index(blocks, i, axis, m, n, b)
  isempty(blocks) && return b
  while true
    trange_b = trailing_block_ranges(blocks, m, n, b)[axis]
    i ∈ trange_b && return b # this b is still the one providing the
                             # constraint.
    prev_b = b == After() ? last_list_index(blocks) :
      @unwrap_or prev_list_index(blocks, b) (return Before())
    trange_prev_b = trailing_block_ranges(blocks, m, n, prev_b)[axis]
    i ∈ trange_prev_b && return prev_b
    b = prev_b
  end
end


function get_cols_first_last!(;
  m::Int,
  n::Int,
  lower_blocks::IndexList{BD},
  max_rl::Int,
  max_drop_l::Int=0,
  upper_blocks::IndexList{BD},
  max_ru::Int,
  max_drop_u::Int=0,
  cols_first_last::AbstractMatrix{Int}
) where {BD<:AbstractBlockData}

  function extend_up_to(first::Int, range::UnitRange)
    @views cols_first_last[1, range] .=
      (x -> min(x, first)).(cols_first_last[1, range])
    return nothing
  end

  function extend_down_to(last::Int, range::UnitRange)
    @views cols_first_last[6, range] .=
      (x -> max(x, last)).(cols_first_last[6, range])
    return nothing
  end

  # For a given column k, the middle indices are the
  # intersection of:
  #
  # 1. the trailing rows of ub where ub is the last block for which k
  # is in the trailing columns of ub.
  #
  # 2. the leading rows of lb where lb is the first block lb for which
  # k is in the leading columns of lb.
  #
  # These set the last upper and first lower indices for each column.
  # The first and last storable are set by reproducing what would fill
  # in curing conversion between leading and trailing with the
  # specified rank bounds.

  # If there are no lower blocks, then there are no constraints imposed by
  # the leading blocks, so the leading block can be the entire matrix.
  # Otherwise start with a 0×0 leading block.
  lb = isempty(lower_blocks) ? After() : Before()

  # For upper_blocks start with a 0×0 leading block in either case.
  ub = Before()

  rows_llb, cols_llb = leading_block_ranges(lower_blocks, m, n, lb)
  rows_ltb, cols_ltb = trailing_block_ranges(lower_blocks, m, n, lb)
  rows_ulb, cols_ulb = leading_block_ranges(upper_blocks, m, n, ub)
  rows_utb, cols_utb = trailing_block_ranges(upper_blocks, m, n, ub)
  Δm_us = Int[]
  Δm_ls = Int[]

  cols_first_last .= 0
  # Zero is not suitable for first storable, since it needs to start
  # at something that can be extended up.
  cols_first_last[1, :] .= m + 1

  for k ∈ 1:n
    ub_tmp = ub
    ub = forward_last_trailing_including_index(upper_blocks, k, 2, m, n, ub)
    if ub_tmp != ub
      # ub changed.
      old_rows_ulb = rows_ulb
      rows_ulb, cols_ulb = leading_block_ranges(upper_blocks, m, n, ub)
      rows_utb, cols_utb = trailing_block_ranges(upper_blocks, m, n, ub)
      Δm_u = length(setdiffᵣ(rows_ulb, old_rows_ulb))
      push!(Δm_us, Δm_u)
      length(Δm_us) > max_drop_u + 1 && popfirst!(Δm_us)
      uncompressed_rows = last(rows_ulb, sum(Δm_us) + max_ru)
      extend_up_to(first(uncompressed_rows), first(cols_utb, max_ru))
    end

    lb_tmp = lb
    lb = forward_first_leading_including_index(lower_blocks, k, 2, m, n, lb)
    if lb_tmp != lb
      # lb changed.
      old_rows_llb = rows_llb
      rows_llb, cols_llb = leading_block_ranges(lower_blocks, m, n, lb)
      rows_ltb, cols_ltb = trailing_block_ranges(lower_blocks, m, n, lb)
      Δm_l = length(setdiffᵣ(rows_llb, old_rows_llb))
      push!(Δm_ls, Δm_l)
      length(Δm_ls) > max_drop_l + 1 && popfirst!(Δm_ls)
    end

    cols_first_last[2, k] = first(rows_utb) # first middle/first inband
    cols_first_last[3, k] = last(rows_ulb) # last upper
    cols_first_last[4, k] = first(rows_ltb) # first lower
    cols_first_last[5, k] = last(rows_llb) # last middle/last inband

    # first storable
    uncompressed_rows = last(rows_ulb, sum(Δm_us) + max_ru)
    compressed_rows = max_drop_u > 0 ? uncompressed_rows : last(rows_ulb, max_ru)
    storable_rows_u =
      k ∈ first(cols_utb, max_ru) ? uncompressed_rows : compressed_rows
    extend_up_to(first(storable_rows_u), k:k)
  end

  # For last storable, we traverse the columns in the opposite direction.
  lb = After()
  rows_llb, cols_llb = leading_block_ranges(lower_blocks, m, n, lb)
  rows_ltb, cols_ltb = trailing_block_ranges(lower_blocks, m, n, lb)
  Δm_ls = Int[]
  for k ∈ n:-1:1
    lb_tmp = lb
    lb = backward_first_leading_including_index(lower_blocks, k, 2, m, n, lb)
    if lb_tmp != lb
      # lb changed.
      old_rows_llb = rows_llb
      rows_llb, cols_llb = leading_block_ranges(lower_blocks, m, n, lb)
      rows_ltb, cols_ltb = trailing_block_ranges(lower_blocks, m, n, lb)
      Δm_l = length(setdiffᵣ(old_rows_llb, rows_llb))
      pushfirst!(Δm_ls, Δm_l)
      length(Δm_ls) > max_drop_l + 1 && pop!(Δm_ls)
      uncompressed_rows = first(rows_ltb, sum(Δm_ls) + max_rl)
      extend_down_to(last(uncompressed_rows), last(cols_llb, max_rl))
    end
    # last storable
    uncompressed_rows = first(rows_ltb, sum(Δm_ls) + max_rl)
    compressed_rows = max_drop_l > 0 ? uncompressed_rows : first(rows_ltb, max_rl)
    storable_rows_l =
      k ∈ last(cols_llb, max_rl) ? uncompressed_rows : compressed_rows
    extend_down_to(last(storable_rows_l), k:k)
  end
  return nothing
end

function get_rows_first_last!(;
  m::Int,
  n::Int,
  lower_blocks::IndexList{BD},
  max_rl::Int,
  max_drop_l::Int=0,
  upper_blocks::IndexList{BD},
  max_ru::Int,
  max_drop_u::Int=0,
  rows_first_last::AbstractMatrix{Int}
) where {BD<:AbstractBlockData}

  function extend_left_to(first::Int, range::UnitRange)
    @views rows_first_last[range, 1] .=
      (x -> min(x, first)).(rows_first_last[range, 1])
    return nothing
  end

  function extend_right_to(last::Int, range::UnitRange)
    @views rows_first_last[range, 6] .=
      (x -> max(x, last)).(rows_first_last[range, 6])
    return nothing
  end

  # For a given row j, the middle indices are the intersection of:
  #
  # 1. the trailing columns of lb where lb is the last block for which
  # j is in the trailing rows of lb.
  #
  # 2. the leading columns of ub where ub is the first block ub for which
  # j is in the leading rows of ub.
  #
  # These set the last upper and first lower indices for each row.
  # The first and last storable are set by reproducing what would fill
  # in curing conversion between leading and trailing with the
  # specified rank bounds.

  # If there are no upper blocks, then there are no constraints imposed by
  # the leading blocks, so the leading block can be the entire matrix.
  # Otherwise start with a 0×0 leading block.
  ub = isempty(upper_blocks) ? After() : Before()

  # For lower_blocks start with a 0×0 leading block in either case.
  lb = Before()

  rows_llb, cols_llb = leading_block_ranges(lower_blocks, m, n, lb)
  rows_ltb, cols_ltb = trailing_block_ranges(lower_blocks, m, n, lb)
  rows_ulb, cols_ulb = leading_block_ranges(upper_blocks, m, n, ub)
  rows_utb, cols_utb = trailing_block_ranges(upper_blocks, m, n, ub)
  Δn_us = Int[] 
  Δn_ls = Int[]

  rows_first_last .= 0
  # Zero is not suitable for first storable, since it needs to start
  # at something that can be extended left.
  rows_first_last[:, 1] .= n + 1

  for j ∈ 1:m
    ub_tmp = ub
    ub = forward_first_leading_including_index(upper_blocks, j, 1, m, n, ub)
    if ub_tmp != ub
      # ub changed.
      old_cols_utb = cols_utb
      rows_ulb, cols_ulb = leading_block_ranges(upper_blocks, m, n, ub)
      rows_utb, cols_utb = trailing_block_ranges(upper_blocks, m, n, ub)
      Δn_u = length(setdiffᵣ(old_cols_utb, cols_utb))
      push!(Δn_us, Δn_u)
      length(Δn_us) > max_drop_u + 1 && popfirst!(Δn_us)
    end

    lb_tmp = lb
    lb = forward_last_trailing_including_index(lower_blocks, j, 1, m, n, lb)
    if lb_tmp != lb
      # lb changed.
      old_cols_llb = cols_llb
      rows_llb, cols_llb = leading_block_ranges(lower_blocks, m, n, lb)
      rows_ltb, cols_ltb = trailing_block_ranges(lower_blocks, m, n, lb)
      Δn_l = length(setdiffᵣ(cols_llb, old_cols_llb))
      push!(Δn_ls, Δn_l)
      length(Δn_ls) > max_drop_l + 1 && popfirst!(Δn_ls)
      uncompressed_cols = last(cols_llb, sum(Δn_ls) + max_rl)
      extend_left_to(first(uncompressed_cols), first(rows_ltb, max_rl))
    end

    rows_first_last[j, 2] = first(cols_ltb) # first middle/first inband
    rows_first_last[j, 3] = last(cols_llb) # last lower
    rows_first_last[j, 4] = first(cols_utb) # first upper
    rows_first_last[j, 5] = last(cols_ulb) # last middle/last inband

    # first storable
    uncompressed_cols = last(cols_llb, sum(Δn_ls) + max_rl)
    compressed_cols = max_drop_l > 0 ? uncompressed_cols : last(cols_llb, max_rl)
    storable_cols_l =
      j ∈ first(cols_ltb, max_rl) ? uncompressed_cols : compressed_cols
    extend_left_to(first(storable_cols_l), j:j)
  end

  # For last storable, we traverse the rows in the opposite direction.
  ub = After()
  rows_ulb, cols_ulb = leading_block_ranges(upper_blocks, m, n, ub)
  rows_utb, cols_utb = trailing_block_ranges(upper_blocks, m, n, ub)
  Δn_us = Int[]
  for j ∈ m:-1:1

    ub_tmp = ub
    ub = backward_first_leading_including_index(upper_blocks, j, 1, m, n, ub)
    if ub_tmp != ub
      # ub changed.
      old_cols_utb = cols_utb
      rows_ulb, cols_ulb = leading_block_ranges(upper_blocks, m, n, ub)
      rows_utb, cols_utb = trailing_block_ranges(upper_blocks, m, n, ub)
      Δn_u = length(setdiffᵣ(cols_utb, old_cols_utb))
      pushfirst!(Δn_us, Δn_u)
      length(Δn_us) > max_drop_u + 1 && pop!(Δn_us)
      uncompressed_cols = first(cols_utb, sum(Δn_us) + max_ru)
      extend_right_to(last(uncompressed_cols), last(rows_ulb, max_ru))
    end
    # last storable
    uncompressed_cols = first(cols_utb, sum(Δn_us) + max_ru)
    compressed_cols = max_drop_u > 0 ? uncompressed_cols : first(cols_utb, max_ru)
    storable_cols_u =
      j ∈ last(rows_ulb, max_ru) ? uncompressed_cols : compressed_cols
    extend_right_to(last(storable_cols_u), j:j)
  end

  return nothing
end

"""
    lower_block_ranges(
      lower_blocks::IndexList{BD},
      m :: Int,
      n :: Int,
      l::Int
    ) where {BD<:AbstractBlockData}

For lower blocks and a given matrix size m×n, compute ranges for lower
block ``l``.
"""
@inline function lower_block_ranges(
  lower_blocks::IndexList{BD},
  m :: Int,
  n :: Int,
  l :: Int,
) where {BD<:AbstractBlockData}
  is_sorted(lower_blocks) || throw(SortedError(lower_blocks, l))
  if l < 1
    (UnitRange(1,m), UnitRange(1,0))
  elseif l > length(lower_blocks)
    (UnitRange(m+1,m), UnitRange(1,n))
  else
    lower_block_ranges(lower_blocks, m, n, ListIndex(l))
  end
end

@inline function lower_block_ranges(
  lower_blocks::IndexList{BD},
  m::Int,
  n::Int,
  le::Result{ListIndex,BeforeAfterError},
) where {BD<:AbstractBlockData}

  err = @unwrap_error_or le (
    return lower_block_ranges(lower_blocks, m, n, unwrap(le))
  )
  if err == BeforeAfterError(Before())
    (UnitRange(1, m), UnitRange(1, 0))
  elseif err == BeforeAfterError(After())
    (UnitRange(m+1,m), UnitRange(1,n))
  else
    throw(err)
  end
end

@inline function lower_block_ranges(
  lower_blocks::IndexList{BD},
  m :: Int,
  :: Int,
  l::ListIndex,
  ) where {BD<:AbstractBlockData}
  j_first = lower_blocks[l].mb + 1
  k_last = lower_blocks[l].nb
  return (UnitRange(j_first, m), UnitRange(1, k_last))
end

"""
    size_lower_block(
      lower_blocks::IndexList{BD},
      m::Int,
      n::Int,
      l::Int,
    ) where {BD<:AbstractBlockData}
  
Compute the size of lower block ``l`` for an m×n matrix using the
lower_block sequence `lower_blocks`.
"""
@inline function size_lower_block(
  lower_blocks::IndexList{BD},
  m::Int,
  n::Int,
  l::Union{Int,ListIndex,Result{ListIndex,BeforeAfterError}}
) where {BD<:AbstractBlockData}
  (rows, cols) = lower_block_ranges(lower_blocks, m, n, l)
  return (last(rows) - first(rows) + 1, last(cols) - first(cols) + 1)
end

"""
    upper_block_ranges(
      upper_blocks::IndexList{BD},
      m :: Int,
      n :: Int,
      l::Int
    ) where {BD<:AbstractBlockData}

For upper blocks and a given matrix size m×n, compute ranges for upper
block ``l``.
"""
function upper_block_ranges(
  upper_blocks::IndexList{BD},
  m::Int,
  n::Int,
  l::Int,
) where {BD<:AbstractBlockData}
  is_sorted(upper_blocks) || throw(SortedError(upper_blocks, l))
  if l < 1
    (UnitRange(1, 0), UnitRange(1, n))
  elseif l > length(upper_blocks)
    (UnitRange(1, m), UnitRange((n + 1):n))
  else
    upper_block_ranges(upper_blocks, m, n, ListIndex(l))
  end
end

function upper_block_ranges(
  upper_blocks::IndexList{BD},
  m::Int,
  n::Int,
  le::Result{ListIndex,BeforeAfterError},
) where {BD<:AbstractBlockData}

  err = @unwrap_error_or le (
    return @inline upper_block_ranges(upper_blocks, m, n, unwrap(le))
  )
  if err == BeforeAfterError(Before())
    (UnitRange(1, 0), UnitRange(1, n))
  elseif err == BeforeAfterError(After())
    (UnitRange(1, m), UnitRange((n + 1):n))
  else
    throw(err)
  end
end

function upper_block_ranges(
  upper_blocks::IndexList{BD},
  :: Int,
  n:: Int,
  l::ListIndex,
  ) where {BD<:AbstractBlockData}
  j_last = upper_blocks[l].mb
  k_first = upper_blocks[l].nb + 1
  return (UnitRange(1, j_last), UnitRange(k_first, n))
end


"""
    size_upper_block(
      upper_blocks::IndexList{BD},
      m::Int,
      n::Int,
      l::Int,
    ) where {BD<:AbstractBlockData}
  
Compute the size of upper block ``l`` for an m×n matrix using the
upper_block sequence `upper_blocks`.
"""
@inline function size_upper_block(
  upper_blocks::IndexList{BD},
  m::Int,
  n::Int,
  l::Union{Int,ListIndex,Result{ListIndex,BeforeAfterError}}
) where {BD<:AbstractBlockData}
  (rows, cols) = upper_block_ranges(upper_blocks, m, n, l)
  return (last(rows) - first(rows) + 1, last(cols) - first(cols) + 1)
end

"""
    function constrain_upper_ranks!(
      mA::Int,
      nA::Int;
      blocks::IndexList{BD},
      ranks::AbstractVector{Int},
    ) where {BD<:AbstractBlockData}

Take a nominal upper rank sequence and constrain it to be minimal and
achievable, i.e. so that any set of upper blocks that had the old
ranks as upper bound is bounded by the new ranks.
"""
function constrain_upper_ranks!(
  ::Int,
  nA::Int;
  blocks::IndexList{BD},
  ranks::AbstractVector{Int},
) where {BD<:AbstractBlockData}

  m(k) = blocks[k].mb
  n(k) = nA - blocks[k].nb
  Δm(k) = m(k) - m(k-1)
  Δn(k) = n(k-1) - n(k)
  p = length(blocks)
  ranks[1] = min(m(1), n(1), ranks[1])
  for k ∈ 2:p
    ranks[k] = min(ranks[k], m(k), n(k), ranks[k-1] + Δm(k))
  end
  for k ∈ p-1:-1:1
    ranks[k] = min(ranks[k], ranks[k+1] + Δn(k+1))
  end
  return nothing
end

function constrain_upper_ranks(
  mA::Int,
  nA::Int;
  blocks::IndexList{BD},
  ranks::AbstractVector{Int},
) where {BD<:AbstractBlockData}
  rs = copy(ranks)
  constrain_upper_ranks!(mA, nA, blocks = blocks, ranks = rs)
  return rs
end

"""
    function constrain_lower_ranks!(
      mA::Int,
      nA::Int;
      blocks::IndexList{BD},
      ranks::AbstractVector{Int},
    ) where {BD<:AbstractBlockData}

Take a nominal lower rank sequence and constrain it to be minimal and
achievable, i.e. so that any set of lower blocks that had the old
ranks as upper bound is bounded by the new ranks.
"""
function constrain_lower_ranks!(
  mA::Int,
  ::Int;
  blocks::IndexList{BD},
  ranks::AbstractVector{Int},
) where {BD<:AbstractBlockData}

  m(k) = mA - blocks[k].mb
  n(k) = blocks[k].nb
  Δm(k) = m(k-1) - m(k)
  Δn(k) = n(k) - n(k-1)
  p = length(blocks)
  ranks[1] = min(m(1), n(1), ranks[1])
  for k ∈ 2:p
    ranks[k] = min(ranks[k], m(k), n(k), ranks[k-1] + Δn(k))
  end
  for k ∈ p-1:-1:1
    ranks[k] = min(ranks[k], ranks[k+1] + Δm(k+1))
  end
  return nothing
end

function constrain_lower_ranks(
  mA::Int,
  nA::Int;
  blocks::IndexList{BD},
  ranks::AbstractVector{Int},
) where {BD<:AbstractBlockData}
  rs = copy(ranks)
  constrain_lower_ranks!(mA, nA, blocks = blocks, ranks = rs)
  return rs
end

"""
    leading_lower_ranks_to_cols_first_last!(
      lower_blocks::IndexList{BD},
      m::Int,
      n::Int,
      cols_first_last::AbstractMatrix{Int},
      rs::AbstractVector{Int},
    ) where {BD<:AbstractBlockData}

Set first_last indices appropriate for a leading decomposition
associated with a given lower rank sequence.
"""
function leading_lower_ranks_to_cols_first_last!(
  lower_blocks::IndexList{BD},
  m::Int,
  n::Int,
  cols_first_last::AbstractMatrix{Int},
  rs::AbstractVector{Int},
) where {BD<:AbstractBlockData}
  num_blocks = length(lower_blocks)
  rs1 = constrain_lower_ranks(m, n, blocks = lower_blocks, ranks = rs)
  for lb = 1:num_blocks
    rows_lb, cols_lb = lower_block_ranges(lower_blocks, m, n, lb)
    # empty if lb+1 > num_blocks
    rows_lb1, _ = lower_block_ranges(lower_blocks, m, n, lb + 1)
    dᵣ = setdiffᵣ(rows_lb, rows_lb1)
    if !isempty(dᵣ)
      cols_first_last[5, last(cols_lb, rs1[lb])] .= last(dᵣ)
    end
  end
  return nothing
end

"""
    trailing_lower_ranks_to_cols_first_last!(
      lower_blocks::IndexList{BD},
      m::Int,
      n::Int,
      cols_first_last::AbstractMatrix{Int},
      rs::AbstractVector{Int},
    ) where {BD<:AbstractBlockData}

Set first_last indices appropriate for a trailing decomposition
associated with a given lower rank sequence.
"""
function trailing_lower_ranks_to_cols_first_last!(
  lower_blocks::IndexList{BD},
  m::Int,
  n::Int,
  cols_first_last::AbstractMatrix{Int},
  rs::AbstractVector{Int},
) where {BD<:AbstractBlockData}
  num_blocks = length(lower_blocks)
  rs1 = constrain_lower_ranks(m, n, blocks = lower_blocks, ranks = rs)
  for lb = num_blocks:-1:1
    rows_lb, cols_lb = lower_block_ranges(lower_blocks, m, n, lb)
    # empty if lb-1 < 1
    _, cols_lb1 = lower_block_ranges(lower_blocks, m, n, lb-1)
    dᵣ = setdiffᵣ(cols_lb, cols_lb1)
    if !isempty(dᵣ)
      rows_lb_first = isempty(rows_lb) ? m : first(rows_lb)
      cols_first_last[5, dᵣ] .= min(m, rows_lb_first + rs1[lb] - 1)
    end
  end
  return nothing
end

"""
    leading_upper_ranks_to_cols_first_last!(
      upper_blocks::IndexList{BD},
      m::Int,
      n::Int,
      cols_first_last::AbstractMatrix{Int},
      rs::AbstractVector{Int},
    ) where {BD<:AbstractBlockData}

Set first_last indices appropriate for a leading decomposition associated
with a given upper rank sequence
"""
function leading_upper_ranks_to_cols_first_last!(
  upper_blocks::IndexList{BD},
  m::Int,
  n::Int,
  cols_first_last::AbstractMatrix{Int},
  rs::AbstractVector{Int},
) where {BD<:AbstractBlockData}

  num_blocks = length(upper_blocks)
  rs1 = constrain_upper_ranks(m, n, blocks = upper_blocks, ranks = rs)
  for ub = 1:num_blocks
    rows_ub, cols_ub = upper_block_ranges(upper_blocks, m, n, ub)
    # empty if ub+1 > num_blocks
    _, cols_ub1 = upper_block_ranges(upper_blocks, m, n, ub+1)
    dᵣ = setdiffᵣ(cols_ub, cols_ub1)
    if !isempty(dᵣ)
      rows_ub_last = isempty(rows_ub) ? 0 : last(rows_ub)
      cols_first_last[2, dᵣ] .= 
        max(1, rows_ub_last - rs1[ub] + 1)
    end
  end
  return nothing
end

"""
    trailing_upper_ranks_to_cols_first_last!(
      bbc::BlockedBandColumn,
      rs::AbstractVector{Int},
    )

Set first_last indices appropriate for a leading decomposition
associated with a given upper rank sequence
"""
function trailing_upper_ranks_to_cols_first_last!(
  upper_blocks::IndexList{BD},
  m::Int,
  n::Int,
  cols_first_last::AbstractMatrix{Int},
  rs::AbstractVector{Int},
) where {BD<:AbstractBlockData}

  num_blocks = length(upper_blocks)
  rs1 = constrain_upper_ranks(m, n, blocks = upper_blocks, ranks = rs)
  for ub = num_blocks:-1:1
    rows_ub, cols_ub = upper_block_ranges(upper_blocks, m, n, ub)
    # empty if ub-1 < 1
    rows_ub1, _ = upper_block_ranges(upper_blocks, m, n, ub - 1)
    dᵣ = setdiffᵣ(rows_ub, rows_ub1)
    if !isempty(dᵣ)
      cols_first_last[2, first(cols_ub, rs1[ub])] .= first(dᵣ)
    end
  end
  return nothing
end


end
