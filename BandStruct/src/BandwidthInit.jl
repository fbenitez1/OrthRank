module BandwidthInit
using Printf
using Random

using ..BandColumnMatrices

export leading_lower_ranks_to_cols_first_last!,
  leading_upper_ranks_to_cols_first_last!,
  leading_constrain_lower_ranks,
  leading_constrain_upper_ranks,
  trailing_lower_ranks_to_cols_first_last!,
  trailing_upper_ranks_to_cols_first_last!,
  trailing_constrain_lower_ranks,
  trailing_constrain_upper_ranks,
  lower_block_ranges,
  upper_block_ranges,
  size_lower_block,
  size_upper_block,
  intersect_lower_block,
  intersect_upper_block,
  setdiffᵣ,
  ∪ᵣ,
  get_cols_first_last,
  get_cols_first_last!,
  get_cols_first_last_lower,
  get_cols_first_last_lower!,
  get_cols_first_last_upper,
  get_cols_first_last_upper!,
  get_rows_first_last,
  get_rows_first_last!,
  get_rows_first_last_lower,
  get_rows_first_last_lower!,
  get_rows_first_last_upper,
  get_rows_first_last_upper!

"""

# AbstractBlockedBandColumn

    AbstractBlockedBandColumn{E,AE,AI} <:
      AbstractBandColumn{NonSub,E,AE,AI}

An AbstractBlockedBandColumn should implement the following:

  get_lower_blocks

"""
abstract type AbstractBlockedBandColumn{E,AE,AI} <:
              AbstractBandColumn{NonSub,E,AE,AI} end


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
    error("setdiffᵣ produces non-UnitRange")
  end
end

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
    error("∪ᵣ produces non-UnitRange")
  end
end

function get_cols_first_last(
  m::Int,
  n::Int,
  upper_blocks::AbstractArray{Int,2},
  lower_blocks::AbstractArray{Int,2},
  r_upper::Int,
  r_lower::Int
)
  cols_first_last = zeros(Int, 6, n)
  get_cols_first_last!(
    m,
    n,
    upper_blocks,
    lower_blocks,
    r_upper,
    r_lower,
    cols_first_last,
  )
  cols_first_last
end

# Compute cols_first_last.  This works for either a leading or
# trailing decomposition and provides enough extra bandwidth for
# conversion between them.
function get_cols_first_last!(
  m::Int,
  n::Int,
  upper_blocks::AbstractArray{Int,2},
  lower_blocks::AbstractArray{Int,2},
  r_upper::Int,
  r_lower::Int,
  cols_first_last::AbstractArray{Int,2}
)
  get_cols_first_last_upper!(m, n, upper_blocks, r_upper, cols_first_last)
  get_cols_first_last_lower!(m, n, lower_blocks, r_lower, cols_first_last)
end

function get_rows_first_last(
  m::Int,
  n::Int,
  upper_blocks::AbstractArray{Int,2},
  lower_blocks::AbstractArray{Int,2},
  r_upper::Int,
  r_lower::Int
)
  rows_first_last = zeros(Int, m, 6)
  get_rows_first_last!(
    m,
    n,
    upper_blocks,
    lower_blocks,
    r_upper,
    r_lower,
    rows_first_last,
  )
  rows_first_last
end

# Compute rows_first_last.  This works for either a leading or
# trailing decomposition and provides enough extra bandwidth for
# conversion between them.
function get_rows_first_last!(
  m::Int,
  n::Int,
  upper_blocks::AbstractArray{Int,2},
  lower_blocks::AbstractArray{Int,2},
  r_upper::Int,
  r_lower::Int,
  rows_first_last::AbstractArray{Int,2}
)
  get_rows_first_last_upper!(m, n, upper_blocks, r_upper, rows_first_last)
  get_rows_first_last_lower!(m, n, lower_blocks, r_lower, rows_first_last)
end


function get_cols_first_last_lower(
  m::Int,
  n::Int,
  lower_blocks::AbstractArray{Int,2},
  r::Int
)
  first_last_lower = zeros(Int, 6, n)
  get_cols_first_last_lower!(m, n, lower_blocks, r, first_last_lower)
  first_last_lower[4:6, :]
end

function get_cols_first_last_lower!(
  m::Int,
  n::Int,
  lower_blocks::AbstractArray{Int,2},
  r::Int,
  first_last_lower::AbstractArray{Int,2}
)
  
  num_blocks = size(lower_blocks, 2)
  first_last_lower[4, :] .= m+1
  first_last_lower[5, :] .= m
  first_last_lower[6, :] .= m
  old_cols_lb = 1:0
  for lb ∈ 1:num_blocks
    (rows_lb, cols_lb) = lower_block_ranges(lower_blocks, m, n, lb)
    if !isempty(rows_lb)
      dᵣ = setdiffᵣ(cols_lb, old_cols_lb)
      first_last_lower[4, dᵣ] .= first(rows_lb)
      first_last_lower[5, dᵣ] .= first(rows_lb) - 1
      first_last_lower[6, dᵣ ∪ᵣ last(old_cols_lb, r)] .=
        min(m, first(rows_lb) + r - 1)
    end
    old_cols_lb = cols_lb
  end
  first_last_lower[6, last(old_cols_lb,r)] .= m
  nothing
end

function get_cols_first_last_upper(
  m::Int,
  n::Int,
  upper_blocks::AbstractArray{Int,2},
  r::Int
)
  first_last_upper = zeros(Int, 6, n)
  get_cols_first_last_upper!(m, n, upper_blocks, r, first_last_upper)
  first_last_upper[1:3, :]
end

function get_cols_first_last_upper!(
  m::Int,
  n::Int,
  upper_blocks::AbstractArray{Int,2},
  r::Int,
  first_last_upper::AbstractArray{Int,2}
)
  num_blocks = size(upper_blocks, 2)
  first_last_upper[1, :] .= 1
  first_last_upper[2, :] .= 1
  first_last_upper[3, :] .= 0
  old_cols_ub = 1:0
  for ub ∈ num_blocks:-1:1
    (rows_ub, cols_ub) = upper_block_ranges(upper_blocks, m, n, ub)
    if !isempty(rows_ub)
      dᵣ = setdiffᵣ(cols_ub, old_cols_ub)
      first_last_upper[3, dᵣ] .= last(rows_ub)
      first_last_upper[2, dᵣ] .= last(rows_ub) + 1
      first_last_upper[1, dᵣ ∪ᵣ first(old_cols_ub, r)] .=
        max(1, last(rows_ub) - r + 1)
    end
    old_cols_ub = cols_ub
  end
  first_last_upper[1, first(old_cols_ub,r)] .= 1
  nothing
end

function get_rows_first_last_lower(
  m::Int,
  n::Int,
  lower_blocks::AbstractArray{Int,2},
  r::Int
)
  first_last_lower = zeros(Int, m, 6)
  get_rows_first_last_lower!(m, n, lower_blocks, r, first_last_lower)
  first_last_lower[:, 1:3]
end

function get_rows_first_last_lower!(
  m::Int,
  n::Int,
  lower_blocks::AbstractArray{Int,2},
  r::Int,
  first_last_lower::AbstractArray{Int,2}
)
  
  num_blocks = size(lower_blocks, 2)
  first_last_lower[:, 1] .= 1
  first_last_lower[:, 2] .= 1
  first_last_lower[:, 3] .= 0
  old_rows_lb = 1:0
  for lb ∈ num_blocks:-1:1
    (rows_lb, cols_lb) = lower_block_ranges(lower_blocks, m, n, lb)
    if !isempty(cols_lb)
      dᵣ = setdiffᵣ(rows_lb, old_rows_lb)
      first_last_lower[dᵣ,3] .= last(cols_lb)
      first_last_lower[dᵣ,2] .= last(cols_lb) + 1
      first_last_lower[dᵣ ∪ᵣ first(old_rows_lb, r), 1] .=
        max(1, last(cols_lb) - r + 1)
    end
    old_rows_lb = rows_lb
  end
  first_last_lower[first(old_rows_lb,r),1] .= 1
  nothing
end

function get_rows_first_last_upper(
  m::Int,
  n::Int,
  upper_blocks::AbstractArray{Int,2},
  r::Int
)
  first_last_upper = zeros(Int, m, 6)
  get_rows_first_last_upper!(m, n, upper_blocks, r, first_last_upper)
  first_last_upper[:, 4:6]
end

function get_rows_first_last_upper!(
  m::Int,
  n::Int,
  upper_blocks::AbstractArray{Int,2},
  r::Int,
  first_last_upper::AbstractArray{Int,2}
)
  
  num_blocks = size(upper_blocks, 2)
  first_last_upper[:, 4] .= n+1
  first_last_upper[:, 5] .= n
  first_last_upper[:, 6] .= n
  old_rows_ub = 1:0
  for ub ∈ 1:num_blocks
    (rows_ub, cols_ub) = upper_block_ranges(upper_blocks, m, n, ub)
    if !isempty(cols_ub)
      dᵣ = setdiffᵣ(rows_ub, old_rows_ub)
      first_last_upper[dᵣ,4] .= first(cols_ub)
      first_last_upper[dᵣ,5] .= first(cols_ub) - 1
      first_last_upper[dᵣ ∪ᵣ last(old_rows_ub, r), 6] .=
        min(n, first(cols_ub) + r - 1)
    end
    old_rows_ub = rows_ub
  end
  first_last_upper[last(old_rows_ub, r), 6] .= n
  nothing
end

"""
    lower_block_ranges(
      lower_blocks::AbstractArray{Int,2},
      m :: Int,
      n :: Int,
      l::Integer
    )

For lower blocks and a given matrix size m×n, compute ranges for lower
block ``l``.
"""
@inline function lower_block_ranges(
  lower_blocks::AbstractArray{Int,2},
  m :: Int,
  n :: Int,
  l::Integer,
)
  if l < 1
    (UnitRange(1,m), UnitRange(1,0))
  elseif l > size(lower_blocks,2)
    (UnitRange(m+1,m), UnitRange(1,n))
  else
    j_first = lower_blocks[1, l] + 1
    k_last = lower_blocks[2, l]
    (UnitRange(j_first, m), UnitRange(1, k_last))
  end
end

"""
    size_lower_block(
      lower_blocks::AbstractArray{Int,2},
      m::Int,
      n::Int,
      l::Int,
    )
  
Compute the size of lower block ``l`` for an m×n matrix using the
lower_block sequence `lower_blocks`.
"""
@inline function size_lower_block(
  lower_blocks::AbstractArray{Int,2},
  m::Int,
  n::Int,
  l::Int,
)
  (rows, cols) = lower_block_ranges(lower_blocks, m, n, l)
  (last(rows) - first(rows) + 1, last(cols) - first(cols) + 1)
end

"""
    intersect_lower_block(
      lower_blocks::Array{Int,2},
      m :: Int,
      n :: Int,
      l::Integer,
      ::Colon,
      k::Int
    )

Determine if column ``k`` intersects with lower block ``l``
in a matrix of size ``m×n``.
"""
@inline function intersect_lower_block(
  lower_blocks::AbstractArray{Int,2},
  m :: Int,
  n :: Int,
  l::Integer,
  ::Colon,
  k::Int
)
  (_, cols) = lower_block_ranges(lower_blocks, m, n, l)
  k ∈ cols
end

"""
    intersect_lower_block(
      lower_blocks::AbstractArray{Int,2},
      m::Int,
      n::Int,
      l::Int,
      j::Int,
      ::Colon,
    )

Determine if row ``j`` intersects with lower block ``l``
in a matrix of size ``m×n``.
"""
@inline function intersect_lower_block(
  lower_blocks::AbstractArray{Int,2},
  m::Int,
  n::Int,
  l::Int,
  j::Int,
  ::Colon,
)
  (rows, _) = lower_block_ranges(lower_blocks, m, n, l)
  j ∈ rows
 end

"""
    upper_block_ranges(
      upper_blocks::AbstractArray{Int,2},
      m :: Int,
      n :: Int,
      l::Integer
    )

For upper blocks and a given matrix size m×n, compute ranges for upper
block ``l``.
"""
@inline function upper_block_ranges(
  upper_blocks::AbstractArray{Int,2},
  m::Int,
  n::Int,
  l::Integer,
)
  if l < 1
    (UnitRange(1, 0), UnitRange(1, n))
  elseif l > size(upper_blocks, 2)
    (UnitRange(1, m), UnitRange((n + 1):n))
  else
    j_last = upper_blocks[1, l]
    k_first = upper_blocks[2, l] + 1
    (UnitRange(1, j_last), UnitRange(k_first, n))
  end
end

"""
    size_upper_block(
      upper_blocks::AbstractArray{Int,2},
      m::Int,
      n::Int,
      l::Int,
    )
  
Compute the size of upper block ``l`` for an m×n matrix using the
upper_block sequence `upper_blocks`.
"""
@inline function size_upper_block(
  upper_blocks::AbstractArray{Int,2},
  m::Int,
  n::Int,
  l::Int,
)
  (rows, cols) = upper_block_ranges(upper_blocks, m, n, l)
  (last(rows) - first(rows) + 1, last(cols) - first(cols) + 1)
end


"""
    intersect_upper_block(
      upper_blocks::AbstractArray{Int,2},
      m :: Int,
      n :: Int,
      l::Integer,
      ::Colon,
      k::Int
    )

Determine if column ``k`` intersects with upper block ``l``
in a matrix of size ``m×n``.
"""
@inline function intersect_upper_block(
  upper_blocks::AbstractArray{Int,2},
  m::Int,
  n::Int,
  l::Integer,
  ::Colon,
  k::Int
)
  (_, cols) = upper_block_ranges(upper_blocks, m, n, l)
  k ∈ cols
 end

"""
    intersect_upper_block(
      upper_blocks::AbstractArray{Int,2},
      m::Int,
      n::Int,
      l::Int,
      j::Int,
      ::Colon,
    )

Determine if row ``j`` intersects with upper block ``l``
in a matrix of size ``m×n``.
"""
@inline function intersect_upper_block(
  upper_blocks::AbstractArray{Int,2},
  m::Int,
  n::Int,
  l::Integer,
  j::Int,
  ::Colon,
)
  (rows, _) = upper_block_ranges(upper_blocks, m, n, l)
  j ∈ rows
 end

"""
    leading_constrain_lower_ranks(
      blocks::AbstractArray{Int,2},
      lower_ranks::AbstractArray{Int,1},
    )

Take a nominal lower rank sequence and constrain it to be
consistent with the preceding ranks in a leading decomposition.
"""
function leading_constrain_lower_ranks(
  blocks::AbstractArray{Int,2},
  m::Int,
  n::Int,
  lower_ranks::AbstractArray{Int,1},
)

  lr = similar(lower_ranks)
  lr .= 0
  num_blocks = size(blocks, 2)
  
  sz_first = size_lower_block(blocks,m,n,1)
  lr[1] = min(minimum(sz_first), lower_ranks[1])
  old_cols_lb = 1:0
  for lb = 2:num_blocks
    rows_lb, cols_lb = lower_block_ranges(blocks, m, n, lb)
    cols_ext = setdiffᵣ(cols_lb, old_cols_lb) ∪ᵣ last(old_cols_lb, lr[lb-1])
    lr[lb] = min(length(rows_lb), length(cols_ext), lower_ranks[lb])
  end
  lr
end

"""
    trailing_constrain_lower_ranks(
      blocks::AbstractArray{Int,2},
      lower_ranks::AbstractArray{Int,1},
    )

Take a nominal lower rank sequence and constrain it to be consistent
with the size of the blocks and preceding ranks in a trailing
decomposition.
"""
function trailing_constrain_lower_ranks(
  blocks::AbstractArray{Int,2},
  m::Int,
  n::Int,
  lower_ranks::AbstractArray{Int,1},
)

  lr = similar(lower_ranks)
  lr .= 0
  num_blocks = size(blocks, 2)

  sz_last = size_lower_block(blocks, m, n, num_blocks)
  lr[num_blocks] = min(minimum(sz_last), lower_ranks[num_blocks])

  old_rows_lb, _ = lower_block_ranges(blocks, m, n, num_blocks)

  for lb = (num_blocks - 1):-1:1
    rows_lb, cols_lb = lower_block_ranges(blocks, m, n, lb)
    rows_ext = setdiffᵣ(rows_lb, old_rows_lb) ∪ᵣ first(old_rows_lb, lr[lb + 1])
    lr[lb] = min(length(cols_lb), length(rows_ext), lower_ranks[lb])
  end
  lr
end

"""
    leading_constrain_upper_ranks(
      blocks::AbstractArray{Int,2},
      upper_ranks::AbstractArray{Int,1},
    )

Take a nominal upper rank sequence and constrain it to be
consistent with the size of block and preceding ranks in
a leading decomposition.
"""
function leading_constrain_upper_ranks(
  blocks::AbstractArray{Int,2},
  m::Int,
  n::Int,
  upper_ranks::AbstractArray{Int,1},
)

  ur = similar(upper_ranks)
  ur .= 0
  num_blocks = size(blocks, 2)

  sz_first = size_upper_block(blocks,m,n,1)
  ur[1] = min(minimum(sz_first), upper_ranks[1])
  old_rows_ub = 1:0
  for ub = 2:num_blocks
    rows_ub, cols_ub = upper_block_ranges(blocks, m, n, ub)
    rows_ext = setdiffᵣ(rows_ub, old_rows_ub) ∪ᵣ last(old_rows_ub, ur[ub - 1])
    ur[ub] = min(length(cols_ub), length(rows_ext), upper_ranks[ub])
  end
  ur
end

"""
    trailing_constrain_upper_ranks(
      blocks::AbstractArray{Int,2},
      upper_ranks::AbstractArray{Int,1},
    )

Take a nominal upper rank sequence and constrain it to be
consistent with the size of block and preceding ranks in
a trailing decomposition.
"""
function trailing_constrain_upper_ranks(
  blocks::AbstractArray{Int,2},
  m::Int,
  n::Int,
  upper_ranks::AbstractArray{Int,1},
)

  ur = similar(upper_ranks)
  ur .= 0
  num_blocks = size(blocks, 2)
  sz_last = size_upper_block(blocks,m,n,num_blocks)

  ur[num_blocks] =
    min(minimum(sz_last), upper_ranks[num_blocks])

  old_cols_ub = 1:0
  for ub = num_blocks-1:-1:1
    rows_ub, cols_ub = upper_block_ranges(blocks, m, n, ub)
    cols_ext = setdiffᵣ(cols_ub, old_cols_ub) ∪ᵣ first(old_cols_ub, ur[ub+1])
    ur[ub] = min(length(rows_ub), length(cols_ext), upper_ranks[ub])
  end
  ur
end

"""
    leading_lower_ranks_to_cols_first_last!(
      lower_blocks::AbstractArray{Int,2},
      m::Int,
      n::Int,
      cols_first_last::AbstractArray{Int,2},
      rs::AbstractArray{Int,1},
    )

Set first_last indices appropriate for a leading decomposition
associated with a given lower rank sequence.
"""
function leading_lower_ranks_to_cols_first_last!(
  lower_blocks::AbstractArray{Int,2},
  m::Int,
  n::Int,
  cols_first_last::AbstractArray{Int,2},
  rs::AbstractArray{Int,1},
)
  num_blocks = size(lower_blocks, 2)
  rs1 = leading_constrain_lower_ranks(lower_blocks, m, n, rs)
  for lb = 1:num_blocks
    rows_lb, cols_lb = lower_block_ranges(lower_blocks, m, n, lb)
    # empty if lb+1 > num_blocks
    rows_lb1, _ = lower_block_ranges(lower_blocks, m, n, lb + 1)
    dᵣ = setdiffᵣ(rows_lb, rows_lb1)
    if !isempty(dᵣ)
      cols_first_last[5, last(cols_lb, rs1[lb])] .= last(dᵣ)
    end
  end
end

"""
    trailing_lower_ranks_to_cols_first_last!(
      lower_blocks::AbstractArray{Int,2},
      m::Int,
      n::Int,
      cols_first_last::AbstractArray{Int,2},
      rs::AbstractArray{Int,1},
    )

Set first_last indices appropriate for a trailing decomposition
associated with a given lower rank sequence.
"""
function trailing_lower_ranks_to_cols_first_last!(
  lower_blocks::AbstractArray{Int,2},
  m::Int,
  n::Int,
  cols_first_last::AbstractArray{Int,2},
  rs::AbstractArray{Int,1},
)
  num_blocks = size(lower_blocks, 2)
  rs1 = trailing_constrain_lower_ranks(lower_blocks, m, n, rs)
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
end

"""
    leading_upper_ranks_to_cols_first_last!(
      upper_blocks::AbstractArray{Int,2},
      m::Int,
      n::Int,
      cols_first_last::AbstractArray{Int,2},
      rs::AbstractArray{Int,1},
    )

Set first_last indices appropriate for a leading decomposition associated
with a given upper rank sequence
"""
function leading_upper_ranks_to_cols_first_last!(
  upper_blocks::AbstractArray{Int,2},
  m::Int,
  n::Int,
  cols_first_last::AbstractArray{Int,2},
  rs::AbstractArray{Int,1},
)

  num_blocks = size(upper_blocks, 2)
  rs1 = leading_constrain_upper_ranks(upper_blocks, m, n, rs)

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
end

"""
    trailing_upper_ranks_to_cols_first_last!(
      bbc::BlockedBandColumn,
      rs::AbstractArray{Int,1},
    )

Set first_last indices appropriate for a leading decomposition
associated with a given upper rank sequence
"""
function trailing_upper_ranks_to_cols_first_last!(
  upper_blocks::AbstractArray{Int,2},
  m::Int,
  n::Int,
  cols_first_last::AbstractArray{Int,2},
  rs::AbstractArray{Int,1},
)

  num_blocks = size(upper_blocks, 2)
  rs1 = trailing_constrain_upper_ranks(upper_blocks, m, n, rs)
  for ub = num_blocks:-1:1
    rows_ub, cols_ub = upper_block_ranges(upper_blocks, m, n, ub)
    # empty if ub-1 < 1
    rows_ub1, _ = upper_block_ranges(upper_blocks, m, n, ub - 1)
    dᵣ = setdiffᵣ(rows_ub, rows_ub1)
    if !isempty(dᵣ)
      cols_first_last[2, first(cols_ub, rs1[ub])] .= first(dᵣ)
    end
  end
end

end
