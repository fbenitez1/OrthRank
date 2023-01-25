module BandwidthInit
using Printf
using Random

using ..BandColumnMatrices
using InPlace

export leading_lower_ranks_to_cols_first_last!,
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
    error("setdiffᵣ produces non-UnitRange")
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
    error("∪ᵣ produces non-UnitRange")
  end
end

function get_cols_first_last(
  m::Int,
  n::Int,
  upper_blocks::AbstractMatrix{Int},
  lower_blocks::AbstractMatrix{Int},
  r_upper::Int,
  r_lower::Int
)
  cols_first_last = similar_zeros(upper_blocks, 6, n)
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
  upper_blocks::AbstractMatrix{Int},
  lower_blocks::AbstractMatrix{Int},
  r_upper::Int,
  r_lower::Int,
  cols_first_last::AbstractMatrix{Int}
)
  get_cols_first_last_upper!(m, n, upper_blocks, r_upper, cols_first_last)
  get_cols_first_last_lower!(m, n, lower_blocks, r_lower, cols_first_last)
end

function get_rows_first_last(
  m::Int,
  n::Int,
  upper_blocks::AbstractMatrix{Int},
  lower_blocks::AbstractMatrix{Int},
  r_upper::Int,
  r_lower::Int
)
  rows_first_last = similar_zeros(upper_blocks, m, 6)
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
  upper_blocks::AbstractMatrix{Int},
  lower_blocks::AbstractMatrix{Int},
  r_upper::Int,
  r_lower::Int,
  rows_first_last::AbstractMatrix{Int}
)
  get_rows_first_last_upper!(m, n, upper_blocks, r_upper, rows_first_last)
  get_rows_first_last_lower!(m, n, lower_blocks, r_lower, rows_first_last)
end

function get_cols_first_last_lower(
  m::Int,
  n::Int,
  lower_blocks::AbstractMatrix{Int},
  r::Int
)
  cols_first_last = similar_zeros(lower_blocks, 6, n)
  get_cols_first_last_lower!(m, n, lower_blocks, r, cols_first_last)
  cols_first_last[4:6, :]
end

# O = Old elements
# X = active region
# N = New elements
#
# One block:
#
# -----------+     -----------+    -----------+
#         OOO|             XXX|    NNNNNNNNNNN|
#         OOO|             XXX|    NNNNNNNNNNN|
#         OOO|             XXX|    NNNNNNNNNNN|
#         OOO|                |               \

# Start: Note there should be a row compression step before this.

# --+               --+               --+            
# OO|               OO|               OO|            
# OO|               OO|               OO|            
# --+--+            --+--+            --+--+         
#  O|OO|             O|OO|             O|OO|         
#  O|OO|             O|OO|             O|OO|         
# --+--+--+         --+--+--+         --+--+--+      
#   | O|OO|           | O|OO|           | O|OO|      
#   | O|OO|           | O|OO|           | O|OO|      
# --+--+--+--+      --+--+--+--+      --+--+--+--+   
#   |  | O|OO|        |  | O|OO|    <   |  | X|XX|   
#   |  | O|OO|        |  | O|OO|    <   |  | X|XX|   
# --+--+--+--+--+   --+--+--+--+--+ < --+--+--+--+--+
#   |  |  | O|OO|     |  | X|XX|NN| <   |  | X|XX|NN|
#   |  |  | O|OO|     |  | X|XX|NN| <   |  |  |  |NN|
#   1  2  3  4  5     1  2  3  4  5     1  2  3  4  5
#        ^^^^^^^

# Generic lower leading to trailing step: ub=3 to ub=2.
#
# uncompress 3:   compress block 2:  Repeat with ub=2:
#
#   
# --+                --+                  --+            
# OO|                OO|                  OO|            
# OO|                OO|                  OO|            
# --+--+             --+--+               --+--+         
#  O|OO|              O|OO|          <     X|XX|         
#  O|OO|              O|OO|          <     X|XX|         
# --+--+--+          --+--+--+       <    --+--+--+      
#   | X|XX|           X|XX|XX|       <     X|XX|NN|      
#   | X|XX|           X|XX|XX|       <      |  |NN|      
# --+--+--+--+       --+--+--+--+    <    --+--+--+--+   
#   | X|XX|NN|        X|XX|XX|NN|    <      |  |NN|NN|   
#   |  |  |NN|         |  |  |NN|           |  |  |NN|   
# --+--+--+--+--+    --+--+--+--+--+      --+--+--+--+--+
#   |  |  |NN|NN|      |  |  |NN|NN|        |  |  |NN|NN|
#   |  |  |  |NN|      |  |  |  |NN|        |  |  |  |NN|
#   1  2  3  4  5      1  2  3  4  5        1  2  3  4  5
#  ^^^^^^^

# End:
#   
# --+              --+            
# OO|              OO|            <    --+            
# OO|              OO|            <    NN|            
# --+--+           --+--+         <    NN|            
#  X|XX|           XX|NN|         <    --+--+         
#  X|XX|           XX|NN|         <    NN|NN|         
# --+--+--+        --+--+--+      <      |NN|         
#  X|XX|NN|        XX|NN|NN|      <    --+--+--+      
#   |  |NN|          |  |NN|             |NN|NN|      
# --+--+--+--+     --+--+--+--+          |  |NN|      
#   |  |NN|NN|       |  |NN|NN|        --+--+--+--+   
#   |  |  |NN|       |  |  |NN|          |  |NN|NN|   
# --+--+--+--+--+  --+--+--+--+--+       |  |  |NN|   
#   |  |  |NN|NN|    |  |  |NN|NN|     --+--+--+--+--+
#   |  |  |  |NN|    |  |  |  |NN|       |  |  |NN|NN|
#   1  2  3  4  5    1  2  3  4  5       |  |  |  |NN|
# ^^^^^                                  1  2  3  4  5

function get_cols_first_last_lower!(
  m::Int,
  n::Int,
  lower_blocks::AbstractMatrix{Int},
  r::Int,
  cols_first_last::AbstractMatrix{Int}
)
  # Trace through a leading to trailing conversion to fill in last
  # storable in cols_first_last suitable for a decomposition with
  # lower ranks bounded by r.  This can be done by tracing through
  # trailing to leading conversion as well.

  function extend_to(k, range)
    @views cols_first_last[6, range] .=
      (x -> max(x, k)).(cols_first_last[6, range])
    nothing
  end

  # default to nothing storable and extend as needed.
  cols_first_last[6, :] .= 0

  num_blocks = size(lower_blocks, 2)

  num_blocks == 0 && return extend_to(m, 1:n)

  rows_lb, cols_lb = lower_block_ranges(lower_blocks, m, n, num_blocks)
  compressed_rows_lb = first(rows_lb, r)
  compressed_cols_lb = last(cols_lb, r)
  last_compressed_row_lb =
    isempty(compressed_rows_lb) ? m : last(compressed_rows_lb)

  extend_to(m, compressed_cols_lb)
  extend_to(m, setdiffᵣ(1:n, cols_lb))

  num_blocks == 1 &&
    return extend_to(last_compressed_row_lb, setdiffᵣ(cols_lb, compressed_cols_lb))

  @views for lb ∈ num_blocks:-1:2
    # At the start block lb is both row and column compressed. After
    # block lb is column uncompressed, block lb-1 has nonzero rows
    # given by hull(compressed_rows_lb, compressed_rows_lb_minus_1).
    # Block lb has nonzero columns given by hull(compressed_cols_lb,
    # compressed_cols_lb_minus_1) extending down to
    # last(compressed_rows_lb).  Set the last storable index in these
    # columns to accommdate the last compressed row elements for block
    # lb.

    rows_lb_minus_1, cols_lb_minus_1 =
      lower_block_ranges(lower_blocks, m, n, lb - 1)
    compressed_rows_lb_minus_1 = first(rows_lb_minus_1, r)
    compressed_cols_lb_minus_1 = last(cols_lb_minus_1, r)
    last_compressed_row_lb_minus_1 =
      isempty(compressed_rows_lb_minus_1) ? m : last(compressed_rows_lb_minus_1)

    hull_of_compressed_cols = hull(compressed_cols_lb, compressed_cols_lb_minus_1)
    extend_to(last_compressed_row_lb, hull_of_compressed_cols)

    rows_lb, cols_lb = rows_lb_minus_1, cols_lb_minus_1
    compressed_rows_lb, compressed_cols_lb =
      compressed_rows_lb_minus_1, compressed_cols_lb_minus_1
    last_compressed_row_lb = last_compressed_row_lb_minus_1

  end
  # fill in the bounds for the column uncompressed block 1
  # so it can accommodate r nonzero rows.  (In case it wasn't already.)
  extend_to(last_compressed_row_lb, cols_lb)

  # Fill in the first lower and last inband elements.

  cols_first_last[4, :] .= m+1   # Default for isempty(rows_lb).
  cols_first_last[5, :] .= m
  old_cols_lb = 1:0
  for lb ∈ 1:num_blocks
    rows_lb, cols_lb = lower_block_ranges(lower_blocks, m, n, lb)
    if !isempty(rows_lb)
      dᵣ = setdiffᵣ(cols_lb, old_cols_lb)
      cols_first_last[4, dᵣ] .= first(rows_lb)
      cols_first_last[5, dᵣ] .= first(rows_lb) - 1
    end
    old_cols_lb = cols_lb
  end
  nothing
end

function get_cols_first_last_upper(
  m::Int,
  n::Int,
  upper_blocks::AbstractMatrix{Int},
  r::Int
)
  cols_first_last = similar_zeros(upper_blocks, 6, n)
  get_cols_first_last_upper!(m, n, upper_blocks, r, cols_first_last)
  cols_first_last[1:3, :]
end

# Upper leading to trailing:
#
# O = Old elements
# X = active region
# N = New elements

# One Block
#
# |                     |                <     |NNN            
# |OOOOOOOOOOOOOOO      |XXX             <     |NNN            
# |OOOOOOOOOOOOOOO      |XXX             <     |NNN            
# |OOOOOOOOOOOOOOO      |XXX             <     |NNN            
# +---------------      +---------------       +---------------
#  ^^^^^^^^^^^^^^^

# First step:

# 1  2  3  4  5       1  2  3  4  5         1  2  3  4  5  
# |OO|  |  |  |       |OO|  |  |  |         |OO|  |  |  |  
# |OO|OO|  |  |       |OO|OO|  |  |         |OO|OO|  |  |  
# +--+--+--+--+--     +--+--+--+--+--       +--+--+--+--+--
#    |OO|  |  |          |OO|  |  |            |OO|  |  |  
#    |OO|OO|  |          |OO|OO|  |            |OO|OO|  |  
#    +--+--+--+--        +--+--+--+--          +--+--+--+--
#       |OO|  |             |OO|  |               |OO|  |  
#       |OO|OO|   <         |OO|OO|XX             |OO|XX|X 
#       +--+--+-- <         +--+--+--             +--+--+--
#          |OO|   <            |OO|XX                |XX|X 
#          |OO|OO <            |OO|XX                |XX|X 
#          +--+-- <            +--+--                +--+--
#             |OO <               |NN                   |NN
#             |OO <               |NN                   |NN
#             +--                 +--                   +--
#                               ^^^^^

# Basic upper leading to trailing step: ub=3 to ub=2.

# row uncompress        col compress        Repeat with ub=2:
# block 3:              block 2:
# 1  2  3  4  5          1  2  3  4  5      1  2  3  4  5  
# |OO|  |  |  |          |OO|  |  |  |      |OO|  |  |  |  
# |OO|OO|  |  |   <      |OO|OO|XX|X |      |OO|XX|X |  |  
# +--+--+--+--+-- <      +--+--+--+--+--    +--+--+--+--+--
#    |OO|  |  |   <         |OO|XX|X |         |XX|X |  |  
#    |OO|XX|X |   <         |OO|XX|X |         |XX|X |  |  
#    +--+--+--+-- <         +--+--+--+--       +--+--+--+--
#       |XX|X |   <            |NN|N |            |NN|N |  
#       |XX|X |   <            |NN|N |            |NN|N |  
#       +--+--+--              +--+--+--          +--+--+--
#          |NN|N                  |NN|N              |NN|N 
#          |NN|N                  |NN|N              |NN|N 
#          +--+--                 +--+--             +--+--
#             |NN                    |NN                |NN
#             |NN                    |NN                |NN
#             +--                    +--                +--
#                            ^^^^^^^  

# End

# 1  2  3  4  5       1  2  3  4  5     1  2  3  4  5  
# |OO|  |  |  |   <   |OO|XX|X |  |     |NN|N |  |  |  
# |OO|XX|X |  |   <   |OO|XX|X |  |     |NN|N |  |  |  
# +--+--+--+--+-- <   +--+--+--+--+--   +--+--+--+--+--
#    |XX|X |  |   <      |NN|N |  |        |NN|N |  |  
#    |XX|X |  |   <      |NN|N |  |        |NN|N |  |  
#    +--+--+--+--        +--+--+--+--      +--+--+--+--
#       |NN|N |             |NN|N |           |NN|N |  

#       |NN|N |             |NN|N |           |NN|N |  
#       +--+--+--           +--+--+--         +--+--+--
#          |NN|N               |NN|N             |NN|N 
#          |NN|N               |NN|N             |NN|N 
#          +--+--              +--+--            +--+--
#             |NN                 |NN               |NN
#             |NN                 |NN               |NN
#             +--                 +--               +--
#                      ^^^^^^^

function get_cols_first_last_upper!(
  m::Int,
  n::Int,
  upper_blocks::AbstractMatrix{Int},
  r::Int,
  cols_first_last::AbstractMatrix{Int}
)

  # Trace through a leading to trailing conversion to fill in first
  # storable in cols_first_last_upper suitable for a decomposition with
  # lower ranks bounded by r.

  function extend_to(k, range)
    @views cols_first_last[1, range] .=
      (x -> min(x, k)).(cols_first_last[1, range])
    nothing
  end

  # default to nothing storable and extend as needed.
  cols_first_last[1, :] .= m+1

  num_blocks = size(upper_blocks, 2)

  num_blocks == 0 && return extend_to(1, 1:n)

  rows_ub, cols_ub = upper_block_ranges(upper_blocks, m, n, num_blocks)
  compressed_rows_ub = last(rows_ub, r)
  compressed_cols_ub = first(cols_ub, r)

  first_compressed_row_ub = isempty(rows_ub) ? 0 : first(compressed_rows_ub)

  extend_to(first_compressed_row_ub, cols_ub)
  num_blocks == 1 && return extend_to(1, compressed_cols_ub)

  @views for ub ∈ num_blocks:-1:2
    # At the start block ub is both row and column compressed. After
    # block ub is row uncompressed, it has nonzero columns
    # given by compressed_cols_ub and nonzero rows given by
    # hull(compressed_rows_ub_minus_1, compressed_rows_ub).
    # Block ub-1 then has nonzero columns given by hull(compressed_cols_ub,
    # compressed_cols_ub_minus_1) and nonzero rows given by
    # compressed_rows_ub_minus_1.  Set the last storable index in these
    # columns to accommdate the first compressed row elements for block
    # ub after it is uncompressed.

    rows_ub_minus_1, cols_ub_minus_1 =
      upper_block_ranges(upper_blocks, m, n, ub - 1)
    compressed_rows_ub_minus_1 = last(rows_ub_minus_1, r)
    compressed_cols_ub_minus_1 = first(rows_ub_minus_1, r)

    hull_of_compressed_cols = hull(compressed_cols_ub, compressed_cols_ub_minus_1)

    first_compressed_row_ub_minus_1 =
      isempty(compressed_rows_ub_minus_1) ? 0 : first(compressed_rows_ub_minus_1)

    extend_to(first_compressed_row_ub_minus_1, hull_of_compressed_cols)

    rows_ub, cols_ub = rows_ub_minus_1, cols_ub_minus_1
    compressed_rows_ub, compressed_cols_ub =
      compressed_rows_ub_minus_1, compressed_cols_ub_minus_1
  end

  # Columns to the left of block 1 go to the top.
  extend_to(1, setdiffᵣ(1:n, cols_ub))

  # Fill in the first inband and last upper elements.
  cols_first_last[2, :] .= 1
  cols_first_last[3, :] .= 0 # Default for isempty(rows_ub).

  old_cols_ub = 1:0
  for ub ∈ num_blocks:-1:1
    rows_ub, cols_ub = upper_block_ranges(upper_blocks, m, n, ub)
    if !isempty(rows_ub)
      dᵣ = setdiffᵣ(cols_ub, old_cols_ub)
      cols_first_last[3, dᵣ] .= last(rows_ub)
      cols_first_last[2, dᵣ] .= last(rows_ub) + 1
      cols_first_last[1, dᵣ ∪ᵣ first(old_cols_ub, r)] .=
        max(1, last(rows_ub) - r + 1)
    end
    old_cols_ub = cols_ub
  end
  cols_first_last[1, first(old_cols_ub,r)] .= 1

  nothing
end

function get_rows_first_last_lower(
  m::Int,
  n::Int,
  lower_blocks::AbstractMatrix{Int},
  r::Int
)
  rows_first_last = similar_zeros(lower_blocks, m, 6)
  get_rows_first_last_lower!(m, n, lower_blocks, r, rows_first_last)
  rows_first_last[:, 1:3]
end

function get_rows_first_last_lower!(
  m::Int,
  n::Int,
  lower_blocks::AbstractMatrix{Int},
  r::Int,
  rows_first_last::AbstractMatrix{Int}
)

  function extend_to(k, range)
    @views rows_first_last[range, 1] .=
      (x -> min(x, k)).(rows_first_last[range, 1])
    nothing
  end

  num_blocks = size(lower_blocks, 2)

  # default to nothing storable and extend as needed.
  rows_first_last[:, 1] .= n+1

  num_blocks == 0 && return extend_to(1, 1:m)

  # Trace through leading to trailing.  (As in get_cols_first_last_lower!)

  rows_lb, cols_lb = lower_block_ranges(lower_blocks, m, n, num_blocks)
  compressed_rows_lb = first(rows_lb, r)
  compressed_cols_lb = last(cols_lb, r)
  
  first_compressed_col_lb =
    isempty(compressed_cols_lb) ? 1 : first(compressed_cols_lb)

  extend_to(first_compressed_col_lb, rows_lb)

  num_blocks == 1 &&
    return extend_to(1, compressed_rows_lb)

  @views for lb ∈ num_blocks:-1:2
    rows_lb_minus_1, cols_lb_minus_1 =
      lower_block_ranges(lower_blocks, m, n, lb - 1)
    compressed_rows_lb_minus_1 = first(rows_lb_minus_1, r)
    compressed_cols_lb_minus_1 = last(cols_lb_minus_1, r)
    first_compressed_col_lb_minus_1 =
      isempty(compressed_cols_lb_minus_1) ? 1 : first(compressed_cols_lb_minus_1)

    hull_of_compressed_rows = hull(compressed_rows_lb, compressed_rows_lb_minus_1)
    extend_to(first_compressed_col_lb_minus_1, hull_of_compressed_rows)

    rows_lb, cols_lb = rows_lb_minus_1, cols_lb_minus_1
    compressed_rows_lb, compressed_cols_lb =
      compressed_rows_lb_minus_1, compressed_cols_lb_minus_1
    first_compressed_col_lb = first_compressed_col_lb_minus_1

  end

  extend_to(1, compressed_rows_lb)
  extend_to(1, setdiffᵣ(1:m, rows_lb))

  # Fill in first inband and last lower indices.
  rows_first_last[:, 2] .= 1
  rows_first_last[:, 3] .= 0
  old_rows_lb = 1:0
  for lb ∈ num_blocks:-1:1
    (rows_lb, cols_lb) = lower_block_ranges(lower_blocks, m, n, lb)
    if !isempty(cols_lb)
      dᵣ = setdiffᵣ(rows_lb, old_rows_lb)
      rows_first_last[dᵣ,3] .= last(cols_lb)
      rows_first_last[dᵣ,2] .= last(cols_lb) + 1
    end
    old_rows_lb = rows_lb
  end
  nothing
end

function get_rows_first_last_upper(
  m::Int,
  n::Int,
  upper_blocks::AbstractMatrix{Int},
  r::Int
)
  rows_first_last = similar_zeros(upper_blocks, m, 6)
  get_rows_first_last_upper!(m, n, upper_blocks, r, rows_first_last)
  rows_first_last[:, 4:6]
end

function get_rows_first_last_upper!(
  m::Int,
  n::Int,
  upper_blocks::AbstractMatrix{Int},
  r::Int,
  rows_first_last::AbstractMatrix{Int}
)

  function extend_to(k, range)
    @views rows_first_last[range, 6] .=
      (x -> max(x, k)).(rows_first_last[range, 6])
    nothing
  end

  num_blocks = size(upper_blocks, 2)
  # default to nothing storable and extend as needed.
  rows_first_last[:, 6] .= 0

  num_blocks == 0 && return extend_to(n, 1:m)

  rows_ub, cols_ub = upper_block_ranges(upper_blocks, m, n, num_blocks)
  compressed_rows_ub = last(rows_ub, r)
  compressed_cols_ub = first(cols_ub, r)
  last_compressed_col_ub = isempty(compressed_cols_ub) ? n : last(compressed_cols_ub)
  
  extend_to(n, setdiffᵣ(1:m, rows_ub))
  extend_to(n, compressed_rows_ub)

  num_blocks == 1 && return extend_to(last_compressed_col_ub, rows_ub)

  @views for ub ∈ num_blocks:-1:2
    rows_ub_minus_1, cols_ub_minus_1 =
      upper_block_ranges(upper_blocks, m, n, ub - 1)
    compressed_rows_ub_minus_1 = last(rows_ub_minus_1, r)
    compressed_cols_ub_minus_1 = first(rows_ub_minus_1, r)

    hull_of_compressed_rows = hull(compressed_rows_ub, compressed_rows_ub_minus_1)
    last_compressed_col_ub_minus_1 =
      isempty(compressed_cols_ub_minus_1) ? n : last(compressed_cols_ub_minus_1)
    
    extend_to(last_compressed_col_ub, hull_of_compressed_rows)
    rows_ub, cols_ub = rows_ub_minus_1, cols_ub_minus_1
    compressed_rows_ub, compressed_cols_ub =
      compressed_rows_ub_minus_1, compressed_cols_ub_minus_1
    last_compressed_col_ub = last_compressed_col_ub_minus_1

  end

  extend_to(last_compressed_col_ub, rows_ub)

  rows_first_last[:, 4] .= n+1
  rows_first_last[:, 5] .= n
  old_rows_ub = 1:0
  for ub ∈ 1:num_blocks
    (rows_ub, cols_ub) = upper_block_ranges(upper_blocks, m, n, ub)
    if !isempty(cols_ub)
      dᵣ = setdiffᵣ(rows_ub, old_rows_ub)
      rows_first_last[dᵣ,4] .= first(cols_ub)
      rows_first_last[dᵣ,5] .= first(cols_ub) - 1
      rows_first_last[dᵣ ∪ᵣ last(old_rows_ub, r), 6] .=
        min(n, first(cols_ub) + r - 1)
    end
    old_rows_ub = rows_ub
  end
  nothing
end

"""
    lower_block_ranges(
      lower_blocks::AbstractMatrix{Int},
      m :: Int,
      n :: Int,
      l::Integer
    )

For lower blocks and a given matrix size m×n, compute ranges for lower
block ``l``.
"""
@inline function lower_block_ranges(
  lower_blocks::AbstractMatrix{Int},
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
      lower_blocks::AbstractMatrix{Int},
      m::Int,
      n::Int,
      l::Int,
    )
  
Compute the size of lower block ``l`` for an m×n matrix using the
lower_block sequence `lower_blocks`.
"""
@inline function size_lower_block(
  lower_blocks::AbstractMatrix{Int},
  m::Int,
  n::Int,
  l::Int,
)
  (rows, cols) = lower_block_ranges(lower_blocks, m, n, l)
  (last(rows) - first(rows) + 1, last(cols) - first(cols) + 1)
end

"""
    intersect_lower_block(
      lower_blocks::Matrix{Int},
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
  lower_blocks::AbstractMatrix{Int},
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
      lower_blocks::AbstractMatrix{Int},
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
  lower_blocks::AbstractMatrix{Int},
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
      upper_blocks::AbstractMatrix{Int},
      m :: Int,
      n :: Int,
      l::Integer
    )

For upper blocks and a given matrix size m×n, compute ranges for upper
block ``l``.
"""
@inline function upper_block_ranges(
  upper_blocks::AbstractMatrix{Int},
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
      upper_blocks::AbstractMatrix{Int},
      m::Int,
      n::Int,
      l::Int,
    )
  
Compute the size of upper block ``l`` for an m×n matrix using the
upper_block sequence `upper_blocks`.
"""
@inline function size_upper_block(
  upper_blocks::AbstractMatrix{Int},
  m::Int,
  n::Int,
  l::Int,
)
  (rows, cols) = upper_block_ranges(upper_blocks, m, n, l)
  (last(rows) - first(rows) + 1, last(cols) - first(cols) + 1)
end


"""
    intersect_upper_block(
      upper_blocks::AbstractMatrix{Int},
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
  upper_blocks::AbstractMatrix{Int},
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
      upper_blocks::AbstractMatrix{Int},
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
  upper_blocks::AbstractMatrix{Int},
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
    function constrain_upper_ranks!(
      mA::Int,
      nA::Int;
      blocks::AbstractMatrix{Int},
      ranks::AbstractVector{Int},
    )

Take a nominal upper rank sequence and constrain it to be minimal and
achievable, i.e. so that any set of upper blocks that had the old
ranks as upper bound is bounded by the new ranks.
"""
function constrain_upper_ranks!(
  mA::Int,
  nA::Int;
  blocks::AbstractMatrix{Int},
  ranks::AbstractVector{Int},
)

  m(k) = blocks[1,k]
  n(k) = nA - blocks[2,k]
  Δm(k) = m(k) - m(k-1)
  Δn(k) = n(k-1) - n(k)
  p = size(blocks, 2)
  ranks[1] = min(m(1), n(1), ranks[1])
  for k ∈ 2:p
    ranks[k] = min(ranks[k], m(k), n(k), ranks[k-1] + Δm(k))
  end
  for k ∈ p-1:-1:1
    ranks[k] = min(ranks[k], ranks[k+1] + Δn(k+1))
  end
  nothing

end

function constrain_upper_ranks(
  mA::Int,
  nA::Int;
  blocks::AbstractMatrix{Int},
  ranks::AbstractVector{Int},
)
  rs = copy(ranks)
  constrain_upper_ranks!(mA, nA, blocks = blocks, ranks = rs)
  rs
end

"""
    function constrain_lower_ranks!(
      mA::Int,
      nA::Int;
      blocks::AbstractMatrix{Int},
      ranks::AbstractVector{Int},
    )

Take a nominal lower rank sequence and constrain it to be minimal and
achievable, i.e. so that any set of lower blocks that had the old
ranks as upper bound is bounded by the new ranks.
"""
function constrain_lower_ranks!(
  mA::Int,
  nA::Int;
  blocks::AbstractMatrix{Int},
  ranks::AbstractVector{Int},
)

  m(k) = mA - blocks[1,k]
  n(k) = blocks[2,k]
  Δm(k) = m(k-1) - m(k)
  Δn(k) = n(k) - n(k-1)
  p = size(blocks, 2)
  ranks[1] = min(m(1), n(1), ranks[1])
  for k ∈ 2:p
    ranks[k] = min(ranks[k], m(k), n(k), ranks[k-1] + Δn(k))
  end
  for k ∈ p-1:-1:1
    ranks[k] = min(ranks[k], ranks[k+1] + Δm(k+1))
  end
  nothing

end

function constrain_lower_ranks(
  mA::Int,
  nA::Int;
  blocks::AbstractMatrix{Int},
  ranks::AbstractVector{Int},
)
  rs = copy(ranks)
  constrain_lower_ranks!(mA, nA, blocks = blocks, ranks = rs)
  rs
end

"""
    leading_lower_ranks_to_cols_first_last!(
      lower_blocks::AbstractMatrix{Int},
      m::Int,
      n::Int,
      cols_first_last::AbstractMatrix{Int},
      rs::AbstractVector{Int},
    )

Set first_last indices appropriate for a leading decomposition
associated with a given lower rank sequence.
"""
function leading_lower_ranks_to_cols_first_last!(
  lower_blocks::AbstractMatrix{Int},
  m::Int,
  n::Int,
  cols_first_last::AbstractMatrix{Int},
  rs::AbstractVector{Int},
)
  num_blocks = size(lower_blocks, 2)
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
end

"""
    trailing_lower_ranks_to_cols_first_last!(
      lower_blocks::AbstractMatrix{Int},
      m::Int,
      n::Int,
      cols_first_last::AbstractMatrix{Int},
      rs::AbstractVector{Int},
    )

Set first_last indices appropriate for a trailing decomposition
associated with a given lower rank sequence.
"""
function trailing_lower_ranks_to_cols_first_last!(
  lower_blocks::AbstractMatrix{Int},
  m::Int,
  n::Int,
  cols_first_last::AbstractMatrix{Int},
  rs::AbstractVector{Int},
)
  num_blocks = size(lower_blocks, 2)
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
end

"""
    leading_upper_ranks_to_cols_first_last!(
      upper_blocks::AbstractMatrix{Int},
      m::Int,
      n::Int,
      cols_first_last::AbstractMatrix{Int},
      rs::AbstractVector{Int},
    )

Set first_last indices appropriate for a leading decomposition associated
with a given upper rank sequence
"""
function leading_upper_ranks_to_cols_first_last!(
  upper_blocks::AbstractMatrix{Int},
  m::Int,
  n::Int,
  cols_first_last::AbstractMatrix{Int},
  rs::AbstractVector{Int},
)

  num_blocks = size(upper_blocks, 2)
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
  upper_blocks::AbstractMatrix{Int},
  m::Int,
  n::Int,
  cols_first_last::AbstractMatrix{Int},
  rs::AbstractVector{Int},
)

  num_blocks = size(upper_blocks, 2)
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
end


end
