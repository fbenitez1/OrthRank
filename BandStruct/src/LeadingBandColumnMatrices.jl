module LeadingBandColumnMatrices
using Printf
using Random
using Base: @propagate_inbounds

using BandStruct.BandColumnMatrices

export LeadingBandColumn,
  leading_lower_ranks_to_cbws!,
  leading_upper_ranks_to_cbws!,
  leading_constrain_lower_ranks,
  leading_constrain_upper_ranks,
  get_lower_block_ranges,
  get_upper_block_ranges,
  size_lower_block,
  size_upper_block,
  intersect_lower_block,
  intersect_upper_block


"""

# LeadingBandColumn

    struct LeadingBandColumn{
      E<:Number,
      AE<:AbstractArray{E,2},
      AI<:AbstractArray{Int,2},
    } <: AbstractBandColumn{E,AE,AI}

A banded matrix with structure defined by leading blocks and stored in
a compressed column-wise format.  This is basically a `BandColumn`
with additional leading block information and without the `roffset`
and `coffset` fields.

# Fields

- `m::Int`: Matrix number of rows.

- `n::Int`: Matrix and elements number of columns.

- `m_els::Int`: Elements number of rows.

- `num_blocks::Int`: Number of leading blocks.

- `upper_bw_max::Int`: Maximum upper bandwidth.

- `middle_bw_max::Int`: Maximum middle bandwidth.

- `lower_bw_max::Int`: Maximum lower bandwidth.

- `rbws::AI`: Row bandwidths and first subdiagonal in each row.  An
   ``m×4`` matrix with each row containing lower, middle, and upper
   bandwidths as well as the first subdiagonal position in that row.

- `cbws::AI`: Column bandwidths and first superdiagonal in each
   column. A ``4×n`` matrix with each column containing lower, middle,
   and upper bandwidths as well as the first superdiagonal position in
   that row.

- `lower_blocks::AI`: A ``2×num_blocks`` matrix.  If
  `j=lower_blocks[1,l]` and `k=lower_blocks[2,l]` then lower block
  ``l`` is in ``lbc[j+1:m, 1:k]``, i.e. the leading block above the
  lower block is ``j×k``.

- `upper_blocks::AI`: A ``2×num_blocks`` matrix.  If
  `j=upper_blocks[1,l]` and `k=upper_blocks[2,l]` then upper block
  ``l`` is in ``lbc[1:j, k+1:n]``, i.e. the leading block to the left
  of the lower block is ``j×k``.  We require that

  `upper_blocks[1,k] <= lower_blocks[1,k]`

  and

  `lower_blocks[2,k] <= upper_blocks[2,k]`

- `band_elements::AE`: Column-wise storage of the band elements with
  dimensions:
   
  ``(upper_bw_max + middle_bw_max lower_bw_max) × n``

It is assumed that the middle bandwidths and the first row subdiagonal
and column superdiagonal will never change.  In the case of a
`LeadingBandColumn`, these are determined by the leading blocks.

# Example

Let

                    1   2       3   4
    A =   X   X   X | U | O   O | N |
                    +---+-----------+
          X   X   X   X | U   U | O |
        1 ------+       |       |   |
          O   L | X   X | U   U | O |
                |       +-------+---+
          O   L | X   X   X   X | U |
        2 ------+---+           +---+
          O   O | L | X   X   X   X |
        3 ------+---+---+           |
          O   O | O | L | X   X   X |
                |   |   |           +
          N   N | N | L | X   X   X 
        4 ------+---+---+-------+   
          N   N | N | N | O   L | X 

where U, X, and L denote upper, middle, and lower band elements.  O
indicates an open location with storage available for expandning
either the lower or upper bandwidth.  N represents a location for
which there is no storage.  The numbering of blocks to the left and
above denotes the lower or upper block number.

The matrix is stored as

    lbc.band_elements = 

    Num rows|Elements
    ----------------------
    ubwmax  |O O O O O O O
            |O O O O U U O
            |O O O U U U U
    ----------------------
    mbwmax+ |X X X X X X X
    lbwmax  |X X X X X X X
            |O L X X X X X
            |O L X X X X X
            |O O L L O L O
            |O O O L O O O

where

    m =                  8
    n =                  7
    m_els =              9
    num_blocks =         4
    upper_bw_max =       3
    middle_bw_max =      4
    lower_bw_max =       2
    cbws =               [ 0  0  0  1  2  2  1  # upper
                           2  2  4  4  4  4  4  # middle
                           0  2  1  2  0  1  0  # lower
                           0  0  0  1  3  3  4 ] # first superdiagonal.
    rbws =               [ 0  3  1  0
                           0  3  2  0
                           1  2  2  2
                           1  4  1  2
                           1  4  0  3
                           1  3  0  4
                           1  3  0  4
                           1  1  0  6 ]
                         # lower, middle, upper first subdiagonal.
    upper_blocks =       [ 1  3  4  6   # rows
                           3  4  6  7 ]  # columns
    lower_blocks =       [ 2  4  5  7   # rows
                           2  3  4  6 ]  # columns
"""
struct LeadingBandColumn{
  E<:Number,
  AE<:AbstractArray{E,2},
  AI<:AbstractArray{Int,2},
} <: AbstractBandColumn{E,AE,AI}
  m::Int
  n::Int
  m_els::Int
  num_blocks::Int
  upper_bw_max::Int
  middle_bw_max::Int
  lower_bw_max::Int
  rbws::AI
  cbws::AI
  upper_blocks::AI
  lower_blocks::AI
  band_elements::AE
end

"""
    LeadingBandColumn(
      ::Type{E},
      m::Int,
      n::Int,
      upper_bw_max::Int,
      lower_bw_max::Int,
      leading_blocks::Array{Int,2},
    ) where {E<:Number}

Construct an empty (all zero) `LeadingBandColumn` structure from the
matrix size, bounds on the upper and lower bandwidth, and blocksizes.
"""
function LeadingBandColumn(
  ::Type{E},
  m::Int,
  n::Int,
  upper_bw_max::Int,
  lower_bw_max::Int,
  upper_blocks::Array{Int,2},
  lower_blocks::Array{Int,2},
) where {E<:Number}

  num_blocks = size(lower_blocks,2)
  cbws = zeros(Int, 4, n)
  rbws = zeros(Int, m, 4)
  lb=0
  ub=0
  for k=1:n
    # If column k intersects with a new upper block, increment.
    while intersect_upper_block(upper_blocks, m, n, ub + 1, :, k)
      ub += 1
    end
    # If we no longer intersect with the current lower block,
    # increment until we do.
    while !intersect_lower_block(lower_blocks, m, n, lb, :, k)
      lb += 1
    end
    (rows_lb, _) = get_lower_block_ranges(lower_blocks, m, n, lb)
    (rows_ub, _) = get_upper_block_ranges(upper_blocks, m, n, ub)
    println(k, " ", lb, " ", ub, " ", rows_ub, " ", rows_lb)
    start = rows_ub.stop+1
    stop = rows_lb.start-1
    println(stop, " ", start)
    cbws[2,k] = max(stop - start + 1,0)
    cbws[4,k] = rows_ub.stop
  end

  lb=0
  ub=0
  for j=1:m
    # If row j intersects with a new lower block, increment.
    while intersect_lower_block(lower_blocks, m, n, lb + 1, j, :)
      lb += 1
    end
    # If we no longer intersect with the current upper block,
    # increment until we do.
    while !intersect_upper_block(upper_blocks, m, n, ub, j, :)
      ub += 1
    end
    (_, cols_lb) = get_lower_block_ranges(lower_blocks, m, n, lb)
    (_, cols_ub) = get_upper_block_ranges(upper_blocks, m, n, ub)
    start = cols_lb.stop+1
    stop = cols_ub.start-1
    rbws[j,2] = max(stop - start + 1,0)
    rbws[j,4] = cols_lb.stop
  end

  middle_bw_max = @views maximum(cbws[2, :])
  m_els = upper_bw_max + middle_bw_max + lower_bw_max
  band_elements = zeros(E, m_els, n)

  LeadingBandColumn(
    m,
    n,
    m_els,
    num_blocks,
    upper_bw_max,
    middle_bw_max,
    lower_bw_max,
    rbws,
    cbws,
    upper_blocks,
    lower_blocks,
    band_elements,
  )
end

"""
    rand!(
      rng::AbstractRNG,
      lbc::LeadingBandColumn{E},
    ) where {E}

Fill a leading band column with random elements.  This assumes
that the bandwidths have already been set.
"""
function rand!(
  rng::AbstractRNG,
  lbc::LeadingBandColumn{E},
) where {E}

  for k = 1:lbc.n
    for j = first_inband_el_storage(lbc, k):last_inband_el_storage(lbc, k)
      lbc.band_elements[j, k] = rand(rng, E)
    end
  end
end

"""
    LeadingBandColumn(
      rng::AbstractRNG,
      T::Type{E},
      m::Int,
      n::Int,
      upper_bw_max::Int,
      lower_bw_max::Int,
      leading_blocks::Array{Int,2},
      upper_ranks::Array{Int,1},
      lower_ranks::Array{Int,1},
    ) where {E<:Number}

Generate a random band column corresponding to a specific rank
structure in the upper and lower blocks.
"""
function LeadingBandColumn(
  rng::AbstractRNG,
  T::Type{E},
  m::Int,
  n::Int,
  upper_bw_max::Int,
  lower_bw_max::Int,
  upper_blocks::Array{Int,2},
  lower_blocks::Array{Int,2},
  upper_ranks::Array{Int,1},
  lower_ranks::Array{Int,1},
) where {E<:Number}
  lbc = LeadingBandColumn(
    T,
    m,
    n,
    upper_bw_max,
    lower_bw_max,
    upper_blocks,
    lower_blocks,
  )

  leading_lower_ranks_to_cbws!(lbc, lower_ranks)
  leading_upper_ranks_to_cbws!(lbc, upper_ranks)
  compute_rbws!(lbc)
  rand!(rng,lbc)
  lbc
end

##
## Functions implementing AbstractBandColumn.
##

@inline BandColumnMatrices.get_m_els(lbc::LeadingBandColumn) = lbc.m_els

@inline BandColumnMatrices.get_m(lbc::LeadingBandColumn) = lbc.m
@inline BandColumnMatrices.get_n(lbc::LeadingBandColumn) = lbc.n

@inline BandColumnMatrices.get_roffset(lbc::LeadingBandColumn) = 0
@inline BandColumnMatrices.get_coffset(lbc::LeadingBandColumn) = 0

@inline BandColumnMatrices.get_rbws(lbc::LeadingBandColumn) = lbc.rbws
@inline BandColumnMatrices.get_cbws(lbc::LeadingBandColumn) = lbc.cbws

@inline BandColumnMatrices.get_upper_bw_max(lbc::LeadingBandColumn) =
  lbc.upper_bw_max
@inline BandColumnMatrices.get_middle_bw_max(lbc::LeadingBandColumn) =
  lbc.middle_bw_max
@inline BandColumnMatrices.get_lower_bw_max(lbc::LeadingBandColumn) =
  lbc.lower_bw_max
@inline BandColumnMatrices.get_band_elements(lbc::LeadingBandColumn) =
  lbc.band_elements

@propagate_inbounds @inline BandColumnMatrices.upper_bw(
  lbc::LeadingBandColumn,
  ::Colon,
  k::Int,
) = lbc.cbws[1, k]
@propagate_inbounds @inline BandColumnMatrices.middle_bw(
  lbc::LeadingBandColumn,
  ::Colon,
  k::Int,
) = lbc.cbws[2, k]
@propagate_inbounds @inline BandColumnMatrices.lower_bw(
  lbc::LeadingBandColumn,
  ::Colon,
  k::Int,
) = lbc.cbws[3, k]
@propagate_inbounds @inline BandColumnMatrices.upper_bw(
  lbc::LeadingBandColumn,
  j::Int,
  ::Colon,
) = lbc.rbws[j, 3]
@propagate_inbounds @inline BandColumnMatrices.middle_bw(
  lbc::LeadingBandColumn,
  j::Int,
  ::Colon,
) = lbc.rbws[j, 2]
@propagate_inbounds @inline BandColumnMatrices.lower_bw(
  lbc::LeadingBandColumn,
  j::Int,
  ::Colon,
) = lbc.rbws[j, 1]

@inline Base.size(lbc::LeadingBandColumn) = (lbc.m, lbc.n)

function Base.show(io::IO, lbc::LeadingBandColumn)
  print(
    io,
    typeof(lbc),
    "(",
    lbc.m,
    ", ",
    lbc.n,
    ", ",
    lbc.m_els,
    ", ",
    lbc.num_blocks,
    ", ",
    lbc.upper_bw_max,
    ", ",
    lbc.middle_bw_max,
    ", ",
    lbc.lower_bw_max,
    ", ",
    lbc.rbws,
    ", ",
    lbc.cbws,
    ", ",
    lbc.lower_blocks,
    ", ",
    lbc.upper_blocks,
    "): ",
  )
  for j = 1:lbc.m
    println()
    for k = 1:lbc.n
      if check_bc_storage_bounds(Bool, lbc, j, k)
        if bc_index_stored(lbc, j, k)
          @printf("%10.2e", lbc[j, k])
        else
          print("         O")
        end
      else
        print("         N")
      end
    end
  end
end

Base.show(io::IO, ::MIME"text/plain", lbc::LeadingBandColumn) = show(io, lbc)

Base.print(io::IO, lbc::LeadingBandColumn) = print(
  io,
  typeof(lbc),
  "(",
  lbc.m,
  ", ",
  lbc.n,
  ", ",
  lbc.m_els,
  ", ",
  lbc.num_blocks,
  ", ",
  lbc.upper_bw_max,
  ", ",
  lbc.middle_bw_max,
  ", ",
  lbc.lower_bw_max,
  ", ",
  lbc.rbws,
  ", ",
  lbc.cbws,
  ", ",
  lbc.band_elements,
  ")",
)

Base.print(io::IO, ::MIME"text/plain", lbc::LeadingBandColumn) = print(io, lbc)

##
## Index operations.  Scalar operations are defined for
## AbstractBandColumn matrices.
##

@inline function BandColumnMatrices.viewbc(
  lbc::LeadingBandColumn,
  i::Tuple{UnitRange{Int},UnitRange{Int}},
)
  (rows, cols) = i
  j0 = rows.start
  j1 = rows.stop
  k0 = cols.start
  k1 = cols.stop

  @boundscheck begin
    checkbounds(lbc, j0, k0)
    checkbounds(lbc, j1, k1)
  end
  BandColumn(
    max(j1 - j0 + 1, 0),
    max(k1 - k0 + 1, 0),
    lbc.m_els,
    j0 - 1,
    k0 - 1,
    lbc.upper_bw_max,
    lbc.middle_bw_max,
    lbc.lower_bw_max,
    view(lbc.rbws, rows, 1:4),
    view(lbc.cbws, 1:4, cols),
    view(lbc.band_elements, 1:4, cols),
  )
end

@inline function Base.getindex(
  lbc::LeadingBandColumn,
  rows::UnitRange{Int},
  cols::UnitRange{Int},
)
  j0 = rows.start
  j1 = rows.stop
  k0 = cols.start
  k1 = cols.stop

  @boundscheck begin
    checkbounds(lbc, j0, k0)
    checkbounds(lbc, j1, k1)
  end
  BandColumn(
    max(j1 - j0 + 1, 0),
    max(k1 - k0 + 1, 0),
    lbc.m_els,
    j0 - 1,
    k0 - 1,
    lbc.upper_bw_max,
    lbc.middle_bw_max,
    lbc.lower_bw_max,
    lbc.rbws[rows, :],
    lbc.cbws[:, cols],
    lbc.band_elements[:, cols],
  )
end

"""
    function get_lower_block_ranges(
      lbc::LeadingBandColumn,
      l::Integer,
    )

Get ranges for lower block ``l``.
"""
@inline function get_lower_block_ranges(
  lbc::LeadingBandColumn,
  l::Int,
)
  
  (m, n) = size(lbc)
  get_lower_block_ranges(lbc.lower_blocks, m, n, l)
end

"""
    get_lower_block_ranges(
      lower_blocks::Array{Int,2},
      m :: Int,
      n :: Int,
      l::Integer
    )

For lower blocks and a given matrix size m×n, compute ranges for lower
block ``l``.
"""
@inline function get_lower_block_ranges(
  lower_blocks::Array{Int,2},
  m :: Int,
  n :: Int,
  l::Integer,
)
  if l < 1
    (UnitRange(1,m), UnitRange(1,0))
  elseif l > size(lower_blocks,2)
    (UnitRange(m+1,m), UnitRange(1,n))
  else
    j1 = lower_blocks[1, l] + 1
    k2 = lower_blocks[2, l]
    (UnitRange(j1, m), UnitRange(1, k2))
  end
end

"""
    size_lower_block(
      lower_blocks::Array{Int,2},
      m::Int,
      n::Int,
      l::Int,
    )
  
Compute the size of lower block ``l`` for an m×n matrix using the
lower_block sequence `lower_blocks`.
"""
@inline function size_lower_block(
  lower_blocks::Array{Int,2},
  m::Int,
  n::Int,
  l::Int,
)
  (rows, cols) = get_lower_block_ranges(lower_blocks, m, n, l)
  (rows.stop - rows.start + 1, cols.stop - cols.start + 1)
end

"""
    size_lower_block(
      lbc::LeadingBandColumn,
      l::Int,
    ) 

Compute the size of lower block ``l`` for a `LeadingBandColumn.
"""
@inline function size_lower_block(
  lbc::LeadingBandColumn,
  l::Int,
)
  (rows, cols) = get_lower_block_ranges(lbc, l)
  (rows.stop - rows.start + 1, cols.stop - cols.start + 1)
end

"""
    function intersect_lower_block(
      lbc::LeadingBandColumn,
      l::Integer,
      ::Colon,
      k::Int
    )

Determine if column ``k`` intersects with lower block ``l``
in a `LeadingBandColumn`.

"""
@inline function intersect_lower_block(
  lbc::LeadingBandColumn,
  l::Integer,
  ::Colon,
  k::Int
)
  (_, cols) = get_lower_block_ranges(lbc, l)
  k ∈ cols
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
  lower_blocks::Array{Int,2},
  m :: Int,
  n :: Int,
  l::Integer,
  ::Colon,
  k::Int
)
  (_, cols) = get_lower_block_ranges(lower_blocks, m, n, l)
  k ∈ cols
 end

"""
    intersect_lower_block(
      lbc::LeadingBandColumn,
      l::Integer,
      j::Int,
      ::Colon,
    )

Determine if row ``j`` intersects with lower block ``l``
in a `LeadingBandColumn`.
"""
@inline function intersect_lower_block(
  lbc::LeadingBandColumn,
  l::Integer,
  j::Int,
  ::Colon,
)
  (rows, _) = get_lower_block_ranges(lbc, l)
  j ∈ rows
 end

"""
    intersect_lower_block(
      lower_blocks::Array{Int,2},
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
  lower_blocks::Array{Int,2},
  m::Int,
  n::Int,
  l::Int,
  j::Int,
  ::Colon,
)
  (rows, _) = get_lower_block_ranges(lower_blocks, m, n, l)
  j ∈ rows
 end

"""
    function get_upper_block_ranges(
      lbc::LeadingBandColumn,
      l::Integer,
    )

Get ranges for upper block ``l``.
"""
@inline function get_upper_block_ranges(lbc::LeadingBandColumn, l::Integer)
  (m, n) = size(lbc)
  get_upper_block_ranges(lbc.upper_blocks, m, n, l)
end

"""
    get_upper_block_ranges(
      upper_blocks::Array{Int,2},
      m :: Int,
      n :: Int,
      l::Integer
    )

For upper blocks and a given matrix size m×n, compute ranges for upper
block ``l``.
"""
@inline function get_upper_block_ranges(
  upper_blocks::Array{Int,2},
  m::Int,
  n::Int,
  l::Integer,
)
  if l < 1
    (UnitRange(1, 0), UnitRange(1, n))
  elseif l > size(upper_blocks, 2)
    (UnitRange(1, m), UnitRange((n + 1):n))
  else
    j2 = upper_blocks[1, l]
    k1 = upper_blocks[2, l] + 1
    (UnitRange(1, j2), UnitRange(k1, n))
  end
end

"""
    size_upper_block(
      upper_blocks::Array{Int,2},
      m::Int,
      n::Int,
      l::Int,
    )
  
Compute the size of upper block ``l`` for an m×n matrix using the
upper_block sequence `upper_blocks`.
"""
@inline function size_upper_block(
  upper_blocks::Array{Int,2},
  m::Int,
  n::Int,
  l::Int,
)
  (rows, cols) = get_upper_block_ranges(upper_blocks, m, n, l)
  (rows.stop - rows.start + 1, cols.stop - cols.start + 1)
end

"""
    size_upper_block(
      lbc::LeadingBandColumn,
      l::Int,
    ) 

Compute the size of lower block ``l`` for a `LeadingBandColumn.
"""
@inline function size_upper_block(
  lbc::LeadingBandColumn,
  l::Int,
)
  (rows, cols) = get_upper_block_ranges(lbc, l)
  (rows.stop - rows.start + 1, cols.stop - cols.start + 1)
end

"""
    function intersect_upper_block(
      lbc::LeadingBandColumn,
      l::Integer,
      ::Colon,
      k::Int
    )

Determine if column ``k`` intersects with upper block ``l``
in a `LeadingBandColumn`.
"""
@inline function intersect_upper_block(
  lbc::LeadingBandColumn,
  l::Integer,
  ::Colon,
  k::Int
)
  (_, cols) = get_upper_block_ranges(lbc, l)
  k ∈ cols
 end

"""
    intersect_upper_block(
      upper_blocks::Array{Int,2},
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
  upper_blocks::Array{Int,2},
  m::Int,
  n::Int,
  l::Integer,
  ::Colon,
  k::Int
)
  (_, cols) = get_upper_block_ranges(upper_blocks, m, n, l)
  k ∈ cols
 end

"""
    intersect_upper_block(
      lbc::LeadingBandColumn,
      l::Integer,
      j::Int,
      ::Colon,
    )

Determine if row ``j`` intersects with upper block ``l``
in a `LeadingBandColumn`.
"""
@inline function intersect_upper_block(
  lbc::LeadingBandColumn,
  l::Integer,
  j::Int,
  ::Colon,
)
  (rows, _) = get_upper_block_ranges(lbc, l)
  j ∈ rows
 end

"""
    intersect_upper_block(
      upper_blocks::Array{Int,2},
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
  upper_blocks::Array{Int,2},
  m::Int,
  n::Int,
  l::Integer,
  j::Int,
  ::Colon,
)
  (rows, _) = get_upper_block_ranges(upper_blocks, m, n, l)
  j ∈ rows
 end

"""
    leading_constrain_lower_ranks(
      blocks::AbstractArray{Int,2},
      lower_ranks::AbstractArray{Int,1},
    )

Take a nominal lower rank sequence and constrain it to be
consistent with the size of the leading blocks and preceding ranks.
If the rank of block 'l' is larger than is consistent with the size
of the corresponding lower block or the rank of the previous block,
this function decreases the rank.
"""
function leading_constrain_lower_ranks(
  blocks::AbstractArray{Int,2},
  m::Int,
  n::Int,
  lower_ranks::AbstractArray{Int,1},
)
  minpair((x,y)) = x < y ? x : y

  lr = similar(lower_ranks)
  lr .= 0
  num_blocks = size(blocks, 2)
  
  lr[1] = min(minpair(size_lower_block(blocks,m,n,1)), lower_ranks[1])

  for l = 2:num_blocks
    j0 = blocks[1, l - 1] + 1
    k0 = blocks[2, l - 1]
    j1 = blocks[1, l] + 1
    k1 = blocks[2, l]
    m1 = m - j1 + 1
    n1 = k1 - k0 + lr[l - 1] # Added k1-k0 columns in next block.
    lr[l] = min(m1, n1, lower_ranks[l])
  end
  lr
end

"""
    leading_constrain_upper_ranks(
      blocks::AbstractArray{Int,2},
      upper_ranks::AbstractArray{Int,1},
    )

Take a nominal upper rank sequence and constrain it to be
consistent with the size of the leading blocks and preceding ranks.
If the rank of block 'l' is larger than is consistent with the size
of the corresponding upper block or the rank of the previous block,
this function decreases the rank.
"""
function leading_constrain_upper_ranks(
  blocks::AbstractArray{Int,2},
  m::Int,
  n::Int,
  upper_ranks::AbstractArray{Int,1},
)
  minpair((x,y)) = x < y ? x : y

  ur = similar(upper_ranks)
  ur .= 0
  num_blocks = size(blocks, 2)
  n = blocks[2,num_blocks]

  ur[1] = min(minpair(size_upper_block(blocks,m,n,1)), upper_ranks[1])

  for l = 2:num_blocks
    j0 = blocks[1, l - 1]
    k0 = blocks[2, l - 1] + 1
    j1 = blocks[1, l]
    k1 = blocks[2, l] + 1
    n1 = n - k1 + 1
    m1 = j1 - j0 + ur[l - 1]
    ur[l] = min(m1, n1, upper_ranks[l])
  end
  ur
end

"""
    leading_lower_ranks_to_cbws!(
      lbc::LeadingBandColumn,
      rs::AbstractArray{Int},
    )

Set lower bandwidth appropriate for a given lower rank sequence.
"""
function leading_lower_ranks_to_cbws!(
  lbc::LeadingBandColumn,
  rs::AbstractArray{Int},
)

  (m, n) = size(lbc)
  lbc.cbws[3, :] .= 0
  rs1 = leading_constrain_lower_ranks(lbc.lower_blocks, m, n, rs)

  for l = 1:lbc.num_blocks
    (rows0, cols0) = get_lower_block_ranges(lbc, l)
    j0=rows0.start
    k0=cols0.stop
    (rows1, cols1) = get_lower_block_ranges(lbc, l + 1)
    j1=rows1.start
    k1=cols1.stop
    d = j1 - j0
    lbc.cbws[3, (k0 - rs1[l] + 1):k0] .+= d
  end
end

"""
    leading_upper_ranks_to_cbws!(
      lbc::LeadingBandColumn,
      rs::AbstractArray{Int},
    )

Set upper bandwidth appropriate for a given upper rank sequence.
"""
function leading_upper_ranks_to_cbws!(
  lbc::LeadingBandColumn,
  rs::AbstractArray{Int},
)

  (m, n) = size(lbc)
  lbc.cbws[1, :] .= 0
  rs1 = leading_constrain_upper_ranks(lbc.upper_blocks, m, n, rs)

  for l = 1:lbc.num_blocks
    (_, cols0) = get_upper_block_ranges(lbc, l)
    k0 = cols0.start
    (_, cols1) = get_upper_block_ranges(lbc, l + 1)
    k1 = cols1.start
    lbc.cbws[1, k0:(k1 - 1)] .= rs1[l]
  end
end

@views function BandColumnMatrices.wilk(lbc::LeadingBandColumn)
  (m, n) = size(lbc)
  num_blocks = lbc.num_blocks - 1 # leave off the full matrix.
  a = fill('N', (2 * m, 2 * n))
  # insert spaces
  fill!(a[2:2:(2 * m), :], ' ')
  fill!(a[:, 2:2:(2 * n)], ' ')
  # insert boundaries for leading blocks.
  # for j = 1:num_blocks
  #   row = lbc.leading_blocks[1, j]
  #   col = lbc.leading_blocks[2, j]
  #   fill!(a[1:(2 * row - 1), 2 * col], '|')
  #   a[2 * row, 2 * col] = '⌋'
  #   fill!(a[2 * row, 1:(2 * col - 1)], '_')
  # end
  for k = 1:n
    kk = 2 * k - 1
    fill!(a[2 .* storable_els_range(lbc, :, k) .- 1, kk], 'O')
    fill!(a[2 .* upper_inband_els_range(lbc, :, k) .- 1, kk], 'U')
    fill!(a[2 .* middle_inband_els_range(lbc, :, k) .- 1, kk], 'X')
    fill!(a[2 .* lower_inband_els_range(lbc, :, k) .- 1, kk], 'L')
  end
  Wilk(a)
end

function Base.copy(lbc::LeadingBandColumn)
  LeadingBandColumn(
    lbc.m,
    lbc.n,
    lbc.m_els,
    lbc.num_blocks,
    lbc.upper_bw_max,
    lbc.middle_bw_max,
    lbc.lower_bw_max,
    copy(lbc.rbws),
    copy(lbc.cbws),
    lbc.upper_blocks,
    lbc.lower_blocks,
    copy(lbc.band_elements),
  )
end


end
