module LeadingBandColumnMatrices
using Printf
using Random

using ..BandColumnMatrices

export LeadingBandColumn,
  leading_lower_ranks_to_cols_first_last!,
  leading_upper_ranks_to_cols_first_last!,
  leading_constrain_lower_ranks,
  leading_constrain_upper_ranks,
  lower_block_ranges,
  upper_block_ranges,
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
    } <: AbstractBandColumn{NonSub,E,AE,AI}

A banded matrix with structure defined by leading blocks and stored in
a compressed column-wise format.  This is basically a `BandColumn`
with additional leading block information and without the `roffset`,
`coffset`, `sub`, `m_nosub`, and `n_nosub` fields.

# Fields

  - `m::Int`: Matrix number of rows.

  - `n::Int`: Matrix and elements number of columns.

  - `roffset::Int`: Uniform column offset, used to identify submatrices.

  - `coffset::Int`: Uniform row offset, used to identify submatrices.

  - `bw_max::Int`: Elements array number of rows.

  - `upper_bw_max::Int`: Maximum upper bandwidth.

  - `middle_lower_bw_max::Int`: Maximum middle + lower bandwidth.

  - `rows_first_last::AI`: `rows_first_last[j,:]` contains
   
      - `rows_first_last[j,1]`: Index of the first storable element in row
        `j`.

      - `rows_first_last[j,2]`: Index of the first inband element in row
        `j`.  If there are no inband elements then
        `rows_first_last[j,2] > rows_first_last[j,5]`.

      - `rows_first_last[j,3]`: Index of the last lower element in row
        `j`.  This is not necessarily an inband element.

      - `rows_first_last[j,4]`: Index of the first upper element in row
        `j`.  This is not necessarily an inband element.

      - `rows_first_last[j,5]`: Index of the last inband element in row
        `j`.  If there are no inband elements then
        `rows_first_last[j,2] > rows_first_last[j,5]`.

      - `rows_first_last[j,6]`: Index of the last storable element in row
        `j`.

  - `cols_first_last::AI`: `cols_first_last[:,k]` contains
   
      - `cols_first_last[1,k]`: Index of the first storable element in column
        `k`.

      - `cols_first_last[2,k]`: Index of the first inband element in column
        `k`.  If there are no inband elements then
        `cols_first_last[2,k] > cols_first_last[5,k]`.

      - `cols_first_last[3,k]`: Index of the last upper element in column
        `k`.  This is not necessarily an inband element.

      - `cols_first_last[4,k]`: Index of the first lower element in column
        `k`.  This is not necessarily an inband element.

      - `cols_first_last[5,k]`: Index of the last inband element in column
        `k`.  If there are no inband elements then
        `cols_first_last[2,k] > cols_first_last[5,k]`.

      - `cols_first_last[6,k]`: Index of the last storable element in column
        `k`.

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

    m =                   8
    n =                   7
    m_els =               9
    num_blocks =          4
    upper_bw_max =        3
    middle_lower_bw_max = 6

    cols_first_last =    [ 1 1 1 1 1 1 2   # first storable
                           1 1 1 1 2 2 4   # first inband
                           0 0 0 1 3 3 4   # last upper
                           3 3 5 6 8 8 9   # first lower
                           2 4 5 7 7 8 8   # last inband
                           6 6 6 7 8 8 8 ] # last storable

    rows_first_last =    [ 1 1 0 4 4 6
                           1 1 0 5 6 7
                           1 2 2 5 6 7
                           1 2 2 7 7 7
                           1 3 3 8 7 7
                           1 4 4 8 7 7
                           4 4 4 8 7 7
                           5 6 6 8 7 7 ]
                         # first storable, first inband, last lower, first upper,
                         # last inband, last storable

    upper_blocks =       [ 1  3  4  6   # rows
                           3  4  6  7 ]  # columns

    lower_blocks =       [ 2  4  5  7   # rows
                           2  3  4  6 ]  # columns
"""
struct LeadingBandColumn{
  E<:Number,
  AE<:AbstractArray{E,2},
  AI<:AbstractArray{Int,2},
} <: AbstractBandColumn{NonSub,E,AE,AI}
  # Fields for the BandColumn structure.
  m::Int
  n::Int
  bw_max::Int
  upper_bw_max::Int
  middle_lower_bw_max::Int
  rows_first_last :: AI
  cols_first_last :: AI
  band_elements::AE
  # Fields for the leading block structure
  num_blocks::Int
  upper_blocks::AI
  lower_blocks::AI
end

"""
    LeadingBandColumn(
      ::Type{E},
      m::Int,
      n::Int;
      upper_bw_max::Int,
      lower_bw_max::Int,
      upper_blocks::Array{Int,2},
      lower_blocks::Array{Int,2},
    ) where {E<:Number}

Construct an empty (all zero) `LeadingBandColumn` structure from the
matrix size, bounds on the upper and lower bandwidth, and blocksizes.
"""
function LeadingBandColumn(
  ::Type{E},
  m::Int,
  n::Int;
  upper_bw_max::Int,
  lower_bw_max::Int,
  upper_blocks::Array{Int,2},
  lower_blocks::Array{Int,2},
) where {E<:Number}

  num_blocks = size(lower_blocks,2)
  cols_first_last = zeros(Int, 6, n)
  rows_first_last = zeros(Int, m, 6)
  lb=0
  ub=0
  middle_bw_max=0
  for k ∈ 1:n
    # If column k intersects with a new upper block, increment.
    while intersect_upper_block(upper_blocks, m, n, ub + 1, :, k)
      ub += 1
    end
    # If we no longer intersect with the current lower block,
    # increment until we do.
    while !intersect_lower_block(lower_blocks, m, n, lb, :, k)
      lb += 1
    end
    (rows_lb, _) = lower_block_ranges(lower_blocks, m, n, lb)
    (rows_ub, _) = upper_block_ranges(upper_blocks, m, n, ub)
    cols_first_last[2,k] = last(rows_ub) + 1
    cols_first_last[3,k] = last(rows_ub)
    cols_first_last[4,k] = first(rows_lb)
    cols_first_last[5,k] = first(rows_lb) - 1
    middle_bw_max = max(first(rows_lb) - last(rows_ub) - 1, middle_bw_max)
  end

  lb=0
  ub=0
  for j = 1:m
    # If row j intersects with a new lower block, increment.
    while intersect_lower_block(lower_blocks, m, n, lb + 1, j, :)
      lb += 1
    end
    # If we no longer intersect with the current upper block,
    # increment until we do.
    while !intersect_upper_block(upper_blocks, m, n, ub, j, :)
      ub += 1
    end
    (_, cols_lb) = lower_block_ranges(lower_blocks, m, n, lb)
    (_, cols_ub) = upper_block_ranges(upper_blocks, m, n, ub)
    rows_first_last[j, 2] = last(cols_lb) + 1
    rows_first_last[j, 3] = last(cols_lb)
    rows_first_last[j, 4] = first(cols_ub)
    rows_first_last[j, 5] = first(cols_ub) - 1
  end
  
  bw_max = upper_bw_max + middle_bw_max + lower_bw_max
  middle_lower_bw_max = middle_bw_max + lower_bw_max
  band_elements = zeros(E, bw_max, n)

  # Set the ranges for storable elements.

  for k = 1:n
    j0 = cols_first_last[3,k] - upper_bw_max
    cols_first_last[1,k] = project(1 + j0, m)
    cols_first_last[6,k] = project(bw_max + j0, m)
  end

  rows_first_last[:,1] .= n+1
  rows_first_last[:,6] .= 0

  for k = 1:n
    j0 = cols_first_last[1, k]
    j1 = cols_first_last[6, k]
    for j = j0:j1
      rows_first_last[j, 1] = min(k, rows_first_last[j, 1])
      rows_first_last[j, 6] = max(k, rows_first_last[j, 6])
    end
  end

  LeadingBandColumn(
    m,
    n,
    bw_max,
    upper_bw_max,
    middle_lower_bw_max,
    rows_first_last,
    cols_first_last,
    band_elements,
    num_blocks,
    upper_blocks,
    lower_blocks,
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
    for j = first_inband_index_storage(lbc, k):last_inband_index_storage(lbc, k)
      lbc.band_elements[j, k] = rand(rng, E)
    end
  end
end

"""
    LeadingBandColumn(
      T::Type{E},
      rng::AbstractRNG,
      m::Int,
      n::Int;
      upper_bw_max::Int,
      lower_bw_max::Int,
      upper_blocks::Array{Int,2},
      lower_blocks::Array{Int,2},
      upper_ranks::Array{Int,1},
      lower_ranks::Array{Int,1},
    ) where {E<:Number}

Generate a random band column corresponding to a specific rank
structure in the upper and lower blocks.
"""
function LeadingBandColumn(
  T::Type{E},
  rng::AbstractRNG,
  m::Int,
  n::Int;
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
    upper_bw_max = upper_bw_max,
    lower_bw_max = lower_bw_max,
    upper_blocks = upper_blocks,
    lower_blocks = lower_blocks,
  )
  leading_lower_ranks_to_cols_first_last!(lbc, lower_ranks)
  leading_upper_ranks_to_cols_first_last!(lbc, upper_ranks)
  compute_rows_first_last!(lbc)
  rand!(rng, lbc)
  lbc
end

@inline BandColumnMatrices.toBandColumn(lbc::LeadingBandColumn) = BandColumn(
  NonSub(),
  lbc.m,
  lbc.n,
  lbc.m,
  lbc.n,
  0,
  0,
  lbc.bw_max,
  lbc.upper_bw_max,
  lbc.middle_lower_bw_max,
  lbc.rows_first_last,
  lbc.cols_first_last,
  lbc.band_elements,
)

#=

Bandwidth functions required by AbstractBandColumn.

=#

@inline BandColumnMatrices.bw_max(lbc::LeadingBandColumn) =
  lbc.bw_max
@inline BandColumnMatrices.upper_bw_max(lbc::LeadingBandColumn) =
  lbc.upper_bw_max
@inline BandColumnMatrices.middle_lower_bw_max(lbc::LeadingBandColumn) =
  lbc.middle_lower_bw_max
@inline BandColumnMatrices.band_elements(lbc::LeadingBandColumn) =
  lbc.band_elements

#=

Index functions required by AbstractBandColumn

=#

Base.@propagate_inbounds function BandColumnMatrices.storable_index_range(
  ::Type{NonSub},
  bc::LeadingBandColumn,
  ::Colon,
  k::Int,
)
  bc.cols_first_last[1,k]:bc.cols_first_last[6,k]
end

Base.@propagate_inbounds function BandColumnMatrices.storable_index_range(
  ::Type{NonSub},
  bc::LeadingBandColumn,
  j::Int,
  ::Colon,
)
  bc.rows_first_last[j,1]:bc.rows_first_last[j,6]
end


Base.@propagate_inbounds function BandColumnMatrices.inband_index_range(
  ::Type{NonSub},
  bc::LeadingBandColumn,
  ::Colon,
  k::Int,
)
  bc.cols_first_last[2, k]:bc.cols_first_last[5, k]
end

Base.@propagate_inbounds function BandColumnMatrices.inband_index_range(
  ::Type{NonSub},
  bc::LeadingBandColumn,
  j::Int,
  ::Colon,
)
  bc.rows_first_last[j, 2]:bc.rows_first_last[j, 5]
end

Base.@propagate_inbounds BandColumnMatrices.first_lower_index(
  ::Type{NonSub},
  lbc::LeadingBandColumn,
  ::Colon,
  k::Int,
) = lbc.cols_first_last[4,k]

Base.@propagate_inbounds BandColumnMatrices.last_lower_index(
  ::Type{NonSub},
  lbc::LeadingBandColumn,
  j::Int,
  ::Colon,
) = lbc.rows_first_last[j,3]

Base.@propagate_inbounds BandColumnMatrices.first_upper_index(
  ::Type{NonSub},
  lbc::LeadingBandColumn,
  j::Int,
  ::Colon,
) = lbc.rows_first_last[j,4]

Base.@propagate_inbounds BandColumnMatrices.last_upper_index(
  ::Type{NonSub},
  lbc::LeadingBandColumn,
  ::Colon,
  k::Int,
) = lbc.cols_first_last[3,k]

@inline function BandColumnMatrices.unsafe_set_first_inband_index!(
  ::Type{NonSub},
  lbc::LeadingBandColumn,
  js::AbstractUnitRange{Int},
  ::Colon,
  k_first::Int,
) 
  lbc.rows_first_last[js, 2] .= k_first
  nothing
end

@inline function BandColumnMatrices.unsafe_set_first_inband_index!(
  ::Type{NonSub},
  lbc::LeadingBandColumn,
  ::Colon,
  ks::AbstractUnitRange{Int},
  j_first::Int,
) 
  lbc.cols_first_last[2, ks] .= j_first
  nothing
end

@inline function BandColumnMatrices.unsafe_set_last_inband_index!(
  ::Type{NonSub},
  lbc::LeadingBandColumn,
  js::AbstractUnitRange{Int},
  ::Colon,
  k_last::Int,
) 
  lbc.rows_first_last[js, 5] .= k_last
  nothing
end

@inline function BandColumnMatrices.unsafe_set_last_inband_index!(
  ::Type{NonSub},
  lbc::LeadingBandColumn,
  ::Colon,
  ks::AbstractUnitRange{Int},
  j_last::Int,
) 
  lbc.cols_first_last[5, ks] .= j_last
  nothing
end

@inline BandColumnMatrices.row_size(lbc::LeadingBandColumn) = lbc.m
@inline BandColumnMatrices.col_size(lbc::LeadingBandColumn) = lbc.n
@inline BandColumnMatrices.row_size(::Type{NonSub}, lbc::LeadingBandColumn) =
  lbc.m
@inline BandColumnMatrices.col_size(::Type{NonSub}, lbc::LeadingBandColumn) =
  lbc.n

function BandColumnMatrices.compute_rows_first_last!(lbc::LeadingBandColumn)
  compute_rows_first_last!(lbc, lbc.rows_first_last)
end

function BandColumnMatrices.validate_rows_first_last(
  lbc::LeadingBandColumn,
)
  rfl = compute_rows_first_last(lbc)
  @views rfl[:,2] == lbc.rows_first_last[:,2]
  @views rfl[:,5] == lbc.rows_first_last[:,5]
end


function Base.show(io::IO, lbc::LeadingBandColumn)
  print(
    io,
    typeof(lbc),
    "(",
    lbc.m,
    ", ",
    lbc.n,
    ", ",
    lbc.bw_max,
    ", ",
    lbc.upper_bw_max,
    ", ",
    lbc.middle_lower_bw_max,
    ", ",
    lbc.rows_first_last,
    ", ",
    lbc.cols_first_last,
    ", ",
    lbc.num_blocks,
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
        if is_inband(lbc, j, k)
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
  lbc.bw_max,
  ", ",
  lbc.upper_bw_max,
  ", ",
  lbc.middle_lower_bw_max,
  ", ",
  lbc.rows_first_last,
  ", ",
  lbc.cols_first_last,
  ", ",
  lbc.band_elements,
  ", ",
  lbc.num_blocks,
  ", ",
  lbc.upper_blocks,
  ", ",
  lbc.lower_blocks,
  ")",
)

Base.print(io::IO, ::MIME"text/plain", lbc::LeadingBandColumn) = print(io, lbc)

##
## Index operations.  Scalar operations are defined for
## AbstractBandColumn matrices.
##

@inline function BandColumnMatrices.viewbc(
  lbc::LeadingBandColumn,
  i::Tuple{AbstractUnitRange{Int},AbstractUnitRange{Int}},
)
  (rows, cols) = i
  j0 = first(rows)
  j1 = last(rows)
  k0 = first(cols)
  k1 = last(cols)
  
  @boundscheck begin
    if j1 >= j0 && k1 >= k0
      checkbounds(lbc, j0, k0)
      checkbounds(lbc, j1, k1)
    end
  end
  BandColumn(
    Sub(),
    lbc.m,
    lbc.n,
    max(j1 - j0 + 1, 0),
    max(k1 - k0 + 1, 0),
    j0 - 1,
    k0 - 1,
    lbc.bw_max,
    lbc.upper_bw_max,
    lbc.middle_lower_bw_max,
    lbc.rows_first_last,
    lbc.cols_first_last,
    lbc.band_elements,
  )
end

@inline function Base.getindex(
  lbc::LeadingBandColumn,
  rows::AbstractUnitRange{Int},
  cols::AbstractUnitRange{Int},
)
  j0 = first(rows)
  j1 = last(rows)
  k0 = first(cols)
  k1 = last(cols)

  @boundscheck begin
    checkbounds(lbc, j0, k0)
    checkbounds(lbc, j1, k1)
  end
  BandColumn(
    Sub(),
    lbc.m,
    lbc.n,
    max(j1 - j0 + 1, 0),
    max(k1 - k0 + 1, 0),
    j0 - 1,
    k0 - 1,
    lbc.bw_max,
    lbc.upper_bw_max,
    lbc.middle_lower_bw_max,
    copy(lbc.rows_first_last),
    copy(lbc.cols_first_last),
    copy(lbc.band_elements),
  )
end

"""
    function lower_block_ranges(
      lbc::LeadingBandColumn,
      l::Integer,
    )

Get ranges for lower block ``l``.
"""
@inline function lower_block_ranges(
  lbc::LeadingBandColumn,
  l::Int,
)
  
  (m, n) = size(lbc)
  lower_block_ranges(lbc.lower_blocks, m, n, l)
end

"""
    lower_block_ranges(
      lower_blocks::Array{Int,2},
      m :: Int,
      n :: Int,
      l::Integer
    )

For lower blocks and a given matrix size m×n, compute ranges for lower
block ``l``.
"""
@inline function lower_block_ranges(
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
    j_first = lower_blocks[1, l] + 1
    k_last = lower_blocks[2, l]
    (UnitRange(j_first, m), UnitRange(1, k_last))
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
  (rows, cols) = lower_block_ranges(lower_blocks, m, n, l)
  (last(rows) - first(rows) + 1, last(cols) - first(cols) + 1)
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
  (rows, cols) = lower_block_ranges(lbc, l)
  (last(rows) - first(rows) + 1, last(cols) - first(cols) + 1)
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
  (_, cols) = lower_block_ranges(lbc, l)
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
  (_, cols) = lower_block_ranges(lower_blocks, m, n, l)
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
  (rows, _) = lower_block_ranges(lbc, l)
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
  (rows, _) = lower_block_ranges(lower_blocks, m, n, l)
  j ∈ rows
 end

"""
    function upper_block_ranges(
      lbc::LeadingBandColumn,
      l::Integer,
    )

Get ranges for upper block ``l``.
"""
@inline function upper_block_ranges(lbc::LeadingBandColumn, l::Integer)
  (m, n) = size(lbc)
  upper_block_ranges(lbc.upper_blocks, m, n, l)
end

"""
    upper_block_ranges(
      upper_blocks::Array{Int,2},
      m :: Int,
      n :: Int,
      l::Integer
    )

For upper blocks and a given matrix size m×n, compute ranges for upper
block ``l``.
"""
@inline function upper_block_ranges(
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
    j_last = upper_blocks[1, l]
    k_first = upper_blocks[2, l] + 1
    (UnitRange(1, j_last), UnitRange(k_first, n))
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
  (rows, cols) = upper_block_ranges(upper_blocks, m, n, l)
  (last(rows) - first(rows) + 1, last(cols) - first(cols) + 1)
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
  (rows, cols) = upper_block_ranges(lbc, l)
  (last(rows) - first(rows) + 1, last(cols) - first(cols) + 1)
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
  (_, cols) = upper_block_ranges(lbc, l)
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
  (_, cols) = upper_block_ranges(upper_blocks, m, n, l)
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
  (rows, _) = upper_block_ranges(lbc, l)
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
  (rows, _) = upper_block_ranges(upper_blocks, m, n, l)
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

Set lower bandwidth appropriate for a leading decomposition associated
with a given lower rank sequence.
"""
function leading_lower_ranks_to_cols_first_last!(
  lbc::LeadingBandColumn,
  rs::AbstractArray{Int},
)

  (m, n) = size(lbc)
  rs1 = leading_constrain_lower_ranks(lbc.lower_blocks, m, n, rs)

  for l = 1:lbc.num_blocks
    _, cols0 = lower_block_ranges(lbc, l)
    k0=last(cols0)
    rows1, _ = lower_block_ranges(lbc, l+1)
    j1=first(rows1)
    lbc.cols_first_last[5, (k0 - rs1[l] + 1):k0] .= j1-1
  end
end

"""
    leading_upper_ranks_to_cbws!(
      lbc::LeadingBandColumn,
      rs::AbstractArray{Int},
    )

Set upper bandwidth appropriate for a leading decomposition associated
with a given upper rank sequence
"""
function leading_upper_ranks_to_cols_first_last!(
  lbc::LeadingBandColumn,
  rs::AbstractArray{Int},
)

  (m, n) = size(lbc)
  rs1 = leading_constrain_upper_ranks(lbc.upper_blocks, m, n, rs)

  for l = 1:lbc.num_blocks
    (rows0, cols0) = upper_block_ranges(lbc, l)
    j0 = last(rows0)
    k0 = first(cols0)
    (_, cols1) = upper_block_ranges(lbc, l + 1)
    k1 = first(cols1)
    lbc.cols_first_last[2, k0:(k1 - 1)] .= j0 - rs1[l] + 1
  end
end

@views function BandColumnMatrices.wilk(lbc::LeadingBandColumn)
  (m, n) = size(lbc)
  a = fill('N', (2 * m, 2 * n))
  # insert spaces
  fill!(a[2:2:(2 * m), :], ' ')
  fill!(a[:, 2:2:(2 * n)], ' ')
  #insert boundaries for lower and upper blocks.
  for i = 1:lbc.num_blocks
    jl = lbc.lower_blocks[1, i]
    kl = lbc.lower_blocks[2, i]
    fill!(a[(2 * jl + 1):(2 * m - 1), 2 * kl], '|')
    a[2 * jl, 2 * kl] = '+'
    fill!(a[2 * jl, 1:(2 * kl - 1)], '-')
    for kk = 1:(2 * n)
      if a[2 * jl, kk] == '-' && a[2 * jl - 1, kk] == '|'
        a[2 * jl, kk] = '+'
      end
    end

    ju = lbc.upper_blocks[1, i]
    ku = lbc.upper_blocks[2, i]
    # fill!(a[(2 * ju + 1):(2 * m), 2 * ku], '|')
    fill!(a[1:(2 * ju - 1), 2 * ku], '|')
    a[2 * ju, 2 * ku] = '+'
    fill!(a[2 * ju, (2 * ku + 1):2*n], '-')
    for jj = 1:(2 * m)
      if a[jj, 2 * ku] == '|' && a[jj, 2 * ku - 1] == '-'
        a[jj, 2 * ku] = '+'
      end
    end
  end
  for k = 1:n
    kk = 2 * k - 1
    fill!(a[2 .* storable_index_range(lbc, :, k) .- 1, kk], 'O')
    fill!(a[2 .* upper_inband_index_range(lbc, :, k) .- 1, kk], 'U')
    fill!(a[2 .* middle_inband_index_range(lbc, :, k) .- 1, kk], 'X')
    fill!(a[2 .* lower_inband_index_range(lbc, :, k) .- 1, kk], 'L')
  end
  Wilk(a)
end

function Base.copy(lbc::LeadingBandColumn)
  LeadingBandColumn(
    lbc.m,
    lbc.n,
    lbc.bw_max,
    lbc.upper_bw_max,
    lbc.middle_lower_bw_max,
    copy(lbc.rows_first_last),
    copy(lbc.cols_first_last),
    copy(lbc.band_elements),
    lbc.num_blocks,
    lbc.upper_blocks,
    lbc.lower_blocks,
  )
end


end
