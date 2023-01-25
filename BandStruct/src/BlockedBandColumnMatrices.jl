module BlockedBandColumnMatrices
using Printf
using Random

using InPlace

using ..BandColumnMatrices
using ..BandwidthInit

export BlockedBandColumn,
  UpperBlock,
  LowerBlock,
  Decomp,
  LeadingDecomp,
  TrailingDecomp,
  get_middle_bw_max,
  get_upper_bw_max,
  get_lower_bw_max,
  view_lower_block,
  view_upper_block

struct UpperBlock end
struct LowerBlock end

abstract type Decomp end

struct LeadingDecomp <: Decomp end
Base.iterate(t::LeadingDecomp) = (t, nothing)
Base.iterate(::LeadingDecomp, ::Any) = nothing

struct TrailingDecomp <: Decomp end
Base.iterate(t::TrailingDecomp) = (t, nothing)
Base.iterate(::TrailingDecomp, ::Any) = nothing


"""

# BlockedBandColumn

    struct BlockedBandColumn{
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
`BlockedBandColumn`, these are determined by the leading blocks.

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

    bbc.band_elements = 

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
struct BlockedBandColumn{
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
    BlockedBandColumn(
      ::Type{E},
      m::Int,
      n::Int;
      upper_bw_max::Int,
      lower_bw_max::Int,
      upper_blocks::Array{Int,2},
      lower_blocks::Array{Int,2},
    ) where {E<:Number}

Construct an empty (all zero) `BlockedBandColumn` structure from the
matrix size, blocksizes, and either upper/lower rank bounds or
upper/lower bandwidth bounds.  With the upper/lower rank bounds, it
provides enough bandwidth for either a leading or trailing
decomposition and extra bandwidth for conversion between them.
"""
function BlockedBandColumn(
  ::Type{E},
  m::Int,
  n::Int;
  upper_bw_max::Union{Nothing,Int}=nothing,
  lower_bw_max::Union{Nothing,Int}=nothing,
  upper_rank_max::Union{Nothing,Int}=nothing,
  lower_rank_max::Union{Nothing,Int}=nothing,
  upper_blocks::Array{Int,2},
  lower_blocks::Array{Int,2},
) where {E<:Number}

  num_blocks = size(lower_blocks, 2)
  cols_first_last = similar_zeros(upper_blocks, 6, n)
  rows_first_last = similar_zeros(upper_blocks, m, 6)
  # Upper and lower ranks
  if isa(upper_rank_max, Int) && isa(lower_rank_max, Int) &&
    isa(upper_bw_max, Nothing) && isa(lower_bw_max, Nothing)

    get_cols_first_last!(
      m,
      n,
      upper_blocks,
      lower_blocks,
      upper_rank_max,
      lower_rank_max,
      cols_first_last,
    )
    get_rows_first_last!(
      m,
      n,
      upper_blocks,
      lower_blocks,
      upper_rank_max,
      lower_rank_max,
      rows_first_last,
    )
    middle_bw_max = get_middle_bw_max(m, n, cols_first_last)
    ubw_max = get_upper_bw_max(m, n, cols_first_last)
    lbw_max = get_lower_bw_max(m, n, cols_first_last)
    bw_max = ubw_max + middle_bw_max + lbw_max

  #Upper and lower bandwidths  
  elseif isa(upper_rank_max, Nothing) && isa(lower_rank_max, Nothing) &&
    isa(upper_bw_max, Int) && isa(lower_bw_max, Int)

    get_cols_first_last!(m, n, upper_blocks, lower_blocks, 0, 0, cols_first_last)
    get_rows_first_last!(m, n, upper_blocks, lower_blocks, 0, 0, rows_first_last)

    middle_bw_max = get_middle_bw_max(m, n, cols_first_last)
    ubw_max = upper_bw_max
    lbw_max = lower_bw_max
    bw_max = ubw_max + middle_bw_max + lbw_max

    for k = 1:n
      j0 = cols_first_last[3,k] - ubw_max
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

  else

    error("BlockedBandColumn must specify ranks or bandwidths (and not both).")

  end
  
  middle_lower_bw_max = middle_bw_max + lbw_max
  band_elements = similar_zeros(upper_blocks, E, bw_max, n)

  # Set the ranges for storable elements.

  BlockedBandColumn(
    m,
    n,
    bw_max,
    ubw_max,
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
    function lower_block_ranges(
      bbc::BlockedBandColumn,
      l::Integer,
    )

Get ranges for lower block ``l``.
"""
@inline function BandwidthInit.lower_block_ranges(
  bbc::BlockedBandColumn,
  l::Int,
)
  
  (m, n) = size(bbc)
  lower_block_ranges(bbc.lower_blocks, m, n, l)
end

"""
    function BandwidthInit.view_lower_block(
      bbc::BlockedBandColumn,
      l::Int,
    )

Get a view of lower block l.
"""
@inline function view_lower_block(
  bbc::BlockedBandColumn,
  l::Int,
)
  (rows, cols) = lower_block_ranges(bbc, l)
  view(bbc, rows, cols)
end

"""
    size_lower_block(
      bbc::BlockedBandColumn,
      l::Int,
    ) 

Compute the size of lower block ``l`` for a `BlockedBandColumn.
"""
@inline function BandwidthInit.size_lower_block(
  bbc::BlockedBandColumn,
  l::Int,
)
  (rows, cols) = lower_block_ranges(bbc, l)
  (last(rows) - first(rows) + 1, last(cols) - first(cols) + 1)
end

"""
    function intersect_lower_block(
      bbc::BlockedBandColumn,
      l::Integer,
      ::Colon,
      k::Int
    )

Determine if column ``k`` intersects with lower block ``l``
in a `BlockedBandColumn`.
"""
@inline function BandwidthInit.intersect_lower_block(
  bbc::BlockedBandColumn,
  l::Integer,
  ::Colon,
  k::Int
)
  (_, cols) = lower_block_ranges(bbc, l)
  k ∈ cols
end

"""
    intersect_lower_block(
      bbc::BlockedBandColumn,
      l::Integer,
      j::Int,
      ::Colon,
    )

Determine if row ``j`` intersects with lower block ``l``
in a `BlockedBandColumn`.
"""
@inline function BandwidthInit.intersect_lower_block(
  bbc::BlockedBandColumn,
  l::Integer,
  j::Int,
  ::Colon,
)
  (rows, _) = lower_block_ranges(bbc, l)
  j ∈ rows
 end

"""
    function upper_block_ranges(
      bbc::BlockedBandColumn,
      l::Integer,
    )

Get ranges for upper block ``l``.
"""
@inline function BandwidthInit.upper_block_ranges(
  bbc::BlockedBandColumn,
  l::Integer,
)
  (m, n) = size(bbc)
  upper_block_ranges(bbc.upper_blocks, m, n, l)
end

"""
    function BandwidthInit.view_upper_block(
      bbc::BlockedBandColumn,
      l::Int,
    )

Get a view of upper block l.
"""
@inline function view_upper_block(
  bbc::BlockedBandColumn,
  l::Int,
)
  (rows, cols) = upper_block_ranges(bbc, l)
  view(bbc, rows, cols)
end

"""
    size_upper_block(
      bbc::BlockedBandColumn,
      l::Int,
    ) 

Compute the size of lower block ``l`` for a `BlockedBandColumn.
"""
@inline function BandwidthInit.size_upper_block(
  bbc::BlockedBandColumn,
  l::Int,
)
  (rows, cols) = upper_block_ranges(bbc, l)
  (last(rows) - first(rows) + 1, last(cols) - first(cols) + 1)
end

"""
    function intersect_upper_block(
      bbc::BlockedBandColumn,
      l::Integer,
      ::Colon,
      k::Int
    )

Determine if column ``k`` intersects with upper block ``l``
in a `BlockedBandColumn`.
"""
@inline function BandwidthInit.intersect_upper_block(
  bbc::BlockedBandColumn,
  l::Integer,
  ::Colon,
  k::Int
)
  (_, cols) = upper_block_ranges(bbc, l)
  k ∈ cols
 end

"""
    intersect_upper_block(
      bbc::BlockedBandColumn,
      l::Integer,
      j::Int,
      ::Colon,
    )

Determine if row ``j`` intersects with upper block ``l``
in a `BlockedBandColumn`.
"""
@inline function BandwidthInit.intersect_upper_block(
  bbc::BlockedBandColumn,
  l::Integer,
  j::Int,
  ::Colon,
)
  (rows, _) = upper_block_ranges(bbc, l)
  j ∈ rows
 end

function get_middle_bw_max(::Int, n::Int, cols_first_last::AbstractArray{Int,2})
  middle_bw_max = 0
  for k ∈ 1:n
    middle_bw_max =
      max(middle_bw_max, cols_first_last[4, k] - cols_first_last[3, k] - 1)
  end
  middle_bw_max
end

function get_upper_bw_max(::Int, n::Int, cols_first_last::AbstractArray{Int,2})
  upper_bw_max = 0
  for k ∈ 1:n
    upper_bw_max =
      max(upper_bw_max, cols_first_last[3, k] - cols_first_last[1, k] + 1)
  end
  upper_bw_max
end

function get_lower_bw_max(::Int, n::Int, cols_first_last::AbstractArray{Int,2})
  lower_bw_max = 0
  for k ∈ 1:n
    lower_bw_max =
      max(lower_bw_max, cols_first_last[6, k] - cols_first_last[4, k] + 1)
  end
  lower_bw_max
end

function Base.fill!(
  bbc::BlockedBandColumn{E},
  x::E
) where {E}

  for k = 1:bbc.n
    for j = inband_index_range_storage(bbc, k)
      bbc.band_elements[j, k] = x
    end
  end
end

function Random.rand!(
  rng::AbstractRNG,
  bbc::BlockedBandColumn{E},
) where {E}

  for k = 1:bbc.n
    for j = inband_index_range_storage(bbc, k)
      bbc.band_elements[j, k] = rand(rng, E)
    end
  end
end

function Random.rand!(
  bbc::BlockedBandColumn{E},
) where {E}

  rand!(Random.default_rng(), bbc)
end

# Random elements for a particular decomposition.
function Random.rand!(
  ::LeadingDecomp,
  rng::AbstractRNG,
  bbc::BlockedBandColumn{E}
) where {E}

  for k = 1:bbc.n
    for j = inband_index_range_storage(bbc, k)
      bbc.band_elements[j, k] = rand(rng, E)
    end
  end
end


"""
    BlockedBandColumn(
      ::Type{E},
      ::LeadingDecomp,
      rng::AbstractRNG,
      m::Int,
      n::Int;
      upper_ranks::Array{Int,1},
      lower_ranks::Array{Int,1},
      upper_rank_max::Int=maximum(upper_ranks),
      lower_rank_max::Int=maximum(lower_ranks),
      upper_blocks::Array{Int,2},
      lower_blocks::Array{Int,2},
    ) where {E<:Number}

Construct a random LeadingDecomp `BlockedBandColumn` structure from
the matrix size, blocksizes, upper/lower rank bounds, and upper/lower
ranks.  It provides enough bandwidth for conversion between leading and
trailing decompositions.
"""
function BlockedBandColumn(
  ::Type{E},
  ::LeadingDecomp,
  rng::AbstractRNG,
  m::Int,
  n::Int;
  upper_ranks::Array{Int,1},
  lower_ranks::Array{Int,1},
  upper_rank_max::Int=maximum(upper_ranks),
  lower_rank_max::Int=maximum(lower_ranks),
  upper_blocks::Array{Int,2},
  lower_blocks::Array{Int,2},
) where {E<:Number}
  bbc = BlockedBandColumn(
    E,
    m,
    n,
    upper_bw_max = nothing,
    lower_bw_max = nothing,
    upper_rank_max = upper_rank_max,
    lower_rank_max = lower_rank_max,
    upper_blocks = upper_blocks,
    lower_blocks = lower_blocks,
  )
  leading_lower_ranks_to_cols_first_last!(bbc, lower_ranks)
  leading_upper_ranks_to_cols_first_last!(bbc, upper_ranks)
  compute_rows_first_last!(bbc)
  rand!(rng, bbc)
  bbc
end

function BlockedBandColumn(
  ::Type{E},
  ::LeadingDecomp,
  x::E,
  m::Int,
  n::Int;
  upper_ranks::Array{Int,1},
  lower_ranks::Array{Int,1},
  upper_rank_max::Int=maximum(upper_ranks),
  lower_rank_max::Int=maximum(lower_ranks),
  upper_blocks::Array{Int,2},
  lower_blocks::Array{Int,2},
) where {E<:Number}
  bbc = BlockedBandColumn(
    E,
    m,
    n,
    upper_bw_max = nothing,
    lower_bw_max = nothing,
    upper_rank_max = upper_rank_max,
    lower_rank_max = lower_rank_max,
    upper_blocks = upper_blocks,
    lower_blocks = lower_blocks,
  )
  leading_lower_ranks_to_cols_first_last!(bbc, lower_ranks)
  leading_upper_ranks_to_cols_first_last!(bbc, upper_ranks)
  compute_rows_first_last!(bbc)
  fill!(bbc, x)
  bbc
end

"""
    BlockedBandColumn(
      ::Type{E},
      ::TrailingDecomp,
      rng::AbstractRNG,
      m::Int,
      n::Int;
      upper_ranks::Array{Int,1},
      lower_ranks::Array{Int,1},
      upper_rank_max::Int=maximum(upper_ranks),
      lower_rank_max::Int=maximum(lower_ranks),
      upper_blocks::Array{Int,2},
      lower_blocks::Array{Int,2},
    ) where {E<:Number}

Construct a random TrailingDecomp `BlockedBandColumn` structure from
the matrix size, blocksizes, upper/lower rank bounds, and upper/lower
ranks.  It provides enough bandwidth for conversion between leading
and trailing decompositions.
"""
function BlockedBandColumn(
  ::Type{E},
  ::TrailingDecomp,
  rng::AbstractRNG,
  m::Int,
  n::Int;
  upper_ranks::Array{Int,1},
  lower_ranks::Array{Int,1},
  upper_rank_max::Int=maximum(upper_ranks),
  lower_rank_max::Int=maximum(lower_ranks),
  upper_blocks::Array{Int,2},
  lower_blocks::Array{Int,2},
) where {E<:Number}
  bbc = BlockedBandColumn(
    E,
    m,
    n,
    upper_bw_max = nothing,
    lower_bw_max = nothing,
    upper_rank_max = upper_rank_max,
    lower_rank_max = lower_rank_max,
    upper_blocks = upper_blocks,
    lower_blocks = lower_blocks,
  )
  trailing_lower_ranks_to_cols_first_last!(bbc, lower_ranks)
  trailing_upper_ranks_to_cols_first_last!(bbc, upper_ranks)
  compute_rows_first_last!(bbc)
  rand!(rng, bbc)
  bbc
end

function BlockedBandColumn(
  ::Type{E},
  ::TrailingDecomp,
  x::E,
  m::Int,
  n::Int;
  upper_ranks::Array{Int,1},
  lower_ranks::Array{Int,1},
  upper_rank_max::Int=maximum(upper_ranks),
  lower_rank_max::Int=maximum(lower_ranks),
  upper_blocks::Array{Int,2},
  lower_blocks::Array{Int,2},
) where {E<:Number}
  bbc = BlockedBandColumn(
    E,
    m,
    n,
    upper_bw_max = nothing,
    lower_bw_max = nothing,
    upper_rank_max = upper_rank_max,
    lower_rank_max = lower_rank_max,
    upper_blocks = upper_blocks,
    lower_blocks = lower_blocks,
  )
  trailing_lower_ranks_to_cols_first_last!(bbc, lower_ranks)
  trailing_upper_ranks_to_cols_first_last!(bbc, upper_ranks)
  compute_rows_first_last!(bbc)
  fill!(bbc, x)
  bbc
end

"""
    BlockedBandColumn(
      ::Type{E},
      D::Decomp,
      m::Int,
      n::Int;
      upper_ranks::Array{Int,1},
      lower_ranks::Array{Int,1},
      upper_rank_max::Int=maximum(upper_ranks),
      lower_rank_max::Int=maximum(lower_ranks),
      upper_blocks::Array{Int,2},
      lower_blocks::Array{Int,2},
    ) where {E<:Number}

Construct a random LeadingDecomp or TrailingDecomp `BlockedBandColumn`
structure from the matrix size, blocksizes, upper/lower rank bounds,
and upper/lower ranks.  It provides enough bandwidth for conversion
between leading and trailing decompositions.  This uses the standard
random number generator.
"""
function BlockedBandColumn(
  ::Type{E},
  D::Decomp,
  m::Int,
  n::Int;
  upper_ranks::Array{Int,1},
  lower_ranks::Array{Int,1},
  upper_rank_max::Int=maximum(upper_ranks),
  lower_rank_max::Int=maximum(lower_ranks),
  upper_blocks::Array{Int,2},
  lower_blocks::Array{Int,2},
) where {E<:Number}
  BlockedBandColumn(
    E,
    D,
    Random.default_rng(),
    m,
    n,
    upper_rank_max = upper_rank_max,
    lower_rank_max = lower_rank_max,
    upper_blocks = upper_blocks,
    lower_blocks = lower_blocks,
    upper_ranks = upper_ranks,
    lower_ranks = lower_ranks,
  )
end

@inline BandColumnMatrices.toBandColumn(bbc::BlockedBandColumn) = BandColumn(
  NonSub(),
  bbc.m,
  bbc.n,
  bbc.m,
  bbc.n,
  0,
  0,
  bbc.bw_max,
  bbc.upper_bw_max,
  bbc.middle_lower_bw_max,
  bbc.rows_first_last,
  bbc.cols_first_last,
  bbc.band_elements,
)

#=

Bandwidth functions required by AbstractBandColumn.

=#

@inline BandColumnMatrices.bw_max(bbc::BlockedBandColumn) =
  bbc.bw_max
@inline BandColumnMatrices.upper_bw_max(bbc::BlockedBandColumn) =
  bbc.upper_bw_max
@inline BandColumnMatrices.middle_lower_bw_max(bbc::BlockedBandColumn) =
  bbc.middle_lower_bw_max
@inline BandColumnMatrices.band_elements(bbc::BlockedBandColumn) =
  bbc.band_elements

#=

Index functions required by AbstractBandColumn

=#

Base.@propagate_inbounds function BandColumnMatrices.storable_index_range(
  ::Type{NonSub},
  bc::BlockedBandColumn,
  ::Colon,
  k::Int,
)
  bc.cols_first_last[1,k]:bc.cols_first_last[6,k]
end

Base.@propagate_inbounds function BandColumnMatrices.storable_index_range(
  ::Type{NonSub},
  bc::BlockedBandColumn,
  j::Int,
  ::Colon,
)
  bc.rows_first_last[j,1]:bc.rows_first_last[j,6]
end


Base.@propagate_inbounds function BandColumnMatrices.inband_index_range(
  ::Type{NonSub},
  bc::BlockedBandColumn,
  ::Colon,
  k::Int,
)
  bc.cols_first_last[2, k]:bc.cols_first_last[5, k]
end

Base.@propagate_inbounds function BandColumnMatrices.inband_index_range(
  ::Type{NonSub},
  bc::BlockedBandColumn,
  j::Int,
  ::Colon,
)
  bc.rows_first_last[j, 2]:bc.rows_first_last[j, 5]
end

Base.@propagate_inbounds BandColumnMatrices.first_lower_index(
  ::Type{NonSub},
  bbc::BlockedBandColumn,
  ::Colon,
  k::Int,
) = bbc.cols_first_last[4,k]

Base.@propagate_inbounds BandColumnMatrices.last_lower_index(
  ::Type{NonSub},
  bbc::BlockedBandColumn,
  j::Int,
  ::Colon,
) = bbc.rows_first_last[j,3]

Base.@propagate_inbounds BandColumnMatrices.first_upper_index(
  ::Type{NonSub},
  bbc::BlockedBandColumn,
  j::Int,
  ::Colon,
) = bbc.rows_first_last[j,4]

Base.@propagate_inbounds BandColumnMatrices.last_upper_index(
  ::Type{NonSub},
  bbc::BlockedBandColumn,
  ::Colon,
  k::Int,
) = bbc.cols_first_last[3,k]

@inline function BandColumnMatrices.unsafe_set_first_inband_index!(
  ::Type{NonSub},
  bbc::BlockedBandColumn,
  js::AbstractUnitRange{Int},
  ::Colon,
  k_first::Int,
) 
  bbc.rows_first_last[js, 2] .= k_first
  nothing
end

@inline function BandColumnMatrices.unsafe_set_first_inband_index!(
  ::Type{NonSub},
  bbc::BlockedBandColumn,
  ::Colon,
  ks::AbstractUnitRange{Int},
  j_first::Int,
) 
  bbc.cols_first_last[2, ks] .= j_first
  nothing
end

@inline function BandColumnMatrices.unsafe_set_last_inband_index!(
  ::Type{NonSub},
  bbc::BlockedBandColumn,
  js::AbstractUnitRange{Int},
  ::Colon,
  k_last::Int,
) 
  bbc.rows_first_last[js, 5] .= k_last
  nothing
end

@inline function BandColumnMatrices.unsafe_set_last_inband_index!(
  ::Type{NonSub},
  bbc::BlockedBandColumn,
  ::Colon,
  ks::AbstractUnitRange{Int},
  j_last::Int,
) 
  bbc.cols_first_last[5, ks] .= j_last
  nothing
end

@inline BandColumnMatrices.row_size(bbc::BlockedBandColumn) = bbc.m
@inline BandColumnMatrices.col_size(bbc::BlockedBandColumn) = bbc.n
@inline BandColumnMatrices.row_size(::Type{NonSub}, bbc::BlockedBandColumn) =
  bbc.m
@inline BandColumnMatrices.col_size(::Type{NonSub}, bbc::BlockedBandColumn) =
  bbc.n

function BandColumnMatrices.compute_rows_first_last!(bbc::BlockedBandColumn)
  compute_rows_first_last!(bbc, bbc.rows_first_last)
end

function BandColumnMatrices.validate_rows_first_last(
  bbc::BlockedBandColumn,
)
  rfl = compute_rows_first_last(bbc)
  @views rfl[:,2] == bbc.rows_first_last[:,2]
  @views rfl[:,5] == bbc.rows_first_last[:,5]
end


function Base.show(io::IO, bbc::BlockedBandColumn)
  print(
    io,
    typeof(bbc),
    "(",
    bbc.m,
    ", ",
    bbc.n,
    ", ",
    bbc.bw_max,
    ", ",
    bbc.upper_bw_max,
    ", ",
    bbc.middle_lower_bw_max,
    ", ",
    bbc.rows_first_last,
    ", ",
    bbc.cols_first_last,
    ", ",
    bbc.num_blocks,
    ", ",
    bbc.lower_blocks,
    ", ",
    bbc.upper_blocks,
    "): ",
  )
  for j = 1:bbc.m
    println()
    for k = 1:bbc.n
      if check_bc_storage_bounds(Bool, bbc, j, k)
        if is_inband(bbc, j, k)
          @printf("%10.2e", bbc[j, k])
        else
          print("         O")
        end
      else
        print("         N")
      end
    end
  end
end

Base.show(io::IO, ::MIME"text/plain", bbc::BlockedBandColumn) = show(io, bbc)

Base.print(io::IO, bbc::BlockedBandColumn) = print(
  io,
  typeof(bbc),
  "(",
  bbc.m,
  ", ",
  bbc.n,
  ", ",
  bbc.bw_max,
  ", ",
  bbc.upper_bw_max,
  ", ",
  bbc.middle_lower_bw_max,
  ", ",
  bbc.rows_first_last,
  ", ",
  bbc.cols_first_last,
  ", ",
  bbc.band_elements,
  ", ",
  bbc.num_blocks,
  ", ",
  bbc.upper_blocks,
  ", ",
  bbc.lower_blocks,
  ")",
)

Base.print(io::IO, ::MIME"text/plain", bbc::BlockedBandColumn) = print(io, bbc)

##
## Index operations.  Scalar operations are defined for
## AbstractBandColumn matrices.
##

@inline function BandColumnMatrices.viewbc(
  bbc::BlockedBandColumn,
  i::Tuple{AbstractUnitRange{Int},AbstractUnitRange{Int}},
)
  (rows, cols) = i
  j0 = first(rows)
  j1 = last(rows)
  k0 = first(cols)
  k1 = last(cols)
  
  @boundscheck begin
    if j1 >= j0 && k1 >= k0
      checkbounds(bbc, j0, k0)
      checkbounds(bbc, j1, k1)
    end
  end
  BandColumn(
    Sub(),
    bbc.m,
    bbc.n,
    max(j1 - j0 + 1, 0),
    max(k1 - k0 + 1, 0),
    j0 - 1,
    k0 - 1,
    bbc.bw_max,
    bbc.upper_bw_max,
    bbc.middle_lower_bw_max,
    bbc.rows_first_last,
    bbc.cols_first_last,
    bbc.band_elements,
  )
end

@inline function Base.getindex(
  bbc::BlockedBandColumn,
  rows::AbstractUnitRange{Int},
  cols::AbstractUnitRange{Int},
)
  j0 = first(rows)
  j1 = last(rows)
  k0 = first(cols)
  k1 = last(cols)

  @boundscheck begin
    checkbounds(bbc, j0, k0)
    checkbounds(bbc, j1, k1)
  end
  BandColumn(
    Sub(),
    bbc.m,
    bbc.n,
    max(j1 - j0 + 1, 0),
    max(k1 - k0 + 1, 0),
    j0 - 1,
    k0 - 1,
    bbc.bw_max,
    bbc.upper_bw_max,
    bbc.middle_lower_bw_max,
    copy(bbc.rows_first_last),
    copy(bbc.cols_first_last),
    copy(bbc.band_elements),
  )
end

"""
    function lower_block_ranges(
      bbc::BlockedBandColumn,
      l::Integer,
    )

Get ranges for lower block ``l``.
"""
@inline function lower_block_ranges(
  bbc::BlockedBandColumn,
  l::Int,
)
  
  (m, n) = size(bbc)
  lower_block_ranges(bbc.lower_blocks, m, n, l)
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
    size_lower_block(
      bbc::BlockedBandColumn,
      l::Int,
    ) 

Compute the size of lower block ``l`` for a `BlockedBandColumn.
"""
@inline function size_lower_block(
  bbc::BlockedBandColumn,
  l::Int,
)
  (rows, cols) = lower_block_ranges(bbc, l)
  (last(rows) - first(rows) + 1, last(cols) - first(cols) + 1)
end

"""
    function intersect_lower_block(
      bbc::BlockedBandColumn,
      l::Integer,
      ::Colon,
      k::Int
    )

Determine if column ``k`` intersects with lower block ``l``
in a `BlockedBandColumn`.
"""
@inline function intersect_lower_block(
  bbc::BlockedBandColumn,
  l::Integer,
  ::Colon,
  k::Int
)
  (_, cols) = lower_block_ranges(bbc, l)
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
      bbc::BlockedBandColumn,
      l::Integer,
      j::Int,
      ::Colon,
    )

Determine if row ``j`` intersects with lower block ``l``
in a `BlockedBandColumn`.
"""
@inline function intersect_lower_block(
  bbc::BlockedBandColumn,
  l::Integer,
  j::Int,
  ::Colon,
)
  (rows, _) = lower_block_ranges(bbc, l)
  j ∈ rows
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
    function upper_block_ranges(
      bbc::BlockedBandColumn,
      l::Integer,
    )

Get ranges for upper block ``l``.
"""
@inline function upper_block_ranges(bbc::BlockedBandColumn, l::Integer)
  (m, n) = size(bbc)
  upper_block_ranges(bbc.upper_blocks, m, n, l)
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
    size_upper_block(
      bbc::BlockedBandColumn,
      l::Int,
    ) 

Compute the size of lower block ``l`` for a `BlockedBandColumn.
"""
@inline function size_upper_block(
  bbc::BlockedBandColumn,
  l::Int,
)
  (rows, cols) = upper_block_ranges(bbc, l)
  (last(rows) - first(rows) + 1, last(cols) - first(cols) + 1)
end

"""
    function intersect_upper_block(
      bbc::BlockedBandColumn,
      l::Integer,
      ::Colon,
      k::Int
    )

Determine if column ``k`` intersects with upper block ``l``
in a `BlockedBandColumn`.
"""
@inline function intersect_upper_block(
  bbc::BlockedBandColumn,
  l::Integer,
  ::Colon,
  k::Int
)
  (_, cols) = upper_block_ranges(bbc, l)
  k ∈ cols
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
      bbc::BlockedBandColumn,
      l::Integer,
      j::Int,
      ::Colon,
    )

Determine if row ``j`` intersects with upper block ``l``
in a `BlockedBandColumn`.
"""
@inline function intersect_upper_block(
  bbc::BlockedBandColumn,
  l::Integer,
  j::Int,
  ::Colon,
)
  (rows, _) = upper_block_ranges(bbc, l)
  j ∈ rows
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
    leading_lower_ranks_to_cols_first_last!(
      bbc::BlockedBandColumn,
      rs::AbstractArray{Int,1},
    )

Set first_last indices appropriate for a leading decomposition
associated with a given lower rank sequence.
"""
function leading_lower_ranks_to_cols_first_last!(
  bbc::BlockedBandColumn,
  rs::AbstractArray{Int,1},
)

  m, n = size(bbc)
  rs1 = constrain_lower_ranks(m, n, blocks = bbc.lower_blocks, ranks = rs)
  for lb = 1:bbc.num_blocks
    rows_lb, cols_lb = lower_block_ranges(bbc, lb)
    rows_lb1, _ = lower_block_ranges(bbc, lb+1) # empty if lb+1 > num_blocks
    dᵣ = setdiffᵣ(rows_lb, rows_lb1)
    if !isempty(dᵣ)
      bbc.cols_first_last[5, last(cols_lb, rs1[lb])] .= last(dᵣ)
    end
  end
end

"""
    trailing_lower_ranks_to_cols_first_last!(
      bbc::BlockedBandColumn,
      rs::AbstractArray{Int,1},
    )

Set first_last indices appropriate for a trailing decomposition
associated with a given lower rank sequence.
"""
function trailing_lower_ranks_to_cols_first_last!(
  bbc::BlockedBandColumn,
  rs::AbstractArray{Int,1},
)

  m, n = size(bbc)
  rs1 = constrain_lower_ranks(m, n, blocks = bbc.lower_blocks, ranks = rs)
  for lb = bbc.num_blocks:-1:1
    rows_lb, cols_lb = lower_block_ranges(bbc, lb)
    _, cols_lb1 = lower_block_ranges(bbc, lb-1) # empty if lb-1 < 1
    dᵣ = setdiffᵣ(cols_lb, cols_lb1)
    if !isempty(dᵣ)
      rows_lb_first = isempty(rows_lb) ? m : first(rows_lb)
      bbc.cols_first_last[5, dᵣ] .= min(m, rows_lb_first + rs1[lb] - 1)
    end
  end
end

"""
    leading_upper_ranks_to_cols_first_last!(
      bbc::BlockedBandColumn,
      rs::AbstractArray{Int,1},
    )

Set first_last indices appropriate for a leading decomposition associated
with a given upper rank sequence
"""
function leading_upper_ranks_to_cols_first_last!(
  bbc::BlockedBandColumn,
  rs::AbstractArray{Int,1},
)

  m, n = size(bbc)
  rs1 = constrain_upper_ranks(m, n, blocks = bbc.upper_blocks, ranks = rs)

  for ub = 1:bbc.num_blocks
    rows_ub, cols_ub = upper_block_ranges(bbc, ub)
    _, cols_ub1 = upper_block_ranges(bbc, ub+1) # empty if ub+1 > num_blocks
    dᵣ = setdiffᵣ(cols_ub, cols_ub1)
    if !isempty(dᵣ)
      rows_ub_last = isempty(rows_ub) ? 0 : last(rows_ub)
      bbc.cols_first_last[2, dᵣ] .= 
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
  bbc::BlockedBandColumn,
  rs::AbstractArray{Int,1},
)

  m, n = size(bbc)
  rs1 = constrain_upper_ranks(m, n, blocks = bbc.upper_blocks, ranks = rs)
  for ub = bbc.num_blocks:-1:1
    rows_ub, cols_ub = upper_block_ranges(bbc, ub)
    rows_ub1, _ = upper_block_ranges(bbc, ub - 1) # empty if ub-1 < 1
    dᵣ = setdiffᵣ(rows_ub, rows_ub1)
    if !isempty(dᵣ)
      bbc.cols_first_last[2, first(cols_ub, rs1[ub])] .= first(dᵣ)
    end
  end
end

@views function BandColumnMatrices.wilk(bbc::BlockedBandColumn)
  (m, n) = size(bbc)
  a = fill('N', (2 * m, 2 * n))
  # insert spaces
  fill!(a[2:2:(2 * m), :], ' ')
  fill!(a[:, 2:2:(2 * n)], ' ')
  #insert boundaries for lower and upper blocks.
  for i = 1:bbc.num_blocks
    jl = bbc.lower_blocks[1, i]
    kl = bbc.lower_blocks[2, i]
    fill!(a[(2 * jl + 1):(2 * m - 1), 2 * kl], '|')
    a[2 * jl, 2 * kl] = '+'
    fill!(a[2 * jl, 1:(2 * kl - 1)], '-')
    for kk = 1:(2 * n)
      if a[2 * jl, kk] == '-' && a[2 * jl - 1, kk] == '|'
        a[2 * jl, kk] = '+'
      end
    end

    ju = bbc.upper_blocks[1, i]
    ku = bbc.upper_blocks[2, i]
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
    fill!(a[2 .* storable_index_range(bbc, :, k) .- 1, kk], 'O')
    fill!(a[2 .* upper_inband_index_range(bbc, :, k) .- 1, kk], 'U')
    fill!(a[2 .* middle_inband_index_range(bbc, :, k) .- 1, kk], 'X')
    fill!(a[2 .* lower_inband_index_range(bbc, :, k) .- 1, kk], 'L')
  end
  Wilk(a)
end

function Base.copy(bbc::BlockedBandColumn)
  BlockedBandColumn(
    bbc.m,
    bbc.n,
    bbc.bw_max,
    bbc.upper_bw_max,
    bbc.middle_lower_bw_max,
    copy(bbc.rows_first_last),
    copy(bbc.cols_first_last),
    copy(bbc.band_elements),
    bbc.num_blocks,
    bbc.upper_blocks,
    bbc.lower_blocks,
  )
end

end
