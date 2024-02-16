module BlockedBandColumnMatrices
using Printf
using Random
using ErrorTypes

using InPlace

using ..BandColumnMatrices
using ..BandwidthInit
using ..IndexLists

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

    struct BlockedBandColumn{E<:Number, BD<:AbstractBlockData} <:
           AbstractBandColumn{NonSub,E,Matrix{E},Matrix{Int}}
      m::Int
      n::Int
      bw_max::Int
      upper_bw_max::Int
      middle_lower_bw_max::Int
      rows_first_last::Matrix{Int}
      cols_first_last::Matrix{Int}
      band_elements::Matrix{E}
      upper_blocks :: IndexList{BD}
      lower_blocks :: IndexList{BD}
    end

A banded matrix with structure defined by leading blocks and stored in
a compressed column-wise format.  This is basically a `BandColumn`
with additional leading block information and without the `roffset`,
`coffset`, `sub`, `m_nosub`, and `n_nosub` fields.  The type `BD`
should be a struct that has fields `mb` and `nb`.  That is, it at
least provides leading block sizes for each partition, possibly along
with additional information.

# Fields

  - `m::Int`: Matrix number of rows.

  - `n::Int`: Matrix and elements number of columns.

  - `bw_max::Int`: Elements array number of rows.

  - `upper_bw_max::Int`: Maximum upper bandwidth.

  - `middle_lower_bw_max::Int`: Maximum middle + lower bandwidth.

  - `rows_first_last::Matrix{Int}`: `rows_first_last[j,:]` contains
   
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

  - `cols_first_last::Matrix{Int}`: `cols_first_last[:,k]` contains
   
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

  - `band_elements::Matrix{E}`: Column-wise storage of the band elements with
     dimensions:
   
     ``(upper_bw_max + middle_bw_max lower_bw_max) × n``

  - `upper_blocks::IndexList{BD}`: List of upper block data.

  - `lower_blocks::IndexList{BD}`: List of lower block data.

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

    upper_blocks =       IndexList[ BlockSize(1,3),
                                    BlockSize(3,4),
                                    BlockSize(4,6),
                                    BlockSize(6,7) ]

    lower_blocks =       IndexList[ BlockSize(2,2),
                                    BlockSize(4,3),
                                    BlockSize(5,4),
                                    BlockSize(7,6) ]
"""
struct BlockedBandColumn{E<:Number, BD<:AbstractBlockData} <:
       AbstractBandColumn{NonSub,E,Matrix{E},Matrix{Int}}
  m::Int
  n::Int
  bw_max::Int
  upper_bw_max::Int
  middle_lower_bw_max::Int
  rows_first_last::Matrix{Int}
  cols_first_last::Matrix{Int}
  band_elements::Matrix{E}
  upper_blocks :: IndexList{BD}
  lower_blocks :: IndexList{BD}
end

"""
    function BlockedBandColumn(
      ::Type{E},
      m::Int,
      n::Int;
      upper_bw_max::Union{Nothing,Int} = nothing,
      lower_bw_max::Union{Nothing,Int} = nothing,
      upper_rank_max::Union{Nothing,Int} = nothing,
      lower_rank_max::Union{Nothing,Int} = nothing,
      upper_blocks::Union{AbstractVector{BD},IndexList{BD}},
      lower_blocks::Union{AbstractVector{BD},IndexList{BD}},
      max_num_blocks::Int = max(
        length(upper_blocks),
        length(lower_blocks),
      ),
    ) where {E<:Number,BD<:AbstractBlockData}

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
  upper_bw_max::Union{Nothing,Int} = nothing,
  lower_bw_max::Union{Nothing,Int} = nothing,
  upper_rank_max::Union{Nothing,Int} = nothing,
  lower_rank_max::Union{Nothing,Int} = nothing,
  upper_blocks::Union{AbstractVector{BD},IndexList{BD}},
  lower_blocks::Union{AbstractVector{BD},IndexList{BD}},
  max_num_blocks::Int = max(
    length(upper_blocks),
    length(lower_blocks),
  ),
) where {E<:Number,BD<:AbstractBlockData}

  num_lower_blocks = length(lower_blocks)
  num_upper_blocks = length(upper_blocks)

  upper_blocks_list = IndexList(upper_blocks, max_length = max_num_blocks)
  lower_blocks_list = IndexList(lower_blocks, max_length = max_num_blocks)

  if isa(max_num_blocks, Nothing)
    num_lower_blocks == num_upper_blocks || error(
      "If max_num_blocks is not provided, BlockedBandColumn " *
      "requires num_lower_blocks == num_upper_blocks",
    )
    max_num_blocks = num_lower_blocks
  end

  cols_first_last = zeros(Int, 6, n)
  rows_first_last = zeros(Int, m, 6)

  # Upper and lower ranks
  if isa(upper_rank_max, Int) &&
     isa(lower_rank_max, Int) &&
     isa(upper_bw_max, Nothing) &&
     isa(lower_bw_max, Nothing)

    get_cols_first_last!(
      m = m,
      n = n,
      upper_blocks = upper_blocks_list,
      lower_blocks = lower_blocks_list,
      max_ru = upper_rank_max,
      max_rl = lower_rank_max,
      cols_first_last = cols_first_last,
    )

    get_rows_first_last!(
      m = m,
      n = n,
      upper_blocks = upper_blocks_list,
      lower_blocks = lower_blocks_list,
      max_ru = upper_rank_max,
      max_rl = lower_rank_max,
      rows_first_last = rows_first_last,
    )
    middle_bw_max = get_middle_bw_max(m, n, cols_first_last)
    ubw_max = get_upper_bw_max(m, n, cols_first_last)
    lbw_max = get_lower_bw_max(m, n, cols_first_last)
    bw_max = ubw_max + middle_bw_max + lbw_max

    #Upper and lower bandwidths
  elseif isa(upper_rank_max, Nothing) &&
         isa(lower_rank_max, Nothing) &&
         isa(upper_bw_max, Int) &&
         isa(lower_bw_max, Int)

    get_cols_first_last!(
      m=m,
      n=n,
      upper_blocks=upper_blocks_list,
      lower_blocks=lower_blocks_list,
      max_ru=0,
      max_rl=0,
      cols_first_last = cols_first_last,
    )

    get_rows_first_last!(
      m = m,
      n = n,
      upper_blocks = upper_blocks_list,
      lower_blocks = lower_blocks_list,
      max_ru = 0,
      max_rl = 0,
      rows_first_last = rows_first_last,
    )

    middle_bw_max = get_middle_bw_max(m, n, cols_first_last)
    ubw_max = upper_bw_max
    lbw_max = lower_bw_max
    bw_max = ubw_max + middle_bw_max + lbw_max

    for k = 1:n
      j0 = cols_first_last[3, k] - ubw_max
      cols_first_last[1, k] = project(1 + j0, m)
      cols_first_last[6, k] = project(bw_max + j0, m)
    end

    rows_first_last[:, 1] .= n + 1
    rows_first_last[:, 6] .= 0

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
  band_elements = zeros(E, bw_max, n)

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
    upper_blocks_list,
    lower_blocks_list,
  )
end

# Basic lower block functions

"""
    function BandwidthInit.lower_block_ranges(
      bbc::BlockedBandColumn,
      l::Union{ListIndex, Int, Result{ListIndex, BeforeAfterError}}
    )

Get ranges for lower block ``l``.
"""
@inline function BandwidthInit.lower_block_ranges(
  bbc::BlockedBandColumn,
  l::Union{ListIndex, Int, Result{ListIndex, BeforeAfterError}}
)
  m, n = size(bbc)
  lower_block_ranges(bbc.lower_blocks, m, n, l)
end

"""
    BandwidthInit.size_lower_block(
      bbc::BlockedBandColumn,
      l::Union{ListIndex, Int, Result{ListIndex, BeforeAfterError}}
    ) 

Compute the size of lower block ``l`` for a `BlockedBandColumn.
"""
@inline function BandwidthInit.size_lower_block(
  bbc::BlockedBandColumn,
  l::Union{ListIndex, Int, Result{ListIndex, BeforeAfterError}}
)
  (rows, cols) = lower_block_ranges(bbc, l)
  (last(rows) - first(rows) + 1, last(cols) - first(cols) + 1)
end

"""
    function view_lower_block(
      bbc::BlockedBandColumn,
      l::Union{ListIndex, Int, Result{ListIndex, BeforeAfterError}}
    )

Get a view of lower block l.
"""
@inline function view_lower_block(
  bbc::BlockedBandColumn,
  l::Union{ListIndex, Int, Result{ListIndex, BeforeAfterError}}
)
  (rows, cols) = lower_block_ranges(bbc, l)
  view(bbc, rows, cols)
end

# Basic upper block functions

"""
    function BandwidthInit.upper_block_ranges(
      bbc::BlockedBandColumn,
      l::Union{ListIndex, Int, Result{ListIndex, BeforeAfterError}}
    )

Get ranges for upper block ``l``.
"""
@inline function BandwidthInit.upper_block_ranges(
  bbc::BlockedBandColumn,
  l::Union{ListIndex, Int, Result{ListIndex, BeforeAfterError}}
)
  (m, n) = size(bbc)
  upper_block_ranges(bbc.upper_blocks, m, n, l)
end

"""
    function view_upper_block(
      bbc::BlockedBandColumn,
      l::Union{ListIndex, Int, Result{ListIndex, BeforeAfterError}}
    )

Get a view of upper block l.
"""
@inline function view_upper_block(
  bbc::BlockedBandColumn,
  l::Union{ListIndex, Int, Result{ListIndex, BeforeAfterError}}
)
  (rows, cols) = upper_block_ranges(bbc, l)
  view(bbc, rows, cols)
end

"""
    BandwidthInit.size_upper_block(
      bbc::BlockedBandColumn,
      l::Union{ListIndex, Int, Result{ListIndex, BeforeAfterError}}
    ) 

Compute the size of lower block ``l`` for a `BlockedBandColumn.
"""
@inline function BandwidthInit.size_upper_block(
  bbc::BlockedBandColumn,
  l::Union{ListIndex, Int, Result{ListIndex, BeforeAfterError}}
)
  (rows, cols) = upper_block_ranges(bbc, l)
  (last(rows) - first(rows) + 1, last(cols) - first(cols) + 1)
end

# Bandwidth functions

function get_middle_bw_max(::Int, n::Int, cols_first_last::AbstractMatrix{Int})
  middle_bw_max = 0
  for k ∈ 1:n
    middle_bw_max =
      max(middle_bw_max, cols_first_last[4, k] - cols_first_last[3, k] - 1)
  end
  middle_bw_max
end

function get_upper_bw_max(::Int, n::Int, cols_first_last::AbstractMatrix{Int})
  upper_bw_max = 0
  for k ∈ 1:n
    upper_bw_max =
      max(upper_bw_max, cols_first_last[3, k] - cols_first_last[1, k] + 1)
  end
  upper_bw_max
end

function get_lower_bw_max(::Int, n::Int, cols_first_last::AbstractMatrix{Int})
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

function Random.randn!(
  rng::AbstractRNG,
  bbc::BlockedBandColumn{E},
) where {E}

  for k = 1:bbc.n
    for j = inband_index_range_storage(bbc, k)
      bbc.band_elements[j, k] = randn(rng, E)
    end
  end
end

function Random.rand!(
  bbc::BlockedBandColumn{E},
) where {E}

  rand!(Random.default_rng(), bbc)
end

function Random.randn!(
  bbc::BlockedBandColumn{E},
) where {E}

  randn!(Random.default_rng(), bbc)
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

function Random.randn!(
  ::LeadingDecomp,
  rng::AbstractRNG,
  bbc::BlockedBandColumn{E}
) where {E}

  for k = 1:bbc.n
    for j = inband_index_range_storage(bbc, k)
      bbc.band_elements[j, k] = randn(rng, E)
    end
  end
end


"""
    function BlockedBandColumn(
      ::Type{E},
      ::LeadingDecomp,
      rng::AbstractRNG,
      m::Int,
      n::Int;
      upper_ranks::AbstractVector{Int},
      lower_ranks::AbstractVector{Int},
      upper_rank_max::Int=maximum(upper_ranks),
      lower_rank_max::Int=maximum(lower_ranks),
      upper_blocks::Union{AbstractVector{BD},IndexList{BD}},
      lower_blocks::Union{AbstractVector{BD},IndexList{BD}},
      max_num_blocks::Int = max(
        length(upper_blocks),
        length(lower_blocks),
      ),
    ) where {E<:Number, BD<:AbstractBlockData}

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
  upper_ranks::AbstractVector{Int},
  lower_ranks::AbstractVector{Int},
  upper_rank_max::Int=maximum(upper_ranks),
  lower_rank_max::Int=maximum(lower_ranks),
  upper_blocks::Union{AbstractVector{BD},IndexList{BD}},
  lower_blocks::Union{AbstractVector{BD},IndexList{BD}},
  max_num_blocks::Int = max(
    length(upper_blocks),
    length(lower_blocks),
  ),
) where {E<:Number, BD<:AbstractBlockData}

  bbc = BlockedBandColumn(
    E,
    m,
    n,
    upper_bw_max = nothing,
    lower_bw_max = nothing,
    upper_rank_max = upper_rank_max,
    lower_rank_max = lower_rank_max,
    max_num_blocks = max_num_blocks,
    upper_blocks = upper_blocks,
    lower_blocks = lower_blocks,
  )
  leading_lower_ranks_to_cols_first_last!(bbc, lower_ranks)
  leading_upper_ranks_to_cols_first_last!(bbc, upper_ranks)
  compute_rows_first_last_inband!(bbc)
  randn!(rng, bbc)
  bbc
end

function BlockedBandColumn(
  ::Type{E},
  ::LeadingDecomp,
  x::E,
  m::Int,
  n::Int;
  upper_ranks::AbstractVector{Int},
  lower_ranks::AbstractVector{Int},
  upper_rank_max::Int=maximum(upper_ranks),
  lower_rank_max::Int=maximum(lower_ranks),
  upper_blocks::Union{AbstractVector{BD},IndexList{BD}},
  lower_blocks::Union{AbstractVector{BD},IndexList{BD}},
  max_num_blocks::Int = max(
    length(upper_blocks),
    length(lower_blocks),
  ),
) where {E<:Number, BD<:AbstractBlockData}

  bbc = BlockedBandColumn(
    E,
    m,
    n,
    upper_bw_max = nothing,
    lower_bw_max = nothing,
    upper_rank_max = upper_rank_max,
    lower_rank_max = lower_rank_max,
    max_num_blocks = max_num_blocks,
    upper_blocks = upper_blocks,
    lower_blocks = lower_blocks,
  )
  leading_lower_ranks_to_cols_first_last!(bbc, lower_ranks)
  leading_upper_ranks_to_cols_first_last!(bbc, upper_ranks)
  compute_rows_first_last_inband!(bbc)
  fill!(bbc, x)
  bbc
end

"""
    function BlockedBandColumn(
      ::Type{E},
      ::TrailingDecomp,
      rng::AbstractRNG,
      m::Int,
      n::Int;
      upper_ranks::AbstractVector{Int},
      lower_ranks::AbstractVector{Int},
      upper_rank_max::Int=maximum(upper_ranks),
      lower_rank_max::Int=maximum(lower_ranks),
      upper_blocks::Union{AbstractVector{BD},IndexList{BD}},
      lower_blocks::Union{AbstractVector{BD},IndexList{BD}},
      max_num_blocks::Int = max(
        length(upper_blocks),
        length(lower_blocks),
      ),
    ) where {E<:Number, BD<:AbstractBlockData}

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
  upper_ranks::AbstractVector{Int},
  lower_ranks::AbstractVector{Int},
  upper_rank_max::Int=maximum(upper_ranks),
  lower_rank_max::Int=maximum(lower_ranks),
  upper_blocks::Union{AbstractVector{BD},IndexList{BD}},
  lower_blocks::Union{AbstractVector{BD},IndexList{BD}},
  max_num_blocks::Int = max(
    length(upper_blocks),
    length(lower_blocks),
  ),
) where {E<:Number, BD<:AbstractBlockData}
  bbc = BlockedBandColumn(
    E,
    m,
    n,
    upper_bw_max = nothing,
    lower_bw_max = nothing,
    upper_rank_max = upper_rank_max,
    lower_rank_max = lower_rank_max,
    max_num_blocks = max_num_blocks,
    upper_blocks = upper_blocks,
    lower_blocks = lower_blocks,
  )
  trailing_lower_ranks_to_cols_first_last!(bbc, lower_ranks)
  trailing_upper_ranks_to_cols_first_last!(bbc, upper_ranks)
  compute_rows_first_last_inband!(bbc)
  randn!(rng, bbc)
  bbc
end

function BlockedBandColumn(
  ::Type{E},
  ::TrailingDecomp,
  x::E,
  m::Int,
  n::Int;
  upper_ranks::AbstractVector{Int},
  lower_ranks::AbstractVector{Int},
  upper_rank_max::Int=maximum(upper_ranks),
  lower_rank_max::Int=maximum(lower_ranks),
  upper_blocks::Union{AbstractVector{BD},IndexList{BD}},
  lower_blocks::Union{AbstractVector{BD},IndexList{BD}},
  max_num_blocks::Int = max(
    length(upper_blocks),
    length(lower_blocks),
  ),
) where {E<:Number, BD<:AbstractBlockData}
  bbc = BlockedBandColumn(
    E,
    m,
    n,
    upper_bw_max = nothing,
    lower_bw_max = nothing,
    upper_rank_max = upper_rank_max,
    lower_rank_max = lower_rank_max,
    max_num_blocks = max_num_blocks,
    upper_blocks = upper_blocks,
    lower_blocks = lower_blocks,
  )
  trailing_lower_ranks_to_cols_first_last!(bbc, lower_ranks)
  trailing_upper_ranks_to_cols_first_last!(bbc, upper_ranks)
  compute_rows_first_last_inband!(bbc)
  fill!(bbc, x)
  bbc
end

"""
    function BlockedBandColumn(
      ::Type{E},
      D::Decomp,
      m::Int,
      n::Int;
      upper_ranks::AbstractVector{Int},
      lower_ranks::AbstractVector{Int},
      upper_rank_max::Int=maximum(upper_ranks),
      lower_rank_max::Int=maximum(lower_ranks),
      upper_blocks::Union{AbstractVector{BD},IndexList{BD}},
      lower_blocks::Union{AbstractVector{BD},IndexList{BD}},
      max_num_blocks::Int = max(
        length(upper_blocks),
        length(lower_blocks),
      ),
    ) where {E<:Number, BD<:AbstractBlockData}

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
  upper_ranks::AbstractVector{Int},
  lower_ranks::AbstractVector{Int},
  upper_rank_max::Int=maximum(upper_ranks),
  lower_rank_max::Int=maximum(lower_ranks),
  upper_blocks::Union{AbstractVector{BD},IndexList{BD}},
  lower_blocks::Union{AbstractVector{BD},IndexList{BD}},
  max_num_blocks::Int = max(
    length(upper_blocks),
    length(lower_blocks),
  ),
) where {E<:Number, BD<:AbstractBlockData}
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
    max_num_blocks = max_num_blocks,
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

function BandColumnMatrices.compute_rows_first_last_inband!(bbc::BlockedBandColumn)
  compute_rows_first_last_inband!(bbc, bbc.rows_first_last)
end

function BandColumnMatrices.validate_rows_first_last(
  bbc::BlockedBandColumn,
)
  rfl = compute_rows_first_last_inband(bbc)
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

function Base.show(io::IO, mime::MIME"text/plain", bbc::BlockedBandColumn)
  println(io, "$(bbc.m)×$(bbc.n) $(typeof(bbc))")
  allfields = get(io, :all, false)::Bool
  if allfields
    println(
      io,
      "(bw_max, upper_bw_max, middle_lower_bw_max):  " *
        "($(bbc.bw_max), $(bbc.upper_bw_max), $(bbc.middle_lower_bw_max))")
    println(io, "rows_first_last: ")
    show(io, mime, bbc.rows_first_last)
    println(io)
    print(io, "cols_first_last: ")
    show(io, mime, bbc.cols_first_last)
    println(io)
  end
  println(io, "upper_blocks: ")
  show(io, mime, bbc.upper_blocks)
  println(io)
  print(io, "lower_blocks: ")
  show(io, mime, bbc.lower_blocks)

  println("Band matrix:")
  show_partial_band_matrix(io, bbc)
end

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
    BandwidthInit.leading_lower_ranks_to_cols_first_last!(
      bbc::BlockedBandColumn,
      rs::AbstractVector{Int},
    )

Set first_last indices appropriate for a leading decomposition
associated with a given lower rank sequence.
"""
function BandwidthInit.leading_lower_ranks_to_cols_first_last!(
  bbc::BlockedBandColumn,
  rs::AbstractVector{Int},
)

  m, n = size(bbc)
  rs1 = constrain_lower_ranks(
    m,
    n,
    blocks = bbc.lower_blocks,
    ranks = rs,
  )
  lb_count = 0
  for lb ∈ bbc.lower_blocks
    lb_count += 1
    rows_lb, cols_lb = lower_block_ranges(bbc, lb)
    # empty if after last index.
    lb1 = next_list_index(bbc.lower_blocks, lb)
    rows_lb1, _ = lower_block_ranges(bbc, lb1)
    dᵣ = setdiffᵣ(rows_lb, rows_lb1)
    if !isempty(dᵣ)
      bbc.cols_first_last[5, last(cols_lb, rs1[lb_count])] .= last(dᵣ)
    end
  end
end

"""
    BandwidthInit.trailing_lower_ranks_to_cols_first_last!(
      bbc::BlockedBandColumn,
      rs::AbstractVector{Int},
    )

Set first_last indices appropriate for a trailing decomposition
associated with a given lower rank sequence.
"""
function BandwidthInit.trailing_lower_ranks_to_cols_first_last!(
  bbc::BlockedBandColumn,
  rs::AbstractVector{Int},
)

  m, n = size(bbc)
  rs1 = constrain_lower_ranks(
    m,
    n,
    blocks = bbc.lower_blocks,
    ranks = rs,
  )
  lb_count = length(bbc.lower_blocks)
  for lb ∈ Iterators.Reverse(bbc.lower_blocks)
    rows_lb, cols_lb = lower_block_ranges(bbc, lb)
    # empty if previous to lb is before the first index.
    lb1 = prev_list_index(bbc.lower_blocks, lb)
    _, cols_lb1 = lower_block_ranges(bbc, lb1)
    dᵣ = setdiffᵣ(cols_lb, cols_lb1)
    if !isempty(dᵣ)
      rows_lb_first = isempty(rows_lb) ? m+1 : first(rows_lb)
      bbc.cols_first_last[5, dᵣ] .= min(m, rows_lb_first + rs1[lb_count] - 1)
    end
    lb_count -= 1
  end
end

"""
    BandwidthInit.leading_upper_ranks_to_cols_first_last!(
      bbc::BlockedBandColumn,
      rs::AbstractVector{Int},
    )

Set first_last indices appropriate for a leading decomposition associated
with a given upper rank sequence
"""
function BandwidthInit.leading_upper_ranks_to_cols_first_last!(
  bbc::BlockedBandColumn,
  rs::AbstractVector{Int},
)

  m, n = size(bbc)
  rs1 = constrain_upper_ranks(
    m,
    n,
    blocks = bbc.upper_blocks,
    ranks = rs,
  )

  ub_count = 0
  for ub ∈ bbc.upper_blocks
    ub_count += 1
    rows_ub, cols_ub = upper_block_ranges(bbc, ub)
    ub1 = next_list_index(bbc.upper_blocks, ub)
    _, cols_ub1 = upper_block_ranges(bbc, ub1) # empty if ub+1 > num_blocks
    dᵣ = setdiffᵣ(cols_ub, cols_ub1)
    if !isempty(dᵣ)
      rows_ub_last = isempty(rows_ub) ? 0 : last(rows_ub)
      bbc.cols_first_last[2, dᵣ] .= 
        max(1, rows_ub_last - rs1[ub_count] + 1)
    end
  end
end

"""
    BandwidthInit.trailing_upper_ranks_to_cols_first_last!(
      bbc::BlockedBandColumn,
      rs::AbstractVector{Int},
    )

Set first_last indices appropriate for a leading decomposition
associated with a given upper rank sequence
"""
function BandwidthInit.trailing_upper_ranks_to_cols_first_last!(
  bbc::BlockedBandColumn,
  rs::AbstractVector{Int},
)

  m, n = size(bbc)
  rs1 = constrain_upper_ranks(
    m,
    n,
    blocks = bbc.upper_blocks,
    ranks = rs,
  )

  ub_count = length(bbc.upper_blocks)
  for ub ∈ Iterators.Reverse(bbc.upper_blocks)
    rows_ub, cols_ub = upper_block_ranges(bbc, ub)
    ub1 = prev_list_index(bbc.upper_blocks, ub)
    rows_ub1, _ = upper_block_ranges(bbc, ub1) # empty if ub1 < 1
    dᵣ = setdiffᵣ(rows_ub, rows_ub1)
    if !isempty(dᵣ)
      bbc.cols_first_last[2, first(cols_ub, rs1[ub_count])] .= first(dᵣ)
    end
    ub_count -= 1
  end
end

@views function BandColumnMatrices.wilk(bbc::BlockedBandColumn)
  (m, n) = size(bbc)
  a = fill('N', (2 * m, 2 * n))
  # insert spaces
  fill!(a[2:2:(2 * m), :], ' ')
  fill!(a[:, 2:2:(2 * n)], ' ')
  #insert boundaries for lower and upper blocks.
  for i ∈ bbc.lower_blocks
    jl = bbc.lower_blocks[i].mb
    kl = bbc.lower_blocks[i].nb
    if jl > 0 && kl > 0
      fill!(a[(2 * jl + 1):(2 * m - 1), 2 * kl], '|')
      a[2 * jl, 2 * kl] = '+'
      fill!(a[2 * jl, 1:(2 * kl - 1)], '-')
      for kk = 1:(2 * n)
        if a[2 * jl, kk] == '-' && a[2 * jl - 1, kk] == '|'
          a[2 * jl, kk] = '+'
        end
      end
    end
  end
  for i = bbc.upper_blocks
    ju = bbc.upper_blocks[i].mb
    ku = bbc.upper_blocks[i].nb
    if ju > 0 && ku > 0
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
    deepcopy(bbc.upper_blocks),
    deepcopy(bbc.lower_blocks),
  )
end

end
