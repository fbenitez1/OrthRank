module BandColumnMatrices

using Printf
using LinearAlgebra
using Base: @propagate_inbounds

export BandColumn,
  AbstractBandColumn,
  # AbstractBandColumn methods requiring implementation.
  viewbc,
  toBandColumn,
  maybe_first,
  maybe_last,
  first_inband_index,
  last_inband_index,
  first_upper_index,
  last_upper_index,
  first_lower_index,
  last_lower_index,
  upper_bw_max,
  middle_lower_bw_max,
  row_offset,
  col_offset,
  bw_max,
  row_size,
  col_size,
  band_elements,
  compute_rows_first_last!, # One method only, to access rows_first_last array
  validate_rows_first_last,
  unsafe_set_first_inband_index!,
  unsafe_set_last_inband_index!,
  # Exceptions
  WellError,
  EmptyUpperRange,
  EmptyLowerRange,
  IndexNotUpper,
  IndexNotLower,
  NoStorageForIndex,
  IndexNotInband,
  SubcolumnIndicesNotInband,
  SubrowIndicesNotInband,
  SubcolumnIndicesNotStorable,
  SubrowIndicesNotStorable,
  # Methods implemented for any AbstractBandColumn
  is_lower_notchable_with_no_well,
  is_upper_notchable_with_no_well,
  is_lower_bulgeable_with_no_well,
  is_upper_bulgeable_with_no_well,
  bulge_lower!,
  bulge_upper!,
  notch_upper!,
  notch_lower!,
  zero_above!,
  zero_below!,
  zero_right!,
  zero_left!,
  bulge!,
  is_upper,
  is_lower,
  first_inband_index_storage,
  last_inband_index_storage,
  inband_index_range_storage,
  upper_inband_index_range_storage,
  middle_inband_index_range_storage,
  lower_inband_index_range_storage,
  first_storable_index,
  last_storable_index,
  storable_index_range,
  inband_index_range,
  upper_inband_index_range,
  middle_inband_index_range,
  lower_inband_index_range,
  storage_offset,
  is_inband,
  check_bc_storage_bounds,
  upper_bw,
  middle_bw,
  lower_bw,
  compute_rows_first_last,
  get_elements,
  setindex_no_bulge!,
  wilk,
  Wilk,
  # utility functions
  hull,
  project,
  # types
  BCFloat64,
  NonSub,
  Sub


@inline function project(j::Int, m::Int)
  min(max(j, 1), m)
end

@inline function project(j::Int, r::AbstractUnitRange{Int})
  min(max(j, first(r)), last(r))
end

"""
  hull(a :: AbstractUnitRange, b :: AbstractUnitRange)

Return the convex hull of two closed intervals, with
the intervals represented by an `AbstractUnitRange`.
"""
@inline function hull(a::AbstractUnitRange, b::AbstractUnitRange)
  isempty(a) ? b :
  (isempty(b) ? a : UnitRange(min(first(a), first(b)), max(last(a), last(b))))
end

"""
    NoStorageForIndex

An exception to indicate that there is no storage available to
represent a particular element of a BandColumn.
"""
struct NoStorageForIndex <: Exception
  arr::Any
  ix::Any
end

"""
    WellError

An exception thrown when an operation would create a well.
"""
struct WellError <: Exception end

struct FirstOfEmptyRange <: Exception end

struct LastOfEmptyRange <: Exception end

struct EmptyUpperRange <: Exception end

struct EmptyLowerRange <: Exception end

struct IndexNotUpper <: Exception 
  j :: Int
  k :: Int
end

struct IndexNotLower <: Exception
  j :: Int
  k :: Int
end

struct IndexNotInband <: Exception
  j :: Int
  k :: Int
end

struct SubcolumnIndicesNotInband <: Exception
  js :: UnitRange{Int}
  k :: Int
end

struct SubrowIndicesNotInband <: Exception
  j :: Int
  ks :: UnitRange{Int}
end

struct SubcolumnIndicesNotStorable <: Exception
  js :: UnitRange{Int}
  k :: Int
end

struct SubrowIndicesNotStorable <: Exception
  j :: Int
  ks :: UnitRange{Int}
end

@inline function maybe_first(js::UnitRange{<:Integer})
  j0 = first(js)
  @boundscheck j0 <= last(js) || throw(FirstOfEmptyRange)
  j0
end

@inline function maybe_last(js::UnitRange{<:Integer})
  j1 = last(js)
  @boundscheck first(js) <= j1 || throw(LastOfEmptyRange)
  j1
end

struct NonSub end
struct Sub end

Base.showerror(io::IO, e::NoStorageForIndex) = print(
  io,
  "NoStorageForIndex:  Attempt to access ",
  typeof(e.arr),
  " at index ",
  e.ix,
  " for which there is no storage.",
)

"""

# AbstractBandColumn

    AbstractBandColumn{S,E,AE,AI} <: AbstractArray{E,2}

An AbstractBandColumn should implement the following:

- `AbstractArray`: `getindex`, `setindex!`, 

- `viewbc`

- `first_inband_index`

- `last_inband_index`

- `first_upper_index`

- `first_lower_index`

- `last_upper_index`

- `last_lower_index`

- `upper_bw_max`

- `middle_lower_bw_max`

- `bw_max`

- `row_offset`

- `col_offset`

- `row_size`

- `col_size`

- `band_elements(bc)`

- `compute_rows_first_last!`: One method only, to access rows_first_last array.

- `validate_rows_first_last`

- `bulge_upper!(bc, j::Int, k::Int)`

- `bulge_lower!(bc, j::Int, k::Int)`

- `notch_upper!(bc, j::Int, k::Int)`

- `notch_lower!(bc, j::Int, k::Int)`

"""
abstract type AbstractBandColumn{S,E,AE,AI} <: AbstractArray{E,2} end

"""

# BandColumn

    BandColumn{S,E<:Number,AE<:AbstractArray{E,2},AI<:AbstractArray{Int,2}}
    <: AbstractBandColumn{S,E,AE,AI}

A simplified band column structure that does not include leading
blocks but does include uniform offsets that can be changed to give
different submatrices.  This can be used to represent submatrices of a
LeadingBandColumn matrix.

# Fields
- `m_nosub::Int`: Full matrix number of rows.

- `n_nosub::Int`: Full matrix and full elements array number of columns.

- `m::Int`: Matrix number of rows.

- `n::Int`: Matrix and elements number of columns.

- `roffset::Int`: Uniform column offset, used to identify submatrices.

- `coffset::Int`: Uniform row offset, used to identify submatrices.

- `bw_max::Int`: Elements array number of rows.

- `upper_bw_max::Int`: Maximum upper bandwidth.

- `middle_lower_bw_max::Int`: Maximum middle + lower bandwidth.

- `rows_first_last::AI`: `rows_first_last[j,:]` contains
   
  - `rows_first_last[j,1]`: Index of the first storable element in row
    `j` for a `NonSub` or for the larger containing matrix of a
    submatrix.

  - `rows_first_last[j,2]`: Index of the first inband element in row
    `j` for a `NonSub` or for the larger containing matrix of a
    submatrix.

  - `rows_first_last[j,3]`: Index of the last lower element in row
    `j` for a `NonSub` or for the larger containing matrix of a
    submatrix.  This is not necessarily an inband element.

  - `rows_first_last[j,4]`: Index of the first upper element in row
    `j` for a `NonSub` or for the larger containing matrix of a
    submatrix.  This is not necessarily an inband element.

  - `rows_first_last[j,5]`: Index of the last inband element in row
    `j` for a `NonSub` or for the larger containing matrix of a
    submatrix.

  - `rows_first_last[j,6]`: Index of the last storable element in row
    `j` for a `NonSub` or for the larger containing matrix of a
    submatrix.

- `cols_first_last::AI`: `cols_first_last[:,k]` contains
   
  - `cols_first_last[1,k]`: Index of the first storable element in column
    `k` for a `NonSub` or for the larger containing matrix of a
    submatrix.

  - `cols_first_last[2,k]`: Index of the first inband element in column
    `k` for a `NonSub` or for the larger containing matrix of a
    submatrix.

  - `cols_first_last[3,k]`: Index of the last upper element in column
    `k` for a `NonSub` or for the larger containing matrix of a
    submatrix.  This is not necessarily an inband element.

  - `cols_first_last[4,k]`: Index of the first lower element in column
    `k` for a `NonSub` or for the larger containing matrix of a
    submatrix.  This is not necessarily an inband element.

  - `cols_first_last[5,k]`: Index of the last inband element in column
    `k` for a `NonSub` or for the larger containing matrix of a
    submatrix.

  - `cols_first_last[6,k]`: Index of the last storable element in column
    `k` for a `NonSub` or for the larger containing matrix of a
    submatrix.

- `band_elements::AE`: Column-wise storage of the band elements with
  dimensions:
   
  ``(upper_bw_max + middle_bw_max lower_bw_max) × n``

It is assumed that the middle bandwidths and the first row subdiagonal
and column superdiagonal will never change.  In the case of a
`LeadingBandColumn`, these are determined by the leading blocks.

# Example

    A = X U O O O N N
        L X O O O N N
        L X U U U O O
        O L X X X O O
        N O O O L O O
        N N O O L U U
        N N O O O L X
        N N N N N L X

where ``U``, ``X``, and ``L`` denote are _inband_ locations for
_upper_, _middle_, and _lower_ band elements.  ``O`` indicates a
_storable_ out of band location with storage available for expanding
either the lower or upper bandwidth.  ``N`` represents an _unstorable_
location, i.e. one for which there is no storage.

The matrix is stored as

    bc.band_elements = 

    Num rows        | Elements
    ----------------------
    upper_bw_max    | O O O O O O O
                    | O O O O O O O
                    | O O O O O O O
                    | O U U U U U U
    ----------------------
    middle_bw_max + | X X X X X L X
    lower_bw_max    | L X O O L L X
                    | L L O O L O O
                    | O O O O O O O

where

    m_nonsub =           8
    n_nonsub =           7
    m =                  8
    n =                  7
    roffset =            0
    coffset =            0
    bw_max =             8
    upper_bw_max =       4
    middle_lowerbw_max = 4


    cols_first_last =    [ 1 1 1 1 1 3 3;
                           1 1 3 3 3 6 6;
                           0 1 3 3 3 6 6;
                           2 4 5 5 5 7 9;
                           3 4 4 4 6 8 8;
                           4 5 6 6 6 8 8 ]
    rows_first_last =    [ 1 1 0 2 2 5;
                           1 1 1 3 2 5;
                           1 1 1 3 5 7;
                           1 2 2 7 6 7;
                           2 5 5 6 5 7;
                           3 5 5 6 7 7;
                           3 6 6 8 7 7;
                           6 6 6 8 7 7 ]
"""
struct BandColumn{S,E<:Number,AE<:AbstractArray{E,2},AI<:AbstractArray{Int,2}} <:
       AbstractBandColumn{S,E,AE,AI}
  sub :: S
  m_nonsub::Int
  n_nonsub::Int
  m::Int
  n::Int
  roffset::Int
  coffset::Int
  bw_max::Int
  upper_bw_max::Int
  middle_lower_bw_max::Int
  rows_first_last :: AI
  cols_first_last :: AI
  band_elements::AE
end

@inline toBandColumn(bc::BandColumn) = bc

const BCFloat64 = BandColumn{NonSub,Float64,Array{Float64,2},Array{Int,2}}

@inline row_size(bc::BandColumn) = bc.m
@inline col_size(bc::BandColumn) = bc.n
@inline row_size(::Type{NonSub}, bc::BandColumn) = bc.m_nonsub
@inline col_size(::Type{NonSub}, bc::BandColumn) = bc.n_nonsub

@inline Base.size(bc::AbstractBandColumn) = (row_size(bc), col_size(bc))

@inline bw_max(bc::BandColumn) = bc.bw_max
@inline upper_bw_max(bc::BandColumn) = bc.upper_bw_max
@inline middle_lower_bw_max(bc::BandColumn) = bc.middle_lower_bw_max
@inline band_elements(bc::BandColumn) = bc.band_elements

"""
    row_offset( bc )

Get the row offset of a `bc`.
"""
@propagate_inbounds @inline row_offset(
  bc::AbstractBandColumn{NonSub},
) = 0

@propagate_inbounds @inline row_offset(
  bc::BandColumn,
) = bc.roffset

"""
    col_offset( bc )

Get the col offset of a `bc`.
"""
@propagate_inbounds @inline col_offset(
  bc::AbstractBandColumn{NonSub},
) = 0

@propagate_inbounds @inline col_offset(
  bc::BandColumn,
) = bc.coffset

"""
    first_inband_index(bc, ::Colon, k::Int)
    first_inband_index(bc, j::Int, ::Colon)

If

    j=first_inband_index(bc,:,k)

then `bc[j,k]` is the first inband element in column ``k``.  If

    k=first_inband_index(bc,j,:)

then `bc[j,k]` is the first inband element in row ``j``.
"""
@propagate_inbounds @inline first_inband_index(
  ::Type{NonSub},
  bc::BandColumn,
  ::Colon,
  k::Int,
) = bc.cols_first_last[2,k]

@propagate_inbounds @inline first_inband_index(
  ::Type{NonSub},
  bc::BandColumn,
  j::Int,
  ::Colon,
) = bc.rows_first_last[j,2]

"""
    last_inband_index(bc, ::Colon, k::Int)
    last_inband_index(bc, j::Int, ::Colon)

If

    j=last_inband_index(bc,:,k)

then `bc[j,k]` is the last inband element in column ``k``.  If

    k=last_inband_index(bc,j,:)

then `bc[j,k]` is the last inband element in row ``j``.
"""
@propagate_inbounds @inline last_inband_index(
  ::Type{NonSub},
  bc::BandColumn,
  ::Colon,
  k::Int,
) = bc.cols_first_last[5,k]

@propagate_inbounds @inline last_inband_index(
  ::Type{NonSub},
  bc::BandColumn,
  j::Int,
  ::Colon,
) = bc.rows_first_last[j,5]

"""
    first_lower_index(bc::BandColumn, ::Colon, k::Int)

If

  j=first_lower_index(bc,:,k)

then `bc[j,k]` is the first, possibly not inband, lower index in
column ``k``.  For a submatrix, this does not necessarily
have to be in the range `1:m` if the first lower index position
of the underlying banded matrix is not in the submatrix.
"""
@propagate_inbounds @inline first_lower_index(
  ::Type{NonSub},
  bc::BandColumn,
  ::Colon,
  k::Int,
) = bc.cols_first_last[4,k]

"""
    last_lower_index(bc::BandColumn, j::Int, ::Colon)

If

  k=last_lower_index(bc,j,:)

then `bc[j,k]` is the last, possibly not inband, lower index in
row ``j``.  For a submatrix, this does not necessarily
have to be in the range `1:n` if the last lower index position
of the underlying banded matrix is not in the submatrix.
"""
@propagate_inbounds @inline last_lower_index(
  ::Type{NonSub},
  bc::BandColumn,
  j::Int,
  ::Colon,
) = bc.rows_first_last[j,3]

"""
    first_upper_index(bc::BandColumn, j::Int, ::Colon)

If

  k=first_upper_index(bc,j,:)

then `bc[j,k]` is the first, possibly not inband, upper index in
row ``j``.  For a submatrix, this does not necessarily
have to be in the range `1:n` if the first lower index position
of the underlying banded matrix is not in the submatrix.
"""
@propagate_inbounds @inline first_upper_index(
  ::Type{NonSub},
  bc::BandColumn,
  j::Int,
  ::Colon,
) = bc.rows_first_last[j,4]

"""
    last_upper_index(bc::BandColumn, ::Colon, k::Int)

If

  j=last_upper_index(bc,:,k)

then `bc[j,k]` is the last, possibly not inband, upper index in
column ``k``.  For a submatrix, this does not necessarily
have to be in the range `1:m` if the first lower index position
of the underlying banded matrix is not in the submatrix.
"""
@propagate_inbounds @inline last_upper_index(
  ::Type{NonSub},
  bc::BandColumn,
  ::Colon,
  k::Int,
) = bc.cols_first_last[3,k]

#=

AbstractBandColumn extensions of the basic NonSub interface.

=#

# col first_inband_index methods
@propagate_inbounds @inline first_inband_index(
  bc::AbstractBandColumn,
  ::Colon,
  k::Int,
) = first(inband_index_range(bc, :, k))

@propagate_inbounds @inline first_inband_index(
  bc::AbstractBandColumn{NonSub},
  ::Colon,
  k::Int,
) = first_inband_index(NonSub,bc,:,k)

# row first_inband_index methods
@propagate_inbounds @inline first_inband_index(
  bc::AbstractBandColumn,
  j::Int,
  ::Colon,
) = first(inband_index_range(bc, j, :))

@propagate_inbounds @inline first_inband_index(
  bc::AbstractBandColumn{NonSub},
  j::Int,
  ::Colon,
) = first_inband_index(NonSub, bc, j, :)


# col last_inband_index methods
@propagate_inbounds @inline last_inband_index(
  bc::AbstractBandColumn,
  ::Colon,
  k::Int,
) = last(inband_index_range(bc, :, k))

@propagate_inbounds @inline last_inband_index(
  bc::BandColumn{NonSub},
  ::Colon,
  k::Int,
) = last_inband_index(NonSub, bc, :, k)

# row last_inband_index methods
@propagate_inbounds @inline last_inband_index(
  bc::AbstractBandColumn,
  j::Int,
  ::Colon,
) = last(inband_index_range(bc, j, :))

@propagate_inbounds @inline last_inband_index(
  bc::AbstractBandColumn{NonSub},
  j::Int,
  ::Colon,
) = last_inband_index(NonSub, bc, j, :)

# first_lower_index methods
@propagate_inbounds @inline first_lower_index(
  bc::AbstractBandColumn,
  ::Colon,
  k::Int,
) = first_lower_index(NonSub,bc,:,k) - row_offset(bc)

@propagate_inbounds @inline first_lower_index(
  bc::AbstractBandColumn{NonSub},
  ::Colon,
  k::Int,
) = first_lower_index(NonSub,bc,:,k)

# last_lower_index methods
@propagate_inbounds @inline last_lower_index(
  bc::AbstractBandColumn,
  j::Int,
  ::Colon,
) = last_lower_index(NonSub,bc,j,:) - col_offset(bc)

@propagate_inbounds @inline last_lower_index(
  bc::AbstractBandColumn{NonSub},
  j::Int,
  ::Colon,
) = last_lower_index(NonSub,bc,j,:)

# first_upper_index methods
@propagate_inbounds @inline first_upper_index(
  bc::AbstractBandColumn,
  j::Int,
  ::Colon,
) = first_upper_index(NonSub,bc,j,:) - col_offset(bc)

@propagate_inbounds @inline first_upper_index(
  bc::AbstractBandColumn{NonSub},
  j::Int,
  ::Colon,
) = first_upper_index(NonSub,bc,j,:)

# last_upper_index methods
@propagate_inbounds @inline last_upper_index(
  bc::AbstractBandColumn,
  ::Colon,
  k::Int,
) = bc.cols_first_last[3,k] - row_offset(bc)

@propagate_inbounds @inline last_upper_index(
  bc::AbstractBandColumn{NonSub},
  ::Colon,
  k::Int,
) = bc.cols_first_last[3,k]

is_upper(bc::AbstractBandColumn, j::Int, k::Int) =
  j <= last_upper_index(bc, :, k)

is_lower(bc::AbstractBandColumn, j::Int, k::Int) =
  j >= first_lower_index(bc, :, k)


"""
    function inband_index_range(
      bc::AbstractBandColumn,
      ::Colon,
      k::Int,
    )

Range of inband elements in column ``k``.
"""
@propagate_inbounds @inline function inband_index_range(
  bc::AbstractBandColumn,
  ::Colon,
  k::Int,
)
  roffs = row_offset(bc)
  (1:row_size(bc)) ∩ (
    (first_inband_index(
      NonSub,
      bc,
      :,
      k,
    ) - roffs):(last_inband_index(NonSub, bc, :, k) - roffs)
  )
end

@propagate_inbounds @inline function inband_index_range(
  ::Type{NonSub},
  bc::AbstractBandColumn,
  ::Colon,
  k::Int,
)
  first_inband_index(NonSub, bc, :, k):last_inband_index(NonSub, bc, :, k)
end

@propagate_inbounds @inline function inband_index_range(
  bc::AbstractBandColumn{NonSub},
  ::Colon,
  k::Int,
)
  inband_index_range(NonSub, bc, :, k)
end

"""
    function inband_index_range(
      bc::AbstractBandColumn,
      j::Int,
      ::Colon,
    )

Range of inband elements in row ``j``.
"""
@propagate_inbounds @inline function inband_index_range(
  bc::AbstractBandColumn,
  j::Int,
  ::Colon,
)
  coffs = col_offset(bc)
  (1:col_size(bc)) ∩ (
    (first_inband_index(
      NonSub,
      bc,
      j,
      :,
    ) - coffs):(last_inband_index(NonSub, bc, j, :) - coffs)
  )
end

@propagate_inbounds @inline function inband_index_range(
  ::Type{NonSub},
  bc::AbstractBandColumn,
  j::Int,
  ::Colon,
)
  first_inband_index(NonSub, bc, j, :):last_inband_index(NonSub, bc, j, :)
end

@propagate_inbounds @inline function inband_index_range(
  bc::AbstractBandColumn{NonSub},
  j::Int,
  ::Colon,
)
  inband_index_range(NonSub, bc, j, :)
end

"""
    function upper_inband_index_range(
      bc::AbstractBandColumn,
      ::Colon,
      k::Int,
    )

Range of inband upper elements in column ``k``.
"""
@propagate_inbounds @inline function upper_inband_index_range(
  bc::AbstractBandColumn,
  ::Colon,
  k::Int,
)
  (1:row_size(bc)) ∩ (first_inband_index(bc, :, k):last_upper_index(bc, :, k))
end

"""
    function upper_inband_index_range(
      bc::AbstractBandColumn,
      j::Int,
      ::Colon,
    )

Range of inband upper elements in row ``j``.
"""
@propagate_inbounds @inline function upper_inband_index_range(
  bc::AbstractBandColumn,
  j::Int,
  ::Colon,
)
  (1:col_size(bc)) ∩ (first_upper_index(bc, j, :):last_inband_index(bc, j, :))
end

"""
    function middle_inband_index_range(
      bc::AbstractBandColumn,
      ::Colon,
      k::Int,
    )

Range of inband middle elements in column ``k``.
"""
@propagate_inbounds @inline function middle_inband_index_range(
  bc::AbstractBandColumn,
  ::Colon,
  k::Int,
)
  (1:row_size(bc)) ∩
  ((last_upper_index(bc, :, k) + 1):(first_lower_index(bc, :, k) - 1))
end

"""
    function middle_inband_index_range(
      bc::AbstractBandColumn,
      j::Int,
      ::Colon,
    )

Range of inband middle elements in row ``j``.
"""
@propagate_inbounds @inline function middle_inband_index_range(
  bc::AbstractBandColumn,
  j::Int,
  ::Colon,
)
  (1:col_size(bc)) ∩
  ((last_lower_index(bc, j, :) + 1):(first_upper_index(bc, j, :) - 1))
end

"""
    function lower_inband_index_range(
      bc::AbstractBandColumn,
      ::Colon,
      k::Int,
    )

Range of inband lower elements in column ``k``.
"""
@propagate_inbounds @inline function lower_inband_index_range(
  bc::AbstractBandColumn,
  ::Colon,
  k::Int,
)
  (1:row_size(bc)) ∩ (first_lower_index(bc, :, k):last_inband_index(bc, :, k))
end

"""
    function lower_inband_index_range(
      bc::AbstractBandColumn,
      j::Int,
      ::Colon,
    )

Range of inband elements in row ``j``.
"""
@propagate_inbounds @inline function lower_inband_index_range(
  bc::AbstractBandColumn,
  j::Int,
  ::Colon,
)
  (1:col_size(bc)) ∩ (first_inband_index(bc, j, :):last_lower_index(bc, j, :))
end

@propagate_inbounds @inline upper_bw(bc::AbstractBandColumn, ::Colon, k::Int) =
  length(upper_inband_index_range(bc, :, k))

@propagate_inbounds @inline middle_bw(bc::AbstractBandColumn, ::Colon, k::Int) =
  length(middle_inband_index_range(bc, :, k))

@propagate_inbounds @inline lower_bw(bc::AbstractBandColumn, ::Colon, k::Int) =
  length(lower_inband_index_range(bc, :, k))

@propagate_inbounds @inline upper_bw(bc::AbstractBandColumn, j::Int, ::Colon) =
  length(upper_inband_index_range(bc, j, :))

@propagate_inbounds @inline middle_bw(bc::AbstractBandColumn, j::Int, ::Colon) =
  length(middle_inband_index_range(bc, j, :))

@propagate_inbounds @inline lower_bw(bc::AbstractBandColumn, j::Int, ::Colon) =
  length(lower_inband_index_range(bc, j, :))

"""
    storage_offset(bc::AbstractBandColumn, k::Int)

Compute a row offset to look into storage:

    d = storage_offset(bc,k)
    bc[j,k] == band_elements(bc)[j - d, k]
"""
@propagate_inbounds @inline storage_offset(bc::AbstractBandColumn, k::Int) =
  last_upper_index(bc,:,k) - upper_bw_max(bc)

"""
    first_inband_index_storage(bc::AbstractBandColumn, k::Int)

If 

    j=first_inband_index_storage(bc::AbstractBandColumn, k::Int)

then `bc.band_elements[j,k]` is the first inband element
in column ``k``.
"""
@propagate_inbounds @inline first_inband_index_storage(
  bc::AbstractBandColumn,
  k::Int,
) = first_inband_index(bc, :, k) - storage_offset(bc, k)

"""
    last_inband_index_storage(bc::AbstractBandColumn, k::Int)

If 

    j=last_inband_index_storage(bc::AbstractBandColumn, k::Int)

then `bc.band_elements[j,k]` is the last inband element
in column ``k``.
"""
@propagate_inbounds @inline last_inband_index_storage(
  bc::AbstractBandColumn,
  k::Int,
) = last_inband_index(bc, :, k) - storage_offset(bc, k)

"""
    first_storable_index(bc::AbstractBandColumn, k::Int)

If 

    j=first_storable_index(bc::AbstractBandColumn, k::Int)

then `bc[j,k]` is the first element in column ``k`` for which there is
available storage.
"""
@propagate_inbounds @inline first_storable_index(
  bc::AbstractBandColumn,
  ::Colon,
  k::Int,
) = project(1 + storage_offset(bc, k), row_size(bc))

"""
    last_storable_index(bc::AbstractBandColumn, k::Int)

If 

    j=last_storable_index(bc::AbstractBandColumn, k::Int)

then `bc[j,k]` is the last element in column ``k`` for which there is
available storage..
"""
@propagate_inbounds @inline last_storable_index(
  bc::AbstractBandColumn,
  ::Colon,
  k::Int,
) = project(bw_max(bc) + storage_offset(bc, k), row_size(bc))

"""
    inband_index_range_storage(
       bc::AbstractBandColumn,
       k::Int,
    )

Compute the range of inband elements in column ``k`` of
`bc.band_elements`.
"""
@propagate_inbounds @inline function inband_index_range_storage(
  bc::AbstractBandColumn,
  k::Int,
)
  inband_index_range(bc,:,k) .- storage_offset(bc,k)
end

"""
    storable_index_range(
      bc::AbstractBandColumn,
      ::Colon,
      k::Int,
    )

The range of elements in column ``k`` of `bc` for which storage is
available.
"""
@propagate_inbounds @inline function storable_index_range(
  bc::AbstractBandColumn,
  ::Colon,
  k::Int,
)
  first_storable_index(bc, :, k):last_storable_index(bc, :, k)
end

"""
    upper_inband_index_range_storage(
       bc::AbstractBandColumn,
       k::Int)

Compute the range of upper inband elements in column ``k`` of
`bc.band_elements`.
"""
@propagate_inbounds @inline function upper_inband_index_range_storage(
  bc::AbstractBandColumn,
  k::Int,
)
  upper_inband_index_range(bc,:,k) .- storage_offset(bc,k)
end

"""
    middle_inband_index_range_storage(
       bc::AbstractBandColumn,
       k::Int,
    )

Compute the range of middle inband elements in column ``k`` of
`bc.band_elements`.
"""
@propagate_inbounds @inline function middle_inband_index_range_storage(
  bc::AbstractBandColumn,
  k::Int,
)
  middle_inband_index_range(bc,:,k) .- storage_offset(bc,k)
end

"""
    lower_inband_index_range_storage(
       bc::AbstractBandColumn,
       k::Int,
    )

Compute the range of lower inband elements in column ``k`` of
`bc.band_elements`.
"""
@propagate_inbounds @inline function lower_inband_index_range_storage(
  bc::AbstractBandColumn,
  k::Int,
)
  lower_inband_index_range(bc,:,k) .- storage_offset(bc,k)
end

"""
    is_inband(
      bc::AbstractBandColumn,
      j::Int,
      k::Int,
    )
Test if `bc[j,k]` is an inband element.
"""
@propagate_inbounds @inline function is_inband(
  bc::AbstractBandColumn,
  j::Int,
  k::Int,
)
  j ∈ inband_index_range(bc,:,k)
end

@propagate_inbounds @inline function is_inband(
  bc::AbstractBandColumn,
  jrange::AbstractRange{Int},
  k::Int,
)
  jrange ⊆ inband_index_range(bc,:,k)
end

"""
    check_bc_storage_bounds(
      ::Type{Bool},
      bc::AbstractBandColumn,
      j::Int,
      k::Int,
    )

  Check whether `bc[j,k]` is a storable element, returning a `Bool`.
"""
@inline function check_bc_storage_bounds(
  ::Type{Bool},
  bc::AbstractBandColumn,
  j::Int,
  k::Int,
)
  j ∈ storable_index_range(bc,:,k)
end

"""
    check_bc_storage_bounds(
      bc::AbstractBandColumn,
      j::Int,
      k::Int,
    )

Check whether `bc[j,k]` is a storable element, throwing
a `NoStorageForIndex` exception if it is not.
"""
@inline check_bc_storage_bounds(bc::AbstractBandColumn, j::Int, k::Int) =
  check_bc_storage_bounds(Bool, bc, j, k) ||
  throw(NoStorageForIndex(bc, (j, k)))

#=

Bulge functions.  Note that bulge_upper!  and bulge_lower! must be
implemented for any BandColumn.  The other methods have
implementations in terms of these.

=#

@inline is_upper_bulgeable_with_no_well(
  bc::AbstractBandColumn{NonSub},
  j::Int,
  k::Int,
) = true

@inline @propagate_inbounds function is_upper_bulgeable_with_no_well(
  bc::AbstractBandColumn,
  j::Int,
  k::Int,
)
  (first_inband_index(NonSub, bc, :, k) - row_offset(bc) <= row_size(bc)) &&
    (last_inband_index(NonSub, bc, j, :) - col_offset(bc) >= 1)
end

@inline is_lower_bulgeable_with_no_well(
  bc::AbstractBandColumn{NonSub},
  j::Int,
  k::Int,
) = true

@inline @propagate_inbounds function is_lower_bulgeable_with_no_well(
  bc::AbstractBandColumn,
  j::Int,
  k::Int,
)
  (last_inband_index(NonSub, bc, :, k) - row_offset(bc) >= 1) &&
    (first_inband_index(NonSub, bc, j, :) - col_offset(bc) <= col_size(bc))
end


"""
    bulge_upper!(
      bc::BandColumn,
      j::Int,
      k::Int,
    )

Given ``j`` and ``k`` in the storable upper triangular part of a
`BandColumn` matrix `bc`, extend the upper bandwidth so that all
elements in the upper triangular part below and to the left of
`bc[j,k]` are inband.

This includes a ``@boundscheck`` block with some more extensive error
checking.  However, if this block is elided, then the function
operates appropriately so long as ``(j,k)`` is storable index in ``bc``.

"""
@inline function bulge_upper!(
  bc::AbstractBandColumn,
  j::Int,
  k::Int,
)
  @boundscheck begin
    checkbounds(bc, j, k)
    check_bc_storage_bounds(bc, j, k)
    is_upper(bc, j, k) || throw(IndexNotUpper(j, k))
    is_upper_bulgeable_with_no_well(bc, j, k) || throw(WellError())
  end
  @inbounds begin
    j_first = first_inband_index(bc, :, k)
    k_last = last_inband_index(bc, j, :)
    # Does nothing if (j,k) is not above the diagonal.
    # j >= j_first ⟹ j:(j_first - 1) is empty.
    unsafe_set_last_inband_index!(bc, j:(j_first - 1), :, k)
    # j >= j_first ⟹ k <= k_last ⟹ (k_last+1):k is empty.
    unsafe_set_first_inband_index!(bc, :, (k_last + 1):k, j)
    nothing
  end
end

"""
    bulge_lower!(
      bc::BandColumn,
      j::Int,
      k::Int,
    )

Given ``j`` and ``k`` in the storable lower triangular part of a
`BandColumn` matrix `bc`, extend the lower bandwidth so that all
elements in the lower triangular part above and to the right of
`bc[j,k]` are inband.

This includes a ``@boundscheck`` block with some more extensive error
checking.  However, if this block is elided, then the function
operates appropriately so long as ``(j,k)`` is storable index in ``bc``.
"""
@inline function bulge_lower!(
  bc::AbstractBandColumn,
  j::Int,
  k::Int,
)
  @boundscheck begin
    checkbounds(bc, j, k)
    check_bc_storage_bounds(bc, j, k)
    is_lower(bc, j, k) || throw(IndexNotLower(j, k))
    is_lower_bulgeable_with_no_well(bc, j, k) || throw(WellError())
  end
  @inbounds begin
    j_last = last_inband_index(bc, :, k)
    k_first = first_inband_index(bc, j, :)
    # Does nothing if (j,k) is not below the diagonal.
    # j <= j_last ⟹ (j_last+1):j is empty.
    unsafe_set_first_inband_index!(bc, (j_last+1):j, :, k)
    # j <= j_last ⟹ k >= k_first ⟹ k:(k_first - 1) is empty.
    unsafe_set_last_inband_index!(bc, :, k:(k_first - 1), j) 
    nothing
  end
end

@inline function unsafe_set_first_inband_index!(
  bc::BandColumn,
  js::AbstractUnitRange{Int},
  ::Colon,
  k_first::Int,
) 
  bc.rows_first_last[first(js):last(js), 2] .= k_first + col_offset(bc)
  nothing
end

@inline function unsafe_set_first_inband_index!(
  bc::BandColumn,
  ::Colon,
  ks::AbstractUnitRange{Int},
  j_first::Int,
) 
  bc.cols_first_last[2, first(ks):last(ks)] .= j_first + row_offset(bc)
  nothing
end

@inline function unsafe_set_last_inband_index!(
  bc::BandColumn,
  js::AbstractUnitRange{Int},
  ::Colon,
  k_last::Int,
) 
  bc.rows_first_last[first(js):last(js), 5] .= k_last + col_offset(bc)
  nothing
end

@inline function unsafe_set_last_inband_index!(
  bc::BandColumn,
  ::Colon,
  ks::AbstractUnitRange{Int},
  j_last::Int,
) 
  bc.cols_first_last[5, first(ks):last(ks)] .= j_last + row_offset(bc)
  nothing
end

"""
    function zero_above!(
      bc::AbstractBandColumn{S,E},
      j::Int,
      k::Int,
    ) where {S,E<:Number}

    function zero_above!(
      bc::AbstractBandColumn{S,E},
      j::Int,
      ks::UnitRange{Int},
    ) where {S,E<:Number}

Insert hard zeros into elements above index ``(j,k)`` or indices
``(j,ks)``.  This does not adjust the structural bandwidth.
"""
@inline function zero_above!(
  bc::AbstractBandColumn{S,E},
  j::Int,
  k::Int,
) where {S,E<:Number}
  @boundscheck begin
    is_inband(bc, j, k) || throw(IndexNotInband(j, k))
  end
  bc_els = band_elements(bc)
  offs = storage_offset(bc, k)
  j0 = first_inband_index(bc, :, k)
  bc_els[j0-offs:j-offs,k] .= zero(E)
end

@inline function zero_above!(
  bc::AbstractBandColumn{S,E},
  j::Int,
  ks::UnitRange{Int},
) where {S,E<:Number}

  @boundscheck begin
    k_first = first(ks)
    k_last = last(ks)
    is_inband(bc, j, k_first) || throw(IndexNotInband(j, k_first))
    is_inband(bc, j, k_last) || throw(IndexNotInband(j, k_last))
  end
  bc_els = band_elements(bc)
  z = zero(E)
  for k ∈ ks
    offs = storage_offset(bc, k)
    j0 = first_inband_index(bc, :, k)
    bc_els[(j0 - offs):(j - offs), k] .= z
  end
end

"""
    function zero_below!(
      bc::AbstractBandColumn{S,E},
      j::Int,
      k::Int,
    ) where {S,E<:Number}

    function zero_below!(
      bc::AbstractBandColumn{S,E},
      j::Int,
      ks::UnitRange{Int},
    ) where {S,E<:Number}

Insert hard zeros into elements below index ``(j,k)`` or indices
``(j,ks)``.  This does not adjust the structural bandwidth.
"""
@inline function zero_below!(
  bc::AbstractBandColumn{S,E},
  j::Int,
  k::Int,
) where {S,E<:Number}

  @boundscheck begin
    is_inband(bc, j, k) || throw(IndexNotInband(j, k))
  end
  bc_els = band_elements(bc)
  offs = storage_offset(bc, k)
  j1 = last_inband_index(bc, :, k)
  bc_els[(j - offs):(j1 - offs), k] .= zero(E)
end

@inline function zero_below!(
  bc::AbstractBandColumn{S,E},
  j::Int,
  ks::UnitRange{Int},
) where {S,E<:Number}

  k_first = first(ks)
  k_last = last(ks)
  @boundscheck begin
    is_inband(bc, j, k_first) || throw(IndexNotInband(j, k_first))
    is_inband(bc, j, k_last) || throw(IndexNotInband(j, k_last))
  end
  bc_els = band_elements(bc)
  z = zero(E)
  for kk ∈ ks
    offs = storage_offset(bc, kk)
    j1 = last_inband_index(bc, :, kk)
    bc_els[(j - offs):(j1 - offs), kk] .= z
  end
end

"""
    function zero_right!(
      bc::AbstractBandColumn{S,E},
      j::Int,
      k::Int,
    ) where {S,E<:Number}

    function zero_right!(
      bc::AbstractBandColumn{S,E},
      js::UnitRange{Int},
      k::Int,
    ) where {S,E<:Number}

Insert hard zeros into elements to the right of index ``(j,k)`` or
indices ``(j,ks)``.  This does not adjust the structural bandwidth.
"""
@inline function zero_right!(
  bc::AbstractBandColumn{S,E},
  js::UnitRange{Int},
  k::Int,
) where {S,E<:Number}

  j_first = first(js)
  j_last = last(js)
  k_first = k
  k_last = last_inband_index(bc, j_last, :)

  @boundscheck begin
    is_inband(bc, j_first, k) || throw(IndexNotInband(j_first, k))
    is_inband(bc, j_last, k) || throw(IndexNotInband(j_last, k))
  end
  bc_els = band_elements(bc)
  z = zero(E)
  for kk ∈ k_first:k_last
    offs = storage_offset(bc, kk)
    j0 = max(j_first, first_inband_index(bc, :, kk)) - offs
    j1 = j_last - offs
    bc_els[j0:j1, kk] .= z
  end
end

@inline function zero_right!(
  bc::AbstractBandColumn{S,E},
  j::Int,
  k::Int,
) where {S,E<:Number}

  k_first = k
  k_last = last_inband_index(bc, j, :)

  @boundscheck begin
    is_inband(bc, j, k) || throw(IndexNotInband(j, k))
  end
  bc_els = band_elements(bc)
  z = zero(E)
  for kk ∈ k_first:k_last
    offs = storage_offset(bc, kk)
    bc_els[j - offs, kk] = z
  end
end

"""
    function zero_left!(
      bc::AbstractBandColumn{S,E},
      j::Int,
      k::Int,
    ) where {S,E<:Number}

    function zero_left!(
      bc::AbstractBandColumn{S,E},
      js::UnitRange{Int},
      k::Int,
    ) where {S,E<:Number}

Insert hard zeros into elements to the left of index ``(j,k)`` or
indices ``(j,ks)``.  This does not adjust the structural bandwidth.
"""
@inline function zero_left!(
  bc::AbstractBandColumn{S,E},
  js::UnitRange{Int},
  k::Int,
) where {S,E<:Number}

  j_first = first(js)
  j_last = last(js)
  k_first = first_inband_index(bc, j_first, :)
  k_last = k

  @boundscheck begin
    is_inband(bc, j_first, k) || throw(IndexNotInband(j_first, k))
    is_inband(bc, j_last, k) || throw(IndexNotInband(j_last, k))
  end
  bc_els = band_elements(bc)
  z = zero(E)
  for kk ∈ k_first:k_last
    offs = storage_offset(bc, kk)
    j0 = j_first - offs
    j1 = min(j_last, last_inband_index(bc, :, kk)) - offs
    bc_els[j0:j1, kk] .= z
  end
end

@inline function zero_left!(
  bc::AbstractBandColumn{S,E},
  j::Int,
  k::Int,
) where {S,E<:Number}

  k_first = first_inband_index(bc, j, :)
  k_last = k

  @boundscheck begin
    is_inband(bc, j, k) || throw(IndexNotInband(j, k))
  end
  bc_els = band_elements(bc)
  z = zero(E)
  for kk ∈ k_first:k_last
    offs = storage_offset(bc, kk)
    bc_els[j - offs, kk] = z
  end
end

"""
    bulge!(
      bc::AbstractBandColumn,
      j::Int,
      k::Int,
    )

Given ``j`` and ``k`` in the storable part of a `BandColumn` matrix
`bc`, extend the lower or upper bandwidth so that `bc[j,k]` is
inband.
"""
@propagate_inbounds @inline function bulge!(
  bc::AbstractBandColumn,
  j::Int,
  k::Int,
)
    is_upper(bc, j, k) && bulge_upper!(bc, j, k)
    is_lower(bc, j, k) && bulge_lower!(bc, j, k)
end

"""
    bulge!(
      bc::AbstractBandColumn,
      j_u::Int,
      k_u::Int,
      j_l::Int,
      k_l::Int,
    )

Extends band structure to include a rectangle with a corner ``(j_l,
k_l)`` in the lower triangular part and a corner ``(j_u, k_u)`` in the
upper triangular part.  This function does not check that ``j_u <=
j_l`` or ``k_u <= k_l``.  As a consequence, it can extend the
bandwidth even if the rectangle has no rows or now columns, so long as
``(j_u,k_u)`` is an upper index or ``(j_l, k_l)`` is a lower index.
This might not be what is desired.  As an alternative, the
``bulge!`` methods that take one or more ``AbstractUnitRange``
parameters check for empty ranges.
"""
@propagate_inbounds @inline function bulge!(
  bc::AbstractBandColumn,
  j_u::Int,
  k_u::Int,
  j_l::Int,
  k_l::Int,
)
  bulge_upper!(bc, j_u, k_u)
  bulge_lower!(bc, j_l, k_l)
end

"""
    bulge!(
      bc::AbstractBandColumn,
      js::AbstractUnitRange{Int},
      ks::AbstractUnitRange{Int},
    )

Given ``j`` and ``k`` in the storable part of a `BandColumn` matrix
`bc`, extend the lower or upper bandwidth so that `bc[j,k]` is
inband.
"""
@propagate_inbounds @inline function bulge!(
  bc::AbstractBandColumn,
  js::AbstractUnitRange{Int},
  ks::AbstractUnitRange{Int},
)
  j_u=first(js)
  k_u=last(ks)
  j_l=last(js)
  k_l=first(ks)
  if !isempty(js) && !isempty(ks)
    bulge_upper!(bc, j_u, k_u)
    bulge_lower!(bc, j_l, k_l)
  end
end

"""
    bulge!(
      bc::AbstractBandColumn,
      js::AbstractUnitRange{Int},
      ::Colon,
    )

Given ``js`` extend the band so that every row in ``js`` has the
minimum first index and maximum last index of any row in the range.
"""
@propagate_inbounds @inline function bulge!(
  bc::AbstractBandColumn,
  js::AbstractUnitRange{Int},
  ::Colon,
)
  j_u = first(js)
  j_l = last(js)
  k_l = first_inband_index(bc, j_u, :)
  k_u = last_inband_index(bc, j_l, :)
  @boundscheck begin
    checkbounds(bc, j_u, k_u)
    checkbounds(bc, j_l, k_l)
    check_bc_storage_bounds(bc, j_u, k_u)
    check_bc_storage_bounds(bc, j_l, k_l)
    is_lower_bulgeable_with_no_well(bc, j_l, k_l) || throw(WellError())
    is_upper_bulgeable_with_no_well(bc, j_u, k_u) || throw(WellError())
  end
  @inbounds if !isempty(j_u:j_l)  && !isempty(k_l:k_u)
    bulge_upper!(bc, j_u, k_u)
    bulge_lower!(bc, j_l, k_l)
  end
end

"""
    bulge!(
      bc::AbstractBandColumn,
      ::Colon,
      ks::AbstractUnitRange{Int},
    )

Given ``ks`` extend the band so that every column in ``ks`` has the
minimum first index and maximum last index of any column in the range.
"""
@propagate_inbounds @inline function bulge!(
  bc::AbstractBandColumn,
  ::Colon,
  ks::AbstractUnitRange{Int},
)
  k_l = first(ks)
  k_u = last(ks)
  j_l = last_inband_index(bc, :, k_u)
  j_u = first_inband_index(bc, :, k_l)
  @boundscheck begin
    checkbounds(bc, j_u, k_u)
    checkbounds(bc, j_l, k_l)
    check_bc_storage_bounds(bc, j_u, k_u)
    check_bc_storage_bounds(bc, j_l, k_l)
    is_lower_bulgeable_with_no_well(bc, j_l, k_l) || throw(WellError())
    is_upper_bulgeable_with_no_well(bc, j_u, k_u) || throw(WellError())
  end
  @inbounds if !isempty(j_u:j_l)  && !isempty(k_l:k_u)
    bulge_upper!(bc, j_u, k_u)
    bulge_lower!(bc, j_l, k_l)
  end
end

#=

Notch Operations

=#

@inline is_upper_notchable_with_no_well(
  bc::AbstractBandColumn{NonSub},
  j::Int,
  k::Int,
) = true

@inline @propagate_inbounds function is_upper_notchable_with_no_well(
  bc::AbstractBandColumn,
  j::Int,
  k::Int,
)
  (first_inband_index(NonSub, bc, :, k) - row_offset(bc) ==
   first_inband_index(bc, :, k)) &&
   (last_inband_index(NonSub, bc, j, :) - col_offset(bc) ==
    last_inband_index(bc, j, :))
end

@inline is_lower_notchable_with_no_well(
  bc::AbstractBandColumn{NonSub},
  j::Int,
  k::Int,
) = true

@inline @propagate_inbounds function is_lower_notchable_with_no_well(
  bc::AbstractBandColumn,
  j::Int,
  k::Int,
)
  (first_inband_index(NonSub, bc, j, :) - col_offset(bc) ==
   first_inband_index(bc, j, :)) &&
   (last_inband_index(NonSub, bc, :, k) - row_offset(bc) ==
    last_inband_index(bc, :, k))
end

"""
    notch_upper!(bc :: AbstractBandColumn, j::Int, k::Int)

Create a notch in the upper triangular part with corner at ``(j,k)``.

Example: if

    A = X U U O O O O
        L X U U O O O
        L X U U U O O
        O L X X X U O
        N O O O L U U
        N N O O L U U
        N N O O O L X
        N N N N N L X

then ``notch_upper!(A, 2, 3)`` results in

    A = X U O O O O O
        L X O O O O O
        L X U U U O O
        O L X X X U O
        N O O O L U U
        N N O O L U U
        N N O O O L X
        N N N N N L X
"""
@inline function notch_upper!(
  bc::AbstractBandColumn,
  j::Int,
  k::Int,
)
  @boundscheck begin
    (m, n) = size(bc)
    checkbounds(bc, j, k)
    is_upper(bc, j, k) || throw(IndexNotUpper(j, k))
    is_inband(bc, j, k) || throw(IndexNotInband(j, k))
    is_upper_notchable_with_no_well(bc, j, k) || throw(WellError())
  end
  @inbounds begin
    j_first = first_inband_index(bc, :, k)
    k_last = last_inband_index(bc, j, :)
    zero_above!(bc, j, k:k_last)
    unsafe_set_last_inband_index!(bc, j_first:j, :, k - 1)
    unsafe_set_first_inband_index!(bc, :, k:k_last, j + 1)
  end
  nothing
end


"""
    notch_lower!(bc :: AbstractBandColumn, j::Int, k::Int)

Create a notch in the lower triangular part with corner at ``(j,k)``.

Example: if

    A = X U U O O O O
        L X U U O O O
        L X U U U O O
        L L X X X U O
        N O L L L U U
        N N O L L U U
        N N O O O L X
        N N N N N L X

then ``notch_lower!(A, 5, 4)`` results in

    A = X U U O O O O
        L X U U O O O
        L X U U U O O
        L L X X X U O
        N O O O L U U
        N N O O L U U
        N N O O O L X
        N N N N N L X
"""
@inline function notch_lower!(
  bc::AbstractBandColumn,
  j::Int,
  k::Int,
)
  @boundscheck begin
    (m, n) = size(bc)
    checkbounds(bc, j, k)
    is_lower(bc, j, k) || throw(IndexNotLower(j, k))
    is_inband(bc, j, k) || throw(IndexNotInband(j, k))
    is_lower_notchable_with_no_well(bc, j, k) || throw(WellError())
  end
  @inbounds begin
    j_last = last_inband_index(bc, :, k)
    k_first = first_inband_index(bc, j, :)
    zero_below!(bc, j, k_first:k)
    unsafe_set_first_inband_index!(bc, j:j_last, :, k + 1)
    unsafe_set_last_inband_index!(bc, :, k_first:k, j - 1)
  end
  nothing
end

#=

Operations for validating and computing rows_first_last.

=#

"""
    compute_rows_first_last!(
      bc::AbstractBandColumn{NonSub},
      first_last::AbstractArray{Int,2},
    )

Compute row first and last elements, filling them into a separate
array.  This is not intended to work on submatrices.
"""
function compute_rows_first_last!(
  bc::AbstractBandColumn{NonSub},
  first_last::AbstractArray{Int,2},
)
  (m, n) = size(bc)
  first_last[:, 2] .= zero(Int)
  first_last[:, 5] .= zero(Int)
  coffs = col_offset(bc)
  for k ∈ n:-1:1
    jrange = inband_index_range(bc, :, k) ∩ (1:m)
    first_last[jrange, 2] .= k+coffs
  end
  for k ∈ 1:n
    jrange = inband_index_range(bc, :, k) ∩ (1:m)
    first_last[jrange, 5] .= k+coffs
  end
end

"""
    compute_rows_first_last!(bc::BandColumn{NonSub})

Compute row first and last elements, filling them into `bc.rows_first_last`.

"""
function compute_rows_first_last!(bc::BandColumn{NonSub})
  compute_rows_first_last!(bc, bc.rows_first_last)
end

"""
    compute_rows_first_last(bc::AbstractBandColumn{NonSub})

Compute row upper and lower bandwidths from column bandwidths, filling
them into a newly allocated array.
"""
function compute_rows_first_last(bc::AbstractBandColumn{NonSub})
  (m, n) = size(bc)
  first_last_arr = zeros(Int,m,6)
  compute_rows_first_last!(bc, first_last_arr)
  first_last_arr
end

"""
    validate_rows_first_last(bc::BandColumn{NonSub})

Check that the upper and lower first and last elements are consistent.
"""
function validate_rows_first_last(bc::BandColumn{NonSub})
  rfl = compute_rows_first_last(bc)
  @views rfl[:,2] == bc.rows_first_last[:,2]
  @views rfl[:,5] == bc.rows_first_last[:,5]
end

#=

Index operations

=#

@inline function Base.getindex(
  bc::AbstractBandColumn{S,E},
  j::Int,
  k::Int,
) where {S, E<:Number}
  @boundscheck begin
    checkbounds(bc, j, k)
    check_bc_storage_bounds(bc, j, k)
    is_inband(bc, j, k) || return zero(E)
  end
  @inbounds begin
    j1 = (j - storage_offset(bc,k))
    getindex(band_elements(bc), j1, k)
  end
end

@inline function Base.setindex!(
  bc::AbstractBandColumn{S,E},
  x::E,
  j::Int,
  k::Int,
) where {S,E<:Number}
  @boundscheck begin
    checkbounds(bc, j, k)
    check_bc_storage_bounds(bc, j, k)
  end
  @inbounds begin
    bulge_upper!(bc, j, k)
    bulge_lower!(bc, j, k)
    j1 = j - storage_offset(bc, k)
    band_elements(bc)[j1, k] = x
  end
end

"""
    setindex_no_bulge!(
      bc::AbstractBandColumn{S,E},
      x::E,
      j::Int,
      k::Int,
    ) where {S,E<:Number}

A version of `setindex!` that does not extend bandwidth.  This is
useful in loops where the bandwidth extension can be done outside
the loop for efficiency.
"""
@inline function setindex_no_bulge!(
  bc::AbstractBandColumn{S, E},
  x::E,
  j::Int,
  k::Int,
) where {S,E<:Number}
  @boundscheck begin
    checkbounds(bc, j, k)
    check_bc_storage_bounds(bc, j, k)
  end
  @inbounds begin
    j1 = j - storage_offset(bc, k)
    band_elements(bc)[j1,k]=x
  end
end

@propagate_inbounds @inline function Base.view(
  bc::AbstractBandColumn,
  I::Vararg{Any,2},
)
  viewbc(bc, I)
end

"""
    viewbc(bc::BandColumn, i::Tuple{AbstractUnitRange{Int},
           AbstractUnitRange{Int}})

Return a bandcolumn submatrix with views of the relevant arrays
wrapped into a BandColumn.
"""
@inline function viewbc(
  bc::BandColumn,
  i::Tuple{AbstractUnitRange{Int},AbstractUnitRange{Int}},
)
  (rows, cols) = i
  j0 = first(rows)
  j1 = last(rows)
  k0 = first(cols)
  k1 = last(cols)

  @boundscheck begin
    checkbounds(bc, j0, k0)
    checkbounds(bc, j1, k1)
  end
  @inbounds BandColumn(
    Sub(),
    bc.m_nonsub,
    bc.n_nonsub,
    max(j1 - j0 + 1, 0),
    max(k1 - k0 + 1, 0),
    bc.roffset + j0 - 1,
    bc.coffset + k0 - 1,
    bc.bw_max,
    bc.upper_bw_max,
    bc.middle_lower_bw_max,
    view(bc.rows_first_last, rows, 1:6),
    view(bc.cols_first_last, 1:6, cols),
    view(bc.band_elements, :, cols),
  )
end

@inline function Base.getindex(
  bc::BandColumn,
  rows::AbstractUnitRange{Int},
  cols::AbstractUnitRange{Int},
)

  j0 = first(rows)
  j1 = last(rows)
  k0 = first(cols)
  k1 = last(cols)

  @boundscheck begin
    checkbounds(bc, j0, k0)
    checkbounds(bc, j1, k1)
  end
  @inbounds BandColumn(
    Sub(),
    bc.m_nonsub,
    bc.n_nonsub,
    max(j1 - j0 + 1, 0),
    max(k1 - k0 + 1, 0),
    bc.roffset + j0 - 1,
    bc.coffset + k0 - 1,
    bc.bw_max,
    bc.upper_bw_max,
    bc.middle_lower_bw_max,
    bc.rows_first_last[rows, :],
    bc.cols_first_last[:, cols],
    bc.band_elements[:, cols],
  )
end

@propagate_inbounds @inline Base.getindex(
  bc::AbstractBandColumn,
  ::Colon,
  cols::AbstractUnitRange{Int},
) = getindex(bc, 1:row_size(bc), cols)

@propagate_inbounds @inline Base.getindex(
  bc::AbstractBandColumn,
  rows::AbstractUnitRange{Int},
  ::Colon,
) = getindex(bc, rows, 1:col_size(bc))

@inline Base.getindex(bc::AbstractBandColumn, ::Colon, ::Colon) = bc

@propagate_inbounds @inline function viewbc(
  bc::AbstractBandColumn,
  i::Tuple{Colon,AbstractUnitRange{Int}}
)
  (_, cols) = i
  viewbc(bc, (1:row_size(bc), cols))
end

@propagate_inbounds @inline function viewbc(
  bc::AbstractBandColumn,
  i::Tuple{AbstractUnitRange{Int},Colon},
)
  (rows, _) = i
  viewbc(bc, (rows, 1:col_size(bc)))
end

@propagate_inbounds @inline function viewbc(
  bc::AbstractBandColumn,
  ::Tuple{Colon,Colon},
)
  viewbc(bc, (1:row_size(bc), 1:col_size(bc)))
end

function LinearAlgebra.Matrix(bc::AbstractBandColumn{S,E}) where {S,E<:Number}
  (m, n) = size(bc)
  a = zeros(E, m, n)
  for k = 1:n
    for j in inband_index_range(bc, :, k)
      a[j, k] = bc[j, k]
    end
  end
  a
end

function Base.eachindex(bc::AbstractBandColumn)
  (_, n) = size(bc)
  (CartesianIndex(j, k) for k = 1:n for j ∈ inband_index_range(bc, :, k))
end

"""
    get_elements(bc::AbstractBandColumn)

Get all the stored elements of `bc` in a generator.
"""
function get_elements(bc::AbstractBandColumn)
  (_, n) = size(bc)
  @inbounds (bc[j, k] for k = 1:n for j ∈ inband_index_range(bc, :, k))
end

#=

 Copying

=#

function Base.copy(bc::BandColumn)
  BandColumn(
    bc.sub,
    bc.m_nonsub,
    bc.n_nonsub,
    bc.m,
    bc.n,
    bc.roffset,
    bc.coffset,
    bc.bw_max,
    bc.upper_bw_max,
    bc.middle_lower_bw_max,
    copy(bc.rows_first_last),
    copy(bc.cols_first_last),
    copy(bc.band_elements),
  )
end

#=

Print and show.

=#

#=

The method show for BandColumn matrices represents elements for which
there is no storage with `N`.  Elements that have available storage
but are not actually stored are represented by `O`.  These are
elements that are outside the current bandwidth but not outside the
maximum bandwidth.

=#
function Base.show(io::IO, bc::BandColumn)
  print(
    io,
    typeof(bc),
    "(",
    bc.m_nonsub,
    ", ",
    bc.n_nonsub,
    ", ",
    bc.m,
    ", ",
    bc.n,
    ", ",
    bc.bw_max,
    ", ",
    bc.roffset,
    ", ",
    bc.coffset,
    ", ",
    bc.upper_bw_max,
    ", ",
    bc.middle_lower_bw_max,
    ", ",
    bc.rows_first_last,
    ", ",
    bc.cols_first_last,
    "): ",
  )
  for j ∈ 1:(bc.m)
    println()
    for k ∈ 1:(bc.n)
      if check_bc_storage_bounds(Bool, bc, j, k)
        if is_inband(bc, j, k)
          @printf("%10.2e", bc[j, k])
        else
          print("         O")
        end
      else
        print("         N")
      end
    end
  end
end

Base.show(io::IO, ::MIME"text/plain", bc::BandColumn) = show(io, bc)

Base.print(io::IO, bc::BandColumn) = print(
    io,
    typeof(bc),
    "(",
    bc.m_nonsub,
    ", ",
    bc.n_nonsub,
    ", ",
    bc.m,
    ", ",
    bc.n,
    ", ",
    bc.bw_max,
    ", ",
    bc.roffset,
    ", ",
    bc.coffset,
    ", ",
    bc.upper_bw_max,
    ", ",
    bc.middle_lower_bw_max,
    ", ",
    bc.rows_first_last,
    ", ",
    bc.cols_first_last,
    ", ",
    bc.band_elements,
    ")",
)

Base.print(io::IO, ::MIME"text/plain", bc::BandColumn) = print(io, bc)

struct Wilk
  arr :: Array{Char,2}
end

function Base.show(io::IO, w::Wilk)
  (m, n) = size(w.arr)
  for j = 1:m
    println(io)
    for k = 1:n
      print(io,w.arr[j, k], " ")
    end
  end
end

function Base.show(w::Wilk)
  (m, n) = size(w.arr)
  for j = 1:m
    println()
    for k = 1:n
      print(w.arr[j, k], " ")
    end
  end
end

"""
    wilk(bc :: AbstractBandColumn)

Generate a Wilkinson diagram for bc.

"""
@views function wilk(bc :: AbstractBandColumn)
  (m,n) = size(bc)
  a = fill('N', (m, n))
  for k ∈ 1:n
    fill!(a[storable_index_range(bc, :, k), k], 'O')
    fill!(a[upper_inband_index_range(bc, :, k), k], 'U')
    fill!(a[middle_inband_index_range(bc, :, k), k], 'X')
    fill!(a[lower_inband_index_range(bc, :, k), k], 'L')
  end
  Wilk(a)
end

end # module
