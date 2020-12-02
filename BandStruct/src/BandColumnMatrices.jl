module BandColumnMatrices

using Printf
using LinearAlgebra
using Base: @propagate_inbounds

export BandColumn,
  AbstractBandColumn,
  WellError,
  EmptyUpperRange,
  EmptyLowerRange,
  first_inband_el_storage,
  last_inband_el_storage,
  inband_els_range_storage,
  upper_inband_els_range_storage,
  middle_inband_els_range_storage,
  lower_inband_els_range_storage,
  first_storable_el,
  last_storable_el,
  storable_els_range,
  first_inband_el,
  last_inband_el,
  inband_els_range,
  upper_inband_els_range,
  middle_inband_els_range,
  lower_inband_els_range,
  NoStorageForIndex,
  storage_offset,
  bc_index_stored,
  check_bc_storage_bounds,
  get_m_els,
  get_m,
  get_n,
  get_roffset,
  get_coffset,
  get_rbws,
  get_cbws,
  get_upper_bw_max,
  get_middle_bw_max,
  get_lower_bw_max,
  get_band_elements,
  upper_bw,
  middle_bw,
  lower_bw,
  first_super,
  first_sub,
  extend_lower_band!,
  extend_upper_band!,
  extend_band!,
  get_elements,
  viewbc,
  hull,
  setindex_noext!,
  compute_rbws!,
  compute_rbws,
  validate_rbws,
  wilk,
  Wilk,
  trim_upper!,
  trim_lower!

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
struct EmptyUpperRange <: Exception end
struct EmptyLowerRange <: Exception end

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

    AbstractBandColumn{E,AE,AI} <: AbstractArray{E,2}

An AbstractBandColumn should implement the following:

- `get_m(bc)`

- `get_n(bc)`

- `get_roffset(bc)`

- `get_coffset(bc)`

- `get_upper_bw_max()`

- `get_lower_bw_max()`

- `get_rbws(bc)`

- `get_cbws(bc)`

- `get_band_elements(bc)`

"""
abstract type AbstractBandColumn{E,AE,AI} <: AbstractArray{E,2} end
# abstract type AbstractBandColumn{E,AE,AI} end

"""

# BandColumn

    BandColumn{E<:Number,AE<:AbstractArray{E,2},AI<:AbstractArray{Int,2}}
    <: AbstractBandColumn{E,AE,AI}

A simplified band column structure that does not include leading
blocks but does include uniform offsets that can be changed to give
different submatrices.  This can be used to represent submatrices of a
LeadingBandColumn matrix.

# Fields
- `m::Int`: Matrix number of rows.

- `n::Int`: Matrix and elements number of columns.

- `m_els::Int`: Elements number of rows.

- `roffset::Int`: Uniform column offset.

- `coffset::Int`: Uniform row offset.

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

    m =                  8
    n =                  7
    m_els =              8
    num_blocks =         6
    upper_bw_max =       4
    middle_bw_max =      2
    lower_bw_max =       2
    cbws =               [ 1 1 1 1 1 1 1;  # upper
                           1 2 1 1 1 0 2;  # middle
                           2 1 0 0 2 1 0;  # lower
                           0 1 3 3 3 6 6 ] # first superdiagonal.
    rbws =               [ 0 1 1 0;
                           1 1 0 1;
                           1 1 3 1;
                           1 3 0 2;
                           1 0 0 5;
                           1 0 2 5;
                           1 1 0 6;
                           1 1 0 6 ]
    leading_blocks =     [ 1 3 4 6 6 8;     # rows
                           1 2 5 5 6 7 ]    # columns


"""
struct BandColumn{E<:Number,AE<:AbstractArray{E,2},AI<:AbstractArray{Int,2}} <:
       AbstractBandColumn{E,AE,AI}
  m::Int
  n::Int
  m_els::Int
  roffset::Int
  coffset::Int
  upper_bw_max::Int
  middle_bw_max::Int
  lower_bw_max::Int
  rbws::AI
  cbws::AI
  band_elements::AE
end

@inline Base.size(bc::BandColumn) = (bc.m, bc.n)

@inline get_m_els(bc::BandColumn) = bc.m_els

@inline get_m(bc::BandColumn) = bc.m
@inline get_n(bc::BandColumn) = bc.n

@inline get_roffset(bc::BandColumn) = bc.roffset
@inline get_coffset(bc::BandColumn) = bc.coffset

@inline get_upper_bw_max(bc::BandColumn) = bc.upper_bw_max
@inline get_middle_bw_max(bc::BandColumn) = bc.middle_bw_max
@inline get_lower_bw_max(bc::BandColumn) = bc.lower_bw_max
@inline get_rbws(bc::BandColumn) = bc.rbws
@inline get_cbws(bc::BandColumn) = bc.cbws

@inline get_band_elements(bc::BandColumn) = bc.band_elements

@propagate_inbounds @inline upper_bw(bc::BandColumn, ::Colon, k::Int) =
  bc.cbws[1, k]
@propagate_inbounds @inline middle_bw(bc::BandColumn, ::Colon, k::Int) =
  bc.cbws[2, k]
@propagate_inbounds @inline lower_bw(bc::BandColumn, ::Colon, k::Int) =
  bc.cbws[3, k]
@propagate_inbounds @inline upper_bw(bc::BandColumn, j::Int, ::Colon) =
  bc.rbws[j, 3]
@propagate_inbounds @inline middle_bw(bc::BandColumn, j::Int, ::Colon) =
  bc.rbws[j, 2]
@propagate_inbounds @inline lower_bw(bc::BandColumn, j::Int, ::Colon) =
  bc.rbws[j, 1]

"""

    first_super(bc, js, ks)

- If `js::Colon` and `ks::Int`, give the first superdiagonal in
  columns `ks`.  Column superdiagonals are numbered going up starting
  from the middle elements of the band structure.


- If `js::Int` and `ks::Colon`, give the first superdiagonal in row
  `js`.  Row superdiagonals are numbered going right from the middle
  elements of the band structure.


"""
@propagate_inbounds @inline first_super(
  bc::AbstractBandColumn,
  ::Colon,
  k::Int,
) = get_cbws(bc)[4, k] - get_roffset(bc)

@propagate_inbounds @inline first_super(
  bc::AbstractBandColumn,
  j::Int,
  ::Colon,
) = first_sub(bc, j, :) + middle_bw(bc, j, :) + 1

"""

    first_sub(bc, js, ks)

- If `js::Colon` and `ks::Int`, give the first subdiagonal in columns
  `ks`.  Column sudiagonals are numbered going down, starting from the
  middle of the band structure.


- If `js::Int` and `ks::Colon`, give the first subdiagonal in row
  `js`.  Row subdiagonals are numbered going left from the middle
  elements of the band structure.

"""
@propagate_inbounds @inline first_sub(bc::AbstractBandColumn, j::Int, ::Colon) =
  get_rbws(bc)[j, 4] - get_coffset(bc)

@propagate_inbounds @inline first_sub(bc::AbstractBandColumn, ::Colon, k::Int) =
  first_super(bc, :, k) + middle_bw(bc, :, k) + 1


"""

    extend_band!(
      bc::AbstractBandColumn,
      j::Int,
      ::Colon,
    )

Merge the band profile of rows ``j`` and ``j+1``, extending the lower
bandwidth of row ``j+1`` and the upper bandwidth of row ``j``.  Note
that this does not extend the lower bandwidth of row ``j`` or the
upper bandwidth of row ``j+1``, which is justified by a "well-free"
assumption.  The column bandwidths are adjusted accordingly to be
consistent with the row bandwidths.

"""
@propagate_inbounds @inline function extend_band!(
  bc::AbstractBandColumn,
  j::Int,
  ::Colon,
)
  krange0 = inband_els_range(bc, j, :)
  krange1 = inband_els_range(bc, j+1, :)
  @boundscheck begin
    checkbounds(bc, j, 1)
    checkbounds(bc, j+1, 1)
    isempty(krange0) || check_bc_storage_bounds(bc, j+1, krange0.start)
    isempty(krange1) || check_bc_storage_bounds(bc, j, krange1.stop)
  end
  cbws = get_cbws(bc)
  rbws = get_rbws(bc)
  rbws[j + 1, 1] = rbws[j, 1] + first_sub(bc, j + 1, :) - first_sub(bc, j, :)
  rbws[j, 3] = rbws[j + 1, 3] + first_super(bc, j + 1, :) - first_super(bc, j, :)
  for l ∈ (krange0.start):(krange1.start - 1)
    cbws[3, l] += 1
  end
  for l ∈ (krange0.stop + 1):(krange1.stop)
    cbws[1, l] += 1
  end
  nothing
end

"""

    extend_band!(
      bc::AbstractBandColumn,
      ::Colon,
      k::Int,
    )

Merge the band profile of columns ``k`` and ``k+1``, extending the
upper bandwidth of column ``k+1`` and the lower bandwidth of column
``k``.  The row bandwidths are adjusted accordingly to be consistent
with the column bandwidths.

"""
@propagate_inbounds @inline function extend_band!(
  bc::AbstractBandColumn,
  ::Colon,
  k::Int,
)
  jrange0 = inband_els_range(bc, :, k)
  jrange1 = inband_els_range(bc, :, k+1)
  @boundscheck begin
    checkbounds(bc, 1, k)
    checkbounds(bc, 1, k+1)
    isempty(jrange0) || check_bc_storage_bounds(bc, jrange0.start, k+1)
    isempty(jrange1) || check_bc_storage_bounds(bc, jrange1.stop, k)
  end
  cbws = get_cbws(bc)
  rbws = get_rbws(bc)
  cbws[1,k+1] = cbws[1,k] + first_super(bc,:,k+1) - first_super(bc,:,k)
  cbws[3,k] = cbws[3,k+1] + first_sub(bc,:,k+1) - first_sub(bc,:,k)
  for l ∈ jrange0.start:jrange1.start - 1
    rbws[l,3] += 1
  end
  for l ∈ jrange0.stop+1:jrange1.stop
    rbws[l,1] += 1
  end
  nothing
end

"""
    extend_upper_band!(
      bc::AbstractBandColumn,
      j::Int,
      k::Int,
    )

Given ``j`` and ``k`` in the storable upper triangular part of a
`BandColumn` matrix `bc`, extend the upper bandwidth so that all
elements in the upper triangular part below and to the left of
`bc[j,k]` are inband.

"""
@propagate_inbounds @inline function extend_upper_band!(
  bc::AbstractBandColumn,
  j::Int,
  k::Int,
)
  @boundscheck begin
    checkbounds(bc, j, k)
    check_bc_storage_bounds(bc, j, k)
  end
  cbws = get_cbws(bc)
  rbws = get_rbws(bc)
  j1 = first_inband_el(bc, :, k) - 1
  k0 = last_inband_el(bc, j, :) + 1
  for l = j:j1
    rbws[l, 3] = max(rbws[l, 3], k - first_super(bc, l, :) + 1)
  end
  for l = k0:k
    cbws[1, l] = max(cbws[1, l], first_super(bc, :, l) - j + 1)
  end
  nothing
end

"""
    extend_lower_band!(
      bc::AbstractBandColumn,
      j::Int,
      k::Int,
    )

Given ``j`` and ``k`` in the storable lower triangular part of a
`BandColumn` matrix `bc`, extend the lower bandwidth so that all
elements in the lower triangular part above and to the right of
`bc[j,k]` are inband.

"""
@propagate_inbounds @inline function extend_lower_band!(
  bc::AbstractBandColumn,
  j::Int,
  k::Int,
)
  @boundscheck begin
    checkbounds(bc, j, k)
    check_bc_storage_bounds(bc, j, k)
  end
  cbws = get_cbws(bc)
  rbws = get_rbws(bc)
  j0 = last_inband_el(bc, :, k) + 1
  k1 = first_inband_el(bc,j,:) - 1
  for l ∈ k:k1
    cbws[3, l] = max(cbws[3, l], j - first_sub(bc, :, l) + 1)
  end
  for l ∈ j0:j
    rbws[l, 1] = max(rbws[l, 1], first_sub(bc, l, :) - k + 1)
  end
  nothing
end


"""
    extend_band!(
      bc::AbstractBandColumn,
      j::Int,
      k::Int,
    )

Given ``j`` and ``k`` in the storable part of a `BandColumn` matrix
`bc`, extend the lower or upper bandwidth so that `bc[j,k]` is
inband.

"""
@propagate_inbounds @inline function extend_band!(
  bc::AbstractBandColumn,
  j::Int,
  k::Int,
)
  extend_upper_band!(bc, j, k)
  extend_lower_band!(bc, j, k)
end

"""
    extend_band!(
      bc::AbstractBandColumn,
      js::UnitRange{Int},
      k::Int,
    )

Given ``js`` and ``k`` in the storable part of a `BandColumn` matrix
`bc`, extend the lower and/or upper bandwidth so that `bc[js,k]` is
inband.

"""
@propagate_inbounds @inline function extend_band!(
  bc::AbstractBandColumn,
  js::UnitRange{Int},
  k::Int,
)
  if !isempty(js)
    extend_upper_band!(bc, js.start, k)
    extend_lower_band!(bc, js.stop, k)
  end
end

"""
    extend_band!(
      bc::AbstractBandColumn,
      j::Int,
      ks::UnitRange{Int},
    )

Given ``j`` and ``ks`` in the storable part of a `BandColumn` matrix
bc`, extend the lower and/or upper bandwidth so that `bc[j,ks]` is
inband.

"""
@propagate_inbounds @inline function extend_band!(
  bc::AbstractBandColumn,
  j::Int,
  ks::UnitRange{Int},
)
  if !isempty(ks)
    extend_upper_band!(bc, j, ks.start)
    extend_lower_band!(bc, j, ks.stop)
  end
end

"""
    extend_band!(
      bc::AbstractBandColumn,
      js::UnitRange{Int},
      ks::UnitRange{Int},
    )

Given ``js`` and ``ks`` in the storable part of a `BandColumn` matrix
bc`, extend the lower and/or upper bandwidth so that `bc[js,ks]` is
inband.

"""
@propagate_inbounds @inline function extend_band!(
  bc::AbstractBandColumn,
  js::UnitRange{Int},
  ks::UnitRange{Int},
)
  if !isempty(js) && !isempty(ks)
    extend_upper_band!(bc, js.start, ks.stop)
    extend_lower_band!(bc, js.stop, ks.start)
  end
end

"""
    compute_rbws!(
      bc::AbstractBandColumn,
      rbws::AbstractArray{Int,2},
    )

Compute row upper and lower bandwidths from column bandwidths, filling
them into a separate array.  Note that this does not fill in
rbws[j,4], which is the first subdiagonal in row j or the middle
bandwidths.  It is assumed that those will never change.  (They are
determined by the leading block sizes).

"""
function compute_rbws!(
  bc::AbstractBandColumn,
  rbws::AbstractArray{Int,2},
)
  (m, n) = size(bc)
  rbws[:, 1:3] .= zero(Int)
  roffs = get_roffset(bc)
  for k ∈ n:-1:1
    jrange = ((lower_inband_els_range(bc, :, k) .+ roffs)) ∩ (1:m)
    rbws[jrange, 1] .+= 1
    jrange = (middle_inband_els_range(bc, :, k) .+ roffs) ∩ (1:m)
    rbws[jrange, 2] .+= 1
    jrange = (upper_inband_els_range(bc, :, k) .+ roffs) ∩ (1:m)
    rbws[jrange, 3] .+= 1
  end
end

"""
    compute_rbws!(bc::AbstractBandColumn)

Compute row upper and lower bandwidths from column bandwidths, filling
them into `bc.rbws`.

"""
function compute_rbws!(bc::AbstractBandColumn)
  compute_rbws!(bc, bc.rbws)
end

"""
    compute_rbws(bc::AbstractBandColumn)

Compute row upper and lower bandwidths from column bandwidths, filling
them into a newly allocated array.

"""
function compute_rbws(bc::AbstractBandColumn)
  (m, n) = size(bc)
  rbws1 = zeros(Int,m,3)
  compute_rbws!(bc, rbws1)
  rbws1
end

"""
    validate_rbws(bc::AbstractBandColumn)

Check that the upper and lower bandwidths in `bc.rbws` and `bc.cbws`
are consistent.

"""
function validate_rbws(bc::AbstractBandColumn)
  (m, n) = size(bc)
  rbws1 = zeros(Int,m,3)
  compute_rbws!(bc, rbws1)
  rbws1 == bc.rbws[:,1:3]
end

"""
    storage_offset(bc::AbstractBandColumn, k::Int)

Compute a row offset to look into storage:

    d = storage_offset(bc,k)
    bc[j,k] == bc.band_elements[j - d, k]
"""
@propagate_inbounds @inline storage_offset(bc::AbstractBandColumn, k::Int) =
  first_super(bc, :, k) - get_upper_bw_max(bc)

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
  j1 = j - storage_offset(bc, k)
  j1 >= 1 && j1 <= get_m_els(bc)
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

"""
    first_inband_el_storage(bc::AbstractBandColumn, k::Int)

If 

    j=first_inband_el_storage(bc::AbstractBandColumn, k::Int)

then `bc.band_elements[j,k]` is the first inband element
in column ``k``.
"""
@propagate_inbounds @inline first_inband_el_storage(
  bc::AbstractBandColumn,
  k::Int,
) = get_upper_bw_max(bc) - upper_bw(bc, :, k) + 1

"""
    last_inband_el_storage(bc::AbstractBandColumn, k::Int)

If 

    j=last_inband_el_storage(bc::AbstractBandColumn, k::Int)

then `bc.band_elements[j,k]` is the last inband element
in column ``k``.
"""
@propagate_inbounds @inline last_inband_el_storage(
  bc::AbstractBandColumn,
  k::Int,
) = get_upper_bw_max(bc) + middle_bw(bc, :, k) + lower_bw(bc, :, k)

"""
    first_storable_el(bc::AbstractBandColumn, k::Int)

If 

    j=first_storable_el(bc::AbstractBandColumn, k::Int)

then `bc[j,k]` is the first element in column ``k`` for which there is
available storage.
"""
@propagate_inbounds @inline first_storable_el(
  bc::AbstractBandColumn,
  ::Colon,
  k::Int,
) = first_super(bc, :, k) - get_upper_bw_max(bc) + 1

"""
    last_storable_el(bc::AbstractBandColumn, k::Int)

If 

    j=last_storable_el(bc::AbstractBandColumn, k::Int)

then `bc[j,k]` is the last element in column ``k`` for which there is
available storage..
"""
@propagate_inbounds @inline last_storable_el(
  bc::AbstractBandColumn,
  ::Colon,
  k::Int,
) = first_super(bc, :, k) +  get_middle_bw_max(bc) + get_lower_bw_max(bc)

"""
    first_inband_el(bc::AbstractBandColumn, ::Colon, k::Int)

If

  j=first_inband_el(bc,:,k)

then `bc[j,k]` is the first inband element in column ``k``.

"""
@propagate_inbounds @inline first_inband_el(
  bc::AbstractBandColumn,
  ::Colon,
  k::Int,
) = first_super(bc, :, k) - upper_bw(bc, :, k) + 1

"""
    first_inband_el(bc::AbstractBandColumn, j::Int, ::Colon)

If

  k=first_inband_el(bc,j,:)

then `bc[j,k]` is the first inband element in row ``j``.
"""
@propagate_inbounds @inline first_inband_el(
  bc::AbstractBandColumn,
  j::Int,
  ::Colon,
) = first_sub(bc, j, :) - lower_bw(bc, j, :) + 1

"""
    last_inband_el(bc::AbstractBandColumn, ::Colon, k::Int)

If

  j=last_inband_el(bc,:,k)

then `bc[j,k]` is the last inband element in column ``k``.

"""
@propagate_inbounds @inline last_inband_el(
  bc::AbstractBandColumn,
  ::Colon,
  k::Int,
) = first_super(bc, :, k) + middle_bw(bc, :, k) + lower_bw(bc, :, k)

"""
    last_inband_el(bc::AbstractBandColumn, j::Int, ::Colon)

If

  k=last_inband_el(bc,j,:)

then `bc[j,k]` is the last inband element in row ``j``.
"""
@propagate_inbounds @inline last_inband_el(
  bc::AbstractBandColumn,
  j::Int,
  ::Colon,
) = first_sub(bc, j, :) + middle_bw(bc, j, :) + upper_bw(bc, j, :)

"""
    function inband_els_range(
      bc::AbstractBandColumn,
      ::Colon,
      k::Int,
    )

Range of inband elements in column ``k``.
"""
@propagate_inbounds @inline function inband_els_range(
  bc::AbstractBandColumn,
  ::Colon,
  k::Int,
)
  (m, _) = size(bc)
  (1:m) ∩ (first_inband_el(bc, :, k):last_inband_el(bc, :, k))
end

"""
    function inband_els_range(
      bc::AbstractBandColumn,
      j::Int,
      ::Colon,
    )

Range of inband elements in row ``j``.
"""
@propagate_inbounds @inline function inband_els_range(
  bc::AbstractBandColumn,
  j::Int,
  ::Colon,
)
  (_, n) = size(bc)
  (1:n) ∩ (first_inband_el(bc, j, :):last_inband_el(bc, j, :))
end

"""
    function upper_inband_els_range(
      bc::AbstractBandColumn,
      ::Colon,
      k::Int,
    )

Range of inband upper elements in column ``k``.
"""
@propagate_inbounds @inline function upper_inband_els_range(
  bc::AbstractBandColumn,
  ::Colon,
  k::Int,
)
  (m, _) = size(bc)
  (1:m) ∩ (first_inband_el(bc, :, k):first_super(bc, :, k))
end

"""
    function upper_inband_els_range(
      bc::AbstractBandColumn,
      j::Int,
      ::Colon,
    )

Range of inband upper elements in row ``j``.
"""
@propagate_inbounds @inline function upper_inband_els_range(
  bc::AbstractBandColumn,
  j::Int,
  ::Colon,
)
  (_, n) = size(bc)
  (1:n) ∩ (first_super(bc, j, :):last_inband_el(bc, j, :))
end

"""
    function middle_inband_els_range(
      bc::AbstractBandColumn,
      ::Colon,
      k::Int,
    )

Range of inband middle elements in column ``k``.
"""
@propagate_inbounds @inline function middle_inband_els_range(
  bc::AbstractBandColumn,
  ::Colon,
  k::Int,
)
  (m, _) = size(bc)
  j = first_super(bc, :, k)
  (1:m) ∩ ((j + 1):(j + middle_bw(bc, :, k)))
end

"""
    function middle_inband_els_range(
      bc::AbstractBandColumn,
      j::Int,
      ::Colon,
    )

Range of inband middle elements in row ``j``.
"""
@propagate_inbounds @inline function middle_inband_els_range(
  bc::AbstractBandColumn,
  j::Int,
  ::Colon,
)
  (_, n) = size(bc)
  k = first_sub(bc, j, :)
  (1:n) ∩ ((k + 1):(k + middle_bw(bc, j, :)))
end

"""
    function lower_inband_els_range(
      bc::AbstractBandColumn,
      ::Colon,
      k::Int,
    )

Range of inband lower elements in column ``k``.
"""
@propagate_inbounds @inline function lower_inband_els_range(
  bc::AbstractBandColumn,
  ::Colon,
  k::Int,
)
  (m, _) = size(bc)
  j = first_super(bc, :, k) + middle_bw(bc, :, k)
  (1:m) ∩ ((j + 1):(j + lower_bw(bc, :, k)))
end

"""
    function lower_inband_els_range(
      bc::AbstractBandColumn,
      j::Int,
      ::Colon,
    )

Range of inband elements in row ``j``.
"""
@propagate_inbounds @inline function lower_inband_els_range(
  bc::AbstractBandColumn,
  j::Int,
  ::Colon,
)
  (_, n) = size(bc)
  (1:n) ∩ (first_inband_el(bc, j, :):first_sub(bc,j,:))
end

"""
    inband_els_range_storage(
       bc::AbstractBandColumn,
       k::Int,
    )

Compute the range of inband elements in column ``k`` of
`bc.band_elements`.
"""
@propagate_inbounds @inline function inband_els_range_storage(
  bc::AbstractBandColumn,
  k::Int,
)
    (1:get_m_els(bc)) ∩
      (first_inband_el_storage(bc, k):last_inband_el_storage(bc, k))
end

"""
    storable_els_range(
      bc::AbstractBandColumn,
      ::Colon,
      k::Int,
    )

The range of elements in column ``k`` of `bc` for which storage is
available.
"""
@propagate_inbounds @inline function storable_els_range(
  bc::AbstractBandColumn,
  ::Colon,
  k::Int,
)
  (1:get_m(bc)) ∩ (first_storable_el(bc, :, k):last_storable_el(bc, :, k))
end

"""
    upper_inband_els_range_storage(
       bc::AbstractBandColumn,
       k::Int)

Compute the range of upper inband elements in column ``k`` of
`bc.band_elements`.
"""
@propagate_inbounds @inline function upper_inband_els_range_storage(
  bc::AbstractBandColumn,
  k::Int,
)
  m = bc.m_els
  j = first_inband_el_storage(bc, k)
  (1:m) ∩ (j:(j + upper_bw(bc, :, k) - 1))
end

"""
    middle_inband_els_range_storage(
       bc::AbstractBandColumn,
       k::Int,
    )

Compute the range of middle inband elements in column ``k`` of
`bc.band_elements`.
"""
@propagate_inbounds @inline function middle_inband_els_range_storage(
  bc::AbstractBandColumn,
  k::Int,
)
  m = bc.m_els
  j = first_inband_el_storage(bc, k) + upper_bw(bc, :, k)
  (1:m) ∩ (j:(j + middle_bw(bc, :, k) - 1))
end

"""
    lower_inband_els_range_storage(
       bc::AbstractBandColumn,
       k::Int,
    )

Compute the range of lower inband elements in column ``k`` of
`bc.band_elements`.
"""
@propagate_inbounds @inline function lower_inband_els_range_storage(
  bc::AbstractBandColumn,
  k::Int,
)
  m = bc.m_els
  j = first_inband_el_storage(bc, k) + upper_bw(bc, :, k) + middle_bw(bc, :, k)
  (1:m) ∩ (j:(j + lower_bw(bc, :, k) - 1))
end

"""
    bc_index_stored(
      bc::AbstractBandColumn,
      j::Int,
      k::Int,
    )
Test if `bc[j,k]` is an inband element.
"""
@propagate_inbounds @inline function bc_index_stored(
  bc::AbstractBandColumn,
  j::Int,
  k::Int,
)
  j1 = j - storage_offset(bc, k)
  j1 >= first_inband_el_storage(bc, k) && j1 <= last_inband_el_storage(bc, k)
end

"""
  hull(a :: UnitRange, b :: UnitRange)

Return the convex hull of two closed intervals, with
the intervals represented by a `UnitRange`.
"""
@inline function hull(a :: UnitRange, b :: UnitRange)
  isempty(a) ? b :
  (isempty(b) ? a : UnitRange(min(a.start, b.start), max(a.stop, b.stop)))
end


##
## Index operations
##

@propagate_inbounds @inline function Base.getindex(
  bc::AbstractBandColumn{E},
  j::Int,
  k::Int,
) where {E}
  @boundscheck begin
    checkbounds(bc, j, k)
    check_bc_storage_bounds(bc, j, k)
    bc_index_stored(bc, j, k) || return zero(E)
  end
  j1 = j - storage_offset(bc,k)
  @inbounds getindex(get_band_elements(bc), j1, k)
end

@propagate_inbounds @inline function Base.setindex!(
  bc::AbstractBandColumn{E},
  x::E,
  j::Int,
  k::Int,
) where {E<:Number}
  @boundscheck begin
    checkbounds(bc, j, k)
    check_bc_storage_bounds(bc, j, k)
  end
  extend_upper_band!(bc, j, k)
  extend_lower_band!(bc, j, k)
  j1 = j - storage_offset(bc, k)
  @inbounds (get_band_elements(bc))[j1,k]=x
end

"""
    setindex_noext!(
      bc::AbstractBandColumn{E},
      x::E,
      j::Int,
      k::Int,
    ) where {E<:Number}

A version of `setindex!` that does not extend bandwidth.  This is
useful in loops where the bandwidth extension can be done outside
the loop for efficiency.
"""
@propagate_inbounds @inline function setindex_noext!(
  bc::AbstractBandColumn{E},
  x::E,
  j::Int,
  k::Int,
) where {E<:Number}
  @boundscheck begin
    checkbounds(bc, j, k)
    check_bc_storage_bounds(bc, j, k)
  end
  j1 = j - storage_offset(bc, k)
  @inbounds get_band_elements(bc)[j1,k]=x
end

"""
    trim_upper!(bc :: AbstractBandColumn, ::Colon, k::Int)
    trim_upper!(bc :: AbstractBandColumn, j::Int, ::Colon)

Remove one upper element from column k or row j.
"""
@propagate_inbounds @inline function trim_upper!(
  bc::AbstractBandColumn{E},
  ::Colon,
  k::Int,
) where {E<:Number}
  jrange = upper_inband_els_range(bc, :, k)
  j = jrange.start
  @boundscheck begin
    (m, n) = size(bc)
    !isempty(jrange) || throw(EmptyUpperRange())
    checkbounds(bc, j, k)
    k == n || first_inband_el(bc, :, k + 1) > j || throw(WellError())
  end
  rbws=get_rbws(bc)
  cbws=get_cbws(bc)
  setindex_noext!(bc,zero(E),j,k)
  rbws[j,3] -= 1
  cbws[1,k] -= 1
  nothing
end

@propagate_inbounds @inline function trim_upper!(
  bc::AbstractBandColumn{E},
  j::Int,
  ::Colon,
) where{E<:Number}
  krange = upper_inband_els_range(bc, j, :)
  k = krange.stop
  @boundscheck begin
    (m, n) = size(bc)
    !isempty(krange) || throw(EmptyUpperRange())
    checkbounds(bc, j, k)
    j == 1 || last_inband_el(bc, j - 1, :) < krange.stop || throw(WellError())
  end
  rbws=get_rbws(bc)
  cbws=get_cbws(bc)
  setindex_noext!(bc,zero(E),j,k)
  rbws[j,3] -= 1
  cbws[1,k] -= 1
  nothing
end

"""
    trim_lower!(bc :: AbstractBandColumn, ::Colon, k::Int)
    trim_lower!(bc :: AbstractBandColumn, j::Int, ::Colon)

Remove one upper element from row j or column k.
"""
@propagate_inbounds @inline function trim_lower!(
  bc::AbstractBandColumn{E},
  ::Colon,
  k::Int,
) where {E<:Number}
  jrange = lower_inband_els_range(bc, :, k)
  j = jrange.stop
  @boundscheck begin
    (m, n) = size(bc)
    !isempty(jrange) || throw(EmptyLowerRange())
    checkbounds(bc, j, k)
    k == 1 || last_inband_el(bc, :, k-1) < j || throw(WellError())
  end
  rbws=get_rbws(bc)
  cbws=get_cbws(bc)
  setindex_noext!(bc,zero(E),j,k)
  rbws[j,1] -= 1
  cbws[3,k] -= 1
  nothing
end

@propagate_inbounds @inline function trim_lower!(
  bc::AbstractBandColumn{E},
  j::Int,
  ::Colon,
) where {E}
  krange = lower_inband_els_range(bc, j, :)
  k = krange.start
  @boundscheck begin
    (m, n) = size(bc)
    !isempty(krange) || throw(EmptyUpperRange())
    checkbounds(bc, j, k)
    j == m || first_inband_el(bc, j + 1, :) > krange.start || throw(WellError())
  end
  rbws=get_rbws(bc)
  cbws=get_cbws(bc)
  setindex_noext!(bc,zero(E),j,k)
  rbws[j,1] -= 1
  cbws[3,k] -= 1
  nothing
end

@inline function Base.view(bc::AbstractBandColumn, I::Vararg{Any,N}) where {N}
  viewbc(bc, I)
end

"""
    viewbc(bc::BandColumn, i::Tuple{UnitRange{Int},UnitRange{Int}})

Return a bandcolumn submatrix with views of the relevant arrays
wrapped into a BandColumn.
"""
@inline function viewbc(bc::BandColumn, i::Tuple{UnitRange{Int},UnitRange{Int}})
  (rows, cols) = i
  j0 = rows.start
  j1 = rows.stop
  k0 = cols.start
  k1 = cols.stop

  @boundscheck begin
    checkbounds(bc, j0, k0)
    checkbounds(bc, j1, k1)
  end
  BandColumn(
    max(j1 - j0 + 1, 0),
    max(k1 - k0 + 1, 0),
    bc.m_els,
    bc.roffset + j0 - 1,
    bc.coffset + k0 - 1,
    bc.upper_bw_max,
    bc.middle_bw_max,
    bc.lower_bw_max,
    view(bc.rbws, rows, 1:4),
    view(bc.cbws, 1:4, cols),
    view(bc.band_elements, 1:4, cols),
  )
end

@inline function Base.getindex(
  bc::BandColumn,
  rows::UnitRange{Int},
  cols::UnitRange{Int},
)

  j0 = rows.start
  j1 = rows.stop
  k0 = cols.start
  k1 = cols.stop

  @boundscheck begin
    checkbounds(bc, j0, k0)
    checkbounds(bc, j1, k1)
  end
  BandColumn(
    max(j1 - j0 + 1, 0),
    max(k1 - k0 + 1, 0),
    bc.m_els,
    bc.roffset + j0 - 1,
    bc.coffset + k0 - 1,
    bc.upper_bw_max,
    bc.middle_bw_max,
    bc.lower_bw_max,
    bc.rbws[rows, :],
    bc.cbws[:, cols],
    bc.band_elements[:, cols],
  )
end

@inline Base.getindex(bc::AbstractBandColumn, ::Colon, cols::UnitRange{Int}) =
  getindex(bc, 1:get_m(bc), cols)

@inline Base.getindex(bc::AbstractBandColumn, rows::UnitRange{Int}, ::Colon) =
  getindex(bc, rows, 1:get_n(bc))

@inline Base.getindex(bc::AbstractBandColumn, ::Colon, ::Colon) = bc

@inline function viewbc(
  bc::AbstractBandColumn,
  i::Tuple{Colon,UnitRange{Int}}
)
  (_, cols) = i
  viewbc(bc, (1:get_m(bc), cols))
end

@inline function viewbc(bc::AbstractBandColumn, i::Tuple{UnitRange{Int},Colon})
  (rows, _) = i
  viewbc(bc, (rows, 1:get_n(bc)))
end

@inline function viewbc(bc::AbstractBandColumn, ::Tuple{Colon,Colon})
  viewbc(bc, (1:get_m(bc), 1:get_n(bc)))
end

function LinearAlgebra.Matrix(bc::AbstractBandColumn{E}) where {E<:Number}
  (m, n) = size(bc)
  a = zeros(E, m, n)
  for k = 1:n
    for j in inband_els_range(bc, :, k)
      a[j, k] = bc[j, k]
    end
  end
  a
end

# TODO: This should probably return CartesianIndices.
function Base.eachindex(bc::AbstractBandColumn)
  (_, n) = size(bc)
  (CartesianIndex(j, k) for k = 1:n for j ∈ inband_els_range(bc, :, k))
end

"""
    get_elements(bc::AbstractBandColumn)

Get all the stored elements of `bc` in a generator.
"""
function get_elements(bc::AbstractBandColumn)
  (_, n) = size(bc)
  (bc[j, k] for k = 1:n for j ∈ inband_els_range(bc, :, k))
end

# Copying

function Base.copy(bc::BandColumn)
  BandColumn(
    bc.m,
    bc.n,
    bc.m_els,
    bc.roffset,
    bc.coffset,
    bc.upper_bw_max,
    bc.middle_bw_max,
    bc.lower_bw_max,
    copy(bc.rbws),
    copy(bc.cbws),
    copy(bc.band_elements),
  )
end

##
## Print and show.
##

# The method show for BandColumn matrices represents elements for
# which there is no storage with `N`.  Elements that have available
# storage but are not actually stored are represented by `O`.  These
# are elements that are outside the current bandwidth but not outside
# the maximum bandwidth.
function Base.show(io::IO, bc::BandColumn)
  print(
    io,
    typeof(bc),
    "(",
    bc.m,
    ", ",
    bc.n,
    ", ",
    bc.m_els,
    ", ",
    bc.roffset,
    ", ",
    bc.coffset,
    ", ",
    bc.upper_bw_max,
    ", ",
    bc.middle_bw_max,
    ", ",
    bc.lower_bw_max,
    ", ",
    bc.rbws,
    ", ",
    bc.cbws,
    "): ",
  )
  for j ∈ 1:(bc.m)
    println()
    for k ∈ 1:(bc.n)
      if check_bc_storage_bounds(Bool, bc, j, k)
        if bc_index_stored(bc, j, k)
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
    bc.m,
    ", ",
    bc.n,
    ", ",
    bc.m_els,
    ", ",
    bc.roffset,
    ", ",
    bc.coffset,
    ", ",
    bc.upper_bw_max,
    ", ",
    bc.middle_bw_max,
    ", ",
    bc.lower_bw_max,
    ", ",
    bc.rbws,
    ", ",
    bc.cbws,
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
    wilk(bc :: BandColumn)

Generate a Wilkinson diagram for bc.

"""
@views function wilk(bc :: BandColumn)
  (m,n) = size(bc)
  a = fill('N', (m, n))
  for k ∈ 1:n
    fill!(a[storable_els_range(bc, :, k), k], 'O')
    fill!(a[upper_inband_els_range(bc, :, k), k], 'U')
    fill!(a[middle_inband_els_range(bc, :, k), k], 'X')
    fill!(a[lower_inband_els_range(bc, :, k), k], 'L')
  end
  Wilk(a)
end

end # module
