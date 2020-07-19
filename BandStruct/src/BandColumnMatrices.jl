module BandColumnMatrices

using MLStyle
using ConstructionBase
using Printf
import LinearAlgebra.Matrix

import Base:
  size,
  getindex,
  setindex!,
  showerror,
  show,
  print,
  @propagate_inbounds,
  copy,
  view,
  eachindex

export BandColumn,
  AbstractBandColumn,
  MatchError,
  first_storage_el,
  last_storage_el,
  storage_els_range,
  upper_storage_els_range,
  middle_storage_els_range,
  lower_storage_els_range,
  first_el,
  last_el,
  els_range,
  storable_els_range,
  upper_els_range,
  middle_els_range,
  lower_els_range,
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
  upper_bw,
  middle_bw,
  lower_bw,
  first_super,
  first_sub,
  get_band_element,
  set_band_element!,
  extend_lower_band!,
  extend_upper_band!,
  extend_band!,
  get_elements,
  viewbc,
  hull,
  zero_to,
  setindex_noext!,
  compute_rbws!,
  compute_rbws,
  validate_rbws,
  wilk,
  Wilk

"MatchError: An exception for pattern match errors in MLStyle."
struct MatchError <: Exception end

"""

NoStorageForIndex: An exception to indicate that there is no
storage available to represent a particular element of a BandColumn.

"""
struct NoStorageForIndex <: Exception
  arr::Any
  ix::Any
end

showerror(io::IO, e::NoStorageForIndex) = print(
  io,
  "NoStorageForIndex:  Attempt to access ",
  typeof(e.arr),
  " at index ",
  e.ix,
  " for which there is no storage.",
)

abstract type AbstractBandColumn{E,AE,AI} <: AbstractArray{E,2} end
# abstract type AbstractBandColumn{E,AE,AI} end

"""

    BandColumn

A simplified band column structure that does not include leading
blocks but does include uniform offsets that can be changed to give
different submatrices.  This can be used to represent submatrices of a
LeadingBandColumn matrix.

"""
struct BandColumn{E<:Number,AE<:AbstractArray{E,2},AI<:AbstractArray{Int,2}} <:
       AbstractBandColumn{E,AE,AI}
  m::Int # Matrix number of rows.
  n::Int # Matrix and elements number of columns.
  m_els::Int # Elements number of rows.
  roffset::Int # uniform column offset.
  coffset::Int # uniform row offset.
  upper_bw_max::Int # maximum upper bandwidth.
  middle_bw_max::Int # maximum middle bandwidth.
  lower_bw_max::Int # maximum lower bandwidth.
  rbws::AI # Row bandwidths and first subdiagonal in each row.
  cbws::AI # Column bandwidths and first superdiagonal in each column.
  band_elements::AE
end

@as_record BandColumn
@as_record UnitRange

##
## Functions that should be implemented as part of the
## AbstractBandColumn interface.
##

@inline get_m_els(bc::BandColumn) = bc.m_els

@inline get_m(bc::BandColumn) = bc.m
@inline get_n(bc::BandColumn) = bc.n

@inline get_roffset(bc::BandColumn) = bc.roffset
@inline get_coffset(bc::BandColumn) = bc.coffset

@inline get_upper_bw_max(bc::BandColumn) = bc.upper_bw_max
@inline get_rbws(bc::BandColumn) = bc.rbws
@inline get_cbws(bc::BandColumn) = bc.cbws

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

* If js::Colon and ks::Int, give the first superdiagonal in columns ks.

* If js::Int and ks::Colon, give the first superdiagonal in row js.

Superdiagonals are numbered starting from the middle of the band
structure.

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

* If js::Colon and ks::Int, give the first subdiagonal in columns ks.

* If js::Int and ks::Colon, give the first subdiagonal in row js.

Sudiagonals are numbered starting from the middle of the band
structure.

"""
@propagate_inbounds @inline first_sub(bc::AbstractBandColumn, j::Int, ::Colon) =
  get_rbws(bc)[j, 4] - get_coffset(bc)

@propagate_inbounds @inline first_sub(bc::AbstractBandColumn, ::Colon, k::Int) =
  first_super(bc, :, k) + middle_bw(bc, :, k) + 1

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
  j1 = first_el(bc, :, k) - 1
  k0 = last_el(bc, j, :) + 1
  for l = j:j1
    rbws[l, 3] = max(rbws[l, 3], k - first_super(bc, l, :) + 1)
  end
  for l = k0:k
    cbws[1, l] = max(cbws[1, l], first_super(bc, :, l) - j + 1)
  end
  nothing
end

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
  j0 = last_el(bc, :, k) + 1
  k1 = first_el(bc,j,:) - 1
  for l ∈ k:k1
    cbws[3, l] = max(cbws[3, l], j - first_sub(bc, :, l) + 1)
  end
  for l ∈ j0:j
    println(l)
    rbws[l, 1] = max(rbws[l, 1], first_sub(bc, l, :) - k + 1)
  end
  nothing
end

@propagate_inbounds @inline function extend_band!(
  bc::AbstractBandColumn,
  j::Int,
  k::Int,
)
  extend_upper_band!(bc, j, k)
  extend_lower_band!(bc, j, k)
end

@propagate_inbounds @inline function extend_band!(
  bc::AbstractBandColumn,
  jrange::UnitRange{Int},
  k::Int,
)
  if !isempty(jrange)
    extend_upper_band!(bc, jrange.start, k)
    extend_lower_band!(bc, jrange.stop, k)
  end
end

@propagate_inbounds @inline function extend_band!(
  bc::AbstractBandColumn,
  j::Int,
  krange::UnitRange{Int},
)
  if !isempty(krange)
    extend_upper_band!(bc, j, krange.start)
    extend_lower_band!(bc, j, krange.stop)
  end
end



@propagate_inbounds @inline get_band_element(bc::BandColumn, j::Int, k::Int) =
  bc.band_elements[j, k]

@propagate_inbounds @inline function set_band_element!(
  bc::BandColumn{E,AE,AI},
  x::E,
  j::Int,
  k::Int,
) where {E,AE,AI}
  bc.band_elements[j, k] = x
end

@inline size(bc::BandColumn) = (bc.m, bc.n)

##
## Generic functions defined for AbstractBandColumn
##

"""

Compute row bandwidths from column bandwidths.  Note that this
does not fill in rbws[j,4], which is the first subdiagonal
in row j.  That information should be obtained from leading block
sizes in a LeadingBandColumn matrix.

"""
function compute_rbws!(
  bc::AbstractBandColumn{E,AE,AI},
  rbws::Array{Int,2},
) where {E,AE,AI} 
  (m, n) = size(bc)
  rbws[:, 1:3] .= zero(Int)
  roffs = get_roffset(bc)
  for k ∈ n:-1:1
    jrange = intersect(lower_els_range(bc, :, k) .+ roffs, 1:m)
    rbws[jrange, 1] .+= 1
    jrange = intersect(middle_els_range(bc, :, k) .+ roffs, 1:m)
    rbws[jrange, 2] .+= 1
    jrange = intersect(upper_els_range(bc, :, k) .+ roffs, 1:m)
    rbws[jrange, 3] .+= 1
  end
end


function compute_rbws!(bc::AbstractBandColumn{E,AE,AI}) where {E,AE,AI}
  compute_rbws!(bc, bc.rbws)
end

function compute_rbws(bc::AbstractBandColumn{E,AE,AI}) where {E,AE,AI}
  (m, n) = size(bc)
  rbws1 = zeros(Int,m,3)
  compute_rbws!(bc, rbws1)
  rbws1
end


function validate_rbws(bc::AbstractBandColumn{E,AE,AI}) where {E,AE,AI}
  (m, n) = size(bc)
  rbws1 = zeros(Int,m,3)
  compute_rbws!(bc, rbws1)
  rbws1 == bc.rbws[:,1:3]
end

"""

Compute a row offset to look into storage.

"""
@propagate_inbounds @inline storage_offset(bc::AbstractBandColumn, k::Int) =
  first_super(bc, :, k) - get_upper_bw_max(bc)

@inline function check_bc_storage_bounds(
  ::Type{Bool},
  bc::AbstractBandColumn,
  j::Int,
  k::Int,
)
  j1 = j - storage_offset(bc, k)
  j1 >= 1 && j1 <= get_m_els(bc)
end

@inline check_bc_storage_bounds(bc::AbstractBandColumn, j::Int, k::Int) =
  check_bc_storage_bounds(Bool, bc, j, k) ||
  throw(NoStorageForIndex(bc, (j, k)))

@propagate_inbounds @inline first_storage_el(bc::AbstractBandColumn, k::Int) =
  get_upper_bw_max(bc) - upper_bw(bc, :, k) + 1

@propagate_inbounds @inline last_storage_el(bc::AbstractBandColumn, k::Int) =
  get_upper_bw_max(bc) + middle_bw(bc, :, k) + lower_bw(bc, :, k)

@propagate_inbounds @inline first_el(bc::AbstractBandColumn, ::Colon, k::Int) =
  first_super(bc, :, k) - upper_bw(bc, :, k) + 1

@propagate_inbounds @inline first_el(bc::AbstractBandColumn, j::Int, ::Colon) =
  first_sub(bc, j, :) - lower_bw(bc, j, :) + 1

@propagate_inbounds @inline last_el(bc::AbstractBandColumn, ::Colon, k::Int) =
  first_super(bc, :, k) + middle_bw(bc, :, k) + lower_bw(bc, :, k)

@propagate_inbounds @inline last_el(bc::AbstractBandColumn, j::Int, ::Colon) =
  first_sub(bc, j, :) + middle_bw(bc, j, :) + upper_bw(bc, j, :)

@propagate_inbounds @inline function els_range(
  bc::AbstractBandColumn,
  ::Colon,
  k::Int,
)
  (m, _) = size(bc)
  intersect(1:m, first_el(bc, :, k):last_el(bc, :, k))
end

@propagate_inbounds @inline function els_range(
  bc::AbstractBandColumn,
  j::Int,
  ::Colon,
)
  (_, n) = size(bc)
  intersect(1:n, first_el(bc, j, :):last_el(bc, j, :))
end

@propagate_inbounds @inline function storable_els_range(
  bc::AbstractBandColumn,
  ::Colon,
  k::Int,
)
  (m, _) = size(bc)
  d = first_super(bc,:,k) - get_upper_bw_max(bc)
  intersect(1:m, (d + 1):(d + get_m_els(bc)))
end

@propagate_inbounds @inline function upper_els_range(
  bc::AbstractBandColumn,
  ::Colon,
  k::Int,
)
  (m, _) = size(bc)
  intersect(1:m, first_el(bc, :, k):first_super(bc, :, k))
end

@propagate_inbounds @inline function upper_els_range(
  bc::AbstractBandColumn,
  j::Int,
  ::Colon,
)
  (_, n) = size(bc)
  intersect(1:n, first_super(bc, j, :):last_el(bc, j, :))
end

@propagate_inbounds @inline function middle_els_range(
  bc::AbstractBandColumn,
  ::Colon,
  k::Int,
)
  (m, _) = size(bc)
  j = first_super(bc, :, k)
  intersect(1:m, (j + 1):(j + middle_bw(bc, :, k)))
end

@propagate_inbounds @inline function middle_els_range(
  bc::AbstractBandColumn,
  j::Int,
  ::Colon,
)
  (_, n) = size(bc)
  k = first_sub(bc, j, :)
  intersect(1:n, (k + 1):(k + middle_bw(bc, j, :)))
end

@propagate_inbounds @inline function lower_els_range(
  bc::AbstractBandColumn,
  ::Colon,
  k::Int,
)
  (m, _) = size(bc)
  j = first_super(bc, :, k) + middle_bw(bc, :, k)
  intersect(1:m, (j + 1):(j + lower_bw(bc, :, k)))
end

@propagate_inbounds @inline function lower_els_range(
  bc::AbstractBandColumn,
  j::Int,
  ::Colon,
)
  (_, n) = size(bc)
  intersect(1:n, first_el(bc, j, :):first_sub(bc,j,:))
end

@propagate_inbounds @inline function storage_els_range(
  bc::AbstractBandColumn,
  k::Int,
)
  intersect(1:get_m_els(bc), first_storage_el(bc, k):last_storage_el(bc, k))
end

@propagate_inbounds @inline function upper_storage_els_range(
  bc::AbstractBandColumn,
  k::Int,
)
  m = bc.m_els
  j = first_storage_el(bc, k)
  intersect(1:m, j:(j + upper_bw(bc, :, k) - 1))
end

@propagate_inbounds @inline function middle_storage_els_range(
  bc::AbstractBandColumn,
  k::Int,
)
  m = bc.m_els
  j = first_storage_el(bc, k) + upper_bw(bc, :, k)
  intersect(1:m, j:(j + middle_bw(bc, :, k) - 1))
end

@propagate_inbounds @inline function lower_storage_els_range(
  bc::AbstractBandColumn,
  k::Int,
)
  m = bc.m_els
  j = first_storage_el(bc, k) + upper_bw(bc, :, k) + middle_bw(bc, :, k)
  intersect(1:m, j:(j + lower_bw(bc, :, k) - 1))
end

@propagate_inbounds @inline function bc_index_stored(
  bc::AbstractBandColumn,
  j::Int,
  k::Int,
)
  j1 = j - storage_offset(bc, k)
  j1 >= first_storage_el(bc, k) && j1 <= last_storage_el(bc, k)
end

" Convex hull of two sets."
@inline function hull(a :: UnitRange, b :: UnitRange)
  isempty(a) ? b :
  (isempty(b) ? a : UnitRange(min(a.start, b.start), max(a.stop, b.stop)))
end


##
## Index operations
##

# (BandColumn{Float64, Array{Float64,2}, Array{Int}}, Int, Int))
@propagate_inbounds @inline function getindex(
  bc::AbstractBandColumn{E,AE,AI},
  j::Int,
  k::Int,
) where {E,AE,AI}
  @boundscheck begin
    checkbounds(bc, j, k)
    check_bc_storage_bounds(bc, j, k)
    bc_index_stored(bc, j, k) || return zero(E)
  end
  j1 = j - storage_offset(bc,k)
  @inbounds get_band_element(bc, j1, k)
end

@propagate_inbounds @inline function setindex!(
  bc::AbstractBandColumn{E,AE,AI},
  x::E,
  j::Int,
  k::Int,
) where {E,AE,AI}
  @boundscheck begin
    checkbounds(bc, j, k)
    check_bc_storage_bounds(bc, j, k)
  end
  extend_upper_band!(bc, j, k)
  extend_lower_band!(bc, j, k)
  j1 = j - storage_offset(bc, k)
  @inbounds set_band_element!(bc, x, j1, k)
end

"""

Setindex without extending bandwidth.  This is useful in loops
where the bandwidth can be extended outside the loop.

"""
@propagate_inbounds @inline function setindex_noext!(
  bc::AbstractBandColumn{E,AE,AI},
  x::E,
  j::Int,
  k::Int,
) where {E,AE,AI}
  @boundscheck begin
    checkbounds(bc, j, k)
    check_bc_storage_bounds(bc, j, k)
  end
  j1 = j - storage_offset(bc, k)
  @inbounds set_band_element!(bc, x, j1, k)
end

"Put in a hard zero and adjust bandwidths as appropriate."
@propagate_inbounds @inline function zero_to(
  bc::AbstractBandColumn{E,AE,AI},
  j::Int,
  k::Int,
) where {E,AE,AI}
  @boundscheck begin
    checkbounds(bc, j, k)
    check_bc_storage_bounds(bc, j, k)
  end
  j1 = j - storage_offset(bc, k)
  @inbounds set_band_element!(bc, zero(E), j1, k)
  rangek = els_range(bc, :, k)
  if bc.cbws[1,k]>0 && j == rangek.start
    bc.cbws[1,k] -= bc.cbws[1,k]
  end
  if bc.cbws[3,k]>0 && j == rangek.stop
    bc.cbws[3,k] -= bc.cbws[3,k]
  end
end

@inline function view(bc::AbstractBandColumn, I::Vararg{Any,N}) where {N}
  viewbc(bc, I)
end

@inline function viewbc(
  bc::BandColumn,
  i::Tuple{UnitRange{Int},UnitRange{Int}},
)
  (rows, cols) = i
  @when let BandColumn(;
                       m,
                       n,
                       m_els,
                       roffset = roffs,
                       coffset = coffs,
                       upper_bw_max = ubw_max,
                       middle_bw_max = mbw_max,
                       lower_bw_max = lbw_max,
                       rbws,
                       cbws,
                       band_elements = els,
                       ) = bc,
    UnitRange(j0, j1) = rows,
    UnitRange(k0, k1) = cols

    @boundscheck begin
      checkbounds(bc, j0, k0)
      checkbounds(bc, j1, k1)
    end
    BandColumn(
      max(j1 - j0 + 1, 0),
      max(k1 - k0 + 1, 0),
      m_els,
      roffs + j0 - 1,
      coffs + k0 - 1,
      ubw_max,
      mbw_max,
      lbw_max,
      view(rbws, rows, 1:4),
      view(cbws, 1:4, cols),
      view(els, 1:4, cols),
    )
    @otherwise
    throw(MatchError)
  end
end

@inline function getindex(
  bc::BandColumn,
  rows::UnitRange{Int},
  cols::UnitRange{Int},
)

  @when let BandColumn(;
                       m,
                       n,
                       m_els,
                       roffset = roffs,
                       coffset = coffs,
                       upper_bw_max = ubw_max,
                       middle_bw_max = mbw_max,
                       lower_bw_max = lbw_max,
                       rbws,
                       cbws,
                       band_elements = els,
                       ) = bc,
    UnitRange(j0, j1) = rows,
    UnitRange(k0, k1) = cols

    @boundscheck begin
      checkbounds(bc, j0, k0)
      checkbounds(bc, j1, k1)
    end
    BandColumn(
      max(j1 - j0 + 1, 0),
      max(k1 - k0 + 1, 0),
      m_els,
      roffs + j0 - 1,
      coffs + k0 - 1,
      ubw_max,
      mbw_max,
      lbw_max,
      rbws[rows, :],
      cbws[:, cols],
      els[:, cols],
    )
    @otherwise
    throw(MatchError)
  end
end

@inline getindex(bc::AbstractBandColumn, ::Colon, cols::UnitRange{Int}) =
  getindex(bc, 1:get_m(bc), cols)

@inline getindex(bc::AbstractBandColumn, rows::UnitRange{Int}, ::Colon) =
  getindex(bc, rows, 1:get_n(bc))

@inline getindex(bc::AbstractBandColumn, ::Colon, ::Colon) = bc

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

function Matrix(bc::AbstractBandColumn{E,AE,AI}) where {E,AE,AI}
  (m, n) = size(bc)
  a = zeros(E, m, n)
  for k = 1:n
    for j in els_range(bc, :, k)
      a[j, k] = bc[j, k]
    end
  end
  a
end

# TODO: This should probably return CartesianIndices.
function eachindex(bc::AbstractBandColumn)
  (_, n) = size(bc)
  (CartesianIndex(j, k) for k = 1:n for j ∈ els_range(bc, :, k))
end

function get_elements(bc::AbstractBandColumn)
  (_, n) = size(bc)
  (bc[j, k] for k = 1:n for j ∈ els_range(bc, :, k))
end

# Copying

function copy(bc::BandColumn)
  @when let BandColumn(
    m,
    n,
    m_els,
    roffs,
    coffs,
    ubw_max,
    mbw_max,
    lbw_max,
    rbws,
    cbws,
    bels,
  ) = bc
    BandColumn(
      m,
      n,
      m_els,
      roffs,
      coffs,
      ubw_max,
      mbw_max,
      lbw_max,
      copy(rbws),
      copy(cbws),
      copy(bels),
    )
    @otherwise
    throw(MatchError)
  end
end

##
## Print and show.
##

"""

The method show for BandColumn matrices represents elements for which
there is no storage with `N`.  Elements that have available storage
but are not actually stored are represented by `O`.  These are
elements that are outside the current bandwidth but not outside the
maximum bandwidth.

"""
function show(io::IO, bc::BandColumn{E,AE,AI}) where {E,AE,AI}
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

show(io::IO, ::MIME"text/plain", bc::BandColumn) = show(io, bc)

print(io::IO, bc::BandColumn) = print(
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

print(io::IO, ::MIME"text/plain", bc::BandColumn) = print(io, bc)

struct Wilk
  arr :: Array{Char,2}
end

function show(io::IO, w::Wilk)
  (m, n) = size(w.arr)
  for j = 1:m
    println(io)
    for k = 1:n
      print(io,w.arr[j, k], " ")
    end
  end
end

function show(w::Wilk)
  (m, n) = size(w.arr)
  for j = 1:m
    println()
    for k = 1:n
      print(w.arr[j, k], " ")
    end
  end
end

@views function wilk(bc :: BandColumn)
  (m,n) = size(bc)
  a = fill('N', (m, n))
  for k ∈ 1:n
    for j ∈ storable_els_range(bc, :, k)
      a[j,k] = 'O'
    end
    for j ∈ upper_els_range(bc, :, k)
      a[j,k] = 'U'
    end
    for j ∈ middle_els_range(bc, :, k)
      a[j,k] = 'M'
    end
    for j ∈ lower_els_range(bc, :, k)
      a[j,k] = 'L'
    end
  end
  Wilk(a)
end

end # module
