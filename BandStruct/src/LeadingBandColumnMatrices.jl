module LeadingBandColumnMatrices
using Printf
using Random
using Base: @propagate_inbounds

using BandStruct.BandColumnMatrices

export LeadingBandColumn,
  size,
  getindex,
  setindex!,
  leading_lower_ranks_to_cbws!,
  leading_upper_ranks_to_cbws!,
  leading_constrain_lower_ranks,
  leading_constrain_upper_ranks,
  get_lower_block

"""
A banded matrix with structure defined by leading blocks and
stored in a compressed column-wise format.

Given

    A = X | U | O   O   O | N | N
        _ ⌋   |           |   |  
        L   X | O   O   O | N | N
              |           |   |  
        L   X | U   U   U | O | O
        _ _ _ ⌋           |   |  
        O   L   X   X   X | O | O
        _ _ _ _ _ _ _ _ _ |   |  
        N   O   O   O   L | O | O
                          |   |  
        N   N   O   O   L | U | U
        _ _ _ _ _ _ _ _ _ ⌋ _ |  
        N   N   O   O   O   L   X

        N   N   N   N   N   L   X

where U, X, and L denote upper, middle, and lower band elements.  O
indicates an open location with storage available for expandning
either the lower or upper bandwidth.  N represents a location for
which there is no storage.

The matrix is stored as

    elements = 

    Num rows|Elements
    ----------------------
    ubwmax  |O O O O O O O
            |O O O O O O O
            |O O O O O O O
            |O U U U U U U
    ----------------------
    mbwmax+ |X X X X X L X
    lbwmax  |L X O O L L X
            |L L O O L O O
            |O O O O O O O

where

    m =                  8
    n =                  7
    m_els =              8
    num_blocks =         6
    upper_bw_max =       4
    middle_bw_max =      2
    lower_bw_max =       2
    cbws =               [ 1, 1, 1, 1, 1, 1, 1;  # upper
                           1, 2, 1, 1, 1, 0, 2;  # middle
                           2, 1, 0, 0, 2, 1, 0;  # lower
                           0, 1, 3, 3, 3, 6, 6 ] # first superdiagonal.
    leading_blocks =     [ 1, 3, 4, 6, 6, 8;     # rows
                           1, 2, 5, 5, 6, 7 ]    # columns
"""
struct LeadingBandColumn{
  E<:Number,
  AE<:AbstractArray{E,2},
  AI<:AbstractArray{Int,2},
} <: AbstractBandColumn{E,AE,AI}
  m::Int             # Matrix number of rows.
  n::Int             # Matrix number of columns.
  m_els::Int         # number of elements rows.
  num_blocks::Int    # Number of leading blocks.
  upper_bw_max::Int  # maximum upper bandwidth.
  middle_bw_max::Int # maximum middle bandwidth.
  lower_bw_max::Int  # maximum lower bandwidth.
  rbws::AI           # mx4 matrix: row-wise lower, middle, and upper bw +
                     # first subdiagonal postion in A.
  cbws::AI           # 4xn matrix: column-wise upper, middle, and lower bw +
                     # first superdiagonal postion in A.
  leading_blocks::AI # 2xn matrix, leading block row and column counts.
  band_elements::AE
end

# Construct an empty (all zero) structure from the matrix size, bounds
# on the upper and lower bandwidth, and blocksizes.
function LeadingBandColumn(
  ::Type{E},
  m::Int,
  n::Int,
  upper_bw_max::Int,
  lower_bw_max::Int,
  leading_blocks::Array{Int,2},
) where {E<:Number}
  num_blocks = size(leading_blocks,2)
  cbws = zeros(Int, 4, n)
  cbws[2, 1] = leading_blocks[1, 1]
  cbws[4, 1] = 0
  block = 1
  for k = 2:n
    cols_in_block = leading_blocks[2,block]
    k <= cols_in_block || while (leading_blocks[2,block] == cols_in_block)
      block += 1
    end
    cbws[2, k] = leading_blocks[1,block] - leading_blocks[1,block-1]
    cbws[4, k] = leading_blocks[1,block-1]
  end
  middle_bw_max = maximum(cbws[2, :])
  # rows
  rbws = zeros(Int, m, 4)
  rbws[1, 2] = leading_blocks[2, 1]
  rbws[1, 4] = 0
  block = 1
  for j = 2:m
    rows_in_block = leading_blocks[1, block]
    j <= rows_in_block || while (leading_blocks[1, block] == rows_in_block)
      block += 1
    end
    rbws[j, 2] = leading_blocks[2, block] - leading_blocks[2, block - 1]
    rbws[j, 4] = leading_blocks[2, block - 1]
  end
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
    leading_blocks,
    band_elements,
  )
end

function rand!(
  rng::AbstractRNG,
  lbc::LeadingBandColumn{E},
) where {E}

  for k = 1:(lbc.n)
    for j = first_storage_el(lbc, k):last_storage_el(lbc, k)
      lbc.band_elements[j, k] = rand(rng,E)
    end
  end

end

function LeadingBandColumn(
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
  lbc = LeadingBandColumn(
    T,
    m,
    n,
    upper_bw_max,
    lower_bw_max,
    leading_blocks,
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


@propagate_inbounds @inline BandColumnMatrices.get_band_element(
  lbc::LeadingBandColumn,
  j::Int,
  k::Int,
) = lbc.band_elements[j, k]

@propagate_inbounds @inline function BandColumnMatrices.set_band_element!(
  lbc::LeadingBandColumn{E},
  x::E,
  j::Int,
  k::Int,
) where {E<:Number}
  lbc.band_elements[j, k] = x
end

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
    lbc.leading_blocks,
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

# Find bounds for lower left block l in A.
@inline function get_lower_block(
  lbc::LeadingBandColumn,
  l::Integer,
)
  (m, _) = size(lbc)
  j1 = lbc.leading_blocks[1, l] + 1
  k2 = lbc.leading_blocks[2, l]
  ((j1, m), (1, k2))
end

# Take lower and upper rank sequences and constrain them to be
# consistent with the size of the blocks (and preceding ranks).
function leading_constrain_lower_ranks(
  blocks::AbstractArray{Int,2},
  lower_ranks::AbstractArray{Int,1},
)
  lr = similar(lower_ranks)
  lr .= 0
  num_blocks = size(blocks, 2)
  m = blocks[1,num_blocks]

  lr[1] = min(m - blocks[1, 1], blocks[2, 1], lower_ranks[1])

  for l = 2:(num_blocks - 1)
    j0 = blocks[1, l - 1] + 1
    k0 = blocks[2, l - 1]
    j1 = blocks[1, l] + 1
    k1 = blocks[2, l]
    m1 = m - j1 + 1
    n1 = k1 - k0 + lr[l - 1]
    lr[l] = min(m1, n1, lower_ranks[l])
  end
  lr
end

function leading_constrain_upper_ranks(
  blocks::AbstractArray{Int,2},
  upper_ranks::AbstractArray{Int,1},
)

  ur = similar(upper_ranks)
  ur .= 0
  num_blocks = size(blocks, 2)
  n = blocks[2,num_blocks]

  ur[1] = min(blocks[1, 1], n - blocks[2, 1], upper_ranks[1])

  for l = 2:(num_blocks - 1)
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

# Find bounds for upper right block l in A.
@inline function get_upper_block(lbc::LeadingBandColumn, l::Integer)
  (_, n) = size(lbc)
  j2 = lbc.leading_blocks[1, l]
  k1 = lbc.leading_blocks[2, l] + 1
  ((1, j2), (k1, n))
end

# Set lower bandwidth appropriate for a given lower rank sequence.
function leading_lower_ranks_to_cbws!(
  lbc::LeadingBandColumn,
  rs::AbstractArray{Int},
)

  (m, n) = size(lbc)
  lbc.cbws[3, :] .= 0
  rs1 = leading_constrain_lower_ranks(lbc.leading_blocks, rs)

  for l = 1:(lbc.num_blocks - 1)
    ((j0, _), (_, k0)) = get_lower_block(lbc, l)
    ((j1, _), (_, k1)) = get_lower_block(lbc, l + 1)
    d = j1 - j0
    lbc.cbws[3, (k0 - rs1[l] + 1):k0] .+= d
  end
end

# Set upper bandwidth appropriate for a given lower rank sequence.
function leading_upper_ranks_to_cbws!(
  lbc::LeadingBandColumn,
  rs::AbstractArray{Int},
)

  (m, n) = size(lbc)
  lbc.cbws[1, :] .= 0
  rs1 = leading_constrain_upper_ranks(lbc.leading_blocks, rs)

  for l = 1:(lbc.num_blocks - 1)
    (_, (k0, _)) = get_upper_block(lbc, l)
    (_, (k1, _)) = get_upper_block(lbc, l + 1)
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
  for j = 1:num_blocks
    row = lbc.leading_blocks[1, j]
    col = lbc.leading_blocks[2, j]
    fill!(a[1:(2 * row - 1), 2 * col], '|')
    a[2 * row, 2 * col] = '⌋'
    fill!(a[2 * row, 1:(2 * col - 1)], '_')
  end
  for k = 1:n
    kk = 2 * k - 1
    fill!(a[2 .* storable_els_range(lbc, :, k) .- 1, kk], 'O')
    fill!(a[2 .* upper_els_range(lbc, :, k) .- 1, kk], 'U')
    fill!(a[2 .* middle_els_range(lbc, :, k) .- 1, kk], 'X')
    fill!(a[2 .* lower_els_range(lbc, :, k) .- 1, kk], 'L')
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
    lbc.leading_blocks,
    copy(lbc.band_elements),
  )
end


end
