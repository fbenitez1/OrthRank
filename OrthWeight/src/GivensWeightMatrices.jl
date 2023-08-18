module GivensWeightMatrices

using BandStruct
using BandStruct.BlockedBandColumnMatrices
using BandStruct.BandwidthInit
using OrthWeight.BasicTypes
using InPlace: apply!, apply_inv!
using Rotations
using LinearAlgebra
using Random

export AbstractGivensWeight,
  GivensWeight,
  GivensBlockData,
  givens_block_sizes,
  to_givens_block_data_index_list,
  get_max_Δn,
  get_max_Δm,
  max_num_rots,
  set_givens_weight_transform_params!,
  get_givens_weight_transform_params,
  get_givens_weight_max_transform_params,
  random_cs

struct UncompressedLowerBlock <: Exception
  block::Int
end

struct UncompressedUpperBlock <: Exception
  block::Int
end

"""
# `GivensBlockData`

    mutable struct GivensBlockData <: AbstractCompressibleData
      mb::Int
      nb::Int
      block_rank::Int
      compressed::Bool
      givens_index::Int
      num_rots::Int
      size::Int
    end

Data associated with matrix off-diagonal blocks and corresponding
rotations stored in a separate matrix.  This describes a set of Givens
rotations that compress a block. 

# Fields

- `mb::Int`: Number of rows in the block.

- `nb::Int`: Number of columns in the block.

- `block_rank::Int`: The rank of the block.

- `compressed::Bool`: Whether the block is compressed or not.

- `givens_index::Int`: The column index for the block in a separate
  matrix of rotations in which each column corresponds to the
  rotations for a single block.

- `num_rots::Int`: The number of rotations associated with the block.

- `size::Int`: The size of the combined transformation, that is the number of
  rows or columns it acts on.

The actual storage for the rotations is elsewhere.

"""
mutable struct GivensBlockData <: AbstractCompressibleData
  mb::Int
  nb::Int
  block_rank::Int
  compressed::Bool
  givens_index::Int
  num_rots::Int
  tsize::Int
end

function Base.show(io::IO, ::MIME"text/plain", gb::GivensBlockData)
    print(
      io,
      "GivensBlockData(mb = $(gb.mb), nb = $(gb.nb), block_rank = $(gb.block_rank), " *
      "compressed = $(gb.compressed), givens_index = $(gb.givens_index), " *
      "num_rots = $(gb.num_rots), tsize = $(gb.tsize))",
    )
end


GivensBlockData(
  mb,
  nb;
  block_rank = 0,
  compressed = true,
  givens_index = 0,
  num_rots = 0,
  tsize = 0,
) =
  GivensBlockData(mb, nb, block_rank, compressed, givens_index, num_rots, tsize)

function to_givens_block_data_index_list(
  blocks::Union{
    AbstractVector{<:AbstractBlockData},
    IndexList{<:AbstractBlockData},
  };
  max_length::Union{Int,Nothing} = nothing,
)
  if blocks isa IndexList
    blocks_B = IndexList([
      let (; mb, nb) = blocks[li]
        GivensBlockData(mb, nb)
      end for li in blocks
        ], max_length = something(max_length, blocks.max_length))
  else
    blocks_B = IndexList([
      let (; mb, nb) = bd
        GivensBlockData(mb, nb)
      end for bd in blocks
        ], max_length = something(max_length, length(blocks)))
  end
  return blocks_B
end

function givens_block_sizes(a::AbstractMatrix{Int})
  v = GivensBlockData[]
  for j ∈ axes(a,2)
    push!(v, GivensBlockData(a[1, j], a[2, j]))
  end
  return v
end

# Givens-weight represenation.

abstract type AbstractGivensWeight <: OrthWeightDecomp end

"""
    GivensWeight{LR,B,UR}

A GivensWeight matrix with weight matrix of type `B` and lower and upper
rotations of type `LR` and `UR` respectively.

# Fields

  - `lower_decomp::Base.RefValue{Union{Nothing,Decomp}}`:
    Type of decomposition in the lower part.

  - `upper_decomp::Base.RefValue{Union{Nothing,Decomp}}`:
    Type of decomposition in the upper part.


  - `lowerRot::LR`: Lower rotations for a banded Givens weight decomposition.

  - `b::B`: Weight matrix.

  - `upperRot::UR`: Upper rotations for a banded Givens weight decomposition.

  - `upper_ranks::Vector{Int}`: Upper ranks

  - `lower_ranks::Vector{Int}`: Lower ranks

Note that block data is stored in the band matrix.
"""
struct GivensWeight{B,E,R} <: AbstractGivensWeight
  lower_decomp::Base.RefValue{Union{Nothing,LeadingDecomp,TrailingDecomp}}
  upper_decomp::Base.RefValue{Union{Nothing,LeadingDecomp,TrailingDecomp}}
  lowerRots::Matrix{Rot{R, E, Int}}
  b::B
  upperRots::Matrix{Rot{R, E, Int}}
  lower_rank_max::Int
  upper_rank_max::Int
end

"""
    GivensWeight(
      ::Type{E},
      m::Int,
      n::Int;
      upper_rank_max::Int,
      lower_rank_max::Int,
      upper_blocks::Array{Int,2},
      lower_blocks::Array{Int,2},
    ) where {E<:Number}

Generic empty `GivensWeight` with zero ranks but room for either a
leading or trailing decomposition with ranks up to `upper_rank_max`
and `lower_rank_max`.

"""
function GivensWeight(
  ::Type{E},
  m::Int,
  n::Int;
  upper_rank_max::Int,
  lower_rank_max::Int,
  upper_blocks::Union{
    AbstractVector{<:AbstractBlockData},
    IndexList{<:AbstractBlockData},
  },
  max_num_upper_blocks = length(upper_blocks),
  lower_blocks::Union{
    AbstractVector{<:AbstractBlockData},
    IndexList{<:AbstractBlockData},
  },
  max_num_lower_blocks = length(lower_blocks),
) where {R<:Real, E <: Union{R, Complex{R}}}

  num_upper_blocks = length(upper_blocks)
  num_lower_blocks = length(lower_blocks)
  max_num_blocks = max(num_upper_blocks, num_lower_blocks)

  upper_blocks_givens = to_givens_block_data_index_list(
    upper_blocks,
    max_length = max_num_upper_blocks,
  )

  lower_blocks_givens = to_givens_block_data_index_list(
    lower_blocks,
    max_length = max_num_lower_blocks,
  )

  bbc = BlockedBandColumn(
    E,
    m,
    n;
    lower_blocks = lower_blocks_givens,
    lower_rank_max = lower_rank_max,
    upper_blocks = upper_blocks_givens,
    upper_rank_max = upper_rank_max,
  )

  (lower_max_num_rots,) = get_givens_weight_max_transform_params(
    Lower(),
    (LeadingDecomp(), TrailingDecomp()),
    m,
    n,
    NumRots();
    upper_blocks = upper_blocks,
    upper_ranks = Consts(num_upper_blocks, upper_rank_max),
    lower_blocks = lower_blocks,
    lower_ranks = Consts(num_lower_blocks, lower_rank_max),
  )

  lowerRots = Matrix{Rot{R,E,Int}}(undef, lower_max_num_rots, max_num_blocks)
  lowerRots .= Rot(zero(R), zero(E), 0)

  j=1
  for lb ∈ lower_blocks_givens
    lower_blocks_givens[lb].givens_index = j
    lower_blocks_givens[lb].block_rank = 0
    lower_blocks_givens[lb].compressed = false
    j += 1
  end

  (upper_max_num_rots,) = get_givens_weight_max_transform_params(
    Upper(),
    (LeadingDecomp(), TrailingDecomp()),
    m,
    n,
    # TransformSizes(),
    NumRots();
    upper_blocks = upper_blocks,
    upper_ranks = Consts(num_upper_blocks, upper_rank_max),
    lower_blocks = lower_blocks,
    lower_ranks = Consts(num_lower_blocks, lower_rank_max),
  )
  
  upperRots = Matrix{Rot{R,E,Int}}(undef, upper_max_num_rots, max_num_blocks)
  upperRots .= Rot(zero(R), zero(E), 0)

  j = 1
  for lb ∈ upper_blocks_givens
    upper_blocks_givens[lb].givens_index = j
    upper_blocks_givens[lb].block_rank = 0
    upper_blocks_givens[lb].compressed = false
    j += 1
  end

  lower_decomp_ref =
    Base.RefValue{Union{Nothing,LeadingDecomp,TrailingDecomp}}(nothing)

  upper_decomp_ref =
    Base.RefValue{Union{Nothing,LeadingDecomp,TrailingDecomp}}(nothing)

  return GivensWeight(
    lower_decomp_ref,
    upper_decomp_ref,
    lowerRots,
    bbc,
    upperRots,
    lower_rank_max,
    upper_rank_max,
  )
end
"""
    function GivensWeight(
      ::Type{E},
      decomp::Union{LeadingDecomp,TrailingDecomp},
      rng::AbstractRNG,
      m::Int,
      n::Int;
      upper_rank_max::Int,
      lower_rank_max::Int,
      upper_ranks::Union{Vector{Int},Nothing} = nothing,
      lower_ranks::Union{Vector{Int},Nothing} = nothing,
      upper_blocks::Union{
        AbstractVector{<:AbstractBlockData},
        IndexList{<:AbstractBlockData},
      },
      max_num_upper_blocks = length(upper_blocks),
      lower_blocks::Union{
        AbstractVector{<:AbstractBlockData},
        IndexList{<:AbstractBlockData},
      },
      max_num_lower_blocks = length(lower_blocks)
    )

Form a random Givens-weight matrix in the form of either a leading or
trailing decomposition.  Space is allocated to hold either one and
convert between them.
"""
function GivensWeight(
  ::Type{E},
  decomp::Union{LeadingDecomp,TrailingDecomp},
  rng::AbstractRNG,
  m::Int,
  n::Int;
  upper_rank_max::Int,
  lower_rank_max::Int,
  upper_ranks::Union{AbstractVector{Int},Nothing} = nothing,
  lower_ranks::Union{AbstractVector{Int},Nothing} = nothing,
  upper_blocks::Union{
    AbstractVector{<:AbstractBlockData},
    IndexList{<:AbstractBlockData},
  },
  max_num_upper_blocks = length(upper_blocks),
  lower_blocks::Union{
    AbstractVector{<:AbstractBlockData},
    IndexList{<:AbstractBlockData},
  },
  max_num_lower_blocks = length(lower_blocks),
) where {R<:Real, E <: Union{R, Complex{R}}}

  num_upper_blocks = length(upper_blocks)
  num_lower_blocks = length(lower_blocks)

  # Assume constant maximum upper ranks if ranks not provided.


  upper_ranks =
    isnothing(upper_ranks) ? Consts(num_upper_blocks, upper_rank_max) :
    upper_ranks

  lower_ranks =
    isnothing(lower_ranks) ? Consts(num_lower_blocks, lower_rank_max) :
    lower_ranks

  # Constrain to be consistent with maximum allowed for block sizes:
  #
  # r_{k-1} -Δn_k <= r_k <= r_{k-1} + Δm_k
  upper_ranks =
    constrain_upper_ranks(m, n, blocks = upper_blocks, ranks = upper_ranks)

  lower_ranks =
    constrain_lower_ranks(m, n, blocks = lower_blocks, ranks = lower_ranks)

  max_num_blocks = max(num_upper_blocks, num_lower_blocks)

  upper_blocks_givens = to_givens_block_data_index_list(
    upper_blocks,
    max_length = max_num_upper_blocks,
  )

  lower_blocks_givens = to_givens_block_data_index_list(
    lower_blocks,
    max_length = max_num_lower_blocks,
  )

  bbc = BlockedBandColumn(
    E,
    decomp,
    rng,
    m,
    n;
    lower_blocks = lower_blocks_givens,
    lower_rank_max = lower_rank_max,
    lower_ranks = lower_ranks,
    upper_blocks = upper_blocks_givens,
    upper_rank_max = upper_rank_max,
    upper_ranks = upper_ranks,
  )


  # Get an overall maximum number of rotations for lower blocks for
  # either leading or trailing decompositions.
  (lower_max_num_rots,) = get_givens_weight_max_transform_params(
    Lower(),
    (LeadingDecomp(), TrailingDecomp()),
    m,
    n,
    NumRots();
    lower_blocks = lower_blocks,
    lower_ranks = Consts(num_lower_blocks, lower_rank_max),
  )

  lowerRots = Matrix{Rot{R,E,Int}}(undef, lower_max_num_rots, max_num_blocks)
  lowerRots .= Rot(zero(R), zero(E), 0)
  
  # Storage for transform sizes and num_rots.
  tsizes = Vector{Int}(undef, max_num_blocks)
  num_rots = Vector{Int}(undef, max_num_blocks)

  # Fill in the actual transform sizes and num_rots for a decmposition of
  # the requested type.

  set_givens_weight_transform_params!(
    Lower(),
    decomp,
    m,
    n;
    lower_blocks = lower_blocks,
    lower_ranks = lower_ranks,
    tsizes = tsizes,
    num_rots = num_rots,
  )

  j = 1
  for lb ∈ lower_blocks_givens
    bbc.lower_blocks[lb].givens_index = j
    bbc.lower_blocks[lb].block_rank = lower_ranks[j]
    bbc.lower_blocks[lb].compressed = true
    bbc.lower_blocks[lb].tsize = tsizes[j]
    bbc.lower_blocks[lb].num_rots = num_rots[j]
    j += 1
  end

  # Get an overall maximum number of rotations for upper blocks for
  # either leading or trailing decompositions.
  (upper_max_num_rots,) = get_givens_weight_max_transform_params(
    Upper(),
    (LeadingDecomp(), TrailingDecomp()),
    m,
    n,
    NumRots();
    upper_blocks = upper_blocks,
    upper_ranks = Consts(num_upper_blocks, upper_rank_max),
  )

  upperRots = Matrix{Rot{R,E,Int}}(undef, upper_max_num_rots, max_num_blocks)
  upperRots .= Rot(zero(R), zero(E), 0)
  
  # Fill in the actual transform sizes and num_rots for a decmposition of
  # the requested type.
  set_givens_weight_transform_params!(
    Upper(),
    decomp,
    m,
    n;
    upper_blocks = upper_blocks,
    upper_ranks = upper_ranks,
    tsizes = tsizes,
    num_rots = num_rots,
  )

  j = 1
  for ub ∈ upper_blocks_givens
    bbc.upper_blocks[ub].givens_index = j
    bbc.upper_blocks[ub].block_rank = upper_ranks[j]
    bbc.upper_blocks[ub].compressed = true
    bbc.upper_blocks[ub].tsize = tsizes[j]
    bbc.upper_blocks[ub].num_rots = num_rots[j]
    j += 1
  end

  lower_decomp =
    Base.RefValue{Union{LeadingDecomp,TrailingDecomp,Nothing}}(decomp)

  upper_decomp =
    Base.RefValue{Union{LeadingDecomp,TrailingDecomp,Nothing}}(decomp)

  gw = GivensWeight(
    lower_decomp,
    upper_decomp,
    lowerRots,
    bbc,
    upperRots,
    lower_rank_max,
    upper_rank_max,
  )

  insert_random_rotations!(rng, gw)

  return gw

end

function random_cs(
  rng::AbstractRNG,
  ::Type{E},
  ::Type{R},
) where {R<:Real,E<:Union{R,Complex{R}}}
  x = zero(R)
  y = zero(E)
  while (x == zero(R) && y == zero(E))
    x = randn(rng, R)
    y = randn(rng, E)
  end
  z = sqrt(abs(x)^2 + abs(y)^2)
  (abs(x) / z, y / z)
end

# assume block IndexLists are sorted.
function insert_random_rotations!(
  rng::AbstractRNG,
  gw::GivensWeight{B,E,R},
) where {B,R<:Real,E<:Union{R,Complex{R}}}

  if gw.lower_decomp[] == LeadingDecomp()
    nb = 0
    r = 0
    for b_ind ∈ gw.b.lower_blocks
      rotnum = 1
      rold = r
      Δn = gw.b.lower_blocks[b_ind].nb - nb
      nb = gw.b.lower_blocks[b_ind].nb
      r = gw.b.lower_blocks[b_ind].block_rank
      gind = gw.b.lower_blocks[b_ind].givens_index
      # loop over diagonals to put zeros a hypothetical upper
      # trapezoidal row space basis of size r × (Δn + rold).  For r=2
      # and Δn + rold = 5, the basis is of the form
      #
      # XXXXX
      #  XXXX
      #
      # with Δn + rold -r = 3 diagonals to be eliminated using
      # r⋅(Δn+rold-r)=6 rotations.
      #
      # Loop over diagonal to be eliminated:
      for d ∈ 1:(Δn + rold - r)
        # Compute elimination rotations for element j,k in the row
        # space basis.  This rotation acts on columns (offs + j + d -
        # 1) and (offs + j + d) where offs = nb - Δn - rold.
        for j ∈ r:-1:1
          k = j + d - 1 # column element to be zeroed in basis.
          offs = nb - Δn - rold # offset into the columns of B.
          c, s = random_cs(rng, E, R)
          gw.lowerRots[rotnum, gind] = Rot(c, s, offs + k)
          rotnum += 1
        end
      end
    end
  elseif gw.lower_decomp[] == TrailingDecomp()
    num_blocks = length(gw.b.lower_blocks)
    mb = gw.b.m
    r = 0
    for b_ind ∈ Iterators.Reverse(gw.b.lower_blocks)
      rotnum = 1
      rold = r
      Δm = mb - gw.b.lower_blocks[b_ind].mb
      mb = gw.b.lower_blocks[b_ind].mb
      r = gw.b.lower_blocks[b_ind].block_rank
      gind = gw.b.lower_blocks[b_ind].givens_index
      # loop over diagonals to put zeros a hypothetical column
      # space basis of size (Δm + rold - r) × r
      # For r=2 and Δm + rold = 5, the basis is of the form
      # 
      # XX
      # XX
      # XX
      # XX
      #  X
      #   
      #
      # with Δm + rold - r = 3 diagonals to be eliminated using
      # r(Δm+rold-r) rotations.
      #
      # Loop over diagonal to be eliminated.
      for d ∈ (Δm + rold - r + 1):-1:2
        # Compute elimination rotations for element j,k in the column
        # space basis.
        for k ∈ 1:r
          j = d + k - 1
          offs = mb # offset into rows of B.
          c, s = random_cs(rng, E, R)
          gw.lowerRots[rotnum, gind] = Rot(c, s, offs + j - 1)
          rotnum += 1
        end
      end
    end
  else
    # Dispatch should prevent this error from occurring.
    error("Decomposition requested is something other than Leading or Trailing")
  end

  if gw.upper_decomp[] == LeadingDecomp()
    mb = 0
    r = 0
    for b_ind ∈ gw.b.upper_blocks
      rotnum = 1
      rold = r
      Δm = gw.b.upper_blocks[b_ind].mb - mb
      mb = gw.b.upper_blocks[b_ind].mb
      r = gw.b.upper_blocks[b_ind].block_rank
      gind = gw.b.upper_blocks[b_ind].givens_index
      # loop over diagonals to put zeros a hypothetical upper
      # trapezoidal column space basis of size (Δm + rold) × r.  For r=2
      # and Δm + rold = 5, the basis is of the form
      #
      # X
      # XX
      # XX
      # XX
      # XX
      #
      # with Δm + rold - r = 3 diagonals to be eliminated using
      # r(Δm+rold-r)=6 rotations.
      #
      # Loop over diagonal to be eliminated.
      for d ∈ 1:(Δm + rold - r)
        # compute elimination rotations for element j,k in the row
        # space basis.
        for k ∈ r:-1:1
          j = k + d - 1 # row of element to be zeroed in basis.
          offs = mb - Δm - rold # offset into the columns of B.
          c, s = random_cs(rng, E, R)
          gw.upperRots[rotnum, gind] = Rot(c, s, offs + j)
          rotnum += 1
        end
      end
    end
  elseif gw.upper_decomp[] == TrailingDecomp()
    num_blocks = length(gw.b.upper_blocks)
    nb = gw.b.n
    r = 0
    for b_ind ∈ Iterators.Reverse(gw.b.upper_blocks)
      rotnum = 1
      rold = r
      Δn = nb - gw.b.upper_blocks[b_ind].nb
      nb = gw.b.upper_blocks[b_ind].nb
      r = gw.b.upper_blocks[b_ind].block_rank
      gind = gw.b.upper_blocks[b_ind].givens_index
      # loop over diagonals to put zeros a hypothetical row
      # space basis of size (Δn + rold - r) × r
      # For r=2 and Δn + rold = 5, the basis is of the form
      # 
      #   
      #  XXXX
      #  XXXXX
      #
      # with Δn + rold - r = 3 diagonals to be eliminated using
      # r(Δn+rold-r) rotations.
      #
      # Loop over diagonal to be eliminated:
      for d ∈ (Δn + rold - r + 1):-1:2
        # Compute elimination rotations for element j,k in the basis.
        for j ∈ 1:r
          k = d + j - 1
          offs = nb # offset into columns of B.
          c, s = random_cs(rng, E, R)
          gw.upperRots[rotnum, gind] = Rot(c, s, offs + k - 1)
          rotnum += 1
        end
      end
    end
  else
    # Dispatch should prevent this error from occurring.
    error("Decomposition requested is something other than Leading or Trailing")
  end

end

"""
    get_max_Δm(m::Int, blocks::AbstractMatrix{<:Int})

Get maximum increment in block row size.
"""
function get_max_Δm(m::Int, blocks::AbstractMatrix{<:Int})
  max_Δm = 0
  for k ∈ axes(blocks, 2)[2:end]
    Δm = blocks[1, k] - blocks[1, k - 1]
    max_Δm = Δm > max_Δm ? Δm : max_Δm
  end
  return max(max_Δm, blocks[1, begin], m - blocks[1, end])
end

"""
    get_max_Δn(n::Int, blocks::AbstractMatrix{<:Int})

Get maximum increment in block column size.
"""
function get_max_Δn(n::Int, blocks::AbstractMatrix{<:Int})
  max_Δn = 0
  for k ∈ axes(blocks, 2)[2:end]
    Δn = blocks[2, k] - blocks[2, k - 1]
    max_Δn = Δn > max_Δn ? Δn : max_Δn
  end
  return max(max_Δn, blocks[2, begin], n - blocks[2, end])
end

"""
    function max_num_rots(
      m::Int,
      n::Int,
      rmax::Int,
      blocks::AbstractMatrix{<:Int},
    )

Get maximum number of rotations required for any block.
"""
function max_num_rots(m::Int, n::Int, rmax::Int, blocks::AbstractMatrix{<:Int})

  Δm = get_max_Δm(m, blocks)
  Δn = get_max_Δn(n, blocks)

  return max(
    rmax * (Δm + 1) + ((rmax - 1) * rmax) ÷ 2,
    rmax * (Δn + 1) + ((rmax - 1) * rmax) ÷ 2,
  )
end

function _forward_get_offset(range, trange)
  # assuming:
  # dᵣ = setdiffᵣ(range, old_range)
  # trange = dᵣ ∪ᵣ last(old_range, old_rank)
  if !isempty(trange)
    # offset to the columns/rows to be compressed in block lb
    first(trange) - 1
  elseif !isempty(range)
    # Since trange is empty, no new columns/rows in range b
    # AND (b-1 has rank bound zero OR b-1 has no columns/rows).  This
    # handles the case in which b-1 has columns/rows and b has the same
    # columns/rows, but no compression is required because the rank bound
    # of b-1 was zero.
    last(range) # skip past all the columns/rows in block b.
  else
    # block b and b-1 have no columns/rows.
    0
  end
end

function _backward_get_offset(n, range, trange)
  # assuming:
  # dᵣ = setdiffᵣ(range, old_range)
  # trange = dᵣ ∪ᵣ first(old_range, old_rank)
  if !isempty(trange)
    # offset to the columns to be compressed in block ub
    first(trange) - 1
  elseif !isempty(range)
    # Since trange is empty, no new columns/rows in range b
    # AND (b+1 has rank bound zero OR b+1 has no columns/rows).  This
    # handles the case in which b+1 has columns/rows and b has the same
    # columns/rows, but no compression is required because the rank bound
    # of b+1 was zero.
    # last(cols_ub)
    first(range) - 2 # first column/row after offset is before first(cols_ub)
  else
    # block b and b+1 have no columns/rows.  Offset through entire matrix.
    n
  end
end

"""
    set_givens_weight_transform_params!(
      side::Union{Left, Right},
      decomp::Union{Decomp, Tuple{Vararg{Decomp}}},
      m::Int,
      n::Int;
      lower_blocks::Union{AbstractArray{Int,2}, Nothing} = nothing,
      lower_ranks::Union{AbstractVector{Int}, Nothing} = nothing,
      upper_blocks::Union{AbstractArray{Int,2}, Nothing} = nothing,
      upper_ranks::Union{AbstractVector{Int}, Nothing} = nothing,
      tsizes::Union{AbstractVector{Int}, Ref{Int}, Nothing}=nothing,
      num_rots::Union{AbstractVector{Int}, Ref{Int}, Nothing}=nothing,
      offsets::Union{AbstractVector{Int}, Nothing}=nothing,
      expand_values::Bool=false,
    )

Compute transform `tsizes`, `num_rots`, `offsets` values in place,
depending on which is provided.  If a `Ref{Int}` is given, compute the
maximum value over the range of indices.  If `expand_values` is
`true` compute maximum of all types of decompositions specified so
that the structure can hold multiple types of decompositions.

"""
function set_givens_weight_transform_params!(
  ::Lower,
  ::LeadingDecomp,
  m::Int,
  n::Int;
  lower_blocks::Union{AbstractVector{<:AbstractBlockData},Nothing} = nothing,
  lower_ranks::Union{AbstractVector{Int},Nothing} = nothing,
  upper_blocks::Union{AbstractVector{<:AbstractBlockData},Nothing} = nothing,
  upper_ranks::Union{AbstractVector{Int},Nothing} = nothing,
  tsizes::Union{AbstractVector{Int},Ref{Int},Nothing} = nothing,
  num_rots::Union{AbstractVector{Int},Ref{Int},Nothing} = nothing,
  offsets::Union{AbstractVector{Int},Nothing} = nothing,
  expand_values::Bool = false,
)

  # start with zeros if not expanding other values to find a maximum.
  maybe_zero(expand_values, tsizes, num_rots)

  # leading lower
  old_cols_lb = 1:0
  old_rank = 0
  for lb ∈ eachindex(lower_blocks, lower_ranks)
    _, cols_lb = lower_block_ranges(lower_blocks, m, n, lb)
    # new columns in block lb.
    dᵣ = setdiffᵣ(cols_lb, old_cols_lb)

    # columns acted on by the rotations associated with block lb,
    # assuming block lb-1 is column compressed.
    trange = dᵣ ∪ᵣ last(old_cols_lb, old_rank)
    tsize = length(trange)

    expand_or_set!(expand_values, tsizes, lb, tsize)

    rlb = lower_ranks[lb]
    # This includes enough extra to do a square triangularization.
    # extra_rots = rlb*(rlb-1) ÷ 2
    extra_rots = 0
    expand_or_set!(
      expand_values,
      num_rots,
      lb,
      (tsize - rlb) * rlb + extra_rots,
    )

    maybe_set!(offsets, lb, _forward_get_offset(cols_lb, trange))

    old_cols_lb = cols_lb
    old_rank = lower_ranks[lb]
  end
  return nothing
end

function set_givens_weight_transform_params!(
  ::Upper,
  ::LeadingDecomp,
  m::Int,
  n::Int;
  lower_blocks::Union{AbstractVector{<:AbstractBlockData},Nothing} = nothing,
  lower_ranks::Union{AbstractVector{Int},Nothing} = nothing,
  upper_blocks::Union{AbstractVector{<:AbstractBlockData},Nothing} = nothing,
  upper_ranks::Union{AbstractVector{Int},Nothing} = nothing,
  tsizes::Union{AbstractVector{Int},Ref{Int},Nothing} = nothing,
  num_rots::Union{AbstractVector{Int},Ref{Int},Nothing} = nothing,
  offsets::Union{AbstractVector{Int},Nothing} = nothing,
  expand_values::Bool = false,
)

  # start with zeros if not expanding other values to find a maximum.
  maybe_zero(expand_values, tsizes, num_rots)

  # leading upper
  old_rows_ub = 1:0
  old_rank = 0
  for ub ∈ eachindex(upper_blocks, upper_ranks)
    rows_ub, _ = upper_block_ranges(upper_blocks, m, n, ub)
    # new rows in block ub.
    dᵣ = setdiffᵣ(rows_ub, old_rows_ub)
    trange = dᵣ ∪ᵣ last(old_rows_ub, old_rank)
    tsize = length(trange)

    expand_or_set!(expand_values, tsizes, ub, tsize)

    rub = upper_ranks[ub]
    # extra_rots = rub * (rub - 1) ÷ 2
    extra_rots = 0

    expand_or_set!(
      expand_values,
      num_rots,
      ub,
      (tsize - rub) * rub + extra_rots,
    )

    maybe_set!(offsets, ub, _forward_get_offset(rows_ub, trange))

    old_rows_ub = rows_ub
    old_rank = upper_ranks[ub]
  end
  return nothing
end

function set_givens_weight_transform_params!(
  ::Upper,
  ::TrailingDecomp,
  m::Int,
  n::Int;
  lower_blocks::Union{AbstractVector{<:AbstractBlockData},Nothing} = nothing,
  lower_ranks::Union{AbstractVector{Int},Nothing} = nothing,
  upper_blocks::Union{AbstractVector{<:AbstractBlockData},Nothing} = nothing,
  upper_ranks::Union{AbstractVector{Int},Nothing} = nothing,
  tsizes::Union{AbstractVector{Int},Ref{Int},Nothing} = nothing,
  num_rots::Union{AbstractVector{Int},Ref{Int},Nothing} = nothing,
  offsets::Union{AbstractVector{Int},Nothing} = nothing,
  expand_values::Bool = false,
)

  maybe_zero(expand_values, tsizes, num_rots)

  # trailing upper
  old_cols_ub = 1:0
  old_rank = 0
  for ub ∈ Iterators.reverse(eachindex(upper_blocks, upper_ranks))
    (_, cols_ub) = upper_block_ranges(upper_blocks, m, n, ub)
    # new columns in block ub relative to ub+1.
    dᵣ = setdiffᵣ(cols_ub, old_cols_ub)
    trange = dᵣ ∪ᵣ first(old_cols_ub, old_rank)
    tsize = length(trange)

    # columns acted on by the rotations associated with block ub,
    # assuming block ub+1 is column compressed.
    expand_or_set!(expand_values, tsizes, ub, tsize)

    rub = upper_ranks[ub]
    # extra_rots = rub * (rub - 1) ÷ 2
    extra_rots = 0

    expand_or_set!(
      expand_values,
      num_rots,
      ub,
      (tsize - rub) * rub + extra_rots,
    )

    maybe_set!(offsets, ub, _backward_get_offset(n, cols_ub, trange))

    old_cols_ub = cols_ub
    old_rank = upper_ranks[ub]
  end

  return nothing
end

function set_givens_weight_transform_params!(
  ::Lower,
  ::TrailingDecomp,
  m::Int,
  n::Int;
  lower_blocks::Union{AbstractVector{<:AbstractBlockData},Nothing} = nothing,
  lower_ranks::Union{AbstractVector{Int},Nothing} = nothing,
  upper_blocks::Union{AbstractVector{<:AbstractBlockData},Nothing} = nothing,
  upper_ranks::Union{AbstractVector{Int},Nothing} = nothing,
  tsizes::Union{AbstractVector{Int},Ref{Int},Nothing} = nothing,
  num_rots::Union{AbstractVector{Int},Ref{Int},Nothing} = nothing,
  offsets::Union{AbstractVector{Int},Nothing} = nothing,
  expand_values::Bool = false,
)

  maybe_zero(expand_values, tsizes, num_rots)

  # trailing lower
  old_rows_lb = 1:0
  old_rank = 0
  for lb ∈ Iterators.reverse(eachindex(lower_blocks, lower_ranks))
    rows_lb, _ = lower_block_ranges(lower_blocks, m, n, lb)
    dᵣ = setdiffᵣ(rows_lb, old_rows_lb)
    trange = dᵣ ∪ᵣ first(old_rows_lb, old_rank)
    tsize = length(trange)

    expand_or_set!(expand_values, tsizes, lb, tsize)

    rlb = lower_ranks[lb]
    #extra_rots = rlb * (rlb - 1) ÷ 2
    extra_rots = 0

    expand_or_set!(
      expand_values,
      num_rots,
      lb,
      (tsize - rlb) * rlb + extra_rots,
    )

    maybe_set!(offsets, lb, _backward_get_offset(m, rows_lb, trange))

    old_rows_lb = rows_lb
    old_rank = lower_ranks[lb]
  end
  return nothing
end

function set_givens_weight_transform_params!(
  lower_upper::Union{Lower,Upper},
  decomp::Union{Decomp,Tuple{Vararg{Decomp}}},
  m::Int,
  n::Int;
  lower_blocks::Union{AbstractVector{<:AbstractBlockData},Nothing} = nothing,
  lower_ranks::Union{AbstractVector{Int},Nothing} = nothing,
  upper_blocks::Union{AbstractVector{<:AbstractBlockData},Nothing} = nothing,
  upper_ranks::Union{AbstractVector{Int},Nothing} = nothing,
  tsizes::Union{AbstractVector{Int},Ref{Int},Nothing} = nothing,
  num_rots::Union{AbstractVector{Int},Ref{Int},Nothing} = nothing,
  offsets::Union{AbstractVector{Int},Nothing} = nothing,
  expand_values::Bool = false,
)

  expand = expand_values

  for d ∈ decomp
    set_givens_weight_transform_params!(
      lower_upper,
      d,
      m,
      n;
      lower_blocks = lower_blocks,
      lower_ranks = lower_ranks,
      upper_blocks = upper_blocks,
      upper_ranks = upper_ranks,
      tsizes = tsizes,
      num_rots = num_rots,
      offsets = offsets,
      expand_values = expand,
    )
    # If set is called more than once, always expand after the first
    # call.
    expand = true
  end

  return nothing
end


"""
    get_givens_transform_params(
      lower_upper::Union{Lower,Upper},
      decomp::Union{Decomp, Tuple{Vararg{Decomp}}},
      m::Int,
      n::Int,
      params::Vararg{Union{TransformSizes,NumRots,Offsets}};
      lower_blocks::Union{AbstractArray{Int,2},Nothing} = nothing,
      lower_ranks::Union{AbstractVector{Int},Nothing} = nothing,
      upper_blocks::Union{AbstractArray{Int,2},Nothing} = nothing,
      upper_ranks::Union{AbstractVector{Int},Nothing} = nothing,
    )

Compute `tsizes`, `num_rots`, `offsets` values, depending on which
selectors are provided as `Vararg` parameters.
"""
function get_givens_weight_transform_params(
  lower_upper::Union{Lower,Upper},
  decomp::Union{Decomp,Tuple{Vararg{Decomp}}},
  m::Int,
  n::Int,
  params::Vararg{Union{TransformSizes,NumRots,Offsets}};
  lower_blocks::Union{AbstractVector{<:AbstractBlockData},Nothing} = nothing,
  lower_ranks::Union{AbstractVector{Int},Nothing} = nothing,
  upper_blocks::Union{AbstractVector{<:AbstractBlockData},Nothing} = nothing,
  upper_ranks::Union{AbstractVector{Int},Nothing} = nothing,
)

  num_blocks = length(something(lower_blocks, upper_blocks))

  tsizes = nothing
  offsets = nothing
  num_rots = nothing

  for p ∈ params
    p == TransformSizes() && (tsizes = zeros(Int, num_blocks))
    p == NumRots() && (num_rots = zeros(Int, num_blocks))
    p == Offsets() && (offsets = zeros(Int, num_blocks))
  end

  set_givens_weight_transform_params!(
    lower_upper,
    decomp,
    m,
    n,
    lower_blocks = lower_blocks,
    lower_ranks = lower_ranks,
    upper_blocks = upper_blocks,
    upper_ranks = upper_ranks,
    tsizes = tsizes,
    num_rots = num_rots,
    offsets = offsets,
    expand_values = false,
  )

  result::Vector{Vector{Int}} = []

  for p ∈ params
    p == TransformSizes() && push!(result, tsizes)
    p == NumRots() && push!(result, num_rots)
    p == Offsets() && push!(result, offsets)
  end

  tuple(result...)

end

"""
    get_givens_weight_max_transform_params(
      lower_upper::Union{Lower,Upper},
      decomp::Union{Decomp, Tuple{Vararg{Decomp}}},
      m::Int,
      n::Int,
      params::Vararg{Union{TransformSizes,NumRots}};
      lower_blocks::Union{AbstractArray{Int,2},Nothing} = nothing,
      lower_ranks::Union{AbstractVector{Int},Nothing} = nothing,
      upper_blocks::Union{AbstractArray{Int,2},Nothing} = nothing,
      upper_ranks::Union{AbstractVector{Int},Nothing} = nothing,
    )

Compute requested maxima for `sizes` and `num_rots` for the type(s) of
decompositions/transforms specified.
"""
function get_givens_weight_max_transform_params(
  lower_upper::Union{Lower,Upper},
  decomp::Union{Decomp,Tuple{Vararg{Decomp}}},
  m::Int,
  n::Int,
  params::Vararg{Union{TransformSizes,NumRots}};
  lower_blocks::Union{AbstractVector{<:AbstractBlockData},Nothing} = nothing,
  lower_ranks::Union{AbstractVector{Int},Nothing} = nothing,
  upper_blocks::Union{AbstractVector{<:AbstractBlockData},Nothing} = nothing,
  upper_ranks::Union{AbstractVector{Int},Nothing} = nothing,
)

  tsizes = nothing
  num_rots = nothing

  for p ∈ params
    p == TransformSizes() && (tsizes = Ref(0))
    p == NumRots() && (num_rots = Ref(0))
  end

  set_givens_weight_transform_params!(
    lower_upper,
    decomp,
    m,
    n;
    lower_blocks = lower_blocks,
    lower_ranks = lower_ranks,
    upper_blocks = upper_blocks,
    upper_ranks = upper_ranks,
    tsizes = tsizes,
    num_rots = num_rots,
    expand_values = false,
  )

  result::Vector{Int} = []

  for p ∈ params
    p == TransformSizes() && push!(result, tsizes[])
    p == NumRots() && push!(result, num_rots[])
  end
  tuple(result...)

end

function LinearAlgebra.Matrix(gw::GivensWeight)
  a = Matrix(gw.b)

  if gw.lower_decomp[] isa TrailingDecomp
    for lb_ind ∈ filter_compressed(gw.b.lower_blocks)
      g_ind = gw.b.lower_blocks[lb_ind].givens_index
      _, cols = lower_block_ranges(gw.b, lb_ind)
      va = view(a, :, cols)
      for j ∈ gw.b.lower_blocks[lb_ind].num_rots:-1:1
        r = gw.lowerRots[j, g_ind]
        apply!(r, va)
      end
    end
  end

  if gw.lower_decomp[] isa LeadingDecomp
    for lb_ind ∈ filter_compressed(Iterators.Reverse(gw.b.lower_blocks))
      g_ind = gw.b.lower_blocks[lb_ind].givens_index
      rows, _ = lower_block_ranges(gw.b, lb_ind)
      va = view(a, rows, :)
      for j ∈ gw.b.lower_blocks[lb_ind].num_rots:-1:1
        r = gw.lowerRots[j, g_ind]
        apply_inv!(va, r)
      end
    end
  end

  if gw.upper_decomp[] isa TrailingDecomp
    for ub_ind ∈ filter_compressed(gw.b.upper_blocks)
      g_ind = gw.b.upper_blocks[ub_ind].givens_index
      rows, _ = upper_block_ranges(gw.b, ub_ind)
      va = view(a, rows, :)
      for j ∈ gw.b.upper_blocks[ub_ind].num_rots:-1:1
        r = gw.upperRots[j, g_ind]
        apply_inv!(va, r)
      end
    end
  end

  if gw.upper_decomp[] isa LeadingDecomp
    for ub_ind ∈ filter_compressed(Iterators.Reverse(gw.b.upper_blocks))
      g_ind = gw.b.upper_blocks[ub_ind].givens_index
      _, cols = upper_block_ranges(gw.b, ub_ind)
      va = view(a, :, cols)
      for j ∈ gw.b.upper_blocks[ub_ind].num_rots:-1:1
        r = gw.upperRots[j, g_ind]
        apply!(r, va)
      end
    end
  end

  return a

end

function Base.show(io::IO, mime::MIME"text/plain", gw::GivensWeight)
  println(io, "$(gw.b.m)×$(gw.b.n) $(typeof(gw))")
  # limited = get(io, :limit, false)::Bool
  println(io, "lower_decomp: $(gw.lower_decomp[])")
  println(io, "upper_decomp: $(gw.upper_decomp[])")
  println(io, "lowerRots:")
  show(io, mime, gw.lowerRots)
  println(io)
  println(io, "upperRots:")
  show(io, mime, gw.upperRots)
  println(io)
  println(io)
  println(io, "Band matrix and Data:")
  show(io, mime, gw.b)
  println(io)
  println(io)
end

end
