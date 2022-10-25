module WY

using Printf
using Random

using LinearAlgebra
using LoopVectorization
using Octavian

import InPlace: product_side, apply!, apply_inv!
using InPlace
using ..Compute

export WYTrans,
  resetWYBlock!,
  resetWYBlocks!,
  reworkWY!,
  WYIndexSubsetError,
  selectWY!,
  throw_WYBlockNotAvailable,
  WYBlockNotAvailable,
  WorkSizeError,
  throw_WorkSizeError,
  throw_ColumnRange_DimensionMismatch,
  throw_RowRange_DimensionMismatch,
  WYIndexSubsetError,
  WYMaxHouseholderError,
  throw_WYMaxHouseholderError,
  SweepForward,
  SweepBackward

"""

# WYTrans

    WYTrans{
      E<:Number,
      AEW<:AbstractArray{E,3},
      AEY<:AbstractArray{E,3},
      AEWork<:AbstractArray{E,1},
      AI<:AbstractArray{Int,1}
    }

A struct for storing multiple WY transformations.

## Fields

  - `max_num_WY::Int`: Maximum storable number of transformations.

  - `max_WY_size::Int`: Maximum number of transformations.

  - `max_num_hs::Int`: maximum number of Householders in each
    transformation.

  - `num_WY::Base.RefValue{Int}`: Actual number of blocks currently stored.

  - `active_WY::Base.RefValue{Int}`: Active block.  Zero if there is
    no active block.

  - `offsets::AI`: Array of length `max_num_WY` giving
    multiplcation offsets for each WY transformation.

  - `sizes::AI`: Array of length `max_num_WY` giving the size of
    each WY transformation.

  - `num_hs::AI`: Array of length `max_num_WY` giving the number
    of Householders in each WY.

  - `W::AEW`: Element array of size `max_WY_size × max_num_hs ×
    max_num_WY`.  This stores `W` for the different
    transformations.

  - `Y::AEW`: Element array of size `max_WY_size × max_num_hs ×
    max_num_WY`.  This stores `Y` for the different
    transformations.

  - `work:AEWork`: Workspace array.  Its size depends on how the WY
    transformation is used.  See the descriptions for the WYTrans
    constructors below.

## Application of WY Transformations

### With `wy.active_WY[]==k`, `wy ⊛ A` is equivalent to

    A₁=A[wy.offsets[k]+1:wy.offsets[k]+wy.sizes[k], :]
    W₁=W[wy.offsets[k]+1:wy.offsets[k]+wy.sizes[k], 1:wy.num_hs[k]]
    Y₁=Y[wy.offsets[k]+1:wy.offsets[k]+wy.sizes[k], 1:wy.num_hs[k]]
    A₁ = A₁ - W₁ Y₁ᴴ A₁

### `wy.active_WY[]==k`, `wy ⊘ A` is equivalent to

    A₁=A[wy.offsets[k]+1:wy.offsets[k]+wy.sizes[k], :]
    W₁=W[wy.offsets[k]+1:wy.offsets[k]+wy.sizes[k], 1:wy.num_hs[k]]
    Y₁=Y[wy.offsets[k]+1:wy.offsets[k]+wy.sizes[k], 1:wy.num_hs[k]]
    A₁ = A₁ - Y₁ W₁ᴴ A₁

### `wy.active_WY[]==k`, `A ⊛ wy` is equivalent to

    A₁=A[:, wy.offsets[k]+1:wy.offsets[k]+wy.sizes[k]]
    W₁=W[wy.offsets[k]+1:wy.offsets[k]+wy.sizes[k], 1:wy.num_hs[k]]
    Y₁=Y[wy.offsets[k]+1:wy.offsets[k]+wy.sizes[k], 1:wy.num_hs[k]]
    A₁ = A₁ - A₁ W₁ Y₁ᴴ

### `wy.active_WY[]==k`, `A ⊘ wy` is equivalent to

    A₁=A[:, wy.offsets[k]+1:wy.offsets[k]+wy.sizes[k]]
    W₁=W[wy.offsets[k]+1:wy.offsets[k]+wy.sizes[k], 1:wy.num_hs[k]]
    Y₁=Y[wy.offsets[k]+1:wy.offsets[k]+wy.sizes[k], 1:wy.num_hs[k]]
    A₁ = A₁ - A₁ Y₁ W₁ᴴ

"""
struct WYTrans{
  E<:Number,
  AEW<:AbstractArray{E,3},
  AEY<:AbstractArray{E,3},
  AEWork<:AbstractArray{E,1},
  AI<:AbstractArray{Int,1}
}
  # Maximum number of blocks.
  max_num_WY::Int
  # Maximum block size.
  max_WY_size::Int
  # maximum number of Householders.
  max_num_hs::Int
  # Number of WY transformations.
  num_WY::Base.RefValue{Int}
  # Active block
  active_WY::Base.RefValue{Int}
  # Offset of each WY.
  offsets::AI
  # Size of each WY.
  sizes::AI
  # num_hs: number of Householders in each WY.
  num_hs::AI
  # W: max_WY_size × max_num_hs × max_num_WY
  W::AEW
  # Y: max_WY_size × max_num_hs × max_num_WY
  Y::AEY
  work::AEWork
end

InPlace.product_side(::Type{<:WYTrans}, _) = InPlace.LeftProduct
InPlace.product_side(::Type{Tuple{W,Int}}, _) where {W <: WYTrans} =
  InPlace.LeftProduct

InPlace.product_side(_, ::Type{<:WYTrans}) = InPlace.RightProduct
InPlace.product_side(_, ::Type{Tuple{W,Int}}) where {W <: WYTrans} =
  InPlace.RightProduct

InPlace.product_side(::Type{<:WYTrans}, ::Type{<:HouseholderTrans}) =
  InPlace.RightProduct
InPlace.product_side(
  ::Type{Tuple{W,Int64}},
  ::Type{<:HouseholderTrans},
) where {W<:WYTrans} = InPlace.RightProduct

InPlace.product_side(::Type{<:HouseholderTrans}, ::Type{<:WYTrans}) =
  InPlace.LeftProduct
InPlace.product_side(
  ::Type{<:HouseholderTrans},
  ::Type{Tuple{W,Int64}},
) where {W<:WYTrans} = InPlace.LeftProduct

InPlace.structure_type(::Type{W}) where {E,W<:WYTrans{E}} = WYTrans{E}
InPlace.structure_type(::Type{Tuple{W,Int}}) where {E,W<:WYTrans{E}} =
  WYTrans{E}

function Random.rand!(rng::AbstractRNG, wy::WYTrans{E}) where {E <: Number}
  v = similar_zeros(wy.W, wy.max_WY_size)
  work = similar_zeros(wy.W, 1)
  @views for k ∈ 1:wy.num_WY[]
    offs = wy.offsets[k]
    n = wy.sizes[k]
    h = HouseholderTrans(one(E), v[1:n], 1, n, offs, work)
    num_hs = wy.num_hs[k]
    wy.num_hs[k] = 0
    wyk = (wy, k)
    for j ∈ 1:num_hs
      rand!(rng, h)
      InPlace.apply!(h, wyk)
    end
  end
  nothing
end

function Random.rand!(wy::WYTrans{E}) where {E <: Number}
  rand!(Random.default_rng(), wy)
end

@inline function selectWY!(wy,k)
  @boundscheck k ∈ 1:wy.num_WY[] || throw_WYBlockNotAvailable(k, wy.num_WY[])
  wy.active_WY[]=k
end

InPlace.product_side(::Type{Tuple{<:WYTrans, Int}}, _) = InPlace.LeftProduct
InPlace.product_side(_, ::Type{Tuple{<:WYTrans, Int}}) = InPlace.RightProduct

InPlace.product_side(::Type{Tuple{<:WYTrans, Int}}, ::Type{<:HouseholderTrans}) =
  InPlace.RightProduct
InPlace.product_side(::Type{<:HouseholderTrans}, ::Type{Tuple{<:WYTrans, Int}}) =
  InPlace.LeftProduct

"""
    WYTrans(
      ::Type{E},
      max_num_WY::Int,
      max_WY_size::Int,
      work_size::Int,
      max_num_hs::Int,
    ) where {E}

Create a new, empty WYTrans with element type `E`.  Defaults to including
all blocks.

The work space required varies with the type of the matrix to which it
is applied.

  - From the left applied to a general m×n matrix:
    `na * max_num_hs` elements.

  - From the right applied to a general m×n matrix:
    `ma * max_num_hs` elements. 

For a band matrix the requirements depend on the details of the structure

  * From the left applied to an m×n band matrix `bc`, in order of decreasing
    specificity/economy:

      * What is actually needed on a single application to a specific block

        ```julia
        (wy_size + num_hs) * 
        length(inband_hull(bc, offs+1:offs+max_WY_size, :))
        ```

      * What is sufficient for an application prior to fill-in:

        ```julia
        (max_WY_size + max_num_hs) *
        max_inband_hull_size(bc, max_WY_size, :)
        ```

      * What is sufficient if the band matrix never fills past what is
        storable:

        ```julia
        (max_WY_size + max_num_hs) *
        max_storable_hull_size(bc, max_WY_size, :)
        ```

      * Overkill:
        ```julia
        (max_WY_size + max_num_hs) * n
        ```

  * From the right applied to an m×n band matrix `bc`, in order of decreasing
    specificity/economy:

      * What is actually needed on a single application to a specific block

        ```julia
        (wy_size + num_hs) * 
        length(inband_hull(bc, :, offs+1:offs+max_WY_size))
        ```

      * What is sufficient for an application prior to fill-in:

        ```julia
        (max_WY_size + max_num_hs) *
        max_inband_hull_size(bc, :, max_WY_size)
        ```

      * What is sufficient if the band matrix never fills past what is
        storable:

        ```julia
        (max_WY_size + max_num_hs) *
        max_storable_hull_size(bc, :, max_WY_size)
        ```

      * Overkill:

        ```julia
        (max_WY_size + max_num_hs) * m
        ```
"""
function WYTrans(
  ::Type{E},
  max_num_WY::Int,
  max_WY_size::Int,
  work_size::Int,
  max_num_hs::Int,
) where {E<:Number}
  WYTrans(
    max_num_WY,
    max_WY_size,
    max_num_hs,
    Ref(max_num_WY), # include all possible blocks.
    Ref(0),
    zeros(Int,max_num_WY),
    zeros(Int,max_num_WY),
    zeros(Int,max_num_WY),
    zeros(E, max_WY_size, max_num_hs, max_num_WY),
    zeros(E, max_WY_size, max_num_hs, max_num_WY),
    zeros(E, work_size),
  )
end

function WYTrans(
  ::Type{E};
  max_num_WY::Int=1,
  max_WY_size::Int,
  work_size::Int,
  max_num_hs::Int,
) where {E<:Number}
  WYTrans(E, max_num_WY, max_WY_size, work_size, max_num_hs)
end

struct WYBlockNotAvailable <: Exception
  message::String
end

throw_WYBlockNotAvailable(block, num_WY) =
  throw(WYBlockNotAvailable(@sprintf(
    "Block %d is not available in WYTrans with num_WY = %d",
    block,
    num_WY
  )))

"""
    resetWYBlock!(
      k::Int
      offset::Int,
      sizeWY::Int,
      wy::WYTrans{E},
    ) where {E<:Number}

Return a block of a WYTrans to an empty state, with an given
offset and block size.

"""
@inline function resetWYBlock!(
  wy::WYTrans,
  k::Int,
  offset::Int,
  sizeWY::Int,
)
  @boundscheck begin
    k ∈ 1:wy.num_WY[] || throw_WYBlockNotAvailable(k, wy.num_WY[])
    sizeWY <= wy.max_WY_size || throw(DimensionMismatch(@sprintf(
      "Requested dimension %d of a WY block exceeds the maximum block size %d.",
      sizeWY,
      wy.max_WY_size
    )))
  end
  @inbounds begin
    wy.offsets[k] = offset
    wy.sizes[k] = sizeWY
    wy.num_hs[k] = 0
  end
  nothing
end

Base.@propagate_inbounds function resetWYBlock!(
  wy::WYTrans;
  block::Int=1,
  offset::Int=wy.offsets[block],
  sizeWY::Int=wy.sizes[block],
)
  resetWYBlock!(wy, block, offset, sizeWY)
end

@inline function resetWYBlocks!(
  offsets_sizes,
  wy::WYTrans,
)
  num_WY = length(offsets_sizes)
  @boundscheck num_WY <= wy.max_num_WY ||
               throw(WYBlockNotAvailable(@sprintf(
    "Block %d is not available in WYTrans with max_num_WY = %d",
    num_WY,
    wy.max_num_WY
  )))

  wy.num_WY[]=num_WY
  k=1
  for (offset, sizeWY) ∈ offsets_sizes
    @boundscheck(
      sizeWY <= wy.max_WY_size || throw(DimensionMismatch(@sprintf(
        "Requested dimension %d of a WY block exceeds the maximum block size %d.",
        sizeWY,
        wy.max_WY_size
      )))
    )
    @inbounds begin
      wy.offsets[k] = offset
      wy.sizes[k] = sizeWY
      wy.num_hs[k] = 0
      k += 1
    end
  end
  nothing
end

"""
    reworkWY!(
      work_size::Int,
      wy::WYTrans{E},
    ) where {E<:Number}

Allocate a new work space for a WYTrans so that it can be applied to a
matrix of a different (larger) opposite side size.
"""
@inline function reworkWY!(wy::WYTrans{E}, work_size::Int) where {E<:Number}
  WYTrans(
    wy.max_num_WY,
    wy.max_WY_size,
    wy.max_num_hs,
    wy.num_WY,
    wy.active_WY,
    wy.offsets,
    wy.sizes,
    wy.num_hs,
    wy.W,
    wy.Y,
    similar_zeros(wy.work, work_size),
  )
end

# Array Updates

struct WorkSizeError <: Exception
  message::String
end

throw_WorkSizeError(ma, na, required, provided) =
  throw(WYBlockNotAvailable(@sprintf(
    """
    An operation on a matrix of dimension %d×%d requires a work space of
    size %d while the available work space is of size %d.
    """,
    ma,
    na,
    required,
    provided
  )))

throw_ColumnRange_DimensionMismatch(ma, na, inds) =
  throw(DimensionMismatch(@sprintf(
      "an operation on a matrix of dimension %d×%d acts on columns %d:%d",
      ma,
      na,
      first(inds),
      last(inds)
    )))

throw_RowRange_DimensionMismatch(ma, na, inds) =
  throw(DimensionMismatch(@sprintf(
      "an operation on a matrix of dimension %d×%d acts on rows %d:%d",
      ma,
      na,
      first(inds),
      last(inds)
    )))

include("./WY/ApplyWY.jl")
include("./WY/ApplyWYHouseholder.jl")
include("./WY/ApplyWYWY.jl")
include("./WY/ApplySweeps.jl")


end # module
