module WY

using Printf
using Random

using LinearAlgebra
using LoopVectorization
using Octavian

import InPlace
using ..Compute

export WYTrans,
  resetWYBlock!,
  resetWYBlocks!,
  reworkWY!,
  WYIndexSubsetError,
  SelectWY,
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

InPlace.product_side(::Type{<:WYTrans}, _) = InPlace.LeftProduct()
InPlace.product_side(::Type{<:WYTrans}, ::Type{<:Integer}, _) =
  InPlace.LeftProduct()
InPlace.product_side(_, ::Type{<:WYTrans}) = InPlace.RightProduct()
InPlace.product_side(_, ::Type{<:WYTrans}, ::Type{<:Integer}) =
  InPlace.RightProduct()

InPlace.product_side(::Type{<:WYTrans}, ::Type{<:HouseholderTrans}) =
  InPlace.RightProduct()
InPlace.product_side(
  ::Type{<:WYTrans},
  ::Type{<:Integer},
  ::Type{<:HouseholderTrans},
) = InPlace.RightProduct()
InPlace.product_side(::Type{<:HouseholderTrans}, ::Type{<:WYTrans}) =
  InPlace.LeftProduct()
InPlace.product_side(
  ::Type{<:HouseholderTrans},
  ::Type{<:WYTrans},
  ::Type{<:Integer},
) = InPlace.LeftProduct()

function Random.rand!(rng::AbstractRNG, wy::WYTrans{E}) where {E <: Number}
  v = zeros(E,wy.max_WY_size)
  work = zeros(E,1)
  @views for k ∈ 1:wy.num_WY[]
    offs = wy.offsets[k]
    n = wy.sizes[k]
    h = HouseholderTrans(one(E), v[1:n], 1, n, offs, work)
    num_hs = wy.num_hs[k]
    wy.num_hs[k] = 0
    for j ∈ 1:num_hs
      rand!(rng, h)
      InPlace.apply!(h, wy, k)
    end
  end
  nothing
end

function Random.rand!(wy::WYTrans{E}) where {E <: Number}
  rand!(Random.default_rng(), wy)
end

struct SelectWY{T}
  wy::T
  block::Int
end

@inline function selectWY!(wy,k)
  @boundscheck k ∈ 1:wy.num_WY[] || throw_WYBlockNotAvailable(k, wy.num_WY[])
  wy.active_WY[]=k
end

InPlace.product_side(::Type{<:SelectWY}, _) = InPlace.LeftProduct()
InPlace.product_side(_, ::Type{<:SelectWY}) = InPlace.RightProduct()

InPlace.product_side(::Type{<:SelectWY}, ::Type{<:HouseholderTrans}) =
  InPlace.RightProduct()
InPlace.product_side(::Type{<:HouseholderTrans}, ::Type{<:SelectWY}) =
  InPlace.LeftProduct()


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
    zeros(E, work_size),
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

@inline function InPlace.apply_right!(
  A::AbstractArray{E,2},
  wy::WYTrans{E},
  k::Int,
) where {E<:Number}

  num_WY = wy.num_WY[]
  @boundscheck k ∈ 1:num_WY || throw_WYBlockNotAvailable(k, num_WY)
  @inbounds begin
    inds = 1:wy.sizes[k]
    offset = wy.offsets[k]
    (ma,na) = size(A)
    num_hs= wy.num_hs[k]
  end

  @boundscheck begin

    inds .+ offset ⊆ 1:na ||
      throw_ColumnRange_DimensionMismatch(ma, na, inds .+ offset)

    length(wy.work) >= ma * num_hs ||
      throw_WorkSizeError(ma, na, ma * num_hs, length(wy.work))

  end

  @inbounds begin
    @views begin
      work = reshape(wy.work[1:ma*num_hs], ma, num_hs)
      W = wy.W[inds, 1:num_hs, k]
      Y = wy.Y[inds, 1:num_hs, k]
      A0 = A[:,inds .+ offset]
    end
    oneE = one(E)
    num_hs > 0 && length(inds) > 0 && begin
      matmul!(work, A0, W)
      matmul!(A0, work, Y', -oneE, oneE)
    end
  end
  nothing
end

Base.@propagate_inbounds function InPlace.apply_right!(
  A::AbstractArray{E,2},
  wyk::SelectWY{<:WYTrans{E}},
) where {E<:Number}
  k=wyk.block
  wy=wyk.wy
  InPlace.apply_right!(A, wy, k)
end

Base.@propagate_inbounds function InPlace.apply_right!(
  A::AbstractArray{E,2},
  wy::WYTrans{E},
) where {E<:Number}
  InPlace.apply_right!(A, wy, wy.active_WY[])
end

@inline function InPlace.apply_right_inv!(
  A::AbstractArray{E,2},
  wy::WYTrans{E},
  k::Int
) where {E<:Number}

  num_WY = wy.num_WY[]
  @boundscheck k ∈ 1:num_WY || throw_WYBlockNotAvailable(k, num_WY)
  @inbounds begin
    num_hs = wy.num_hs[k]
    offset = wy.offsets[k]
    inds = 1:wy.sizes[k]
    (ma,na) = size(A)
  end

  @boundscheck begin

    inds .+ offset ⊆ 1:na ||
      throw_ColumnRange_DimensionMismatch(ma, na, inds .+ offset)

    length(wy.work) >= ma * num_hs ||
      throw_WorkSizeError(ma, na, ma * num_hs, length(wy.work))

  end

  @inbounds begin
    @views begin
      work = reshape(wy.work[1:ma*num_hs], ma, num_hs)
      W = wy.W[inds, 1:num_hs, k]
      Y = wy.Y[inds, 1:num_hs, k]
      A0 = A[:, inds .+ offset]
    end
    oneE = one(E)
    num_hs > 0 && length(inds) > 0 && begin
      matmul!(work, A0, Y)
      matmul!(A0, work, W', -oneE, oneE)
    end
  end
  nothing
end

Base.@propagate_inbounds function InPlace.apply_right_inv!(
  A::AbstractArray{E,2},
  wyk::SelectWY{<:WYTrans{E}},
) where {E<:Number}
  k=wyk.block
  wy=wyk.wy
  InPlace.apply_right_inv!(A, wy, k)
end

Base.@propagate_inbounds function InPlace.apply_right_inv!(
  A::AbstractArray{E,2},
  wy::WYTrans{E},
) where {E<:Number}
  apply_right_inv!(A, wy, wy.active_WY[])
end

@inline function InPlace.apply_left!(
  wy::WYTrans{E},
  k::Int,
  A::AbstractArray{E,2},
) where {E<:Number}

  num_WY = wy.num_WY[]
  @boundscheck k ∈ 1:num_WY || throw_WYBlockNotAvailable(k, num_WY)
  @inbounds begin
    num_hs = wy.num_hs[k]
    offset = wy.offsets[k]
    inds = 1:wy.sizes[k]
    (ma, na) = size(A)
  end

  @boundscheck begin

    inds .+ offset ⊆ 1:ma ||
      throw_RowRange_DimensionMismatch(ma, na, inds .+ offset)

    length(wy.work) >= na * num_hs ||
      throw_WorkSizeError(ma, na, na * num_hs, length(wy.work))

  end

  @inbounds begin
    @views begin
      work = reshape(wy.work[1:na*num_hs],num_hs,na)
      W = wy.W[inds, 1:num_hs, k]
      Y = wy.Y[inds, 1:num_hs, k]
      A0 = A[inds .+ offset, :]
    end
    oneE = one(E)
    num_hs > 0 && length(inds) > 0 && begin
      matmul!(work, Y', A0)
      matmul!(A0, W, work, -oneE, oneE)
    end
  end
  nothing
end

Base.@propagate_inbounds function InPlace.apply_left!(
  wyk::SelectWY{<:WYTrans{E}},
  A::AbstractArray{E,2},
) where {E<:Number}
  k=wyk.block
  wy=wyk.wy
  InPlace.apply_left!(wy, k, A)
end

Base.@propagate_inbounds function InPlace.apply_left!(
  wy::WYTrans{E},
  A::AbstractArray{E,2},
) where {E<:Number}
  InPlace.apply_left!(wy, wy.active_WY[], A)
end

@inline function InPlace.apply_left_inv!(
  wy::WYTrans{E},
  k::Int,
  A::AbstractArray{E,2},
) where {E<:Number}

  num_WY = wy.num_WY[]
  @boundscheck k ∈ 1:num_WY || throw_WYBlockNotAvailable(k, num_WY)

  @inbounds begin
    num_hs = wy.num_hs[k]
    offset = wy.offsets[k]
    inds = 1:wy.sizes[k]
    (ma, na) = size(A)
  end
  @boundscheck begin
    inds .+ offset ⊆ 1:ma ||
      throw_RowRange_DimensionMismatch(ma, na, inds .+ offset)

    length(wy.work) >= na * num_hs ||
      throw_WorkSizeError(ma, na, na * num_hs, length(wy.work))

  end

  @inbounds begin
    @views begin
      work = reshape(wy.work[1:na*num_hs],num_hs,na)
      W = wy.W[inds, 1:num_hs, k]
      Y = wy.Y[inds, 1:num_hs, k]
      A0 = A[inds .+ offset, :]
    end
    oneE = one(E)
    num_hs > 0 && length(inds) > 0 && begin
      matmul!(work, W', A0)
      matmul!(A0, Y, work, -oneE, oneE)
    end
  end
  nothing
end

Base.@propagate_inbounds function InPlace.apply_left_inv!(
  wyk::SelectWY{<:WYTrans{E}},
  A::AbstractArray{E,2},
) where {E<:Number}
  k=wyk.block
  wy=wyk.wy
  InPlace.apply_left_inv!(wy, k, A)
end

Base.@propagate_inbounds function InPlace.apply_left_inv!(
  wy::WYTrans{E},
  A::AbstractArray{E,2},
) where {E<:Number}
  InPlace.apply_left_inv!(wy, wy.active_WY[], A)
end

# Updating functions for adding a Householder.

struct WYIndexSubsetError <: Exception end

struct WYMaxHouseholderError <: Exception
  message::String
end

throw_WYMaxHouseholderError(block) =
  throw(WYMaxHouseholderError(@sprintf(
    "Too many Householders for block %d.",
    block
  )))

@inline function InPlace.apply_right!(
  wy::WYTrans{E},
  k::Int,
  h::HouseholderTrans{E},
) where {E<:Number}
  
  num_WY = wy.num_WY[]
  active_WY = wy.active_WY[]

  @boundscheck k ∈ 1:wy.num_WY[] || throw_WYBlockNotAvailable(k, wy.num_WY[])
  
  @inbounds begin
    num_hs = wy.num_hs[k]
    wy_offset=wy.offsets[k]
    indsh = (h.offs + 1):(h.offs + h.size)
    indswy = 1:wy.sizes[k]
    indshwy = (h.offs - wy_offset + 1):(h.offs - wy_offset + h.size)
  end

  @boundscheck begin
    indsh ⊆ (indswy .+ wy_offset) || throw(WYIndexSubsetError)
    num_hs < wy.max_num_hs || throw_WYMaxHouseholderError(k)
  end

  @inbounds begin
    @views begin
      work = reshape(wy.work[1:num_hs],num_hs,1)

      W0 = wy.W[indswy, 1:num_hs, k]
      W1 = wy.W[indswy, (num_hs + 1):(num_hs + 1), k]
      Y0 = wy.Y[indswy, 1:num_hs, k]
      Y1 = wy.Y[indswy, (num_hs + 1):(num_hs + 1), k]
    end
    W1[:,:] .= zero(E)
    Y1[:,:] .= zero(E)
    W1[indshwy,:] .= h.β .* h.v
    Y1[indshwy,:] .= h.v
    num_hs > 0 && length(indswy) > 0 && begin
      matmul!(work, Y0', W1)
      matmul!(W1, W0, work, -one(E), one(E))
    end
    wy.num_hs[k] += 1
  end
  nothing
end

Base.@propagate_inbounds function InPlace.apply_right!(
  wyk::SelectWY{<:WYTrans{E}},
  h::HouseholderTrans{E},
) where {E<:Number}

  k=wyk.block
  wy=wyk.wy
  InPlace.apply_right!(wy, k, h)
end

Base.@propagate_inbounds function InPlace.apply_right!(
  wy::WYTrans{E},
  h::HouseholderTrans{E},
) where {E<:Number}

  InPlace.apply_right!(wy, wy.active_WY[], h)
end

@inline function InPlace.apply_right_inv!(
  wy::WYTrans{E},
  k::Int,
  h::HouseholderTrans{E},
) where {E<:Number}

  num_WY = wy.num_WY[]
  @boundscheck k ∈ 1:num_WY || throw_WYBlockNotAvailable(k, num_WY)

  @inbounds begin
    num_hs = wy.num_hs[k]
    wy_offset=wy.offsets[k]

    indsh = (h.offs + 1):(h.offs + h.size)
    indswy = 1:wy.sizes[k]
    indshwy = (h.offs - wy_offset + 1):(h.offs - wy_offset + h.size)
  end

  @boundscheck begin
    indsh ⊆ (indswy .+ wy_offset) || throw(WYIndexSubsetError)
    num_hs < wy.max_num_hs || throw_WYMaxHouseholderError(k)
  end

  @inbounds begin
    @views begin
      work = reshape(wy.work[1:num_hs],num_hs,1)

      W0 = wy.W[indswy, 1:num_hs, k]
      W1 = wy.W[indswy, (num_hs + 1):(num_hs + 1), k]
      Y0 = wy.Y[indswy, 1:num_hs, k]
      Y1 = wy.Y[indswy, (num_hs + 1):(num_hs + 1), k]
    end

    W1[:,:] .= zero(E)
    Y1[:,:] .= zero(E)
    W1[indshwy,:] .= conj(h.β) .* h.v
    Y1[indshwy,:] = h.v
    num_hs > 0 && length(indswy) > 0 && begin
      matmul!(work, Y0', W1)
      matmul!(W1, W0, work, -one(E), one(E))
    end

    wy.num_hs[k] += 1
  end
  nothing
end

Base.@propagate_inbounds function InPlace.apply_right_inv!(
  wyk::SelectWY{<:WYTrans{E}},
  h::HouseholderTrans{E},
) where {E<:Number}
  
  k=wyk.block
  wy=wyk.wy
  InPlace.apply_right_inv!(wy, k, h)
end

Base.@propagate_inbounds function InPlace.apply_right_inv!(
  wy::WYTrans{E},
  h::HouseholderTrans{E},
) where {E<:Number}
  
  InPlace.apply_right_inv!(wy, wy.active_WY[], h)
end

@inline function InPlace.apply_left!(
  h::HouseholderTrans{E},
  wy::WYTrans{E},
  k::Int,
) where {E<:Number}

  num_WY = wy.num_WY[]
  @boundscheck k ∈ 1:num_WY || throw_WYBlockNotAvailable(k, num_WY)

  @inbounds begin
    num_hs = wy.num_hs[k]
    wy_offset = wy.offsets[k]
    v=reshape(h.v,length(h.v),1)
    indsh = (h.offs + 1):(h.offs + h.size)
    indswy = 1:wy.sizes[k]
    indshwy = (h.offs - wy_offset + 1):(h.offs - wy_offset + h.size)
  end

  @boundscheck begin
    indsh ⊆ (indswy .+ wy_offset) || throw(WYIndexSubsetError)
    num_hs < wy.max_num_hs || throw_WYMaxHouseholderError(k)
  end

  @inbounds begin

    @views begin
      work = reshape(wy.work[1:num_hs],num_hs,1)

      W0 = wy.W[indswy, 1:num_hs, k]
      W1 = wy.W[indswy, num_hs + 1:num_hs+1, k]
      Y0 = wy.Y[indswy, 1:num_hs, k]
      Y1 = wy.Y[indswy, num_hs + 1:num_hs+1, k]
    end
    W1[:,:] .= zero(E)
    Y1[:,:] .= zero(E)
    W1[indshwy,:] = v
    Y1[indshwy,:] .= conj(h.β) .* v
    
    num_hs > 0 && length(indswy) > 0 && begin
      matmul!(work, W0', Y1)
      matmul!(Y1, Y0, work, -one(E), one(E))
    end

    wy.num_hs[k] += 1
  end
  nothing
end

Base.@propagate_inbounds function InPlace.apply_left!(
  h::HouseholderTrans{E},
  wyk::SelectWY{<:WYTrans{E}},
) where {E<:Number}

  k=wyk.block
  wy=wyk.wy

  InPlace.apply_left!(h, wy, k)
end

Base.@propagate_inbounds function InPlace.apply_left!(
  h::HouseholderTrans{E},
  wy::WYTrans{E},
) where {E<:Number}

  InPlace.apply_left!(h, wy, wy.active_WY[])
end

@inline function InPlace.apply_left_inv!(
  h::HouseholderTrans{E},
  wy::WYTrans{E},
  k::Int,
) where {E<:Number}

  num_WY = wy.num_WY[]
  @boundscheck k ∈ 1:num_WY || throw_WYBlockNotAvailable(k, num_WY)

  @inbounds begin
    num_hs = wy.num_hs[k]
    wy_offset = wy.offsets[k]

    v=reshape(h.v,length(h.v),1)

    indsh = (h.offs + 1):(h.offs + h.size)
    indswy = 1:wy.sizes[k]
    indshwy = (h.offs - wy_offset + 1):(h.offs - wy_offset + h.size)
  end

  @boundscheck begin
    indsh ⊆ (indswy .+ wy_offset) || throw(WYIndexSubsetError)
    num_hs < wy.max_num_hs || throw_WYMaxHouseholderError(k)
  end

  @inbounds begin
    @views begin
      work = reshape(wy.work[1:num_hs],num_hs,1)

      W0 = wy.W[indswy, 1:num_hs, k]
      W1 = wy.W[indswy, num_hs + 1:num_hs+1, k]
      Y0 = wy.Y[indswy, 1:num_hs, k]
      Y1 = wy.Y[indswy, num_hs + 1:num_hs+1, k]
    end
    W1[:,:] .= zero(E)
    Y1[:,:] .= zero(E)
    W1[indshwy,:] = v
    Y1[indshwy,:] .= h.β .* v

    num_hs > 0 && length(indswy) > 0 && begin
      matmul!(work, W0', Y1)
      matmul!(Y1, Y0, work, -one(E), one(E))
    end

    wy.num_hs[k] += 1
  end
  nothing

end

Base.@propagate_inbounds function InPlace.apply_left_inv!(
  h::HouseholderTrans{E},
  wyk::SelectWY{<:WYTrans{E}},
) where {E<:Number}

  k=wyk.block
  wy=wyk.wy
  InPlace.apply_left_inv!(h, wy, k)
end

Base.@propagate_inbounds function InPlace.apply_left_inv!(
  h::HouseholderTrans{E},
  wy::WYTrans{E},
) where {E<:Number}

  InPlace.apply_left_inv!(h, wy, wy.active_WY[])
end

# WY times WY

# Apply transform l2 of wy2 to transform l1 of wy1.
# function apply_right!(
#   wy1::WYTrans{E},
#   l1,
#   wy2::WYTrans{E},
#   l2,
# ) where {E<:Number}

#   num_WY1 = wy1.num_WY[]
#   num_WY2 = wy2.num_WY[]
  
#   wy1_num_hs = wy1.num_hs[l1]
#   wy2_num_hs = wy2.num_hs[l2]

#   @boundscheck l1 ∈ 1:num_WY1 || throw_WYBlockNotAvilable(l1, num_WY1)
#   @boundscheck l2 ∈ 1:num_WY2 || throw_WYBlockNotAvilable(l2, num_WY2)

#   wy1_offset = wy1.offsets[l1]
#   wy2_offset = wy2.offsets[l2]

#   wy1_size=wy1.sizes[l1]
#   wy2_size=wy2.sizes[l2]
  
#   # Active indices for each transformation.
#   wy1_inds = (wy1_offset + 1):(wy1_offset + wy1_size)
#   wy2_inds = (wy2_offset + 1):(wy2_offset + wy2_size)

#   # Indices of wy2 within wy1, without the wy1_offset.
#   inds_wy12 = wy2_inds .- wy1_offset

#   @boundscheck begin
#     wy2_inds ⊆ wy1_inds || throw(WYIndexSubsetError)
#     wy1_num_hs + wy2_num_hs <= wy1.max_num_hs || throw_WYMaxHouseholderError(l1)
#   end

#   @inbounds begin
#     @views begin
#       wy2.offsets[l2] = wy2_offset - wy1_offset
#       apply_inv!(wy2, l2, wy1.Y[1:wy1_size, 1:wy1_num_hs, l1])
#       wy2.offsets[l2] = wy2_offset

#       wy1.num_hs[l1] = wy1_num_hs + wy2_num_hs
#       col_inds2 = wy1_num_hs + 1 : wy1_num_hs + wy2_num_hs
#       wy1.W[1:wy1_size, col_inds2, l1] .= zero(E)
#       wy1.Y[1:wy1_size, col_inds2, l1] .= zero(E)
#       wy1.W[inds_wy12, col_inds2, l1] .= wy2.W[1:wy2_size, 1:wy2_num_hs, l2]
#       wy1.Y[inds_wy12, col_inds2, l1] .= wy2.Y[1:wy2_size, 1:wy2_num_hs, l2]
#     end

#   end
  
# end


# Sweeps

"""
    SweepForward{T}

A struct for a forward product of WY transformations.  If
`wy=SweepForward.wy` and `n=wy.num_WY[]`, then this is conceptually
equivalent to the product

    wy₁ * wy₂ * ⋯ * wyₙ.
"""
struct SweepForward{T}
  wy::T
end

"""
    SweepBackward{T}

A struct for a forward product of WY transformations.  If
`wy=SweepForward.wy` and `n=wy.num_WY[]`, then this is conceptually
equivalent to the product

    wyₙ * wyₙ₋₁ * ⋯ * wy₁.
"""
struct SweepBackward{T}
  wy::T
end

InPlace.product_side(::Type{<:SweepForward}, _) = InPlace.LeftProduct()
InPlace.product_side(::Type{<:SweepBackward}, _) = InPlace.LeftProduct()
InPlace.product_side(_, ::Type{<:SweepForward}) = InPlace.RightProduct()
InPlace.product_side(_, ::Type{<:SweepBackward}) = InPlace.RightProduct()

@inline function InPlace.apply_right!(
  A::AbstractArray{E,2},
  sfwy::SweepForward{<:WYTrans{E}},
) where {E<:Number}

  wy = sfwy.wy
  n = wy.num_WY[]
  for k ∈ 1:n
    InPlace.apply_right!(A, wy, k)
  end
end

@inline function InPlace.apply_left!(
  sfwy::SweepForward{<:WYTrans{E}},
  A::AbstractArray{E,2},
) where {E<:Number}

  wy = sfwy.wy
  n = wy.num_WY[]
  for k ∈ n:-1:1
    InPlace.apply_left!(wy, k, A)
  end
end

@inline function InPlace.apply_right_inv!(
  A::AbstractArray{E,2},
  sfwy::SweepForward{<:WYTrans{E}},
) where {E<:Number}

  wy = sfwy.wy
  n = wy.num_WY[]
  for k ∈ n:-1:1
    InPlace.apply_right_inv!(A, wy, k)
  end
end

@inline function InPlace.apply_left_inv!(
  sfwy::SweepForward{<:WYTrans{E}},
  A::AbstractArray{E,2},
) where {E<:Number}

  wy = sfwy.wy
  n = wy.num_WY[]
  for k ∈ 1:n
    InPlace.apply_left_inv!(wy, k, A)
  end
end

@inline function InPlace.apply_right!(
  A::AbstractArray{E,2},
  sfwy::SweepBackward{<:WYTrans{E}},
) where {E<:Number}

  wy = sfwy.wy
  n = wy.num_WY[]
  for k ∈ n:-1:1
    InPlace.apply_right!(A, wy, k)
  end
end

@inline function InPlace.apply_left!(
  sfwy::SweepBackward{<:WYTrans{E}},
  A::AbstractArray{E,2},
) where {E<:Number}

  wy = sfwy.wy
  n = wy.num_WY[]
  for k ∈ 1:n
    InPlace.apply_left!(wy, k, A)
  end
end

@inline function InPlace.apply_right_inv!(
  A::AbstractArray{E,2},
  sfwy::SweepBackward{<:WYTrans{E}},
) where {E<:Number}

  wy = sfwy.wy
  n = wy.num_WY[]
  for k ∈ 1:n
    InPlace.apply_right_inv!(A, wy, k)
  end
end

@inline function InPlace.apply_left_inv!(
  sfwy::SweepBackward{<:WYTrans{E}},
  A::AbstractArray{E,2},
) where {E<:Number}

  wy = sfwy.wy
  n = wy.num_WY[]
  for k ∈ n:-1:1
    InPlace.apply_left_inv!(wy, k, A)
  end
end

end # module
