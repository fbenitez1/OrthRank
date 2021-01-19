module WY

using Base: @propagate_inbounds
using Printf

using LinearAlgebra
import InPlace

using ..Compute

export WYTrans,
  resetWYBlock!, resetWYBlocks!, reworkWY!, WYIndexSubsetError, SelectWY, apply!, selectWY!

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

  - `A_other_dim::Int`: Other dimension of A (determines the size of
    the workspace).

  - `max_num_hs::Int`: maximum number of Householders in each
    transformation.

  - `num_WY::Ref{Int}`: Actual number of blocks currently stored.

  - `active_WY::Ref{Int}`: Active block.  Zero if no active block.

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

  - `work:AEWork`: Element array of length `A_other_dim * max_num_hs`.

## Application of WY Transformations

### `selectWY(wy,k) ⊛ A` is equivalent to

    A₁=A[wy.offsets[k]+1:wy.offsets[k]+wy.sizes[k], :]
    W₁=W[wy.offsets[k]+1:wy.offsets[k]+wy.sizes[k], 1:wy.num_hs[k]]
    Y₁=Y[wy.offsets[k]+1:wy.offsets[k]+wy.sizes[k], 1:wy.num_hs[k]]
    A₁ = A₁ - W₁ Y₁ᴴ A₁

### `selectWY(wy,k) ⊘ A` is equivalent to

    A₁=A[wy.offsets[k]+1:wy.offsets[k]+wy.sizes[k], :]
    W₁=W[wy.offsets[k]+1:wy.offsets[k]+wy.sizes[k], 1:wy.num_hs[k]]
    Y₁=Y[wy.offsets[k]+1:wy.offsets[k]+wy.sizes[k], 1:wy.num_hs[k]]
    A₁ = A₁ - Y₁ W₁ᴴ A₁

### `A ⊛ selectWY(wy,k)` is equivalent to

    A₁=A[:, wy.offsets[k]+1:wy.offsets[k]+wy.sizes[k]]
    W₁=W[wy.offsets[k]+1:wy.offsets[k]+wy.sizes[k], 1:wy.num_hs[k]]
    Y₁=Y[wy.offsets[k]+1:wy.offsets[k]+wy.sizes[k], 1:wy.num_hs[k]]
    A₁ = A₁ - A₁ W₁ Y₁ᴴ

### `A ⊘ selectWY(wy,k)` is equivalent to

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
  # Other dimension of A (determines the size of the workspace).
  A_other_dim::Int
  # maximum number of Householders.
  max_num_hs::Int
  # Number of blocks
  num_WY::Ref{Int}
  # Active block
  active_WY::Ref{Int}
  offsets::AI
  # Size of the individual block transformations
  sizes::AI
  # Number of Householders in each WY.
  num_hs::AI
  # sizeA_h×max_k array.
  W::AEW
  # sizeA_h×max_k array.
  Y::AEY
  # sizeA_other*max_k work array.
  work::AEWork
end

struct SelectWY{T}
  wy::T
  block::Int
end

@inline function selectWY!(wy,k)
  wy.active_WY[]=k
end


"""
    WYTrans(
      ::Type{E},
      max_num_WY::Int,
      max_WY_size::Int,
      A_other_dim::Int,
      max_num_hs::Int,
    ) where {E}

Create a new, empty WYTrans with element type `E`.  Defaults to including
all blocks.

"""
function WYTrans(
  ::Type{E},
  max_num_WY::Int,
  max_WY_size::Int,
  A_other_dim::Int,
  max_num_hs::Int,
) where {E<:Number}
  WYTrans(
    max_num_WY,
    max_WY_size,
    A_other_dim,
    max_num_hs,
    Ref(max_num_WY), # include all possible blocks.
    Ref(0),
    zeros(Int,max_num_WY),
    zeros(Int,max_num_WY),
    zeros(Int,max_num_WY),
    zeros(E, max_WY_size, max_num_hs, max_num_WY),
    zeros(E, max_WY_size, max_num_hs, max_num_WY),
    zeros(E, A_other_dim * max_num_hs),
  )
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
      offsets::Int,
      sizeWY::Int,
      wy::WYTrans{E},
    ) where {E<:Number}

Return a block of a WYTrans to an empty state, with an given
offsetset and block size.

"""
@inline function resetWYBlock!(
  k::Int,
  offsets::Int,
  sizeWY::Int,
  wy::WYTrans,
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
    wy.offsets[k] = offsets
    wy.sizes[k] = sizeWY
    wy.num_hs[k] = 0
  end
  nothing
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
  for (offsets, sizeWY) ∈ offsets_sizes
    @boundscheck(
      sizeWY <= wy.max_WY_size || throw(DimensionMismatch(@sprintf(
        "Requested dimension %d of a WY block exceeds the maximum block size %d.",
        sizeWY,
        wy.max_WY_size
      )))
    )
    @inbounds begin
      wy.offsets[k] = offsets
      wy.sizes[k] = sizeWY
      wy.num_hs[k] = 0
      k += 1
    end
  end
  nothing
end

"""
    reworkWY!(
      A_other_dim::Int,
      wy::WYTrans{E},
    ) where {E<:Number}

Allocate a new work space for a WYTrans so that it can be applied to a
matrix of a different (larger) opposite side size.
"""
@inline function reworkWY!(A_other_dim::Int, wy::WYTrans{E}) where {E<:Number}
  WYTrans(
    wy.max_num_WY,
    wy.max_WY_size,
    A_other_dim,
    wy.max_num_hs,
    wy.num_WY,
    wy.active_WY,
    wy.offsets,
    wy.sizes,
    wy.num_hs,
    wy.W,
    wy.Y,
    zeros(E, wy.A_other_dim * wy.max_num_hs),
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

@inline function apply!(
  A::AbstractArray{E,2},
  wy::WYTrans{E},
  k::Int,
) where {E<:Number}

  num_WY = wy.num_WY[]
  @boundscheck k ∈ 1:num_WY || throw_WYBlockNotAvailable(k, num_WY)
  @inbounds begin
    inds = 1:wy.sizes[k]
    offsets = wy.offsets[k]
    (ma,na) = size(A)
    num_hs= wy.num_hs[k]
  end

  @boundscheck begin

    inds .+ offsets ⊆ 1:na ||
      throw_ColumnRange_DimensionMismatch(ma, na, inds)

    length(wy.work) >= ma * num_hs ||
      throw_WorkSizeError(ma, na, ma * num_hs, length(wy.work))

  end

  @inbounds begin
    @views begin
      work = reshape(wy.work[1:ma*num_hs], ma, num_hs)
      W = wy.W[inds, 1:num_hs, k]
      Y = wy.Y[inds, 1:num_hs, k]
      A0 = A[:,inds .+ offsets]
    end
    oneE = one(E)
    mul!(work, A0, W)
    mul!(A0, work, Y', -oneE, oneE)
  end
  nothing
end

@propagate_inbounds function InPlace.:⊛(
  A::AbstractArray{E,2},
  wyk::SelectWY{<:WYTrans{E}},
) where {E<:Number}
  k=wyk.block
  wy=wyk.wy
  apply!(A, wy, k)
end

@propagate_inbounds function InPlace.:⊛(
  A::AbstractArray{E,2},
  wy::WYTrans{E},
) where {E<:Number}
  apply!(A, wy, wy.active_WY[])
end

@inline function apply_inv!(
  A::AbstractArray{E,2},
  wy::WYTrans{E},
  k::Int
) where {E<:Number}

  num_WY = wy.num_WY[]
  @boundscheck k ∈ 1:num_WY || throw_WYBlockNotAvailable(k, num_WY)
  @inbounds begin
    num_hs = wy.num_hs[k]
    offsets = wy.offsets[k]
    inds = 1:wy.sizes[k]
    (ma,na) = size(A)
  end

  @boundscheck begin

    inds .+ offsets ⊆ 1:na ||
      throw_ColumnRange_DimensionMismatch(ma, na, inds)

    length(wy.work) >= ma * num_hs ||
      throw_WorkSizeError(ma, na, ma * num_hs, length(wy.work))

  end

  @inbounds begin
    @views begin
      work = reshape(wy.work[1:ma*num_hs], ma, num_hs)
      W = wy.W[inds, 1:num_hs, k]
      Y = wy.Y[inds, 1:num_hs, k]
      A0 = A[:, inds .+ offsets]
    end
    oneE = one(E)
    mul!(work, A0, Y)
    mul!(A0, work, W', -oneE, oneE)
  end
  nothing
end

@propagate_inbounds function InPlace.:⊘(
  A::AbstractArray{E,2},
  wyk::SelectWY{<:WYTrans{E}},
) where {E<:Number}
  k=wyk.block
  wy=wyk.wy
  apply_inv!(A, wy, k)
end

@propagate_inbounds function InPlace.:⊘(
  A::AbstractArray{E,2},
  wy::WYTrans{E},
) where {E<:Number}
  apply_inv!(A, wy, wy.active_WY[])
end

@inline function apply!(
  wy::WYTrans{E},
  k::Int,
  A::AbstractArray{E,2},
) where {E<:Number}

  num_WY = wy.num_WY[]
  @boundscheck k ∈ 1:num_WY || throw_WYBlockNotAvailable(k, num_WY)
  @inbounds begin
    num_hs = wy.num_hs[k]
    offsets = wy.offsets[k]
    inds = 1:wy.sizes[k]
    (ma, na) = size(A)
  end

  @boundscheck begin

    inds .+ offsets ⊆ 1:ma ||
      throw_RowRange_DimensionMismatch(ma, na, inds)

    length(wy.work) >= na * num_hs ||
      throw_WorkSizeError(ma, na, na * num_hs, length(wy.work))

  end

  @inbounds begin
    @views begin
      work = reshape(wy.work[1:na*num_hs],num_hs,na)
      W = wy.W[inds, 1:num_hs, k]
      Y = wy.Y[inds, 1:num_hs, k]
      A0 = A[inds .+ offsets, :]
    end
    oneE = one(E)
    mul!(work, Y', A0)
    mul!(A0, W, work, -oneE, oneE)
  end
  nothing
end

@propagate_inbounds function InPlace.:⊛(
  wyk::SelectWY{<:WYTrans{E}},
  A::AbstractArray{E,2},
) where {E<:Number}
  k=wyk.block
  wy=wyk.wy
  apply!(wy, k, A)
end

@propagate_inbounds function InPlace.:⊛(
  wy::WYTrans{E},
  A::AbstractArray{E,2},
) where {E<:Number}
  apply!(wy, wy.active_WY[], A)
end

@inline function apply_inv!(
  wy::WYTrans{E},
  k::Int,
  A::AbstractArray{E,2},
) where {E<:Number}

  num_WY = wy.num_WY[]
  @boundscheck k ∈ 1:num_WY || throw_WYBlockNotAvailable(k, num_WY)

  @inbounds begin
    num_hs = wy.num_hs[k]
    offsets = wy.offsets[k]
    inds = 1:wy.sizes[k]
    (ma, na) = size(A)
  end
  @boundscheck begin

    inds .+ offsets ⊆ 1:ma ||
      throw_RowRange_DimensionMismatch(ma, na, inds)

    length(wy.work) >= na * num_hs ||
      throw_WorkSizeError(ma, na, na * num_hs, length(wy.work))

  end

  @inbounds begin
    @views begin
      work = reshape(wy.work[1:na*num_hs],num_hs,na)
      W = wy.W[inds, 1:num_hs, k]
      Y = wy.Y[inds, 1:num_hs, k]
      A0 = A[inds .+ offsets, :]
    end
    oneE = one(E)
    mul!(work, W', A0)
    mul!(A0, Y, work, -oneE, oneE)
  end
  nothing
end

@propagate_inbounds function InPlace.:⊘(
  wyk::SelectWY{<:WYTrans{E}},
  A::AbstractArray{E,2},
) where {E<:Number}
  k=wyk.block
  wy=wyk.wy
  apply_inv!(wy, k, A)
end

@propagate_inbounds function InPlace.:⊘(
  wy::WYTrans{E},
  A::AbstractArray{E,2},
) where {E<:Number}
  apply_inv!(wy, wy.active_WY[], A)
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

@inline function apply!(
  wy::WYTrans{E},
  k::Int,
  h::HouseholderTrans{E},
) where {E<:Number}
  num_WY = wy.num_WY[]
  @boundscheck k ∈ 1:num_WY || throw_WYBlockNotAvailable(k, num_WY)
  
  @inbounds begin
    num_hs = wy.num_hs[k]
    wy_offsets=wy.offsets[k]
    v=reshape(h.v,length(h.v),1)
    indsh = (h.offs + 1):(h.offs + h.size)
    indswy = 1:wy.sizes[k]
    indshwy = (h.offs - wy_offsets + 1):(h.offs - wy_offsets + h.size)
  end

  @boundscheck begin
    indsh ⊆ (indswy .+ wy_offsets) || throw(WYIndexSubsetError)
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
    W1[indshwy,:] = v
    Y1[indshwy,:] = v
    mul!(work, Y0', W1)
    mul!(W1, W0, work, -h.β, h.β)

    wy.num_hs[k] += 1
  end
  nothing
end

@propagate_inbounds function InPlace.:⊛(
  wyk::SelectWY{<:WYTrans{E}},
  h::HouseholderTrans{E},
) where {E<:Number}

  k=wyk.block
  wy=wyk.wy
  apply!(wy, k, h)
end

@propagate_inbounds function InPlace.:⊛(
  wy::WYTrans{E},
  h::HouseholderTrans{E},
) where {E<:Number}

  apply!(wy, wy.active_WY[], h)
end

@inline function apply_inv(
  wy::WYTrans{E},
  k::Int,
  h::HouseholderTrans{E},
) where {E<:Number}

  num_WY = wy.num_WY[]
  @boundscheck k ∈ 1:num_WY || throw_WYBlockNotAvailable(k, num_WY)

  @inbounds begin
    num_hs = wy.num_hs[k]
    wy_offsets=wy.offsets[k]

    v=reshape(h.v,length(h.v),1)

    indsh = (h.offs + 1):(h.offs + h.size)
    indswy = 1:wy.sizes[k]
    indshwy = (h.offs - wy_offsets + 1):(h.offs - wy_offsets + h.size)
  end

  @boundscheck begin
    indsh ⊆ (indswy .+ wy_offsets) || throw(WYIndexSubsetError)
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
    W1[indshwy,:] = v
    Y1[indshwy,:] = v
    mul!(work, Y0', W1)
    mul!(W1, W0, work, -conj(h.β), conj(h.β))
    wy.num_hs[k] += 1
  end
  nothing
end

@propagate_inbounds function InPlace.:⊘(
  wyk::SelectWY{<:WYTrans{E}},
  h::HouseholderTrans{E},
) where {E<:Number}
  
  k=wyk.block
  wy=wyk.wy
  apply_inv!(wy, k, h)
end

@propagate_inbounds function InPlace.:⊘(
  wy::WYTrans{E},
  h::HouseholderTrans{E},
) where {E<:Number}
  
  apply_inv!(wy, wy.active_WY[], h)
end

@inline function apply!(
  h::HouseholderTrans{E},
  wy::WYTrans{E},
  k::Int,
) where {E<:Number}

  num_WY = wy.num_WY[]
  @boundscheck k ∈ 1:num_WY || throw_WYBlockNotAvailable(k, num_WY)

  @inbounds begin
    num_hs = wy.num_hs[k]
    wy_offsets = wy.offsets[k]

    v=reshape(h.v,length(h.v),1)

    indsh = (h.offs + 1):(h.offs + h.size)
    indswy = 1:wy.sizes[k]
    indshwy = (h.offs - wy_offsets + 1):(h.offs - wy_offsets + h.size)
  end

  @boundscheck begin
    indsh ⊆ (indswy .+ wy_offsets) || throw(WYIndexSubsetError)
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
    Y1[indshwy,:] = v

    mul!(work, W0', Y1)
    mul!(Y1, Y0, work, -conj(h.β), conj(h.β))

    wy.num_hs[k] += 1
  end
  nothing
end

@propagate_inbounds function InPlace.:⊛(
  h::HouseholderTrans{E},
  wyk::SelectWY{<:WYTrans{E}},
) where {E<:Number}

  k=wyk.block
  wy=wyk.wy

  apply!(h, wy, k)
end

@propagate_inbounds function InPlace.:⊛(
  h::HouseholderTrans{E},
  wy::WYTrans{E},
) where {E<:Number}

  apply!(h, wy, wy.active_WY[])
end

@inline function apply_inv!(
  h::HouseholderTrans{E},
  wy::WYTrans{E},
  k::Int,
) where {E<:Number}

  num_WY = wy.num_WY[]
  @boundscheck k ∈ 1:num_WY || throw_WYBlockNotAvailable(k, num_WY)

  @inbounds begin
    num_hs = wy.num_hs[k]
    wy_offsets = wy.offsets[k]

    v=reshape(h.v,length(h.v),1)

    indsh = (h.offs + 1):(h.offs + h.size)
    indswy = 1:wy.sizes[k]
    indshwy = (h.offs - wy_offsets + 1):(h.offs - wy_offsets + h.size)
  end

  @boundscheck begin
    indsh ⊆ (indswy .+ wy_offsets) || throw(WYIndexSubsetError)
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
    Y1[indshwy,:] = v

    mul!(work, W0', Y1)
    mul!(Y1, Y0, work, -h.β, h.β)

    wy.num_hs[k] += 1
  end
  nothing

end

@propagate_inbounds function InPlace.:⊘(
  h::HouseholderTrans{E},
  wyk::SelectWY{<:WYTrans{E}},
) where {E<:Number}

  k=wyk.block
  wy=wyk.wy
  apply_inv!(h, wy, k)
end

@propagate_inbounds function InPlace.:⊘(
  h::HouseholderTrans{E},
  wy::WYTrans{E},
) where {E<:Number}

  apply_inv!(h, wy, wy.active_WY[])
end

end # module
