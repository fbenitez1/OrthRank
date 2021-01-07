module WY

using Base: @propagate_inbounds
using Printf

using LinearAlgebra
import InPlace

using ..Compute

export WYTrans,
  resetWYBlock!, resetWYBlocks!, reworkWY!, WYIndexSubsetError, SelectBlock

"""

# WYTrans

## `wy ⊛ A` is equivalent to

    A₁=A[wy.offs+1:wy.offs+wy.sizeWY, :]
    W₁=W[wy.offs+1:wy.offs+wy.sizeWY, 1:wy.num_h]
    Y₁=Y[wy.offs+1:wy.offs+wy.sizeWY, 1:wy.num_h]
    A₁ = A₁ - W₁ Y₁ᴴ A₁

## `wy ⊘ A` is equivalent to

    A₁=A[wy.offs+1:wy.offs+wy.sizeWY, :]
    W₁=W[wy.offs+1:wy.offs+wy.sizeWY, 1:wy.num_h]
    Y₁=Y[wy.offs+1:wy.offs+wy.sizeWY, 1:wy.num_h]
    A₁ = A₁ - Y₁ W₁ᴴ A₁

## `A ⊛ wy` is equivalent to

    A₁=A[:, wy.offs+1:wy.offs+wy.sizeWY]
    W₁=W[wy.offs+1:wy.offs+wy.sizeWY, 1:wy.num_h]
    Y₁=Y[wy.offs+1:wy.offs+wy.sizeWY, 1:wy.num_h]
    A₁ = A₁ - A₁ W₁ Y₁ᴴ

## `A ⊘ wy` is equivalent to

    A₁=A[:, wy.offs+1:wy.offs+wy.sizeWY]
    W₁=W[wy.offs+1:wy.offs+wy.sizeWY, 1:wy.num_h]
    Y₁=Y[wy.offs+1:wy.offs+wy.sizeWY, 1:wy.num_h]
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
  max_num_blocks::Int
  # Maximum block size.
  max_block_size::Int
  # Other dimension of A (determines the size of the workspace).
  A_other_dim::Int
  # maximum number of Householders.
  max_num_hs::Int
  # Number of blocks
  num_blocks::Ref{Int}
  # Offset into A.
  offs::AI
  # Size of the individual block transformations
  sizeWYs::AI
  # Number of Householders in each WY.
  num_hs::AI
  # sizeA_h×max_k array.
  W::AEW
  # sizeA_h×max_k array.
  Y::AEY
  # sizeA_other*max_k work array.
  work::AEWork
end

struct SelectBlock{T}
  wy::T
  block::Int
end
  

"""
    WYTrans(
      ::Type{E},
      max_num_blocks::Int,
      max_block_size::Int,
      A_other_dim::Int,
      max_num_hs::Int,
    ) where {E}

Create a new, empty WYTrans with element type `E`.  Defaults to including
all blocks.

"""
function WYTrans(
  ::Type{E},
  max_num_blocks::Int,
  max_block_size::Int,
  A_other_dim::Int,
  max_num_hs::Int,
) where {E<:Number}
  WYTrans(
    max_num_blocks,
    max_block_size,
    A_other_dim,
    max_num_hs,
    Ref(max_num_blocks), # include all possible blocks.
    zeros(Int,max_num_blocks),
    zeros(Int,max_num_blocks),
    zeros(Int,max_num_blocks),
    zeros(E, max_block_size, max_num_hs, max_num_blocks),
    zeros(E, max_block_size, max_num_hs, max_num_blocks),
    zeros(E, A_other_dim * max_num_hs),
  )
end

struct WYBlockNotAvailable <: Exception
  message::String
end

throw_WYBlockNotAvailable(block, num_blocks) = 
  throw(WYBlockNotAvailable(@sprintf(
    "Block %d is not available in WYTrans with num_blocks = %d",
    block,
    num_blocks
  )))

"""
    resetWYBlock!(
      k::Int
      offs::Int,
      sizeWY::Int,
      wy::WYTrans{E},
    ) where {E<:Number}

Return a block of a WYTrans to an empty state, with an given offset
and block size.

"""
@inline function resetWYBlock!(
  k::Int,
  offs::Int,
  sizeWY::Int,
  wy::WYTrans,
)
  @boundscheck begin
    k ∈ 1:wy.num_blocks[] || throw_WYBlockNotAvailable(k, wy.num_blocks[])
    sizeWY <= wy.max_block_size || throw(DimensionMismatch(@sprintf(
      "Requested dimension %d of a WY block exceeds the maximum block size %d.",
      sizeWY,
      wy.max_block_size
    )))
 end
  wy.offs[k] = offs
  wy.sizeWYs[k] = sizeWY
  wy.num_hs[k] = 0
  nothing
end

@inline function resetWYBlocks!(
  offs_sizes,
  wy::WYTrans,
)
  num_blocks = length(offs_sizes)
  @boundscheck num_blocks <= wy.max_num_blocks ||
               throw(WYBlockNotAvailable(@sprintf(
    "Block %d is not available in WYTrans with max_num_blocks = %d",
    num_blocks,
    wy.max_num_blocks
  )))

  wy.num_blocks[]=num_blocks
  k=1
  for (offs,sizeWY) ∈ offs_sizes
    @boundscheck(
      sizeWY <= wy.max_block_size || throw(DimensionMismatch(@sprintf(
        "Requested dimension %d of a WY block exceeds the maximum block size %d.",
        sizeWY,
        wy.max_block_size
      )))
    )
    wy.offs[k] = offs
    wy.sizeWYs[k] = sizeWY
    wy.num_hs[k] = 0
    k += 1
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
    wy.max_num_blocks,
    wy.max_block_size,
    A_other_dim,
    wy.max_num_hs,
    wy.num_blocks,
    wy.offs,
    wy.sizeWYs,
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

@inline function InPlace.:⊛(
  A::AE2,
  wyk::SelectBlock{<:WYTrans{E}},
) where {E<:Number,AE2<:AbstractArray{E,2}}
  k=wyk.block
  wy=wyk.wy
  num_blocks = wy.num_blocks[]
  @boundscheck k ∈ 1:num_blocks || throw_WYBlockNotAvailable(k, num_blocks)
  inds = 1:wy.sizeWYs[k]
  offs = wy.offs[k]
  (ma,na) = size(A)
  num_hs= wy.num_hs[k]

  @boundscheck begin

    inds .+ offs ⊆ 1:na || throw_ColumnRange_DimensionMismatch(ma, na, inds)

    length(wy.work) >= ma * num_hs ||
      throw_WorkSizeError(ma, na, ma * num_hs, length(wy.work))

  end

  @inbounds begin
    @views begin
      work = reshape(wy.work[1:ma*num_hs], ma, num_hs)
      W = wy.W[inds, 1:num_hs, k]
      Y = wy.Y[inds, 1:num_hs, k]
      A0 = A[:,inds .+ offs]
    end
    oneE = one(E)
    mul!(work, A0, W)
    mul!(A0, work, Y', -oneE, oneE)
  end
end

@propagate_inbounds @inline function InPlace.:⊛(
  A::AE2,
  wy::WYTrans{E},
) where {E<:Number,AE2<:AbstractArray{E,2}}
  for k ∈ 1:wy.num_blocks[]
    A ⊛ SelectBlock(wy, k)
  end
end

@inline function InPlace.:⊘(
  A::AE2,
  wyk::SelectBlock{<:WYTrans{E}},
) where {E<:Number,AE2<:AbstractArray{E,2}}
  k=wyk.block
  wy=wyk.wy
  num_blocks = wy.num_blocks[]
  @boundscheck k ∈ 1:num_blocks || throw_WYBlockNotAvailable(k, num_blocks)
  num_hs = wy.num_hs[k]
  offs = wy.offs[k]
  inds = 1:wy.sizeWYs[k]
  (ma,na) = size(A)

  @boundscheck begin

    inds .+ offs ⊆ 1:na || throw_ColumnRange_DimensionMismatch(ma, na, inds)

    length(wy.work) >= ma * num_hs ||
      throw_WorkSizeError(ma, na, ma * num_hs, length(wy.work))

  end

  @inbounds begin
    @views begin
      work = reshape(wy.work[1:ma*num_hs], ma, num_hs)
      W = wy.W[inds, 1:num_hs, k]
      Y = wy.Y[inds, 1:num_hs, k]
      A0 = A[:, inds .+ offs]
    end
    oneE = one(E)
    mul!(work, A0, Y)
    mul!(A0, work, W', -oneE, oneE)
  end
end

@propagate_inbounds @inline function InPlace.:⊘(
  A::AE2,
  wy::WYTrans{E},
) where {E<:Number,AE2<:AbstractArray{E,2}}
  for k ∈ 1:wy.num_blocks[]
    A ⊘ SelectBlock(wy, k)
  end
end

@inline function InPlace.:⊛(
  wyk::SelectBlock{<:WYTrans{E}},
  A::AE2,
) where {E<:Number,AE2<:AbstractArray{E,2}}
  k=wyk.block
  wy=wyk.wy
  num_blocks = wy.num_blocks[]
  @boundscheck k ∈ 1:num_blocks || throw_WYBlockNotAvailable(k, num_blocks)
  num_hs = wy.num_hs[k]
  offs = wy.offs[k]
  inds = 1:wy.sizeWYs[k]
  (ma, na) = size(A)

  @boundscheck begin

    inds .+ offs ⊆ 1:ma || throw_RowRange_DimensionMismatch(ma, na, inds)

    length(wy.work) >= na * num_hs ||
      throw_WorkSizeError(ma, na, na * num_hs, length(wy.work))

  end

  @inbounds begin
    @views begin
      work = reshape(wy.work[1:na*num_hs],num_hs,na)
      W = wy.W[inds, 1:num_hs, k]
      Y = wy.Y[inds, 1:num_hs, k]
      A0 = A[inds .+ offs, :]
    end
    oneE = one(E)
    mul!(work, Y', A0)
    mul!(A0, W, work, -oneE, oneE)
  end
end

@propagate_inbounds @inline function InPlace.:⊛(
  wy::WYTrans{E},
  A::AE2,
) where {E<:Number,AE2<:AbstractArray{E,2}}
  for k ∈ 1:wy.num_blocks[]
    SelectBlock(wy,k) ⊛ A
  end
end

@inline function InPlace.:⊘(
  wyk::SelectBlock{<:WYTrans{E}},
  A::AE2,
) where {E<:Number,AE2<:AbstractArray{E,2}}
  k=wyk.block
  wy=wyk.wy
  num_blocks = wy.num_blocks[]
  @boundscheck k ∈ 1:num_blocks || throw_WYBlockNotAvailable(k, num_blocks)
  num_hs = wy.num_hs[k]
  offs = wy.offs[k]
  inds = 1:wy.sizeWYs[k]
  (ma, na) = size(A)

  @boundscheck begin

    inds .+ offs ⊆ 1:ma || throw_RowRange_DimensionMismatch(ma, na, inds)

    length(wy.work) >= na * num_hs ||
      throw_WorkSizeError(ma, na, na * num_hs, length(wy.work))

  end

  @inbounds begin
    @views begin
      work = reshape(wy.work[1:na*num_hs],num_hs,na)
      W = wy.W[inds, 1:num_hs, k]
      Y = wy.Y[inds, 1:num_hs, k]
      A0 = A[inds .+ offs, :]
    end
    oneE = one(E)
    mul!(work, W', A0)
    mul!(A0, Y, work, -oneE, oneE)
  end
end

@propagate_inbounds @inline function InPlace.:⊘(
  wy::WYTrans{E},
  A::AE2,
) where {E<:Number,AE2<:AbstractArray{E,2}}
  for k ∈ 1:wy.num_blocks[]
    SelectBlock(wy,k) ⊘ A
  end
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


@inline function InPlace.:⊛(
  wyk::SelectBlock{<:WYTrans{E}},
  h::HouseholderTrans{E},
) where {E<:Number}

  k=wyk.block
  wy=wyk.wy
  num_blocks = wy.num_blocks[]
  @boundscheck k ∈ 1:num_blocks || throw_WYBlockNotAvailable(k, num_blocks)

  num_hs = wy.num_hs[k]
  wy_offs=wy.offs[k]
  v=reshape(h.v,length(h.v),1)
  indsh = (h.offs + 1):(h.offs + h.size)
  indswy = 1:wy.sizeWYs[k]
  indshwy = (h.offs - wy_offs + 1):(h.offs - wy_offs + h.size)

  @boundscheck begin
    indsh ⊆ (indswy .+ wy_offs) || throw(WYIndexSubsetError)
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
end

@inline function InPlace.:⊘(
  wyk::SelectBlock{<:WYTrans{E}},
  h::HouseholderTrans{E},
) where {E<:Number}

  k=wyk.block
  wy=wyk.wy
  num_blocks = wy.num_blocks[]
  @boundscheck k ∈ 1:num_blocks || throw_WYBlockNotAvailable(k, num_blocks)

  num_hs = wy.num_hs[k]
  wy_offs=wy.offs[k]

  v=reshape(h.v,length(h.v),1)
  
  indsh = (h.offs + 1):(h.offs + h.size)
  indswy = 1:wy.sizeWYs[k]
  indshwy = (h.offs - wy_offs + 1):(h.offs - wy_offs + h.size)

  @boundscheck begin
    indsh ⊆ (indswy .+ wy_offs) || throw(WYIndexSubsetError)
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
end

@inline function InPlace.:⊛(
  h::HouseholderTrans{E},
  wyk::SelectBlock{<:WYTrans{E}},
) where {E<:Number}

  k=wyk.block
  wy=wyk.wy
  num_blocks = wy.num_blocks[]
  @boundscheck k ∈ 1:num_blocks || throw_WYBlockNotAvailable(k, num_blocks)

  num_hs = wy.num_hs[k]
  wy_offs = wy.offs[k]

  v=reshape(h.v,length(h.v),1)
  
  indsh = (h.offs + 1):(h.offs + h.size)
  indswy = 1:wy.sizeWYs[k]
  indshwy = (h.offs - wy_offs + 1):(h.offs - wy_offs + h.size)

  @boundscheck begin
    indsh ⊆ (indswy .+ wy_offs) || throw(WYIndexSubsetError)
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
end

@inline function InPlace.:⊘(
  h::HouseholderTrans{E},
  wyk::SelectBlock{<:WYTrans{E}},
) where {E<:Number}

  k=wyk.block
  wy=wyk.wy
  num_blocks = wy.num_blocks[]
  @boundscheck k ∈ 1:num_blocks || throw_WYBlockNotAvailable(k, num_blocks)

  num_hs = wy.num_hs[k]
  wy_offs = wy.offs[k]

  v=reshape(h.v,length(h.v),1)
  
  indsh = (h.offs + 1):(h.offs + h.size)
  indswy = 1:wy.sizeWYs[k]
  indshwy = (h.offs - wy_offs + 1):(h.offs - wy_offs + h.size)

  @boundscheck begin
    indsh ⊆ (indswy .+ wy_offs) || throw(WYIndexSubsetError)
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
end

end # module
