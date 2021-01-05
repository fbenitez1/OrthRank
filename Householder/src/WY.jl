module WY

using Base: @propagate_inbounds

using LinearAlgebra
import InPlace

using ..Compute

export WYTrans, resetWY!, reworkWY!, WYIndexSubsetError

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
mutable struct WYTrans{
  E<:Number,
  AEW<:AbstractArray{E,2},
  AEY<:AbstractArray{E,2},
  AEWork<:AbstractArray{E,1},
}
  # Offset
  offs::Int
  # Size of the individual block transformation
  sizeWY::Int
  # Number of Householders.
  num_h::Int
  # Size of the full matrix transformed on the side of the
  # transformation.
  sizeA_h::Int
  # Other dimension of a.
  sizeA_other::Int
  # maximum number of Householders.
  max_k::Int
  # sizeA_h×max_k array.
  W::AEW
  # sizeA_h×max_k array.
  Y::AEY
  # sizeA_other*max_k work array.
  work::AEWork
end

"""
    WYTrans(
      ::Type{E},
      sizeA_h::Int,
      sizeA_other::Int,
      max_k::Int,
    ) where {E}

Create a new, empty WYTrans with element type `E`.
"""
function WYTrans(
  ::Type{E},
  sizeA_h::Int,
  sizeA_other::Int,
  max_k::Int,
) where {E}
  WYTrans(
    0,
    0,
    0,
    sizeA_h,
    sizeA_other,
    max_k,
    zeros(E, sizeA_h, max_k),
    zeros(E, sizeA_h, max_k),
    zeros(E, sizeA_other * max_k),
  )
end

"""
    resetWY!(
      offs::Int,
      sizeWY::Int,
      wy::WYTrans{E},
    ) where {E<:Number}

Return a WYTrans to an empty state, with an given offset and block size.
"""
@inline function resetWY!(
  offs::Int,
  sizeWY::Int,
  wy::WYTrans{E},
) where {E<:Number}
  wy.offs = offs
  wy.sizeWY = sizeWY
  wy.num_h=0
  nothing
end

"""
    reworkWY!(
      sizeA_other,
      wy::WYTrans{E},
    ) where {E<:Number}

Allocate a new work space for a WYTrans.
"""
@inline function reworkWY!(
  sizeA_other,
  wy::WYTrans{E},
) where {E<:Number}
  wy.num_h=0
  wy.sizeA_other = sizeA_other
  wy.work = zeros(E, sizeA_other * wy.max_k)
end

# Array Updates

@propagate_inbounds @inline function InPlace.:⊛(
  A::AE2,
  wy::WYTrans{E},
) where {E<:Number,AE2<:AbstractArray{E,2}}
  k = wy.num_h
  inds = (wy.offs + 1):(wy.offs + wy.sizeWY)
  ma = size(A, 1)
  work = reshape(view(wy.work, 1:ma*k), ma, k)
  W = view(wy.W, inds, 1:k)
  Y = view(wy.Y, inds, 1:k)
  A0 = view(A, :, inds)
  oneE = one(E)
  mul!(work, A0, W)
  mul!(A0, work, Y', -oneE, oneE)
end

@propagate_inbounds @inline function InPlace.:⊘(
  A::AE2,
  wy::WYTrans{E},
) where {E<:Number,AE2<:AbstractArray{E,2}}
  k = wy.num_h
  inds = (wy.offs + 1):(wy.offs + wy.sizeWY)
  ma = size(A, 1)
  work = reshape(view(wy.work, 1:ma * k), ma, k)
  W = view(wy.W, inds, 1:k)
  Y = view(wy.Y, inds, 1:k)
  A0 = view(A, :, inds)
  oneE = one(E)
  mul!(work, A0, Y)
  mul!(A0, work, W', -oneE, oneE)
end

@inline function InPlace.:⊛(
  wy::WYTrans{E},
  A::AE2,
) where {E<:Number,AE2<:AbstractArray{E,2}}
  k = wy.num_h
  inds = (wy.offs + 1):(wy.offs + wy.sizeWY)
  na = size(A, 2)
  work = reshape(view(wy.work, 1:na, 1:k),k,na)
  W = view(wy.W, inds, 1:k)
  Y = view(wy.Y, inds, 1:k)
  A0 = view(A, inds, :)
  oneE = one(E)
  mul!(work, Y', A0)
  mul!(A0, W, work, -oneE, oneE)
end

@propagate_inbounds @inline function InPlace.:⊘(
  wy::WYTrans{E},
  A::AE2,
) where {E<:Number,AE2<:AbstractArray{E,2}}
  k = wy.num_h
  inds = (wy.offs + 1):(wy.offs + wy.sizeWY)
  na = size(A, 2)
  work = reshape(view(wy.work, 1:na*k),k,na)
  W = view(wy.W, inds, 1:k)
  Y = view(wy.Y, inds, 1:k)
  A0 = view(A, inds, :)
  oneE = one(E)
  zeroE = zero(E)
  mul!(work, W', A0)
  mul!(A0, Y, work, -oneE, oneE)
end

# Updating functions for adding a Householder.

struct WYIndexSubsetError <: Exception end

@propagate_inbounds @inline function InPlace.:⊛(
  wy::WYTrans{E},
  h::HouseholderTrans{E},
) where {E<:Number}

  k = wy.num_h
  v=reshape(h.v,length(h.v),1)
  
  indsh = (h.offs + 1):(h.offs + h.size)
  indswy = (wy.offs + 1):(wy.offs + wy.sizeWY)
  indshwy = (h.offs - wy.offs + 1):(h.offs - wy.offs + h.size)

  indsh ⊆ indswy || throw(WYIndexSubsetError)

  work = reshape(view(wy.work, 1:k),k,1)

  W0 = view(wy.W, indswy, 1:k)
  W1 = view(wy.W, indswy, (k + 1):(k + 1))
  Y0 = view(wy.Y, indswy, 1:k)
  Y1 = view(wy.Y, indswy, (k + 1):(k + 1))

  W1[:,:] .= zero(E)
  Y1[:,:] .= zero(E)
  W1[indshwy,:] = v
  Y1[indshwy,:] = v

  mul!(work, Y0', W1)
  mul!(W1, W0, work, -h.β, h.β)

  wy.num_h = wy.num_h + 1
end

@propagate_inbounds @inline function InPlace.:⊘(
  wy::WYTrans{E},
  h::HouseholderTrans{E},
) where {E<:Number}

  k = wy.num_h
  v=reshape(h.v,length(h.v),1)
  
  indsh = (h.offs + 1):(h.offs + h.size)
  indswy = (wy.offs + 1):(wy.offs + wy.sizeWY)
  indshwy = (h.offs - wy.offs + 1):(h.offs - wy.offs + h.size)

  indsh ⊆ indswy || throw(WYIndexSubsetError)

  work = reshape(view(wy.work, 1:k),k,1)

  W0 = view(wy.W, indswy, 1:k)
  W1 = view(wy.W, indswy, k + 1:k+1)
  Y0 = view(wy.Y, indswy, 1:k)
  Y1 = view(wy.Y, indswy, k + 1:k+1)

  W1[:,:] .= zero(E)
  Y1[:,:] .= zero(E)
  W1[indshwy,:] = v
  Y1[indshwy,:] = v

  mul!(work, Y0', W1)
  mul!(W1, W0, work, -conj(h.β), conj(h.β))

  wy.num_h = wy.num_h + 1
end

@propagate_inbounds @inline function InPlace.:⊛(
  h::HouseholderTrans{E},
  wy::WYTrans{E},
) where {E<:Number}

  k = wy.num_h
  v=reshape(h.v,length(h.v),1)
  
  indsh = (h.offs + 1):(h.offs + h.size)
  indswy = (wy.offs + 1):(wy.offs + wy.sizeWY)
  indshwy = (h.offs - wy.offs + 1):(h.offs - wy.offs + h.size)

  indsh ⊆ indswy || throw(WYIndexSubsetError)

  work = reshape(view(wy.work, 1:k),k,1)

  W0 = view(wy.W, indswy, 1:k)
  W1 = view(wy.W, indswy, k + 1:k+1)
  Y0 = view(wy.Y, indswy, 1:k)
  Y1 = view(wy.Y, indswy, k + 1:k+1)

  W1[:,:] .= zero(E)
  Y1[:,:] .= zero(E)
  W1[indshwy,:] = v
  Y1[indshwy,:] = v

  mul!(work, W0', Y1)
  mul!(Y1, Y0, work, -conj(h.β), conj(h.β))

  wy.num_h = wy.num_h + 1
end

@propagate_inbounds @inline function InPlace.:⊘(
  h::HouseholderTrans{E},
  wy::WYTrans{E},
) where {E<:Number}

  k = wy.num_h
  v = reshape(h.v, length(h.v), 1)

  indsh = (h.offs + 1):(h.offs + h.size)
  indswy = (wy.offs + 1):(wy.offs + wy.sizeWY)
  indshwy = (h.offs - wy.offs + 1):(h.offs - wy.offs + h.size)

  indsh ⊆ indswy || throw(WYIndexSubsetError)

  work = reshape(view(wy.work, 1:k),k,1)

  W0 = view(wy.W, indswy, 1:k)
  W1 = view(wy.W, indswy, (k + 1):(k + 1))
  Y0 = view(wy.Y, indswy, 1:k)
  Y1 = view(wy.Y, indswy, (k + 1):(k + 1))

  W1[:, :] .= zero(E)
  Y1[:, :] .= zero(E)
  W1[indshwy, :] = v
  Y1[indshwy, :] = v

  mul!(work, W0', Y1)
  mul!(Y1, Y0, work, -h.β, h.β)

  wy.num_h = wy.num_h + 1
end

end # module
