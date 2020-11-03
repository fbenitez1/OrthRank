module WY

using LinearAlgebra
import InPlace

if isdefined(@__MODULE__, :LanguageServer)
  include("src/Compute.jl")
  using .Compute
else
  using Householder.Compute
end

export WYTrans, resetWY!, reworkWY!, WYIndexSubsetError

"""

  I - W Yᴴ

"""
mutable struct WYTrans{
  E<:Number,
  AEW<:AbstractArray{E,2},
  AEY<:AbstractArray{E,2},
  AEWork<:AbstractArray{E,1},
}
  offs::Int # Offset
  sizeWY::Int # Size of the individual block transformation
  num_h::Int # Number of Householders.
  sizeA_h::Int # Size of the full matrix transformed on the
  # side of the transformation.
  sizeA_other::Int # Other dimension of a.
  max_k::Int # maximum number of Householders.
  W::AEW # sizeA_h×max_k array.
  Y::AEY # sizeA_h×max_k array.
  work::AEWork # sizeA_other*max_k work array.
end

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
allocate a new work space
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

@inline function InPlace.:⊛(
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

@inline function InPlace.:⊘(
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

@inline function InPlace.:⊘(
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

@inline function InPlace.:⊛(
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

@inline function InPlace.:⊘(
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

@inline function InPlace.:⊛(
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

@inline function InPlace.:⊘(
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
