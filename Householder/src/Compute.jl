module Compute

using Printf
using Random

using LinearAlgebra
import InPlace
using LoopVectorization

export HouseholderTrans,
  update_norm,
  lhouseholder,
  rhouseholder,
  householder,
  column_nonzero!,
  row_nonzero!

"""

# HouseholderTrans

    HouseholderTrans{E,AEV<:AbstractArray{E,1},AEW<:AbstractArray{E,1}}

A Householder data structure.

## `h ⊛ A` is equivalent to

    v = reshape(h.v, h.size, 1)
    A₁ = A[(h.offs + 1) : (h.offs+h.size), :]
    A₁ = A₁ - h.β * v[1:h.size,1] * ( v[1:h.size,1]' * A₁ )

## `h ⊘ A` is equivalent to

    v = reshape(h.v, h.size, 1)
    A₁ = A[(h.offs + 1) : (h.offs+h.size), :]
    A₁ = A₁ - conj(h.β) * v[1:h.size,1] * ( v[1:h.size,1]' * A₁ )

## `A ⊛ h` is equivalent to

    v = reshape(h.v, h.size, 1)
    A₁ = A[:, (h.offs + 1) : (h.offs+h.size)]
    A₁ = A₁ - h.β * ( A₁ * v[1:h.size,1] ) * v[1:h.size,1]'

## `A ⊘ h` is equivalent to

    v = reshape(h.v, h.size, 1)
    A₁ = A[:, (h.offs + 1) : (h.offs+h.size)]
    A₁ = A₁ - conj(h.β) * ( A₁ * v[1:h.size,1] ) * v[1:h.size,1]'

"""
struct HouseholderTrans{E,AEV<:AbstractArray{E,1},AEW<:AbstractArray{E,1}}
  β::E
  # Householder vector.
  v::AEV
  # element to leave nonzero.
  l::Int64
  # size of transformation.
  size::Int64
  # offset for applying to a matrix.
  offs::Int64
  # Size = opposite side size of A.  For m × n A:
  # h ⊛ A requires work space of size n.
  # A ⊛ h requires work space of size m.
  work::AEW
end

InPlace.product_side(::Type{<:HouseholderTrans}, _) = InPlace.LeftProduct()
InPlace.product_side(::Type{<:HouseholderTrans}, _, _) = InPlace.LeftProduct()
InPlace.product_side(_, ::Type{<:HouseholderTrans}) = InPlace.RightProduct()
InPlace.product_side(_, _, ::Type{<:HouseholderTrans}) = InPlace.RightProduct()

function Random.rand!(rng::AbstractRNG, h::HouseholderTrans)
  @views begin
    rand!(rng, h.v[1:h.size])
    x = norm(h.v[1:h.size])
    if !iszero(h.β)
      h.v[1:h.size] .= h.v[1:h.size] .* (sqrt(2/h.β) / x)
    end
  end
end

function Random.rand!(h::HouseholderTrans)
  rand!(Random.default_rng(), h)
end

@inline function update_norm(a::R, b::E) where {R<:Real,E<:Union{R,Complex{R}}}
  a_abs = abs(a)
  b_abs = abs(b)
  z = max(a_abs, b_abs)
  iszero(z) ? z : z * sqrt((a_abs / z)^2 + (b_abs / z)^2)
end


@inline function maybe_complex(::Type{E}, a::E, ::E) where {E<:Real}
  a
end

@inline function maybe_complex(
  ::Type{E},
  a::R,
  b::R,
) where {R<:Real,E<:Complex{R}}
  complex(a, b)
end

"""

  Compute a Householder such that for h = I - β v vᴴ,
  hᴴ * a = ||a|| eₗ.

"""
function lhouseholder(
  a::AbstractArray{E,1},
  l::Int64,
  offs::Int64,
  work::AbstractArray{E,1}
) where {E<:Number}

  m = length(a)
  a1 = a[l]
  if m == 1
    a[1] = 1
    sign_a1 = iszero(a1) ? one(a1) : sign(a1)
    β = (sign_a1 - one(a1)) / sign_a1
    HouseholderTrans(conj(β), a, l, m, offs, work)
  else
    norm_a2 = @views update_norm(norm(a[1:(l - 1)]), norm(a[(l + 1):m]))
    norm_a = update_norm(norm_a2, a1)
    if iszero(norm_a)
      HouseholderTrans(zero(E), a, l, m, offs, work)
    elseif iszero(norm_a2)
      a[l] = 1
      sign_a1 = iszero(a1) ? one(a1) : sign(a1)
      β = (sign_a1 - one(a1)) / sign_a1
      HouseholderTrans(conj(β), a, l, m, offs, work)
    else
      alpha = if real(a1) <= 0
        a1 - norm_a
      else
        a1i = imag(a1)
        a1r = real(a1)
        x = update_norm(norm_a2, a1i)
        y = a1r + norm_a
        z = y / x
        maybe_complex(E, -x, a1i * z) / z
      end
      β = -conj(alpha) / norm_a
      a[l] = one(a1)
      rdiv!(view(a,1:(l-1)), alpha)
      rdiv!(view(a,(l+1):m), alpha)
      HouseholderTrans(conj(β), a, l, m, offs, work)
    end
  end
end

Base.@propagate_inbounds function lhouseholder(
  a::AbstractArray{E,1},
  l::Int64,
  offs::Int64,
  work_size::Int64,
) where {E<:Number}
  work = zeros(E, work_size)
  lhouseholder(a,l,offs,work)
end

"""

  Compute a Householder such that for h = I - β v vᴴ,
  a * h = ||a|| eₗᵀ.

"""
function rhouseholder(
  a::AbstractArray{E,1},
  l::Int64,
  offs::Int64,
  work::AbstractArray{E,1},
) where {E<:Number}
  m = length(a)
  a1 = a[l]
  if m == 1
    a[1] = 1
    sign_a1 = iszero(a1) ? one(a1) : sign(a1)
    β = (sign_a1 - one(a1)) / sign_a1
    conj!(a)
    HouseholderTrans(β, a, l, m, offs, work)
  else
    norm_a2 = @views update_norm(norm(a[1:(l - 1)]), norm(a[(l + 1):m]))
    norm_a = update_norm(norm_a2, a1)
    if iszero(norm_a)
      HouseholderTrans(zero(E), a, l, m, offs, work)
    elseif iszero(norm_a2)
      a[l] = 1
      sign_a1 = iszero(a1) ? one(a1) : sign(a1)
      β = (sign_a1 - one(a1)) / sign_a1
      conj!(a)
      HouseholderTrans(β, a, l, m, offs, work)
    else
      alpha = if real(a1) <= 0
        a1 - norm_a
      else
        a1i = imag(a1)
        a1r = real(a1)
        x = update_norm(norm_a2, a1i)
        y = a1r + norm_a
        z = y / x
        maybe_complex(E, -x, a1i * z) / z
      end

      β = -conj(alpha) / norm_a
      a[l] = one(a1)
      rdiv!(view(a,1:(l-1)), alpha)
      rdiv!(view(a,(l+1):m), alpha)
      conj!(a)
      HouseholderTrans(β, a, l, m, offs, work)
    end
  end
end

Base.@propagate_inbounds function rhouseholder(
  a::AbstractArray{E,1},
  l::Int64,
  offs::Int64,
  work_size::Int64,
) where {E<:Number}
  work = zeros(E, work_size)
  rhouseholder(a,l,offs,work)
end

# No keywords, explicit vector and work arrays.
Base.@propagate_inbounds function householder(
  A::AbstractArray{E,2},
  js::UnitRange{Int},
  k::Int,
  nonzero_index::Int,
  offset::Int,
  vector::AbstractArray{E,1},
  work::AbstractArray{E,1},
) where {E<:Number}
  ljs=length(js)
  @views begin
    vjs = vector[1:ljs]
    vjs[:] = A[js,k]
  end
  lhouseholder(vjs, nonzero_index, offset, work)
end

Base.@propagate_inbounds function householder(
  A::AbstractArray{E,2},
  js::UnitRange{Int},
  k::Int,
  vector::AbstractArray{E,1},
  work::AbstractArray{E,1};
  nonzero_index = 1,
  offset = first(js)-1,
) where {E<:Number}
  householder(A, js, k, nonzero_index, offset, vector, work)
end

Base.@propagate_inbounds function householder(
  A::AbstractArray{E,2},
  js::UnitRange{Int},
  k::Int,
  work::AbstractArray{E,1};
  nonzero_index = 1,
  offset = first(js) - 1,
) where {E<:Number}
  vector=zeros(E,length(js))
  householder(A, js, k, nonzero_index, offset, vector, work)
end

Base.@propagate_inbounds function householder(
  A::AbstractArray{E,2},
  js::UnitRange{Int},
  k::Int,
  vector::AbstractArray{E,1},
  work_size::Int;
  nonzero_index::Int = 1,
  offset::Int = first(js)-1,
) where {E<:Number}
  work=zeros(E,work_size)
  householder(A, js, k, nonzero_index, offset, vector, work)
end

Base.@propagate_inbounds function householder(
  A::AbstractArray{E,2},
  js::UnitRange{Int},
  k::Int,
  work_size::Int;
  nonzero_index::Int = 1,
  offset::Int = first(js)-1,
) where {E<:Number}
  work=zeros(E,work_size)
  vector=zeros(E,length(js))
  householder(A, js, k, nonzero_index, offset, vector, work)
end

Base.@propagate_inbounds function householder(
  A::AbstractArray{E,2},
  j::Int,
  ks::UnitRange{Int},
  nonzero_index::Int,
  offset::Int,
  vector::AbstractArray{E,1},
  work::AbstractArray{E,1},
) where {E<:Number}
  lks=length(ks)
  @views begin
    vks = vector[1:lks]
    vks[:] = A[j,ks]
  end
  lhouseholder(vks, nonzero_index, offset, work)
end

Base.@propagate_inbounds function householder(
  A::AbstractArray{E,2},
  j::Int,
  ks::UnitRange{Int},
  vector::AbstractArray{E,1},
  work::AbstractArray{E,1};
  nonzero_index = 1,
  offset = first(ks) - 1,
) where {E<:Number}
  householder(A, j, ks, nonzero_index, offset, vector, work)
end

Base.@propagate_inbounds function householder(
  A::AbstractArray{E,2},
  j::Int,
  ks::UnitRange{Int},
  work::AbstractArray{E,1};
  nonzero_index = 1,
  offset = first(ks) - 1,
) where {E<:Number}
  vector=zeros(E,length(ks))
  householder(A, j, ks, nonzero_index, offset, vector, work)
end

Base.@propagate_inbounds function householder(
  A::AbstractArray{E,2},
  j::Int,
  ks::UnitRange{Int},
  vector::AbstractArray{E,1},
  work_size::Int;
  nonzero_index = 1,
  offset = first(ks) - 1,
) where {E<:Number}
  work=zeros(E,work_size)
  householder(A, j, ks, nonzero_index, offset, vector, work)
end

Base.@propagate_inbounds function householder(
  A::AbstractArray{E,2},
  j::Int,
  ks::UnitRange{Int},
  work_size::Int;
  nonzero_index = 1,
  offset = first(ks) - 1,
) where {E<:Number}
  work=zeros(E,work_size)
  vector=zeros(E,length(ks))
  householder(A, j, ks, nonzero_index, offset, vector, work)
end

@inline function column_nonzero!(
  A::AbstractArray{E,2},
  l::Int,
  k::Int,
) where {E<:Number}
  m = size(A,1)
  A[1:(l - 1), k] .= zero(E)
  A[(l + 1):m, k] .= zero(E)
end

@inline function row_nonzero!(
  A::AbstractArray{E,2},
  j::Int,
  l::Int,
) where {E<:Number}
  n = size(A,2)
  A[j, 1:(l - 1)] .= zero(E)
  A[j, (l + 1):n] .= zero(E)
end

include("./Compute/ApplyHouseholder.jl")

end # module
