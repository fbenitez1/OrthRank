module Compute

export HouseholderTrans,
  update_norm, lhouseholder, rhouseholder, column_nonzero!, row_nonzero!

using LinearAlgebra
import InPlace

"""

A Householder data structure: h ⊛ A is equivalent to

  A[h.j:h.j+h.size] = A[h.j:h.j+h.size] - h.beta * h.v * (h.v' * A[h.j:h.j+h.size])

"""
struct HouseholderTrans{E,AE}
  β::E
  v::AE          # Householder vector.
  l::Int64       # element to leave nonzero.
  size::Int64    # size of transformation.
  offs::Int64    # offset for applying to a matrix.
  # Size = opposite side size of A.  For m × n A:
  # h ⊛ A requires work space of size n.
  # A ⊛ h requires work space of size m.
  work::AE       
end

@inline function update_norm(a::R, b::E) where {R<:Real,E<:Union{R,Complex{R}}}
  a_abs = abs(a)
  b_abs = abs(b)
  z = max(a_abs, b_abs)
  z * sqrt((a_abs / z)^2 + (b_abs / z)^2)
end


@inline function maybe_complex(::Type{E}, a::E, b::E) where {E<:Real}
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
  a::A,
  l::Int64,
  offs::Int64,
  work::A
) where {R<:Real,E<:Union{R,Complex{R}},A<:AbstractArray{E,1}}
  m = length(a)
  a1 = a[l]
  if m == 1
    a[1] = 1
    sign_a1 = a1 == zero(a1) ? one(a1) : sign(a1)
    β = (sign_a1 - one(a1)) / sign_a1
    HouseholderTrans(conj(β), a, l, m, offs, work)
  else
    norm_a2 = update_norm(norm(view(a, 1:(l - 1))), norm(view(a,(l + 1):m)))
    norm_a = update_norm(norm_a2, a1)
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
    HouseholderTrans(conj(β), a, l, m, offs,work)
  end
end

function lhouseholder(
  a::A,
  l::Int64,
  offs::Int64,
  work_size::Int64,
) where {R<:Real,E<:Union{R,Complex{R}},A<:AbstractArray{E,1}}
  work = zeros(E, work_size)
  lhouseholder(a,l,offs,work)
end

"""

  Compute a Householder such that for h = I - β v vᴴ,
  a * h = ||a|| eₗᵀ.

"""
function rhouseholder(
  a::A,
  l::Int64,
  offs::Int64,
  work::A,
) where {R<:Real,E<:Union{R,Complex{R}},A<:AbstractArray{E,1}}
  m = length(a)
  a1 = a[l]
  if m == 1
    a[1] = 1
    sign_a1 = a1 == zero(a1) ? one(a1) : sign(a1)
    β = (sign_a1 - one(a1)) / sign_a1
    conj!(a)
    HouseholderTrans(β, a, l, m, offs, work)
  else
    norm_a2 = update_norm(norm(view(a,1:(l - 1))), norm(view(a,(l + 1):m)))
    norm_a = update_norm(norm_a2, a1)
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

function rhouseholder(
  a::A,
  l::Int64,
  offs::Int64,
  work_size::Int64,
) where {R<:Real,E<:Union{R,Complex{R}},A<:AbstractArray{E,1}}
  work = zeros(E, work_size)
  rhouseholder(a,l,offs,work)
end

@inline function column_nonzero!(
  a::AbstractArray{E,2},
  l::Int,
  k::Int,
) where {E<:Number}
  a[1:(l - 1), k] .= zero(E)
  a[(l + 1):end, k] .= zero(E)
end

@inline function row_nonzero!(
  a::AbstractArray{E,2},
  j::Int,
  l::Int,
) where {E<:Number}
  a[j, 1:(l - 1)] .= zero(E)
  a[j, (l + 1):end] .= zero(E)
end
  
@inline function InPlace.:⊛(
  h::HouseholderTrans{E,AE1},
  a::AE2,
) where {E<:Number,AE1<:AbstractArray{E,1},AE2<:AbstractArray{E,2}}
  m = h.size
  na = size(a,2)
  v = reshape(h.v, m, 1)
  offs = h.offs
  for k ∈ 1:na
    x = zero(E)
    for j ∈ 1:m
      x = x + conj(v[j]) * a[offs+j,k]
    end
    for j ∈ 1:m
      a[offs+j,k] -= h.β * v[j] * x
    end
  end
  nothing
end

@inline function InPlace.:⊘(
  h::HouseholderTrans{E,AE1},
  a::AE2,
) where {E<:Number,AE1<:AbstractArray{E,1},AE2<:AbstractArray{E,2}}
  m = h.size
  na = size(a,2)
  v = reshape(h.v, m, 1)
  offs = h.offs
  for k ∈ 1:na
    x = zero(E)
    for j ∈ 1:m
      x = x + conj(v[j]) * a[offs+j,k]
    end
    for j ∈ 1:m
      a[offs+j,k] -= conj(h.β) * v[j] * x
    end
  end
  nothing
end


@inline function InPlace.:⊛(
  a::AE2,
  h::HouseholderTrans{E,AE1},
) where {E<:Number,AE1<:AbstractArray{E,1},AE2<:AbstractArray{E,2}}
  m = h.size
  v = reshape(h.v, m, 1)
  offs = h.offs
  work = h.work
  ma = size(a,1)
  work[1:ma] .= zero(E)
  for k ∈ 1:m
    for j ∈ 1:ma
      work[j] += a[j,k+offs] * v[k]
    end
  end
  for k ∈ 1:m
    for j ∈ 1:ma
      a[j,k+offs] -= h.β * work[j] * conj(v[k])
    end
  end
  nothing
end

@inline function InPlace.:⊘(
  a::AE2,
  h::HouseholderTrans{E,AE1},
) where {E<:Number,AE1<:AbstractArray{E,1},AE2<:AbstractArray{E,2}}
  m = h.size
  v = reshape(h.v, m, 1)
  offs = h.offs
  work = h.work
  ma = size(a,1)
  work[1:ma] .= zero(E)
  for k ∈ 1:m
    for j ∈ 1:ma
      work[j] += a[j,k+offs] * v[k]
    end
  end
  for k ∈ 1:m
    for j ∈ 1:ma
      a[j,k+offs] -= conj(h.β) * work[j] * conj(v[k])
    end
  end
  nothing
end

end # module
