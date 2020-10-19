module Compute

export HouseholderTrans,
  update_norm, lhouseholder, rhouseholder, column_nonzero!, row_nonzero!

using LinearAlgebra
import InPlace

"""

A Householder data structure: h ⊛ A is equivalent to

  A[h.j:h.j+h.m] = A[h.j:h.j+h.m] - h.beta * h.v * (h.v' * A[h.j:h.j+h.m])

"""
struct HouseholderTrans{E,AE}
  beta::E
  v::AE          # Householder vector.
  l::Int64       # element to leave nonzero.
  m::Int64       # size of transformation.
  offs::Int64    # offset for applying to a matrix.
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

  Compute a Householder vector such that for h = I - beta v v',
  h^H * a = ||a|| e_1.

"""
function lhouseholder(
  a::A,
  l::Int64,
  offs::Int64,
) where {R<:Real,E<:Union{R,Complex{R}},A<:AbstractArray{E,1}}
  m = length(a)
  a1 = a[l]
  if m == 1
    a[1] = 1
    sign_a1 = a1 == zero(a1) ? one(a1) : sign(a1)
    beta = (sign_a1 - one(a1)) / sign_a1
    HouseholderTrans(conj(beta), a, l, m, offs)
  else
    norm_a2 = update_norm(norm(a[1:(l - 1)]), norm(a[(l + 1):m]))
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
    beta = -conj(alpha) / norm_a
    a[l] = one(a1)
    a[1:(l - 1)] = a[1:(l - 1)] / alpha
    a[(l + 1):m] = a[(l + 1):m] / alpha
    HouseholderTrans(conj(beta), a, l, m, offs)
  end
end

function rhouseholder(
  a::A,
  l::Int64,
  offs::Int64,
) where {R<:Real,E<:Union{R,Complex{R}},A<:AbstractArray{E,1}}
  m = length(a)
  a1 = a[l]
  if m == 1
    a[1] = 1
    sign_a1 = a1 == zero(a1) ? one(a1) : sign(a1)
    beta = (sign_a1 - one(a1)) / sign_a1
    conj!(a)
    HouseholderTrans(beta, a, l, m, offs)
  else
    norm_a2 = update_norm(norm(a[1:(l - 1)]), norm(a[(l + 1):m]))
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
    beta = -conj(alpha) / norm_a
    a[l] = one(a1)
    a[1:(l - 1)] = a[1:(l - 1)] / alpha
    a[(l + 1):m] = a[(l + 1):m] / alpha
    conj!(a)
    HouseholderTrans(beta, a, l, m, offs)
  end
end

# Specialized Householder function for real case.  Note that
# the general version above also works for real.
function lhouseholder(
  a::AR,
  l::Int64,
  offs::Int64,
) where {R<:Real,AR<:AbstractArray{R,1}}
  m = length(a)
  a1 = a[l]
  if m==1
    a[1]=1
    sign_a1 = a1 == zero(a1) ? one(a1) : sign(a1)
    beta = (sign_a1 - one(a1))/sign_a1
    HouseholderTrans(beta, a, l, m, offs)
  else
    norm_a2 = update_norm(norm(a[1:l - 1]), norm(a[l + 1:m]))
    norm_a = update_norm(norm_a2, a1)
    alpha = a1 <= 0 ? a1 - norm_a : -(norm_a2 / (a1 + norm_a))*norm_a2
    beta = -alpha / norm_a
    a[l] = one(a1)
    a[1:l - 1] = a[1:l - 1] / alpha
    a[l + 1:m] = a[l + 1:m] / alpha
    HouseholderTrans(beta, a, l, m, offs)
  end
end

# Specialized Householder function for real case.  Note that
# the general version above also works for real.
function rhouseholder(
  a::AR,
  l::Int64,
  offs::Int64,
) where {R<:Real,AR<:AbstractArray{R,1}}
  m = length(a)
  a1 = a[l]
  if m==1
    a[1]=1
    sign_a1 = a1 == zero(a1) ? one(a1) : sign(a1)
    beta = (sign_a1 - one(a1))/sign_a1
    HouseholderTrans(beta, a, l, m, offs)
  else
    norm_a2 = update_norm(norm(a[1:l - 1]), norm(a[l + 1:m]))
    norm_a = update_norm(norm_a2, a1)
    alpha = a1 <= 0 ? a1 - norm_a : -(norm_a2 / (a1 + norm_a))*norm_a2
    beta = -alpha / norm_a
    a[l] = one(a1)
    a[1:l - 1] = a[1:l - 1] / alpha
    a[l + 1:m] = a[l + 1:m] / alpha
    HouseholderTrans(beta, a, l, m, offs)
  end
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
  m = h.m
  v = reshape(h.v, m, 1)
  offs = h.offs
  a[(offs + 1):(offs + m), :] =
    view(a, (offs + 1):(offs + m), :) -
    (h.beta * v) * (v' * view(a, (offs + 1):(offs + m), :))
  nothing
end

@inline function InPlace.:⊛(
  a::AE2,
  h::HouseholderTrans{E,AE1},
) where {E<:Number,AE1<:AbstractArray{E,1},AE2<:AbstractArray{E,2}}
  m = h.m
  v = reshape(h.v, m, 1)
  offs = h.offs
  a[:, (offs + 1):(offs + m)] =
    view(a, :, (offs + 1):(offs + m)) -
    (h.beta * (view(a, :, (offs + 1):(offs + m)) * v)) * v'
  nothing
end

@inline function InPlace.:⊘(
  h::HouseholderTrans{E,AE1},
  a::AE2,
) where {E<:Number,AE1<:AbstractArray{E,1},AE2<:AbstractArray{E,2}}
  m = h.m
  v = reshape(h.v, m, 1)
  offs = h.offs
  a[(offs + 1):(offs + m), :] =
    view(a, (offs + 1):(offs + m), :) -
    conj(h.beta) * v * (v' * view(a, (offs + 1):(offs + m), :))
  nothing
end

@inline function InPlace.:⊘(
  a::AE2,
  h::HouseholderTrans{E,AE1},
) where {E<:Number,AE1<:AbstractArray{E,1},AE2<:AbstractArray{E,2}}
  m = h.m
  v = reshape(h.v, m, 1)
  offs = h.offs
  a[:, (offs + 1):(offs + m)] =
    view(a, :, (offs + 1):(offs + m)) -
    (conj(h.beta) * (view(a, :, (offs + 1):(offs + m)) * v)) * v'
  nothing
end


end # module
