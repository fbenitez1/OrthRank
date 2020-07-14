module Givens

export Rot, lgivens, lgivens1, rgivens, rgivens1, ⊛, ⊘

using LinearAlgebra

"""

A rotation data structure with real cosine.  The sine might or might
not be complex.  The rotation acts in rows (or columns) j1 and j2.
 
This should be interpreted as a matrix

[  c conj(s) ;
  -s c       ]

"""
struct Rot{R,T}
  c::R
  s::T
  j1::Int64
  j2::Int64
end

"""

Compute a rotation r to introduce a zero from the left into the second
element of r ⊘ [x;y].

"""
@inline function lgivens(
  x::T,
  y::T,
  j1::Integer,
  j2::Integer,
) where {R<:AbstractFloat,T<:Union{R,Complex{R}}}
  xmag = abs(x)
  ymag = abs(y)
  if xmag == 0
    Rot(zero(R), one(T), j1, j2)
  else
    scale = 1 / (xmag + ymag) # scale to avoid possible overflow.
    xr = real(x) * scale
    xi = imag(x) * scale
    yr = real(y) * scale
    yi = imag(y) * scale
    normxy = sqrt(xr * xr + xi * xi + yr * yr + yi * yi) / scale
    signx = x / xmag
    c = xmag / normxy
    s = -conj(signx)*y / normxy
    Rot(c, s, j1, j2)
  end::Rot{R,T}
end

"""

Compute a rotation r to introduce a zero from the left into the first
element of r ⊘ [x;y].

"""
@inline function lgivens1(
  x::T,
  y::T,
  j1::Integer,
  j2::Integer,
) where {R<:AbstractFloat,T<:Union{R,Complex{R}}}
  xmag = abs(x)
  ymag = abs(y)
  if ymag == 0
    Rot(zero(R), one(T), j1, j2)
  else
    scale = 1 / (xmag + ymag) # scale to avoid possible overflow.
    xr = real(x) * scale
    xi = imag(x) * scale
    yr = real(y) * scale
    yi = imag(y) * scale
    normxy = sqrt(xr * xr + xi * xi + yr * yr + yi * yi) / scale
    signy = y / ymag
    c = ymag / normxy
    s = signy * conj(x) / normxy
    Rot(c, s, j1, j2)
  end::Rot{R,T}
end

"""

Compute a rotation r to introduce a zero from the right into the 
second component of  [x,y] ⊛ r.

"""
@inline function rgivens(
  x::T,
  y::T,
  j1::Integer,
  j2::Integer,
) where {R<:AbstractFloat,T<:Union{R,Complex{R}}}
  xmag = abs(x)
  ymag = abs(y)
  if xmag == 0
    Rot(zero(R), one(T), j1, j2)
  else
    scale = 1 / (xmag + ymag) # scale to avoid possible overflow.
    xr = real(x) * scale
    xi = imag(x) * scale
    yr = real(y) * scale
    yi = imag(y) * scale
    normxy = sqrt(xr * xr + xi * xi + yr * yr + yi * yi) / scale
    signx = x / xmag
    c = xmag / normxy
    s = -signx * conj(y) / normxy
    Rot(c, s, j1, j2)
  end::Rot{R,T}
end

"""

Compute a rotation r to introduce a zero from the right into the 
first component of  [x,y] ⊛ r.

"""
@inline function rgivens1(
  x::T,
  y::T,
  j1::Integer,
  j2::Integer,
) where {R<:AbstractFloat,T<:Union{R,Complex{R}}}
  xmag = abs(x)
  ymag = abs(y)
  if ymag == 0
    Rot(zero(R), one(T), j1, j2)
  else
    scale = 1 / (xmag + ymag) # scale to avoid possible overflow.
    xr = real(x) * scale
    xi = imag(x) * scale
    yr = real(y) * scale
    yi = imag(y) * scale
    normxy = sqrt(xr * xr + xi * xi + yr * yr + yi * yi) / scale
    signy = y / ymag
    c = ymag / normxy
    s = conj(signy) * x / normxy
    Rot(c, s, j1, j2)
  end::Rot{R,T}
end

# Apply a rotation from the left.  This acts in-place, modifying a.
@inbounds @inline function ⊛(
  r::Rot{R,T},
  a::AbstractArray{T,2},
) where {R<:AbstractFloat,T<:Union{R,Complex{R}}}
  c = r.c
  s = r.s
  (_, n) = size(a)
  j1 = r.j1
  j2 = r.j2
  for k = 1:n
    tmp = a[j1, k]
    a[j1, k] = c * tmp + conj(s) * a[j2, k]
    a[j2, k] = -s * tmp + c * a[j2, k]
  end
  nothing
end

@inbounds @inline function ⊛(
  a::AbstractArray{T,2},
  r::Rot{R,T},
) where {R<:AbstractFloat,T<:Union{R,Complex{R}}}
  c = r.c
  s = r.s
  (m, _) = size(a)
  k1 = r.j1
  k2 = r.j2
  for j = 1:m
    tmp = a[j, k1]
    a[j, k1] = c * tmp - s * a[j, k2]
    a[j, k2] = conj(s) * tmp + c * a[j, k2]
  end
  nothing
end

@inbounds @inline function ⊘(
  r::Rot{R,T},
  a::AbstractArray{T,2},
) where {R<:AbstractFloat,T<:Union{R,Complex{R}}}
  c = r.c
  s = r.s
  (_, n) = size(a)
  j1 = r.j1
  j2 = r.j2
  for k = 1:n
    tmp = a[j1, k]
    a[j1, k] = c * tmp - conj(s) * a[j2, k]
    a[j2, k] = +s * tmp + c * a[j2, k]
  end
  nothing
end

@inbounds @inline function ⊘(
  a::AbstractArray{T,2},
  r::Rot{R,T},
) where {R<:AbstractFloat,T<:Union{R,Complex{R}}}
  c = r.c
  s = r.s
  (m, _) = size(a)
  k1 = r.j1
  k2 = r.j2
  for j = 1:m
    tmp = a[j, k1]
    a[j, k1] = c * tmp + s * a[j, k2]
    a[j, k2] = -conj(s) * tmp + c * a[j, k2]
  end
  nothing
end


end # module
