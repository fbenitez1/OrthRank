module Givens

export Rot, AdjRot, lgivens, lgivens1, rgivens, rgivens1

using LinearAlgebra
import InPlace
using InPlace

"""

A rotation data structure with real cosine.  The sine might or might
not be complex.  The rotation acts in rows (or columns) j1 and j2.
 
This should be interpreted as a matrix

[  c       s ;
  -conj(s) c ]

"""
struct Rot{R,T}
  c::R
  s::T
  j1::Int64
  j2::Int64
end

InPlace.product_side(::Type{<:Rot}, _) = InPlace.LeftProduct()
InPlace.product_side(_, ::Type{<:Rot}) = InPlace.RightProduct()
InPlace.product_side(::Type{<:Rot}, _, _) = InPlace.LeftProduct()
InPlace.product_side(_, _, ::Type{<:Rot}) = InPlace.RightProduct()

"""

A rotation data structure with real cosine and acting on adjacent rows
or columns.  The sine might or might not be complex.  The rotation
acts in rows (or columns) j and j+1..
 
This should be interpreted as a matrix

[  c       s ;
  -conj(s) c ]

"""
struct AdjRot{R,T}
  c::R
  s::T
  j::Int64
end

InPlace.product_side(::Type{<:AdjRot}, _) = InPlace.LeftProduct()
InPlace.product_side(_, ::Type{<:AdjRot}) = InPlace.RightProduct()
InPlace.product_side(::Type{<:AdjRot}, _, _) = InPlace.LeftProduct()
InPlace.product_side(_, _, ::Type{<:AdjRot}) = InPlace.RightProduct()

@inline function lgivens(
  x::T,
  y::T,
) where {R<:AbstractFloat,T<:Union{R,Complex{R}}}
  xmag = abs(x)
  ymag = abs(y)
  if xmag == 0
    (zero(R), one(T))
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
    (c,s)
  end
end

"""

Compute a rotation r to introduce a zero from the left into the first
element of r ⊘ [x;y].

"""
@inline function lgivens1(
  x::T,
  y::T,
) where {R<:AbstractFloat,T<:Union{R,Complex{R}}}
  xmag = abs(x)
  ymag = abs(y)
  if ymag == 0
    (zero(R), one(T))
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
    (c,s)
  end
end

"""

Compute a rotation r to introduce a zero from the right into the 
second component of  [x,y] ⊛ r.

"""
@inline function rgivens(
  x::T,
  y::T,
) where {R<:AbstractFloat,T<:Union{R,Complex{R}}}
  xmag = abs(x)
  ymag = abs(y)
  if xmag == 0
    (zero(R), one(T))
  else
    scale = 1 / (xmag + ymag) # scale to avoid possible overflow.
    xr = real(x) * scale
    xi = imag(x) * scale
    yr = real(y) * scale
    yi = imag(y) * scale
    normxy = sqrt(xr * xr + xi * xi + yr * yr + yi * yi) / scale
    signx = x / xmag
    c = xmag / normxy
    s = -conj(signx) * y / normxy
    (c,s)
  end
end

"""

Compute a rotation r to introduce a zero from the right into the 
first component of  [x,y] ⊛ r.

"""
@inline function rgivens1(
  x::T,
  y::T,
) where {R<:AbstractFloat,T<:Union{R,Complex{R}}}
  xmag = abs(x)
  ymag = abs(y)
  if ymag == 0
    (zero(R), one(T))
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
    (c,s)
  end
end

####
## Nonadjacent rotations
####

@inline function lgivens(
  x::T,
  y::T,
  j1::Integer,
  j2::Integer,
) where {R<:AbstractFloat,T<:Union{R,Complex{R}}}
  (c,s)=lgivens(x,y)
  Rot(c, s, j1, j2)
end

@inline function lgivens1(
  x::T,
  y::T,
  j1::Integer,
  j2::Integer,
) where {R<:AbstractFloat,T<:Union{R,Complex{R}}}
  (c,s)=lgivens1(x,y)
  Rot(c, s, j1, j2)
end

@inline function rgivens(
  x::T,
  y::T,
  j1::Integer,
  j2::Integer,
) where {R<:AbstractFloat,T<:Union{R,Complex{R}}}
  (c,s)=rgivens(x,y)
  Rot(c, s, j1, j2)
end

@inline function rgivens1(
  x::T,
  y::T,
  j1::Integer,
  j2::Integer,
) where {R<:AbstractFloat,T<:Union{R,Complex{R}}}
  (c, s) = rgivens1(x, y)
  Rot(c, s, j1, j2)
end

####
## Adjacent rotations
####

@inline function lgivens(
  x::T,
  y::T,
  j::Integer,
) where {R<:AbstractFloat,T<:Union{R,Complex{R}}}
  (c,s)=lgivens(x,y)
  AdjRot(c, s, j)
end

@inline function lgivens1(
  x::T,
  y::T,
  j::Integer,
) where {R<:AbstractFloat,T<:Union{R,Complex{R}}}
  (c,s)=lgivens1(x,y)
  AdjRot(c, s, j)
end

@inline function rgivens(
  x::T,
  y::T,
  j::Integer,
) where {R<:AbstractFloat,T<:Union{R,Complex{R}}}
  (c,s)=rgivens(x,y)
  AdjRot(c, s, j)
end

@inline function rgivens1(
  x::T,
  y::T,
  j::Integer,
) where {R<:AbstractFloat,T<:Union{R,Complex{R}}}
  (c, s) = rgivens1(x, y)
  AdjRot(c, s, j)
end

"""

Apply a rotation, acting in-place to modify a.

"""
@inline function InPlace.apply_left!(
  ::Type{GeneralMatrix{T}},
  r::Rot{R,T},
  a::AbstractArray{T,2};
  offset = 0
) where {R<:AbstractFloat,T<:Union{R,Complex{R}}}
  c = r.c
  s = r.s
  (_, n) = size(a)
  j1 = r.j1 + offset
  j2 = r.j2 + offset
  @inbounds for k = 1:n
    tmp = a[j1, k]
    a[j1, k] = c * tmp + s * a[j2, k]
    a[j2, k] = -conj(s) * tmp + c * a[j2, k]
  end
  nothing
end

@inline function InPlace.apply_right!(
  ::Type{GeneralMatrix{T}},
  a::AbstractArray{T,2},
  r::Rot{R,T};
  offset = 0
) where {R<:AbstractFloat,T<:Union{R,Complex{R}}}
  c = r.c
  s = r.s
  (m, _) = size(a)
  k1 = r.j1 + offset
  k2 = r.j2 + offset
  @inbounds for j = 1:m
    tmp = a[j, k1]
    a[j, k1] = c * tmp - conj(s) * a[j, k2]
    a[j, k2] = s * tmp + c * a[j, k2]
  end
  nothing
end

@inline function InPlace.apply_left!(
  ::Type{GeneralMatrix{T}},
  r::AdjRot{R,T},
  a::AbstractArray{T,2};
  offset = 0
) where {R<:AbstractFloat,T<:Union{R,Complex{R}}}
  c = r.c
  s = r.s
  (_, n) = size(a)
  j = r.j + offset
  @inbounds for k = 1:n
    tmp = a[j, k]
    a[j, k] = c * tmp + s * a[j + 1, k]
    a[j + 1, k] = -conj(s) * tmp + c * a[j + 1, k]
  end
  nothing
end

@inline function InPlace.apply_right!(
  ::Type{GeneralMatrix{T}},
  a::AbstractArray{T,2},
  r::AdjRot{R,T};
  offset = 0
) where {R<:AbstractFloat,T<:Union{R,Complex{R}}}
  c = r.c
  s = r.s
  (m, _) = size(a)
  k = r.j + offset
  for j = 1:m
    tmp = a[j, k]
    a[j, k] = c * tmp - conj(s) * a[j, k + 1]
    a[j, k + 1] = s * tmp + c * a[j, k + 1]
  end
  nothing
end

"""

Apply an inverse rotation, acting in-place to modify a.

"""
@inline function InPlace.apply_left_inv!(
  ::Type{GeneralMatrix{T}},
  r::Rot{R,T},
  a::AbstractArray{T,2};
  offset = 0
) where {R<:AbstractFloat,T<:Union{R,Complex{R}}}
  c = r.c
  s = r.s
  (_, n) = size(a)
  j1 = r.j1 + offset
  j2 = r.j2 + offset
  @inbounds for k = 1:n
    tmp = a[j1, k]
    a[j1, k] = c * tmp - s * a[j2, k]
    a[j2, k] = conj(s) * tmp + c * a[j2, k]
  end
  nothing
end

@inline function InPlace.apply_right_inv!(
  ::Type{GeneralMatrix{T}},
  a::AbstractArray{T,2},
  r::Rot{R,T};
  offset = 0
) where {R<:AbstractFloat,T<:Union{R,Complex{R}}}
  c = r.c
  s = r.s
  (m, _) = size(a)
  k1 = r.j1 + offset
  k2 = r.j2 + offset
  @inbounds for j = 1:m
    tmp = a[j, k1]
    a[j, k1] = c * tmp + conj(s) * a[j, k2]
    a[j, k2] = -s * tmp + c * a[j, k2]
  end
  nothing
end

@inline function InPlace.apply_left_inv!(
  ::Type{GeneralMatrix{T}},
  r::AdjRot{R,T},
  a::AbstractArray{T,2};
  offset = 0
) where {R<:AbstractFloat,T<:Union{R,Complex{R}}}
  c = r.c
  s = r.s
  (_, n) = size(a)
  j = r.j + offset
  @inbounds for k = 1:n
    tmp = a[j, k]
    a[j, k] = c * tmp - s * a[j + 1, k]
    a[j + 1, k] = conj(s) * tmp + c * a[j + 1, k]
  end
  nothing
end

@inline function InPlace.apply_right_inv!(
  ::Type{GeneralMatrix{T}},
  a::AbstractArray{T,2},
  r::AdjRot{R,T};
  offset = 0
) where {R<:AbstractFloat,T<:Union{R,Complex{R}}}
  c = r.c
  s = r.s
  (m, _) = size(a)
  k = r.j + offset
  @inbounds for j = 1:m
    tmp = a[j, k]
    a[j, k] = c * tmp + conj(s) * a[j, k + 1]
    a[j, k + 1] = -s * tmp + c * a[j, k + 1]
  end
  nothing
end

end # module
