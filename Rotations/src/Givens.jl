module Givens

export Rot,
  AdjRot,
  get_inds,
  lgivens,
  lgivens1,
  rgivens,
  rgivens1,
  check_inplace_rotation_types

using LinearAlgebra
import InPlace
using InPlace

"""
A flexible rotation struct.  The sine and cosine might or might
not be complex.  It can act either in a set of adjacent indices
if J = Int or independent indices if J = Tuple{Int,Int}.
 
It should be interpreted as a matrix

[  c       s       ;
  -conj(s) conj(c) ]
"""
struct Rot{TS,TC,J}
  c::TC
  s::TS
  inds::J
  function Rot(
    c::TC,
    s::TS,
    j::Int,
  ) where {R<:Real,TC<:Union{R,Complex{R}},TS<:Union{R,Complex{R}}}
    new{TS,TC,Int}(c, s, j)
  end
  function Rot(
    c::TC,
    s::TS,
    inds::Tuple{Int,Int}
  ) where {R<:Real,TC<:Union{R,Complex{R}},TS<:Union{R,Complex{R}}}
    new{TS,TC,Tuple{Int,Int}}(c, s, inds)
  end
end

const AdjRot{TS,TC} = Rot{TS,TC,Int}

@inline function get_inds(r::Rot{TS,TC,Int}) where {TS,TC,Int}
  r.inds, r.inds + 1
end

@inline function get_inds(r::Rot{TS,TC,Tuple{Int,Int}}) where {TS,TC,Int}
  r.inds[1], r.inds[2]
end

InPlace.product_side(::Type{<:Rot}, _) = InPlace.LeftProduct()
InPlace.product_side(_, ::Type{<:Rot}) = InPlace.RightProduct()
InPlace.product_side(::Type{<:Rot}, _, _) = InPlace.LeftProduct()
InPlace.product_side(_, _, ::Type{<:Rot}) = InPlace.RightProduct()

"""
    function lgivens(
      x::T,
      y::T,
    ) where {R<:AbstractFloat,T<:Union{R,Complex{R}}}

Compute a rotation r to introduce a zero from the left into the second
element of r ⊘ [x;y].
"""
@inline function lgivens(
  x::T,
  y::T,
) where {R<:AbstractFloat,T<:Union{R,Complex{R}}}
  xmag = abs(x)
  ymag = abs(y)
  if xmag == 0
    zero(R), one(T)
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
    c, s
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
    zero(R), one(T)
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
    c, s
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
    zero(R), one(T)
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
    c, s
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
    zero(R), one(T)
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
    c, s
  end
end

@inline function lgivens(
  x::T,
  y::T,
  j1::Integer,
  j2::Integer,
) where {R<:AbstractFloat,T<:Union{R,Complex{R}}}
  c, s = lgivens(x,y)
  Rot(c, s, j1, j2)
end

@inline function lgivens1(
  x::T,
  y::T,
  j1::Integer,
  j2::Integer,
) where {R<:AbstractFloat,T<:Union{R,Complex{R}}}
  c, s = lgivens1(x, y)
  Rot(c, s, j1, j2)
end

@inline function rgivens(
  x::T,
  y::T,
  j1::Integer,
  j2::Integer,
) where {R<:AbstractFloat,T<:Union{R,Complex{R}}}
  c, s=rgivens(x,y)
  Rot(c, s, j1, j2)
end

@inline function rgivens1(
  x::T,
  y::T,
  j1::Integer,
  j2::Integer,
) where {R<:AbstractFloat,T<:Union{R,Complex{R}}}
  c, s = rgivens1(x, y)
  Rot(c, s, j1, j2)
end

@inline function lgivens(
  x::T,
  y::T,
  j::Integer,
) where {R<:AbstractFloat,T<:Union{R,Complex{R}}}
  c, s = lgivens(x,y)
  Rot(c, s, j)
end

@inline function lgivens1(
  x::T,
  y::T,
  j::Integer,
) where {R<:AbstractFloat,T<:Union{R,Complex{R}}}
  c, s = lgivens1(x,y)
  Rot(c, s, j)
end

@inline function rgivens(
  x::T,
  y::T,
  j::Integer,
) where {R<:AbstractFloat,T<:Union{R,Complex{R}}}
  c, s = rgivens(x,y)
  Rot(c, s, j)
end

@inline function rgivens1(
  x::T,
  y::T,
  j::Integer,
) where {R<:AbstractFloat,T<:Union{R,Complex{R}}}
  (c, s) = rgivens1(x, y)
  Rot(c, s, j)
end

"""
    check_inplace_rotation_types(TS,TC,E)

Check that a `Rot{TS,TC}` can be applied inplace to an array with
element type `E`.  I would like to require directly that
`E<:Union{TS,Complex{TS}}`.  But this fails if `TS == Complex{R}`
because `Complex{TS}` requires `TS<:R`.  This fails even if `E ==
Complex{R} == TS`.  By making it a method, JET.jl can pick up on
applying a complex rotation to a real array.
"""
check_inplace_rotation_types(
  ::Type{Complex{R}},
  ::Type{R},
  ::Type{Complex{R}},
) where {R<:Real} = nothing
check_inplace_rotation_types(
  ::Type{Complex{R}},
  ::Type{Complex{R}},
  ::Type{Complex{R}},
) where {R<:Real} = nothing
check_inplace_rotation_types(
  ::Type{R},
  ::Type{Complex{R}},
  ::Type{Complex{R}},
) where {R<:Real} = nothing
check_inplace_rotation_types(
  ::Type{R},
  ::Type{R},
  ::Type{Complex{R}},
) where {R<:Real} = nothing
check_inplace_rotation_types(::Type{R}, ::Type{R}, ::Type{R}) where {R<:Real} =
  nothing

"""
Apply a rotation, acting in-place to modify a.
"""
@inline function InPlace.apply_left!(
  ::Type{GeneralMatrix{E}},
  r::Rot{TS,TC},
  a::AbstractArray{E,2};
  offset = 0,
) where {TS<:Number,TC<:Number,E<:Number}

  check_inplace_rotation_types(TS, TC, E)
  j1, j2 = get_inds(r)
  c = r.c
  s = r.s
  (_, n) = size(a)
  j1 = j1 + offset
  j2 = j2 + offset
  @inbounds for k = 1:n
    tmp = a[j1, k]
    a[j1, k] = c * tmp + s * a[j2, k]
    a[j2, k] = -conj(s) * tmp + conj(c) * a[j2, k]
  end
  nothing
end

@inline function InPlace.apply_right!(
  ::Type{GeneralMatrix{E}},
  a::AbstractArray{E,2},
  r::Rot{TS,TC};
  offset = 0
) where {TS<:Number,TC<:Number,E<:Number}

  check_inplace_rotation_types(TS, TC, E)
  j1, j2 = get_inds(r)
  c = r.c
  s = r.s
  (m, _) = size(a)
  k1 = j1 + offset
  k2 = j2 + offset
  @inbounds for j = 1:m
    tmp = a[j, k1]
    a[j, k1] = c * tmp - conj(s) * a[j, k2]
    a[j, k2] = s * tmp + conj(c) * a[j, k2]
  end
  nothing
end

"""
Apply an inverse rotation, acting in-place to modify a.
"""
@inline function InPlace.apply_left_inv!(
  ::Type{GeneralMatrix{E}},
  r::Rot{TS,TC},
  a::AbstractArray{E,2};
  offset = 0
) where {TS<:Number, TC<:Number, E<:Number}

  check_inplace_rotation_types(TS, TC, E)
  j1, j2 = get_inds(r)
  c = r.c
  s = r.s
  (_, n) = size(a)
  j1 = j1 + offset
  j2 = j2 + offset
  @inbounds for k = 1:n
    tmp = a[j1, k]
    a[j1, k] = conj(c) * tmp - s * a[j2, k]
    a[j2, k] = conj(s) * tmp + c * a[j2, k]
  end
  nothing
end

@inline function InPlace.apply_right_inv!(
  ::Type{GeneralMatrix{E}},
  a::AbstractArray{E,2},
  r::Rot{TS,TC};
  offset = 0
) where {TS<:Number,TC<:Number,E<:Number}

  check_inplace_rotation_types(TS, TC, E)
  j1, j2 = get_inds(r)
  c = r.c
  s = r.s
  (m, _) = size(a)
  k1 = j1 + offset
  k2 = j2 + offset
  @inbounds for j = 1:m
    tmp = a[j, k1]
    a[j, k1] = conj(c) * tmp + conj(s) * a[j, k2]
    a[j, k2] = -s * tmp + c * a[j, k2]
  end
  nothing
end

end # module
