module Givens

export Rot,
  AdjRot,
  RotationBoundsError,
  get_inds,
  lgivens,
  lgivens1,
  rgivens,
  rgivens1,
  lgivensPR,
  lgivensPR1,
  rgivensPR,
  rgivensPR1,
  check_inplace_rotation_types

using LinearAlgebra
import InPlace: product_side, apply!, apply_inv!
using InPlace
using LoopVectorization

macro real_turbo(t, ex)
  return esc(quote
               if $t <: Real
                 @turbo $ex
               else
                 @inbounds $ex
               end
             end)
end

struct RotationBoundsError <: Exception
  a::Any
  row_or_col::Any
  i::Any
  j::Any
  RotationBoundsError() = new()
  RotationBoundsError(a) = new(a)
  RotationBoundsError(a,row_or_col,i,j) = new(a,row_or_col,i,j)
end

function Base.showerror(io::IO, ex::RotationBoundsError)
  print(io, "RotationBoundsError")
  if isdefined(ex, :a)
    print(io, ": attempt to apply rotation to ")
    summary(io, ex.a)
    if isdefined(ex, :row_or_col)
      print(io, " at ", ex.row_or_col, " ", ex.i, " and ", ex.j)
    end
  end
end

"""
A flexible rotation struct.  The sine and cosine might or might
not be complex.  It can act either in a set of adjacent indices
if J = Int or independent indices if J = Tuple{Int,Int}.
 
It should be interpreted as a matrix

[  c       s       ;
  -conj(s) conj(c) ]
"""
struct Rot{TC,TS,J}
  c::TC
  s::TS
  inds::J
  function Rot(
    c::TC,
    s::TS,
    j::Int,
  ) where {R<:Real,TC<:Union{R,Complex{R}},TS<:Union{R,Complex{R}}}
    new{TC,TS,Int}(c, s, j)
  end
  function Rot(
    c::TC,
    s::TS,
    inds::Tuple{Int,Int}
  ) where {R<:Real,TC<:Union{R,Complex{R}},TS<:Union{R,Complex{R}}}
    new{TC,TS,Tuple{Int,Int}}(c, s, inds)
  end
end

const AdjRot{TC,TS} = Rot{TC,TS,Int}

function Base.show(io::IO, r::AdjRot{C,S}) where {C,S}
  comp = get(io, :compact, false)::Bool
  if comp
    print(io, "(")
    show(io, r.c)
    print(io, ", ")
    show(io, r.s)
    print(io, ", ")
    show(io, r.inds)
    print(io, ")")
  else
    print(io, "AdjRot{")
    show(io, C)
    print(io, ",")
    show(io, S)
    print(io, "}(")
    show(io, r.c)
    print(io, ", ")
    show(io, r.s)
    print(io, ", ")
    show(io, r.inds)
    print(io, ")")
  end
end

function Base.show(io::IO, r::Rot{S,C,J}) where {C,S,J}
  comp = get(io, :compact, false)::Bool
  if comp
    print(io, "(")
    show(io, r.c)
    print(io, ", ")
    show(io, r.s)
    print(io, ", ")
    show(io, r.inds)
    print(io, ")")
  else
    print(io, "Rot{")
    show(io, C)
    print(io, ",")
    show(io, S)
    print(io, ",")
    show(io, J)
    print(io, "}(")
    show(io, r.s)
    print(io, ", ")
    show(io, r.c)
    print(io, ", ")
    show(io, r.inds)
    print(io, ")")
  end
end

# Make rotations act like a scalar for broadcasting.
Base.broadcastable(r::Rot) = Ref(r)

function get_inds(r::Rot{TC,TS,Int}) where {TC,TS,Int}
  r.inds, r.inds + 1
end

function get_inds(r::Rot{TC,TS,Tuple{Int,Int}}) where {TC,TS,Int}
  r.inds[1], r.inds[2]
end

InPlace.product_side(::Type{<:Rot}, _) = InPlace.LeftProduct
InPlace.product_side(_, ::Type{<:Rot}) = InPlace.RightProduct

InPlace.product_side(::Type{<:Rot}, _, _) = InPlace.LeftProduct
InPlace.product_side(_, _, ::Type{<:Rot}) = InPlace.RightProduct

"""
    function lgivens(
      x::T,
      y::T,
    ) where {R<:AbstractFloat,T<:Union{R,Complex{R}}}

Compute a rotation r to introduce a zero from the left into the second
element of r ⊘ [x;y].
"""
function lgivens(
  x::T,
  y::T,
) where {R<:AbstractFloat,T<:Union{R,Complex{R}}}
  xmag = abs(x)
  ymag = abs(y)
  if xmag == 0
    ymag == 0 ? (one(R), zero(T)) : (zero(R), -conj(y)/abs(y))
  else
    scale = 1 / (xmag + ymag)
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

function lgivensPR(
  x::T,
  y::T,
) where {R<:AbstractFloat,T<:Union{R,Complex{R}}}
  xmag = abs(x)
  ymag = abs(y)
  if xmag == 0
    ymag == 0 ? (one(T), zero(T)) : (zero(T), -conj(y)/abs(y))
  else
    scale = 1 / (xmag + ymag)
    xr = real(x) * scale
    xi = imag(x) * scale
    yr = real(y) * scale
    yi = imag(y) * scale
    normxy = sqrt(xr * xr + xi * xi + yr * yr + yi * yi) / scale
    c = x / normxy
    s = -conj(y) / normxy
    c, s
  end
end

"""
Compute a rotation r to introduce a zero from the left into the first
element of r ⊘ [x;y].
"""
function lgivens1(
  x::T,
  y::T,
) where {R<:AbstractFloat,T<:Union{R,Complex{R}}}
  xmag = abs(x)
  ymag = abs(y)
  if ymag == 0
    xmag == 0 ? (one(R), zero(T)) : (zero(R), x/abs(x))
  else
    scale = 1 / (xmag + ymag)
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

function lgivensPR1(
  x::T,
  y::T,
) where {R<:AbstractFloat,T<:Union{R,Complex{R}}}
  xmag = abs(x)
  ymag = abs(y)
  if ymag == 0
    xmag == 0 ? (one(T), zero(T)) : (zero(T), x/abs(x))
  else
    scale = 1 / (xmag + ymag)
    xr = real(x) * scale
    xi = imag(x) * scale
    yr = real(y) * scale
    yi = imag(y) * scale
    normxy = sqrt(xr * xr + xi * xi + yr * yr + yi * yi) / scale
    c = conj(y) / normxy
    s = x / normxy
    c, s
  end
end

"""
Compute a rotation r to introduce a zero from the right into the 
second component of  [x,y] ⊛ r.
"""
function rgivens(
  x::T,
  y::T,
) where {R<:AbstractFloat,T<:Union{R,Complex{R}}}
  xmag = abs(x)
  ymag = abs(y)
  if xmag == 0
    ymag == 0 ? (one(R), zero(T)) : (zero(R), -y/ymag)
  else
    scale = 1 / (xmag + ymag)
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

function rgivensPR(
  x::T,
  y::T,
) where {R<:AbstractFloat,T<:Union{R,Complex{R}}}
  xmag = abs(x)
  ymag = abs(y)
  if xmag == 0
    ymag == 0 ? (one(T), zero(T)) : (zero(T), -y/ymag)
  else
    scale = 1 / (xmag + ymag)
    xr = real(x) * scale
    xi = imag(x) * scale
    yr = real(y) * scale
    yi = imag(y) * scale
    normxy = sqrt(xr * xr + xi * xi + yr * yr + yi * yi) / scale
    c = conj(x) / normxy
    s = -y / normxy
    c, s
  end
end

"""
Compute a rotation r to introduce a zero from the right into the 
first component of  [x,y] ⊛ r.
"""
function rgivens1(
  x::T,
  y::T,
) where {R<:AbstractFloat,T<:Union{R,Complex{R}}}
  xmag = abs(x)
  ymag = abs(y)
  if ymag == 0
    xmag == 0 ? (one(R), zero(T)) : (zero(R), conj(x)/xmag)
  else
    scale = 1 / (xmag + ymag)
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

function rgivensPR1(
  x::T,
  y::T,
) where {R<:AbstractFloat,T<:Union{R,Complex{R}}}
  xmag = abs(x)
  ymag = abs(y)
  if ymag == 0
    xmag == 0 ? (one(T), zero(T)) : (zero(T), conj(x)/xmag)
  else
    scale = 1 / (xmag + ymag)
    xr = real(x) * scale
    xi = imag(x) * scale
    yr = real(y) * scale
    yi = imag(y) * scale
    normxy = sqrt(xr * xr + xi * xi + yr * yr + yi * yi) / scale
    c = y / normxy
    s = conj(x) / normxy
    c, s
  end
end

function lgivens(
  x::T,
  y::T,
  inds::J,
) where {R<:AbstractFloat,T<:Union{R,Complex{R}},J<:Union{Int,Tuple{Int,Int}}}
  @inline c, s = lgivens(x, y)
  Rot(c, s, inds)
end

function lgivensPR(
  x::T,
  y::T,
  inds::J,
) where {R<:AbstractFloat,T<:Union{R,Complex{R}},J<:Union{Int,Tuple{Int,Int}}}
  @inline c, s = lgivensPR(x, y)
  Rot(c, s, inds)
end

function lgivens1(
  x::T,
  y::T,
  inds::J
) where {R<:AbstractFloat,T<:Union{R,Complex{R}},J<:Union{Int,Tuple{Int,Int}}}
  @inline c, s = lgivens1(x, y)
  Rot(c, s, inds)
end

function lgivensPR1(
  x::T,
  y::T,
  inds::J
) where {R<:AbstractFloat,T<:Union{R,Complex{R}},J<:Union{Int,Tuple{Int,Int}}}
  @inline c, s = lgivensPR1(x, y)
  Rot(c, s, inds)
end

function rgivens(
  x::T,
  y::T,
  inds::J
) where {R<:AbstractFloat,T<:Union{R,Complex{R}},J<:Union{Int,Tuple{Int,Int}}}
  @inline c, s = rgivens(x,y)
  Rot(c, s, inds)
end

function rgivensPR(
  x::T,
  y::T,
  inds::J,
) where {R<:AbstractFloat,T<:Union{R,Complex{R}},J<:Union{Int,Tuple{Int,Int}}}
  @inline c, s = rgivensPR(x,y)
  Rot(c, s, inds)
end

function rgivens1(
  x::T,
  y::T,
  inds::J
) where {R<:AbstractFloat,T<:Union{R,Complex{R}},J<:Union{Int,Tuple{Int,Int}}}
  @inline c, s = rgivens1(x, y)
  Rot(c, s, inds)
end

function rgivensPR1(
  x::T,
  y::T,
  inds::J
) where {R<:AbstractFloat,T<:Union{R,Complex{R}},J<:Union{Int,Tuple{Int,Int}}}
  @inline c, s = rgivensPR1(x, y)
  Rot(c, s, inds)
end


"""
    check_inplace_rotation_types(TC,TS,E)

Check that a `Rot{TC,TS}` can be applied inplace to an array with
element type `E`.  I would like to require directly that
`E<:Union{TS,Complex{TS}}`.  But this fails if `TS == Complex{R}`
because `Complex{TS}` requires `TS<:R`.  This fails even if `E ==
Complex{R} == TS`.  By making it a method, JET.jl can pick up on
applying a complex rotation to a real array.
"""
check_inplace_rotation_types(
  ::Type{R},
  ::Type{Complex{R}},
  ::Type{Complex{R}},
) where {R<:Real} = nothing
check_inplace_rotation_types(
  ::Type{Complex{R}},
  ::Type{Complex{R}},
  ::Type{Complex{R}},
) where {R<:Real} = nothing
check_inplace_rotation_types(
  ::Type{Complex{R}},
  ::Type{R},
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
Base.@propagate_inbounds function InPlace.apply!(
  ::Type{LeftProduct},
  ::Type{GeneralMatrix{E}},
  r::Rot{TC,TS},
  a::AbstractArray{E,2};
  offset = 0,
) where {TC<:Number,TS<:Number,E<:Number}

  check_inplace_rotation_types(TC, TS, E)
  j1, j2 = get_inds(r)
  c = r.c
  s = r.s
  j1 = j1 + offset
  j2 = j2 + offset
  @boundscheck begin
    m = size(a,1)
    (j1 >= 1 && j1 <= m && j2 >= 1 && j2 <= m) ||
      throw(RotationBoundsError(a, "rows", j1, j2))
  end
  @real_turbo E for k ∈ axes(a,2)
    tmp = a[j1, k]
    a[j1, k] = c * tmp + s * a[j2, k]
    a[j2, k] = -conj(s) * tmp + conj(c) * a[j2, k]
  end
  nothing
end

Base.@propagate_inbounds function InPlace.apply!(
  ::Type{RightProduct},
  ::Type{GeneralMatrix{E}},
  a::AbstractArray{E,2},
  r::Rot{TC,TS};
  offset = 0
) where {TC<:Number,TS<:Number,E<:Number}

  check_inplace_rotation_types(TC, TS, E)
  j1, j2 = get_inds(r)
  c = r.c
  s = r.s
  k1 = j1 + offset
  k2 = j2 + offset
  @boundscheck begin
    n = size(a,2)
    (k1 >= 1 && k1 <= n && k2 >= 1 && k2 <= n) ||
      throw(RotationBoundsError(a, "columns", k1, k1))
  end
  @real_turbo E for j = axes(a,1)
    tmp = a[j, k1]
    a[j, k1] = c * tmp - conj(s) * a[j, k2]
    a[j, k2] = s * tmp + conj(c) * a[j, k2]
  end
  nothing
end

"""
Apply an inverse rotation, acting in-place to modify a.
"""
Base.@propagate_inbounds function InPlace.apply_inv!(
  ::Type{LeftProduct},
  ::Type{GeneralMatrix{E}},
  r::Rot{TC,TS},
  a::AbstractArray{E,2};
  offset = 0
) where {TC<:Number, TS<:Number, E<:Number}

  check_inplace_rotation_types(TC, TS, E)
  j1, j2 = get_inds(r)
  c = r.c
  s = r.s
  j1 = j1 + offset
  j2 = j2 + offset
  @boundscheck begin
    m = size(a,1)
    (j1 >= 1 && j1 <= m && j2 >= 1 && j2 <= m) ||
      throw(RotationBoundsError(a, "rows", j1, j2))
  end
  @real_turbo E for k = axes(a,2)
    tmp = a[j1, k]
    a[j1, k] = conj(c) * tmp - s * a[j2, k]
    a[j2, k] = conj(s) * tmp + c * a[j2, k]
  end
  nothing
end

Base.@propagate_inbounds function InPlace.apply_inv!(
  ::Type{RightProduct},
  ::Type{GeneralMatrix{E}},
  a::AbstractArray{E,2},
  r::Rot{TC,TS};
  offset = 0
) where {TC<:Number,TS<:Number,E<:Number}

  check_inplace_rotation_types(TC, TS, E)
  j1, j2 = get_inds(r)
  c = r.c
  s = r.s
  k1 = j1 + offset
  k2 = j2 + offset
  @boundscheck begin
    n = size(a,2)
    (k1 >= 1 && k1 <= n && k2 >= 1 && k2 <= n) ||
      throw(RotationBoundsError(a, "columns", k1, k2))
  end
  @real_turbo E for j ∈ axes(a,1)
    tmp = a[j, k1]
    a[j, k1] = conj(c) * tmp + conj(s) * a[j, k2]
    a[j, k2] = -s * tmp + c * a[j, k2]
  end
  nothing
end

end
