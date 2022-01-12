# Operators for in place multiplication by a linear transformation and
# its inverse.  For a given linear transformation type it is normal to
# define apply_right!, apply_left!, apply_left_inv!, and
# apply_right_inv!.  apply! and apply_inv! can be defined directly for
# transformations in which the transform and what is transformed are
# clearly distinguishable.
module InPlace

export ⊛,
  ⊘,
  Linear,
  apply!,
  apply_inv!,
  apply_right!,
  apply_right_inv!,
  apply_left!,
  apply_left_inv!,
  LeftProduct,
  RightProduct,
  ProductSide,
  product_side

abstract type ProductSide end

struct LeftProduct <: ProductSide end
struct RightProduct <: ProductSide end

struct Linear{A}
  trans :: A
end

product_side(::Type{Linear{A}}, _) where A = LeftProduct()
product_side(_, ::Type{Linear{A}}) where A = RightProduct()

apply!(a::A, b::B; offset = 0) where {A,B} =
  apply!(product_side(A, B), a, b, offset = offset)
apply!(a::A, b::B, c::C, offset = 0) where {A,B,C} =
  apply!(product_side(A, B, C), a, b, c, offset = offset)

apply!(::LeftProduct, a, b; offset = 0) = apply_left!(a, b, offset = offset)
apply!(::LeftProduct, a, b, c; offset = 0) =
  apply_left!(a, b, c, offset = offset)

apply!(::RightProduct, a, b; offset = 0) = apply_right!(a, b, offset = offset)
apply!(::RightProduct, a, b, c; offset = 0) =
  apply_right!(a, b, c, offset = offset)

apply_inv!(a::A, b::B; offset = 0) where {A,B} =
  apply_inv!(product_side(A, B), a, b, offset = offset)
apply_inv!(a::A, b::B, c::C; offset = 0) where {A,B,C} =
  apply_inv!(product_side(A, B, C), a, b, c, offset = offset)

apply_inv!(::LeftProduct, a, b; offset = 0) =
  apply_left_inv!(a, b, offset = offset)
apply_inv!(::LeftProduct, a, b, c; offset = 0) =
  apply_left_inv!(a, b, c, offset = offset)

apply_inv!(::RightProduct, a, b; offset = 0) =
  apply_right_inv!(a, b, offset = offset)
apply_inv!(::RightProduct, a, b, c; offset = 0) =
  apply_right_inv!(a, b, c, offset = offset)

@inline function ⊛(a, b)
  apply!(a, b)
end

@inline function ⊛(a, b, c)
  apply!(a, b, c)
end

@inline function ⊘(a, b)
  apply_inv!(a,b)
end

@inline function ⊘(a, b, c)
  apply_inv!(a, b, c)
end

@inline function apply!(t::Linear{A}, b; offset = 0) where {A}
  apply_left!(t.trans, b, offset = offset)
end

@inline function apply!(b, t::Linear{A}; offset = 0) where {A}
  apply_right!(b, t.trans, offset = offset)
end

@inline function apply_inv!(t::Linear{A}, b; offset = 0) where {A}
  apply_left_inv!(t.trans, b, offset = offset)
end

@inline function apply_inv!(b, t::Linear{A}; offset = 0) where {A}
  apply_right_inv!(b, t.trans, offset = offset)
end

# TODO: Make these minimally allocating.
@inline function apply_left!(
  t::AbstractArray{E,2},
  b::AbstractArray{E,2};
  offset = 0,
) where {E<:Number}
  @views b[(offset + 1):end, (offset + 1):end] =
    t * b[(offset + 1):end, (offset + 1):end]
  nothing
end

@inline function apply_right!(
  b::AbstractArray{E,2},
  t::AbstractArray{E,2};
  offset = 0,
) where {E<:Number}
  @views b[(offset + 1):end, (offset + 1):end] =
    b[(offset + 1):end, (offset + 1):end] * t
  nothing
end

@inline function apply_left_inv!(
  t::AbstractArray{E,2},
  b::AbstractArray{E,2};
  offset = 0,
) where {E<:Number}
  @views b[(offset + 1):end, (offset + 1):end] =
    t \ b[(offset + 1):end, (offset + 1):end]
  nothing
end

@inline function apply_right_inv!(
  b::AbstractArray{E,2},
  t::AbstractArray{E,2};
  offset = 0,
) where {E<:Number}
  @views b[(offset + 1):end, (offset + 1):end] =
    b[(offset + 1):end, (offset + 1):end] / t
  nothing
end

end # module
