# Operators for in place multiplication by a linear transformation and
# its inverse.  For a given linear transformation type it is normal to
# define apply_right!, apply_left!, apply_left_inv!, and
# apply_right_inv!.  apply! and apply_inv! can be defined directly for
# transformations in which the transform and what is transformed are
# clearly distinguishable.
module InPlace
import SnoopPrecompile

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
  product_side,
  GeneralArray,
  GeneralMatrix,
  GeneralVector,
  structure_type,
  substructure_type,
  similar_leftI,
  similar_rightI,
  similarI,
  similar_left_zeros,
  similar_right_zeros,
  similar_zeros

using LinearAlgebra

# A trait for deciding what is a transformation and what is transformed.
abstract type ProductSide end

struct LeftProduct <: ProductSide end
struct RightProduct <: ProductSide end

similarI(A, x...) = copyto!(similar(A, x...), I)

similar_leftI(A::AbstractArray{E,2}) where {E} =
  similarI(A, size(A, 1), size(A, 1))

similar_rightI(A::AbstractArray{E,2}) where {E} =
  similarI(A, size(A, 2), size(A, 2))

similar_fill(A, e, x...) = fill!(similar(A, x...), e)

similar_zeros(A, x...) = similar_fill(A, zero(eltype(A)), x...)

similar_left_zeros(A::AbstractArray{E,2}) where {E} =
  fill!(similar(A, size(A, 1), size(A, 1)), zero(E))

similar_right_zeros(A::AbstractArray{E,2}) where {E} =
  fill!(similar(A, size(A, 2), size(A, 2)), zero(E))

# A trait for classifying matrix structure, notably general matrices
# with no particular constraints on stored elements.  This is
# motivated by the fact that there is no obvious supertype for such
# matrices.  AbstractMatrix{Float64} includes things like
# Hermitian{Float64, Matrix{Float64}}.  However DenseMatrix{Float64}
# fails to include things like Adjoint{Float64, Matrix{Float64}} that
# can be handled by many functions for general matrices.
#
# For any mix of adjoints, subarrays, etc. of a DenseArray{E,M},
# structure_type returns GeneralArray{E,N} of appropriate dimension.
# It is more comprehensive than trying to dispatch on a Union of
# DenseArray and one layer of Adjoints or SubArrays.

struct GeneralArray{E,N} end
const GeneralMatrix{E} = GeneralArray{E,2}
const GeneralVector{E} = GeneralArray{E,1}

structure_type(::Type{A}) where {E, N, A<:DenseArray{E,N}} = GeneralArray{E,N}
structure_type(::Type{A}) where {E,N,P,A<:SubArray{E,N,P}} =
  substructure_type(E, Val(N), P)
structure_type(::Type{A}) where {E,N,P,A<:Base.ReshapedArray{E,N,P}} =
  substructure_type(E, Val(N), P)
structure_type(::Type{A}) where {E,P,A<:Adjoint{E,P}} =
  structure_type(P)


substructure_type(::Type{E}, ::Val{N}, ::Type{A}) where {E,N,A<:DenseArray{E}} =
  GeneralArray{E,N}
substructure_type(::Type{E}, ::Val{N}, ::Type{A}) where {E,N,P,A<:Adjoint{E,P}} =
  substructure_type(E, Val(N), P)
substructure_type(::Type{E}, ::Val{N}, ::Type{A}) where {E,N,M,P,A<:SubArray{E,M,P}} =
  substructure_type(E, Val(M), P)
substructure_type(
  ::Type{E},
  ::Val{N},
  ::Type{A},
) where {E,N,M,P,A<:Base.ReshapedArray{E,M,P}} =
  substructure_type(E, Val(M), P)

struct GeneralHermitianMatrix{E} end

structure_type(::Type{H}) where {E,H<:Hermitian{E,<:DenseMatrix{E}}} =
  GeneralHermitianMatrix{E}


# Marker for a linear transformation.  It can be used to determine
# product side.
struct Linear{A}
  trans :: A
end

product_side(::Type{Linear{A}}, _) where A = LeftProduct()
product_side(_, ::Type{Linear{A}}) where A = RightProduct()

apply!(a::A, b::B; offset = 0) where {A,B} =
  apply!(product_side(A, B), a, b, offset = offset)

apply!(::LeftProduct, a, b; offset = 0) = apply_left!(a, b, offset = offset)

apply!(::RightProduct, a, b; offset = 0) = apply_right!(a, b, offset = offset)

apply_inv!(a::A, b::B; offset = 0) where {A,B} =
  apply_inv!(product_side(A, B), a, b, offset = offset)

apply_inv!(::LeftProduct, a, b::B; offset = 0) where {B} =
  apply_left_inv!(structure_type(B), a, b, offset = offset)

apply_inv!(::RightProduct, a::A, b::B; offset = 0) where {A, B} =
  apply_right_inv!(structure_type(A), a, b, offset = offset)

@inline function ⊛(a, b)
  apply!(a, b)
end

@inline function ⊘(a, b)
  apply_inv!(a,b)
end

@inline function apply!(t::Linear{A}, b::B; offset = 0) where {A,B}
  apply_left!(structure_type(B), t.trans, b, offset = offset)
end

@inline function apply!(b::B, t::Linear{A}; offset = 0) where {A,B}
  apply_right!(structure_type(B), b, t.trans, offset = offset)
end

@inline function apply_inv!(t::Linear{A}, b::B; offset = 0) where {A,B}
  apply_left_inv!(structure_type(B), t.trans, b, offset = offset)
end

@inline function apply_inv!(b::B, t::Linear{A}; offset = 0) where {A,B}
  apply_right_inv!(structure_type(B), b, t.trans, offset = offset)
end


@inline function apply_left!(a::A, b::B; offset = 0) where {A,B}
  apply_left!(structure_type(B), a, b, offset = offset)
end

@inline function apply_right!(a::A, b::B; offset = 0) where {A,B}
  apply_right!(structure_type(A), a, b, offset = offset)
end

@inline function apply_left_inv!(a::A, b::B; offset = 0) where {A,B}
  apply_left_inv!(structure_type(B), a, b, offset = offset)
end

@inline function apply_right_inv!(a::A, b::B; offset = 0) where {A,B}
  apply_right_inv!(structure_type(A), a, b, offset = offset)
end

# TODO: Make these minimally allocating.
@inline function apply_left!(
  ::Type{GeneralMatrix{E}},
  t::AbstractArray{E,2},
  b::AbstractArray{E,2};
  offset = 0,
) where {E<:Number}
  @views b[(offset + 1):end, (offset + 1):end] =
    t * b[(offset + 1):end, (offset + 1):end]
  nothing
end

@inline function apply_right!(
  ::Type{GeneralMatrix{E}},
  b::AbstractArray{E,2},
  t::AbstractArray{E,2};
  offset = 0,
) where {E<:Number}
  @views b[(offset + 1):end, (offset + 1):end] =
    b[(offset + 1):end, (offset + 1):end] * t
  nothing
end

@inline function apply_left_inv!(
  ::Type{GeneralArray{E,2}},
  t::AbstractArray{E,2},
  b::AbstractArray{E,2};
  offset = 0,
) where {E<:Number}
  @views b[(offset + 1):end, (offset + 1):end] =
    t \ b[(offset + 1):end, (offset + 1):end]
  nothing
end

@inline function apply_right_inv!(
  ::Type{GeneralArray{E,2}},
  b::AbstractArray{E,2},
  t::AbstractArray{E,2};
  offset = 0,
) where {E<:Number}
  @views b[(offset + 1):end, (offset + 1):end] =
    b[(offset + 1):end, (offset + 1):end] / t
  nothing
end

SnoopPrecompile.@precompile_all_calls begin
  include("precompile.jl")
end

end # module
