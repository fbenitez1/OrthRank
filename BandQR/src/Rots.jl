module Rots

export Rot, lgivens, ⊛

using LinearAlgebra

# import Base: * # Required to overload the multiplication operator.

# A rotation data structure in which the cosine is real.  The sine
# might or might not be complex.  The rotation acts in rows (or
# columns) j1 and j2.
struct Rot{R,T}
    c :: R
    s :: T
    j1 :: Int64
    j2 :: Int64
end

# Compute a rotation r to introduce a zero from the left into the
# vector r*[x;y].
function lgivens( x :: T,
                  y :: T,
                  j1 :: Integer,
                  j2 :: Integer ) where { R <: AbstractFloat,
                                          T <: Union{R, Complex{R}} }
    xmag = abs(x)
    ymag = abs(y)
    if (xmag == 0)
        # zero and one return zero and one values of the appropriate
        # type so that this branch of the "if" returns a value of the
        # same type the "else" branch.  This can be important for
        # compiler optimization.  It seems like a good idea to attach
        # the type assertion to all return values to help check for
        # potential errors of this type..
        Rot(zero(R),one(T),j1,j2)
    else
        scale = 1/(xmag+ymag) # scale to avoid possible overflow
        # in squaring.
        xr = real(x)*scale
        xi = imag(x)*scale
        yr = real(y)*scale
        yi = imag(y)*scale
        normxy = sqrt(xr*xr + xi * xi + yr * yr + yi * yi)/scale
        signx = x / xmag
        c = xmag/normxy
        s = y / (signx * normxy)
        Rot(c,s,j1,j2)
    end :: Rot{R,T}
end

# Overload multiplication to allow multiplication by rotations from
# the left.  This acts in-place, modifying a.  Bounds checking is
# disabled by the @inbounds macro.  @inline inlines the function.
# @traitfn allows the use of the type constraint RealOf{X,R}.
@inbounds @inline function
    ⊛( r :: Rot{R,T},
       a :: AbstractArray{T,2} ) where { R <: AbstractFloat,
                                         T <: Union{R, Complex{R}} }
    begin
        c = r.c
        s = r.s
        (_,n)=size(a)
        j1 = r.j1
        j2 = r.j2
        for k=1:n
            tmp=a[j1,k]
            a[j1,k]=c*tmp + conj(s)*a[j2,k]
            a[j2,k]=-s*tmp + c*a[j2,k]
        end
        nothing
    end :: Nothing
end
end # module
