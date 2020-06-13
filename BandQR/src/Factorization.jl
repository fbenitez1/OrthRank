module Factorization
using LinearAlgebra

using Rotations

export bandQR

# Take the banded matrix and eliminate the subdiagonal.  This is not
# very efficient or useful, but it is a nice benchmark of a basic
# operation that will be used heavily in the NSF project.
function bandQR( a :: Array{X,2},
                 lbw::Integer,
                 ubw::Integer ) where {X <: Number}
    (ma,na) = size(a)
    for k=1:na-1
        j1 = min(k+lbw-1,na-1)
        for j=j1:-1:k
            r=lgivens(a[j,k], a[j+1,k], j, j+1)
            # view(a,jj,kk) is similar to a[jj,kk], except that the
            # latter copies, so we would not actually modify a.
            r ⊘ view(a, :, k:min(na,k+lbw+ubw))
        end
    end
    nothing :: Nothing
end

# Precompile to get specialized functions for monomorphic types that I
# will use.  Without pre-compilation, compile time would be measured
# as part of the time required for the first run of a benchmark.
precompile(bandQR,(Array{Float64,2}, Int, Int))
precompile(bandQR,(Array{Complex{Float64},2}, Int, Int))

end
