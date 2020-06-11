using RandRank
using LinearAlgebra

p=1000

h=1/p

# Create p random points in the rectangle [a0,a1] x [b0,b1].
function points_in_region(p :: Integer, a0 :: R, a1 :: R,
                          b0 :: R, b1 :: R) where {R <: AbstractFloat}
    x=rand(R,2,p)
    x[1,:]=(a1-a0)*x[1,:] .+ a0
    x[2,:]=(b1-b0)*x[2,:] .+ b0
    x
end

xs=points_in_region(p,0.0,1-h,0.0,h^3)
ys=points_in_region(p,1.0,2.0,4*h,4*h+h^3)

function get_A( xs :: AbstractArray{T,2},
                ys :: AbstractArray{T,2} ) where { X <: AbstractFloat,
                                                   T <: Union{X,Complex{X}} }
    begin
        p=size(xs,2)
        a=zeros(T,p,p)
        for j=1:p
            for k=1:p
                z = -log((xs[1,j]-ys[1,k])^2 + (xs[2,j]-ys[2,k])^2)/(4*pi)
                a[j,k]=z
            end
        end
        a 
    end :: AbstractArray{T,2}
end

rmax=50
print("getting A:  ")
@time A=get_A(xs,ys)
A=A/norm(A,Inf)
println()

print("Computing rand_svd: ")
@time (U,S,V)=rand_svd(A,rmax)
println("Error:  ", norm(U*Diagonal(S)*V' - A, Inf))
(_,r)=size(U)
println("Rank: ", r)
println()

print("Computing rand_rank:  ")
@time r=rand_rank(A,rmax,rel=false)
println("Rank:  ", r)
println()

print("Computing rand_QB: ")
@time (Q,B)=rand_QB(A,rmax)
println("Error:  ", norm(Q*B - A, Inf))
(_,r)=size(Q)
println("Rank: ", r)
println()
