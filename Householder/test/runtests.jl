if isdefined(@__MODULE__, :LanguageServer)
  include("src/Compute.jl")
  using .Compute
  include("src/WY.jl")
  using .WY
else
  using Householder.Compute
  using Householder.WY
end

using LinearAlgebra
using Random
using InPlace
using ShowTests

tol = 1e-14
l=2
m=3

include("HouseholderGeneral.jl")

tol = 1e-14
maxk=2
m=10
n=10
E=Float64
a=rand(E,m,n) .- 0.5
a0=copy(a)
wy1=WYTrans(E,m,n,maxk)
wy1=resetWY(0,m,wy1)
wy2=WYTrans(E,m,n,maxk)
wy2=resetWY(0,m,wy2)
q=Matrix{E}(I,m,m)
for j=1:2
  h = lhouseholder(a[j:m,j],1,j-1)
  h ⊘ a
  q ⊛ h
  wy1 ⊛ h
  h ⊘ wy2 
end
q1 = Matrix{E}(I,m,m)
q1 ⊛ wy1
q2 = Matrix{E}(I,m,m)
q2 ⊘ wy2
println(norm(q*a-a0))
println(norm(q1*a-a0))
println(norm(q2*a-a0))
