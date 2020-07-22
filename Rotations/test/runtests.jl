if isdefined(@__MODULE__, :LanguageServer)
  include("src/Givens.jl")
  using .Givens
else
  using Rotations.Givens
end

using LinearAlgebra
using Random

using BandStruct
using BandStruct.BandColumnMatrices
using BandStruct.LeadingBandColumnMatrices

function show_equality_result(testname, a, b)
  if a == b
    println("Success: ", testname)
  else
    println("Failure: ", testname)
  end
end

function show_bool_result(testname, b)
  if b
    println("Success: ", testname)
  else
    println()
    println("****  Failure: ", testname)
    println()
  end
end

function show_error_result(testname, err, tol)
  abserr = abs(err)
  if abserr < tol
    println("Success: ", testname, ", error: ", abserr)
  else
    println()
    println("****  Failure: ", testname, ", error: ", abserr)
    println()
  end
end

tol = 1e-15

include("general.jl")

blocks = [
  1 3 4 6 7 9 9 11 12
  1 2 5 5 6 7 8 9 10
]

lbc0 = LeadingBandColumn(
  MersenneTwister(0),
  Float64,
  12,
  10,
  2,
  2,
  blocks,
  [1, 1, 1, 2, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1, 1, 1],
)

println()
println("Testing BandStruct Rotations for a matrix with structure:")
println()

show(wilk(lbc0))
println()
bc0 = lbc0[1:end,1:end]
mx_bc0 = Matrix(bc0)

bc = copy(bc0)
mx_bc = Matrix(bc)
r = rgivens(bc[3,4], bc[3,5],4)
bc ⊛ r
mx_bc ⊛ r
show_error_result("Real BandStruct Givens zero test", abs(bc[3,5]), tol)
show_equality_result("Real BandStruct Givens transformation test", Matrix(bc), mx_bc)
bc[3,5]=1.0
zero_to(bc,3,5)
show_error_result("zero_to element test", abs(bc[3,5]), tol)
show_equality_result("zero_to equality test", Matrix(bc), mx_bc)


bc = copy(bc0)
mx_bc = Matrix(bc)
r = rgivens(bc[5,5], bc[5,6],5)
res=false
try 
  bc ⊛ r
catch e
  if isa(e, NoStorageForIndex)
    global res=true
  end
end
show_bool_result("NoStorageForIndex rotation test", res)
