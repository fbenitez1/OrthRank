module Precompile

using Random
using LinearAlgebra
using InPlace

function run_inplace(::Type{E}) where {E}
  a = Matrix{E}(I,3,3)
  b = Matrix{E}(I,3,3)
  Linear(a) ⊛ b
  Linear(a) ⊘ b
  b ⊛ Linear(a)
  b ⊘ Linear(a)
end

function run_all()
  run_inplace(Float64)
  run_inplace(Complex{Float64})
end

end

