using Random
using LinearAlgebra

function run_inplace(::Type{E}) where {E}
  a = randn(E, 3, 3)
  b = randn(E, 3, 3)
  Linear(a) ⊛ b
  # Linear(a) ⊘ b
  b ⊛ Linear(a)
  # b ⊘ Linear(a)
end

function run_cases()
  run_inplace(Float64)
  run_inplace(Complex{Float64})
end

run_cases()
