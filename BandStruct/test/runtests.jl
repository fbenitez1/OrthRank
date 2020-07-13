if isdefined(@__MODULE__, :LanguageServer)
  include("src/BandColumnMatrices.jl")
  using .BandColumnMatrices
  include("src/LeadingBandColumnMatrices.jl")
  using .LeadingBandColumnMatrices
else
  using BandStruct.BandColumnMatrices
  using BandStruct.LeadingBandColumnMatrices
end
using Random

function show_error_result(testname, err, tol)
  abserr = abs(err)
  if abserr < tol
    println("Success: ", testname, ", error: ", abserr)
  else
    println("Failure: ", testname, ", error: ", abserr)
  end
end

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
    println("Failure: ", testname)
  end
end

function rand_range(bc :: AbstractBandColumn)
  (m,n)= size(bc)
  j1 = rand(1:m)
  j2 = rand(1:m)
  k1 = rand(1:n)
  k2 = rand(1:n)
  (UnitRange(j1,j2), UnitRange(k1,k2))
end

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

println("Testing BandStruct operations for a matrix with structure:")
println()


print_wilk(lbc0)
println()

bc0 = lbc0[1:end, 1:end]
mx_bc0 = Matrix(bc0)
bc = copy(bc0)
lbc = copy(lbc0)
mx_bc = copy(mx_bc0)

println()
show_equality_result(
  "eachindex equality BC/LBC",
  collect(eachindex(bc)),
  collect(eachindex(lbc)),
)
show_equality_result(
  "Get elements equality BC/LBC",
  collect(get_elements(bc)),
  collect(get_elements(lbc)),
)

for ix in (ix for ix ∈ eachindex(bc) if rand() > 0.75)
  x = rand()
  bc[ix] = x
  lbc[ix] = x
  mx_bc[ix] = x
end
show_equality_result("Leading BC Matrix Set Index", Matrix(bc), mx_bc)
show_equality_result("Leading LBC Matrix Set Index", Matrix(lbc), mx_bc)

res = true
for j ∈ 1:100
  global res
  (rows, cols) = rand_range(bc)
  if Matrix(bc[rows, cols]) != mx_bc[rows, cols]
    res = false
    println()
    println("BC submatrix test failed with: ")
    println("Size: ", size(bc))
    println("Ranges: ", rows, ", ", cols)
    break
  end
end
show_bool_result("BC submatrix test", res)

bc_ = bc[1:end, 1:end]
bc__ = view(bc, 1:2, 1:2)

res = true
for j ∈ 1:100
  global res
  (rows, cols) = rand_range(bc)
  if Matrix(view(bc, rows, cols)) != view(mx_bc, rows, cols)
    res = false
    println()
    println("BC View test failed with: ")
    println("Size: ", size(bc))
    println("Ranges: ", rows, ", ", cols)
    break
  end
end
show_bool_result("BC View test", res)

res = true
for j ∈ 1:100
  global res
  (rows, cols) = rand_range(lbc)
  if Matrix(lbc[rows, cols]) != mx_bc[rows, cols]
    res = false
    println()
    println("LBC submatrix test failed with: ")
    println("Size: ", size(lbc))
    println("Ranges: ", rows, ", ", cols)
    break
  end
end
show_bool_result("LBC submatrix test", res)

res = true
for j ∈ 1:100
  global res
  (rows, cols) = rand_range(lbc)
  if Matrix(view(lbc, rows, cols)) != view(mx_bc, rows, cols)
    res = false
    println()
    println("LBC View test failed with: ")
    println("Size: ", size(lbc))
    println("Ranges: ", rows, ", ", cols)
    break
  end
end
show_bool_result("LBC View test", res)
