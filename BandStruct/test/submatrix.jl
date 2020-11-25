bc = copy(bc0)
lbc = copy(lbc0)
mx_bc = copy(mx_bc0)

println("""

******************
** Submatrix tests
******************

""")

function rand_range(bc :: AbstractBandColumn)
  (m,n)= size(bc)
  j1 = rand(1:m)
  j2 = rand(1:m)
  k1 = rand(1:n)
  k2 = rand(1:n)
  (UnitRange(j1,j2), UnitRange(k1,k2))
end

res = true
for j ∈ 1:1
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

res = true
for j ∈ 1:1
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
for j ∈ 1:1
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
for j ∈ 1:1
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
