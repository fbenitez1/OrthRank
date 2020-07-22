println("""

******************
** Submatrix tests
******************

""")

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
