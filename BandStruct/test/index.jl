bc = copy(bc0)
lbc = copy(lbc0)
mx_bc = copy(mx_bc0)

println("""

**************
** Index tests
**************

""")

println()
show_equality_result(
  "eachindex equality BC/LBC",
  ==,
  collect(eachindex(bc)),
  collect(eachindex(lbc)),
)
show_equality_result(
  "Get elements equality BC/LBC",
  ==,
  collect(get_elements(bc)),
  collect(get_elements(lbc)),
)

resbc = true
reslbc = true
for ix in (ix for ix ∈ eachindex(bc) if rand() > 0.75)
  global resbc, reslbc
  x = rand()
  bc[ix] = x
  lbc[ix] = x
  mx_bc[ix] = x
  resbc = resbc && validate_rows_first_last(bc)
  reslbc = reslbc && validate_rows_first_last(lbc)
end
show_bool_result("Validate BC bandwidths after setindex", resbc)
show_bool_result("Validate LBC bandwidths after setindex", reslbc)
show_equality_result("Leading BC Matrix Set Index", ==, Matrix(bc), mx_bc)
show_equality_result("Leading LBC Matrix Set Index", ==, Matrix(lbc), mx_bc)
