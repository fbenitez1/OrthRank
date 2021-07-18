bc = copy(bc0)
bbc = copy(bbc0)
mx_bc = copy(mx_bc0)

println("""

**************
** Index tests
**************

""")

println()
show_equality_result(
  "eachindex equality BC/BBC",
  ==,
  collect(eachindex(bc)),
  collect(eachindex(bbc)),
)
show_equality_result(
  "Get elements equality BC/BBC",
  ==,
  collect(get_elements(bc)),
  collect(get_elements(bbc)),
)

resbc = true
resbbc = true
for ix in (ix for ix ∈ eachindex(bc) if rand() > 0.75)
  global resbc, resbbc
  x = rand()
  bc[ix] = x
  bbc[ix] = x
  mx_bc[ix] = x
  resbc = resbc && validate_rows_first_last(bc)
  resbbc = resbbc && validate_rows_first_last(bbc)
end
show_bool_result("Validate BC bandwidths after setindex", resbc)
show_bool_result("Validate BBC bandwidths after setindex", resbbc)
show_equality_result("Leading BC Matrix Set Index", ==, Matrix(bc), mx_bc)
show_equality_result("Leading BBC Matrix Set Index", ==, Matrix(bbc), mx_bc)
