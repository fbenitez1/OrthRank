function show_equality_result(testname, a, b)
  if a == b
    println("Success: ", testname)
  else
    println()
    println("**** Failure: ", testname)
    println()
  end
end
