function show_equality_result(testname, a, b)
  if a == b
    println("Success: ", testname)
  else
    println()
    println("**** Failure: ", testname)
    println()
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
