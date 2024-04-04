using Pkg

packages = [
  ("InPlace", []),
  ("Rotations", ["InPlace"]),
  ("Householder", ["InPlace"]),
  ("BandStruct", ["InPlace", "Rotations",  "Householder"]),
  ("OrthWeight", ["InPlace", "Rotations",  "Householder", "BandStruct"]),
  (
    "GivensWeightQR",
    ["InPlace", "Rotations", "Householder", "BandStruct", "OrthWeight"]
  )
]
println("Update dependencies? (Y/n) [n]")
response = replace(readline(stdin), r" " => "")

if match(r"^$", response) != nothing
  up = false
else
  up = match(r"(Yes|yes|Y)", response) != nothing
  noup = match(r"(n|No|no|N)", response) != nothing
  xor(up, noup) || error("Answer y or n.")
end

for (p, devlist) in packages
  cd("$(p)")
  println("Testing in $(pwd())")
  Pkg.activate(".")
  if !isfile("Manifest.toml")
    for d in devlist
      Pkg.rm(d)
    end
    for d in devlist
      Pkg.develop(path="../$d")
    end
  end
  Pkg.resolve()
  Pkg.instantiate()
  up && Pkg.update()
  try
    Pkg.test()
  catch
    @error "Test failure."
  end
  cd("..")
end
