using Pkg

packages = [
  "InPlace",
  "Rotations",
  "Householder",
  "BandStruct",
  "OrthWeight",
]

for p in packages
  cd("$(p)")
  println("Testing in $(pwd())")
  Pkg.activate(".")
  # Pkg.resolve()
  # Pkg.instantiate()
  # Pkg.update() 
  try
    Pkg.test()
  catch
    @error "Test failure."
  end
  cd("..")
end
