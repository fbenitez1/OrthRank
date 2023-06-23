module Rotations

export Rot,
  AdjRot,
  get_inds,
  lgivens,
  lgivens1,
  rgivens,
  rgivens1,
  lgivensPR,
  lgivensPR1,
  rgivensPR,
  rgivensPR1,
  check_inplace_rotation_types

include("Givens.jl")
using .Givens

using PrecompileTools
@setup_workload begin
  include("Precompile.jl")
  import .Precompile
  @compile_workload begin
    Precompile.run_all()
  end
end

end # module
