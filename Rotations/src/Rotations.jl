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

include("Precompile.jl")
import .Precompile

import SnoopPrecompile
SnoopPrecompile.@precompile_all_calls begin
  Precompile.run_all()
end

end # module
