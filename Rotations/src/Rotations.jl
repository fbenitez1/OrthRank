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

import SnoopPrecompile
SnoopPrecompile.@precompile_all_calls begin
  include("precompile.jl")
end

end # module
