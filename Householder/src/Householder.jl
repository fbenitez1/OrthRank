module Householder

using Reexport

include("Compute.jl")
@reexport using .Compute

include("WY.jl")
@reexport using .WY

include("Factor.jl")
@reexport using .Factor

include("Precompile.jl")
import .Precompile

import SnoopPrecompile
SnoopPrecompile.@precompile_all_calls begin
  Precompile.run_all()
end

end # module
