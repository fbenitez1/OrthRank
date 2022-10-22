module Householder

using Reexport

include("Compute.jl")
@reexport using .Compute

include("WY.jl")
@reexport using .WY

include("Factor.jl")
@reexport using .Factor

import SnoopPrecompile
SnoopPrecompile.@precompile_all_calls begin
  include("precompile.jl")
end

end # module
