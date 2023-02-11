module BandStruct

using Reexport

include("BandColumnMatrices.jl")
@reexport using .BandColumnMatrices

include("IndexLists.jl")
@reexport using .IndexLists

include("BandwidthInit.jl")
@reexport using .BandwidthInit

include("BlockedBandColumnMatrices.jl")
@reexport using .BlockedBandColumnMatrices

include("BandRotations.jl")
@reexport using .BandRotations

include("BandHouseholder.jl")
@reexport using .BandHouseholder

include("BandFactor.jl")
@reexport using .BandFactor

using Random
include("../test/standard_test_case.jl")

include("Precompile.jl")
import .Precompile

import SnoopPrecompile
SnoopPrecompile.@precompile_all_calls begin
  Precompile.run_all()
end

end
