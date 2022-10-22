module BandStruct

using Reexport

include("BandColumnMatrices.jl")
@reexport using .BandColumnMatrices

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

import SnoopPrecompile
SnoopPrecompile.@precompile_all_calls begin
  include("../test/standard_test_case.jl")
  include("precompile.jl")
end

end
