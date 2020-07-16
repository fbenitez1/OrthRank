module BandStruct

export BandColumnMatrices, LeadingBandColumnMatrices

include("BandColumnMatrices.jl")
using BandStruct.BandColumnMatrices

include("LeadingBandColumnMatrices.jl")
using BandStruct.LeadingBandColumnMatrices


end # module
