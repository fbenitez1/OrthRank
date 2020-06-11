module MultiBand

export makeMultiBand, applyLevelTrans, printRanks, printSV

include("MakeMultiBand.jl")
using MultiBand.MakeMultiBand

include("PrintRanks.jl")
using MultiBand.PrintRanks

end # module
