module OrthWeight

using Reexport

include("WYWeightMatrices.jl")
@reexport using .WYWeightMatrices

end # module
