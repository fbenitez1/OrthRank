module OrthWeight

using Reexport

include("BasicTypes.jl")
@reexport using .BasicTypes

include("WYWeightMatrices.jl")
@reexport using .WYWeightMatrices

# include("WeightConvert.jl")
# @reexport using .WeightConvert

include("GivensWeightMatrices.jl")
@reexport using .GivensWeightMatrices

include("GivensWeightConvert.jl")
@reexport using .GivensWeightConvert

using PrecompileTools
@setup_workload begin
  include("Precompile.jl")
  import .Precompile
  @compile_workload begin
    Precompile.run_all()
  end
end

end # module
