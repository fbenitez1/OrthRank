module Householder

using Reexport

include("Compute.jl")
@reexport using .Compute

include("WY.jl")
@reexport using .WY

include("Factor.jl")
@reexport using .Factor

using PrecompileTools
@setup_workload begin
  include("Precompile.jl")
  import .Precompile
  @compile_workload begin
    Precompile.run_all()
  end
end

end # module
