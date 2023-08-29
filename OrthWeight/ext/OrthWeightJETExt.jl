module OrthWeightJETExt

using OrthWeight
using OrthWeight.GivensWeightMatrices
using JET

functions = [(:run_all, :(())), (:run_matrix, :(())), (:run_convert, :(()))]

# OrthWeight.Precompile.run_all(:opt)
# OrthWeight.Precompile.run_all(:call)

for (func, params) in functions
  @eval begin

    function OrthWeight.Precompile.$func(
      s::Symbol,
      xs...;
      ignore_opt = (Base,),
      ignore_call = nothing,
      target = nothing,
    )
      OrthWeight.Precompile.$func(
        s,
        xs,
        ignore_opt = ignore_opt,
        ignore_call = ignore_call,
        target = target,
      )
    end

    function OrthWeight.Precompile.$func(
      s::Symbol,
      $params::Tuple;
      ignore_opt = (Base,),
      ignore_call = nothing,
      target = nothing,
    )
      if s == :opt
        JET.@report_opt ignored_modules = ignore_opt target_modules = target (
          OrthWeight.Precompile.$func($params...)
        )
      else
        JET.@report_call ignored_modules = ignore_opt target_modules = target (
          OrthWeight.Precompile.$func($params...)
        )
      end
    end
  end
end

end
