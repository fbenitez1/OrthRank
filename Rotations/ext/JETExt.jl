module JETExt

using Rotations
using JET

functions = [(:run_givens, :((a,))), (:run_all, :(()))]

for (func, params) in functions
  @eval begin
    function Rotations.Precompile.$func(
      $params,
      s::Symbol;
      ignore_opt = (Base,),
      ignore_call = nothing,
      target = nothing,
    )
      if s == :opt
        JET.@report_opt ignored_modules = ignore_opt target_modules = target (
          Rotations.Precompile.$func($params...)
        )
      else
        JET.@report_call ignored_modules = ignore_opt target_modules = target (
          Rotations.Precompile.$func($params...)
        )
      end
    end
  end
end

end
