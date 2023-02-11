module JETExt

using InPlace
using JET

functions = [(:run_inplace, :((a,))), (:run_all, :(()))]

for (func, params) in functions
  @eval begin
    function InPlace.Precompile.$func(
      $params,
      s::Symbol;
      ignore_opt = (Base,),
      ignore_call = nothing,
      target = nothing,
    )
      if s == :opt
        JET.@report_opt ignored_modules = ignore_opt target_modules = target (
          InPlace.Precompile.$func($params...)
        )
      else
        JET.@report_call ignored_modules = ignore_opt target_modules = target (
          InPlace.Precompile.$func($params...)
        )
      end
    end
  end
end

end
