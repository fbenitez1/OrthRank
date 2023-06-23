module InPlaceJETExt

using InPlace
using JET

functions = [(:run_inplace, :((a,))), (:run_all, :(()))]

for (func, params) in functions
  @eval begin

    function InPlace.Precompile.$func(
      s::Symbol,
      xs...;
      ignore_opt = (Base,),
      ignore_call = nothing,
      target = nothing,
    )
      InPlace.Precompile.$func(
        s,
        xs,
        ignore_opt = ignore_opt,
        ignore_call = ignore_call,
        target = target,
      )
    end

    function InPlace.Precompile.$func(
      s::Symbol,
      $params::Tuple;
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
