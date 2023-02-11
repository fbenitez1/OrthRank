module JETExt

using BandStruct
using JET

functions = [
  (:run_first_last_init, :(())),
  (:run_all, :(())),
  (:run_wilkinson, :((a,))),
  (:run_index,:((a,))),
  (:run_submatrix, :((a,))),
  (:run_range, :((a,))),
  (:run_bulge, :((a,))),
  (:run_notch, :((a,))),
  (:run_rotations, :((a,))),
  (:run_householder, :((a,))),
  (:run_WY, :((a,)))
]

for (func, params) in functions
  @eval begin
    function BandStruct.Precompile.$func(
      $params,
      s::Symbol;
      ignore_opt = (Base,),
      ignore_call = nothing,
      target = nothing,
    )
      if s == :opt
        JET.@report_opt ignored_modules = ignore_opt target_modules = target (
          BandStruct.Precompile.$func($params...)
        )
      else
        JET.@report_call ignored_modules = ignore_opt target_modules = target (
          BandStruct.Precompile.$func($params...)
        )
      end
    end
  end
end

end
