module JETExt

using Householder
using JET

functions = [
  (:run_all, :(())),
  (:run_householder, :((a,))),
  (:run_WY_general,:((a,))),
  (:run_WYWY, :((a,))),
  (:run_small_QR, :((a,))),
  (:run_qrWY, :((a,))),
  (:run_qrWYSweep, :((a,)))
]

for (func, params) in functions
  @eval begin
    function Householder.Precompile.$func(
      $params,
      s::Symbol;
      ignore_opt = (Base,),
      ignore_call = nothing,
      target = nothing,
    )
      if s == :opt
        JET.@report_opt ignored_modules = ignore_opt target_modules = target (
          Householder.Precompile.$func($params...)
        )
      else
        JET.@report_call ignored_modules = ignore_opt target_modules = target (
          Householder.Precompile.$func($params...)
        )
      end
    end
  end
end

end
