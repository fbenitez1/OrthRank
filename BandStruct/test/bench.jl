using BandStruct.BandColumnMatrices
using BandStruct.BlockedBandColumnMatrices
using BandStruct.BandRotations
using BandStruct.BandHouseholder
using BandStruct.BandFactor
using BenchmarkTools

# Benchmark for Float64
# BenchmarkTools.Trial: 7 samples with 1 evaluation.
#  Range (min … max):  573.696 ms … 614.020 ms  ┊ GC (min … max): 0.00% … 0.00%
#  Time  (median):     582.349 ms               ┊ GC (median):    0.00%
#  Time  (mean ± σ):   587.407 ms ±  14.737 ms  ┊ GC (mean ± σ):  0.00% ± 0.00%

#   ▁   █        ▁         ▁              ▁                     ▁  
#   █▁▁▁█▁▁▁▁▁▁▁▁█▁▁▁▁▁▁▁▁▁█▁▁▁▁▁▁▁▁▁▁▁▁▁▁█▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█ ▁
#   574 ms           Histogram: frequency by time          614 ms <

#  Memory estimate: 3.93 MiB, allocs estimate: 169875.
# Benchmark for ComplexF64
# BenchmarkTools.Trial: 2 samples with 1 evaluation.
#  Range (min … max):  2.450 s …  2.457 s  ┊ GC (min … max): 0.00% … 0.00%
#  Time  (median):     2.454 s             ┊ GC (median):    0.00%
#  Time  (mean ± σ):   2.454 s ± 5.468 ms  ┊ GC (mean ± σ):  0.00% ± 0.00%

#   █                                                      █  
#   █▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█ ▁
#   2.45 s        Histogram: frequency by time        2.46 s <

#  Memory estimate: 6.04 MiB, allocs estimate: 180511.


m=5000
lbw=400
ubw=400
bs = 16
for E ∈ [ Float64,
          Complex{Float64} ]

  println("Benchmark for $E")

  B = makeBForQR(E, m, lbw, ubw, 32)

  B0=copy(B)
  wy = get_WY(B, lbw, ubw, block_size=bs)
  q,r = qrBWYSweep(wy, B, lbw, ubw, block_size=bs)
  display(
    @benchmark qrBWYSweep(wy1, B1, $lbw, $ubw, block_size = $bs) evals = 1 setup =
    begin
    B1 = copy($B0)
    wy1 = get_WY(B1, $lbw, $ubw, block_size = $bs)
    end)
  GC.gc()

end
