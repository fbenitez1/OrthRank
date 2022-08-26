using BandStruct.BandColumnMatrices
using BandStruct.BlockedBandColumnMatrices
using BandStruct.BandRotations
using BandStruct.BandHouseholder
using BandStruct.BandFactor
using BenchmarkTools

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
