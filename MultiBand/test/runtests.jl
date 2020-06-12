using MultiBand
using DoubleFloats
using GenericSVD

n=256
A0 = makeMultiBand(n,1, Double64)

A = applyLevelTrans(copy(A0),1)

#printRanks(A, 1e-14)
println("Computing QR factorization.")
@time qrA = qr(A)
println("Computing Inverse")
@time Ai = inv(A)
println("Normalizing Inverse")
@time Ai = Ai/norm(Ai, Inf)
# printRanks(qrA.R', 1e-14)
# printRanks(Ai, 1e-14)

println("Assembling Qa")
@time qa = Matrix(qrA.Q)
println

println("Assembling Rat.")
@time rat = qrA.R'
println

#printRanks(qa, 1e-14)

@time printRanks(qa', 1e-14)

#printSV(A*A,5,1)
# printSV(Ai,7,2)
# @time printSV(rat,5,10)
# @time printSV(rat,4,3)
println

printSV(qa',5,5)

# println("Computing backward error.")
# @time println(norm(qa*qrA.R - A, Inf))

println("Computing Inf condition.")
@time println(norm(A,Inf)*norm(Ai,Inf))
println

GC.gc()
