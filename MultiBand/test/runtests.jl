using MultiBand
using DoubleFloats
using GenericSVD

n=256
A0 = makeMultiBand(n,1, Double64)

A = applyLevelTrans(copy(A0),1)

#printRanks(A, 1e-14)
print("Computing QR factorization\n")
@time qrA = qr(A)
print("Computing Inverse\n")
@time Ai = inv(A)
print("Normalizing Inverse\n")
@time Ai = Ai/norm(Ai, Inf)
# printRanks(qrA.R', 1e-14)
# printRanks(Ai, 1e-14)

print("Assembling Qa\n")
@time qa = Matrix(qrA.Q)
print("\n")

print("Assembling Rat\n")
@time rat = qrA.R'
print("\n")

#printRanks(qa, 1e-14)

@time printRanks(qa', 1e-14)

#printSV(A*A,5,1)
# printSV(Ai,7,2)
# @time printSV(rat,5,10)
# @time printSV(rat,4,3)
print("\n")

printSV(qa',5,5)

# print("Computing backward error.")
# @time print(norm(qa*qrA.R - A, Inf))
# print("\n")

print("Computing Inf condition\n")
@time print(norm(A,Inf)*norm(Ai,Inf))
print("\n")

GC.gc()
