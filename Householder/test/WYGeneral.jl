print("""

Testing Real WY Transformations

""")


tol = 1e-14
maxk=3
m=10
n=10

E=Float64
A=randn(E,m,n)
A0=copy(A)
wy1=WYTrans(E,m,n,maxk)
resetWY!(0,m,wy1)
wy2=WYTrans(E,m,n,maxk)
resetWY!(0,m,wy2)
Im=Matrix{E}(I,m,m)
Q=copy(Im)
work=zeros(E,m)
for j=1:3
  local h = lhouseholder(A[j:m,j],1,j-1,work)
  h ⊘ A
  Q ⊛ h
  wy1 ⊛ h
  h ⊘ wy2 
end
Q1 = Matrix{E}(I,m,m)
Q1 ⊛ wy1
Q2 = Matrix{E}(I,m,m)
Q2 ⊘ wy2
show_error_result(
  "WY factorization error using ⊛, Real",
  norm(Q1*A-A0),
  tol,
)
show_error_result(
  "WY factorization error using ⊘, Real",
  norm(Q2*A-A0),
  tol,
)

print("""

Testing Complex WY Transformations

""")

E=Complex{Float64}
A=randn(E,m,n)
A0=copy(A)
wy1=WYTrans(E,m,n,maxk)
resetWY!(0,m,wy1)
wy2=WYTrans(E,m,n,maxk)
resetWY!(0,m,wy2)
Im=Matrix{E}(I,m,m)
Q=copy(Im)
work=zeros(E,m)
for j=1:3
  local h = lhouseholder(A[j:m,j],1,j-1,work)
  h ⊘ A
  Q ⊛ h
  wy1 ⊛ h
  h ⊘ wy2 
end
Q1 = Matrix{E}(I,m,m)
Q1 ⊛ wy1
Q2 = Matrix{E}(I,m,m)
Q2 ⊘ wy2
show_error_result(
  "WY factorization error using ⊛, Complex",
  norm(Q1*A-A0),
  tol,
)
show_error_result(
  "WY factorization error using ⊘, Complex",
  norm(Q2*A-A0),
  tol,
)

function qrH(A::AbstractArray{E,2}) where {E<:Number}
  (m, n) = size(A)
  # blocks, rem = divrem(n, bs)
  # blocks = rem > 0 ? blocks + 1 : blocks
  Q = Matrix{E}(I, m, m)
  v = zeros(E, m)
  work = zeros(E, m)
  @views for k ∈ 1:n
    vk = v[1:(m - k + 1)]
    vk[:] = A[k:m, k]
    local h = lhouseholder(vk, 1, k - 1, work)
    h ⊘ A[:, k:n]
    A[(k + 1):m, k] .= zero(E)
    Q ⊛ h
  end
  (Q, A)
end

@inbounds function qrWY(A::Array{E,2}, bs::Int64) where {E<:Number}
  m, n = size(A)
  blocks, rem = divrem(n, bs)
  blocks = rem > 0 ? blocks + 1 : blocks
  Q = Matrix{E}(I, m, m)
  v = zeros(E, m)
  wy=WYTrans(E,m,m,bs+2)
  workh = zeros(E, m)
  @views for b ∈ 1:blocks
    offs = (b-1)*bs
    resetWY!(offs,m-offs,wy)
    block_end = min(b*bs,n)
    for k ∈ ((b - 1) * bs + 1):block_end
      vk = v[1:(m - k + 1)]
      vk[:] = A[k:m, k]
      local h = lhouseholder(vk, 1, k - 1, workh)
      h ⊘ A[:, k:block_end]
      A[(k + 1):m, k] .= zero(E)
      wy ⊛ h
    end
    Q ⊛ wy
    wy ⊘ A[:,block_end+1:n]
  end
  (Q, A)
end

function qrLA(A::AbstractArray{E,2}) where {E<:Number}
  m, n = size(A)
  qrA=qr(A)
  Q = Matrix{E}(I, m, m)
  Q = qrA.Q * Q
  R=zeros(E,m,n)
  R[1:n,1:n] = qrA.R
  (Q,R)
end

m=500
n=400
E=Float64
A=randn(E,m,n)
A0=copy(A)
print("""

Benchmarking basic Householder QR:
""")
A[:,:]=A0
(Q,R) = qrH(A)
println("Backward error: ", norm(Q*R-A0))
@btime begin
  A[:,:]=A0
  (Q,R) = qrH(A)
end

print("""

Benchmarking LAPACK QR:
""")
A[:,:]=A0
(Q,R) = qrLA(A)
println("Backward error: ", norm(Q*R-A0))
@btime begin
  A[:,:]=A0
  (Q,R) = qrLA(A)
end

print("""

Benchmarking WY QR:
""")
A[:,:]=A0
(Q,R) = qrWY(A,32)
println("Backward error: ", norm(Q*R-A0))
@btime begin
  A[:,:]=A0
  (Q,R) = qrWY(A,32)
end

