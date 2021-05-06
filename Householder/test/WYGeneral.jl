print("""

Testing Real WY Transformations

""")


tol = 1e-14
max_num_hs=3
m=10
n=10

E = Float64
A = randn(E, m, n)
A0 = copy(A)
wy1 = WYTrans(
  E,
  max_WY_size = m,
  work_size = n * max_num_hs,
  max_num_hs = max_num_hs,
)
resetWYBlock!(wy1, offset=0, sizeWY=m)
wy2 = WYTrans(
  E,
  max_WY_size = m,
  work_size = n * max_num_hs,
  max_num_hs = max_num_hs,
)
resetWYBlock!(wy2, offset=0, sizeWY=m)
Im=Matrix{E}(I,m,m)
Q=copy(Im)
work=zeros(E,m)
for j=1:3
  local h = lhouseholder(A[j:m,j],1,j-1,work)
  h ⊘ A
  Q ⊛ h
  SelectWY(wy1,1) ⊛ h
  h ⊘ SelectWY(wy2, 1)
end
Q1 = Matrix{E}(I,m,m)
Q1 ⊛ SelectWY(wy1,1)
Q2 = Matrix{E}(I,m,m)
Q2 ⊘ SelectWY(wy2,1)
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

# print("""

# Testing Complex WY Transformations

# """)

E=Complex{Float64}
A=randn(E,m,n)
A0=copy(A)
wy1 = WYTrans(
  E,
  max_WY_size = m,
  work_size = n * max_num_hs,
  max_num_hs = max_num_hs,
)
resetWYBlock!(wy1, offset = 0, sizeWY = m)
wy2 = WYTrans(
  E,
  max_WY_size = m,
  work_size = n * max_num_hs,
  max_num_hs = max_num_hs,
)
resetWYBlock!(wy2, offset = 0, sizeWY = m)
Im=Matrix{E}(I,m,m)
Q=copy(Im)
work=zeros(E,m)
for j=1:3
  local h = lhouseholder(A[j:m,j],1,j-1,work)
  h ⊘ A
  Q ⊛ h
  SelectWY(wy1,1) ⊛ h
  h ⊘ SelectWY(wy2,1)
end
Q1 = Matrix{E}(I,m,m)
Q1 ⊛ SelectWY(wy1,1)
Q2 = Matrix{E}(I,m,m)
Q2 ⊘ SelectWY(wy2,1)
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
  Q = Matrix{E}(I, m, m)
  v = zeros(E, m)
  work = zeros(E, m)
  @inbounds @views for k ∈ 1:n
    vk = v[1:(m - k + 1)]
    vk[:] = A[k:m, k]
    h = lhouseholder(vk, 1, k - 1, work)
    h ⊘ A[:, k:n]
    A[(k + 1):m, k] .= zero(E)
    Q ⊛ h
  end
  (Q, A)
end

function qrWY(A::Array{E,2}; block_size::Int=32) where {E<:Number}
  m, n = size(A)
  blocks, rem = divrem(n, block_size)
  blocks = rem > 0 ? blocks + 1 : blocks
  Q = Matrix{E}(I, m, m)
  v = zeros(E, m)
  wy = WYTrans(
    E,
    max_WY_size = m,
    work_size = m * (block_size + 2),
    max_num_hs = block_size + 2,
  )
  workh = zeros(E, m)
  selectWY!(wy, 1)
  @inbounds @views for b ∈ 1:blocks
    offs = (b - 1) * block_size
    resetWYBlock!(wy, offset = offs, sizeWY = m - offs)
    block_end = min(b * block_size, n)
    for k ∈ ((b - 1) * block_size + 1):block_end
      vk = v[1:(m - k + 1)]
      h = householder(A, k:m, k, vk, workh)
      h ⊘ A[:, k:block_end]
      A[(k + 1):m, k] .= zero(E)
      wy ⊛ h
    end
    Q ⊛ wy
    wy ⊘ A[:, (block_end + 1):n]
  end
  (Q, A)
end

function qrWYSweep(A::Array{E,2}; block_size::Int=32) where {E<:Number}
  m, n = size(A)
  blocks, rem = divrem(n, block_size)
  blocks = rem > 0 ? blocks + 1 : blocks
  v = zeros(E, m)
  wy = WYTrans(
    E,
    max_num_WY = blocks,
    max_WY_size = m,
    work_size = m * (block_size + 2),
    max_num_hs = block_size + 2,
  )
  workh = zeros(E, m)
  @inbounds @views for b ∈ 1:blocks
    offs = (b - 1) * block_size
    selectWY!(wy, b)
    resetWYBlock!(wy, block = b, offset = offs, sizeWY = m - offs)
    block_end = min(b * block_size, n)
    for k ∈ ((b - 1) * block_size + 1):block_end
      vk = v[1:(m - k + 1)]
      h = householder(A, k:m, k, vk, workh)
      h ⊘ A[:, k:block_end]
      A[(k + 1):m, k] .= zero(E)
      wy ⊛ h
    end
    wy ⊘ A[:, (block_end + 1):n]
  end
  (SweepForward(wy), A)
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

m=1000
n=900
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
(Q,R) = qrWY(A)
println("Backward error: ", norm(Q*R-A0))
@btime begin
  A[:,:]=A0
  (Q,R) = qrWY(A)
  nothing
end

print("""

Testing WY QR with Q Returned as a SweepForward{WYTrans}:
""")
A[:,:]=A0
(Q,R) = qrWYSweep(A)
Q⊛R
println("Backward error: ", norm(R-A0))
