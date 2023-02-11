module Precompile

using InPlace
using Random
using LinearAlgebra
using Householder

function run_householder(E)
  function run_householder_matrix(A)
    n = size(A,2)
    h =  lhouseholder(A[:,1],1,0,n)
    h⊘A
    column_nonzero!(A,1,1)
    h = rhouseholder(A[1,:],1,0,n)
    A⊛h
    row_nonzero!(A,1,1)
  end
  A = ones(E, 8, 8)
  run_householder_matrix(A)
  run_householder_matrix(A')
  run_householder_matrix(view(A,1:4,1:4))
  run_householder_matrix(view(A',1:4,1:4))
  run_householder_matrix(view(A,1:4,1:4)')
end

function run_WY_general(E)
  function run_WY_general_matrix(a)
    m = size(a,1)
    n = size(a,2)
    max_num_hs=2
    num_blocks=2
    bs = m ÷ num_blocks
    wy1 = WYTrans(
      eltype(a),
      max_num_WY = num_blocks,
      max_WY_size = bs,
      work_size = n * max_num_hs,
      max_num_hs = max_num_hs,
    )
    for l ∈ 1:num_blocks
      resetWYBlock!(
        wy1,
        block = l,
        offset = (l - 1) * bs,
        sizeWY = bs,
      )
      wy1.num_hs[l] = max_num_hs
    end
    rand!(wy1)
    SweepForward(wy1) ⊛ a
    SweepForward(wy1) ⊘ a
  end
  A = ones(E, 8, 8)
  run_WY_general_matrix(A)
  run_WY_general_matrix(A')
  run_WY_general_matrix(view(A,1:4,1:4))
  run_WY_general_matrix(view(A',1:4,1:4))
  run_WY_general_matrix(view(A,1:4,1:4)')
end

function run_WYWY(E)
  max_num_hs = 2
  m = 18

  range1 = 3:13
  size1 = length(range1)
  offset1 = first(range1) - 1
  num_hs1 = 3
  max_num_wy1 = 3
  l1 = 3

  range2 = 5:10
  offset2 = first(range2) - 1
  size2 = length(range2)
  num_hs2 = 2
  max_num_wy2 = 4
  l2 = 2

  max_num_hs1 = num_hs1 + 4 * num_hs2
  max_num_hs2 = num_hs2

  wy1 = WYTrans(
    E,
    max_num_WY = max_num_wy1,
    max_WY_size = m,
    work_size = m * max_num_hs1,
    max_num_hs = max_num_hs1,
  )

  wy2 = WYTrans(
    E,
    max_num_WY = max_num_wy2,
    max_WY_size = m,
    work_size = m * max_num_hs1,
    max_num_hs = max_num_hs2,
  )
  resetWYBlock!(wy1, block = l1, offset = offset1, sizeWY = size1)
  selectWY!(wy1, l1)
  wy1.num_hs[l1] = num_hs1
  rand!(wy1)

  resetWYBlock!(wy2, block = l2, offset = offset2, sizeWY = size2)
  selectWY!(wy2, l2)
  wy2.num_hs[l2] = num_hs2
  rand!(wy2)
  wy1 ⊛ Linear(wy2)
  wy1 ⊘ Linear(wy2)
  Linear(wy2) ⊛ wy1
  Linear(wy2) ⊘ wy1
end

function run_small_QR(E)
  max_num_hs = 3
  m = 10
  n = 10
  A = randn(E, m, n)
  A0 = copy(A)
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
  Iₘ = Matrix{E}(I, m, m)
  work = zeros(E, m)

  for j = 1:3
    h = lhouseholder(A[j:m, j], 1, j - 1, work)
    h ⊘ A
    (wy1, 1) ⊛ h
    h ⊘ (wy2, 1)
  end

  Q1 = copy(Iₘ)
  Q1 ⊛ (wy1, 1)
  Q2 = copy(Iₘ)
  Q2 ⊘ (wy2, 1)
end

function run_qrWY(E)
  m = 100
  n = 90
  A = randn(E, m, n)
  A0 = copy(A)
  A[:, :] = A0
  (Q, R) = qrWY(A)
end

function run_qrWYSweep(E)
  m = 100
  n = 90
  A = randn(E, m, n)
  A0 = copy(A)
  A[:, :] = A0
  (Q, R) = qrWYSweep(A)
  Q ⊛ R
end

function run_all()

  run_householder(Float64)  
  run_householder(Complex{Float64})

  run_WY_general(Float64)
  run_WY_general(Complex{Float64})

  run_WYWY(Float64)
  run_WYWY(Complex{Float64})

  run_small_QR(Float64)
  run_qrWY(Float64)
  run_qrWYSweep(Float64)
  
  run_small_QR(Complex{Float64})
  run_qrWY(Complex{Float64})
  run_qrWYSweep(Complex{Float64})

end

end
