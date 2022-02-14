using InPlace
using Random
using LinearAlgebra

function run_cases()

  function run_householder(a)
    n = size(a,2)
    h = lhouseholder(a[:,1],1,0,n)
    h⊘a
    column_nonzero!(a,1,1)
    h = rhouseholder(a[1,:],1,0,n)
    a⊛h
    row_nonzero!(a,1,1)
  end
  A = ones(Float64, 8, 8)
  run_householder(A)
  run_householder(A')
  run_householder(view(A,1:4,1:4))
  run_householder(view(A',1:4,1:4))
  run_householder(view(A,1:4,1:4)')

  A = ones(Complex{Float64}, 8, 8)
  run_householder(A)
  run_householder(A')
  run_householder(view(A,1:4,1:4))
  run_householder(view(A',1:4,1:4))
  run_householder(view(A,1:4,1:4)')

  function run_WY_general(a)
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
  A = ones(Float64, 8, 8)
  run_WY_general(A)
  run_WY_general(A')
  run_WY_general(view(A,1:4,1:4))
  run_WY_general(view(A',1:4,1:4))
  run_WY_general(view(A,1:4,1:4)')

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

  run_WYWY(Float64)
  run_WYWY(Complex{Float64})

end

run_cases()
