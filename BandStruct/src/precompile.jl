using InPlace
using Rotations
using Householder
using Random
using LinearAlgebra

function run_first_last_init()
  blocks = transpose([
    2 2
    5 5
  ])
  get_cols_first_last(11, 14, blocks, blocks, 2, 2)
  get_rows_first_last(11, 14, blocks, blocks, 2, 2)
end

function run_wilkinson(::Type{E}) where {E}
  bc0, bbc0 = BandStruct.standard_test_case(E)
  wilk(toBandColumn(bbc0))
  wilk(toBandColumn(bc0))
  bc0T, bbc0T =
    BandStruct.standard_test_case(E, decomp_type = TrailingDecomp)
  wilk(toBandColumn(bbc0T))
  wilk(toBandColumn(bc0T))
end

function run_index(::Type{E}) where {E}
  bc0, bbc0 =
    BandStruct.standard_test_case(E, upper_rank_max = 3, lower_rank_max = 2)
  eachindex(bc0)
  eachindex(bbc0)
  bc0[1,1] = zero(E)
  bbc0[1,1] = zero(E)
end

function run_submatrix(::Type{E}) where {E}
  bc0, bbc0 =
    BandStruct.standard_test_case(E, upper_rank_max = 3, lower_rank_max = 2)
  Matrix(bc0[1:2,1:2])
  Matrix(view(bc0,1:2,1:2))
  Matrix(bbc0[1:2,1:2])
  Matrix(view(bbc0,1:2,1:2))

  Matrix(bc0[:,1:2])
  Matrix(view(bc0,:,1:2))
  Matrix(bbc0[:,1:2])
  Matrix(view(bbc0,:,1:2))

  Matrix(bc0[1:2,:])
  Matrix(view(bc0,1:2,:))
  Matrix(bbc0[1:2,:])
  Matrix(view(bbc0,1:2,:))
end

function run_range(::Type{E}) where {E}
  bc0, bbc0 =
    BandStruct.standard_test_case(E, upper_rank_max = 3, lower_rank_max = 2)
  inband_index_range(bc0, :, 1)
  upper_inband_index_range(bc0, :, 1)
  middle_inband_index_range(bc0, :, 1)
  lower_inband_index_range(bc0, :, 1)
  inband_index_range(bc0, 1, :)
  upper_inband_index_range(bc0, 1, :)
  middle_inband_index_range(bc0, 1, :)
  lower_inband_index_range(bc0, 1, :)

  inband_index_range(bbc0, :, 1)
  upper_inband_index_range(bbc0, :, 1)
  middle_inband_index_range(bbc0, :, 1)
  lower_inband_index_range(bbc0, :, 1)
  inband_index_range(bbc0, 1, :)
  upper_inband_index_range(bbc0, 1, :)
  middle_inband_index_range(bbc0, 1, :)
  lower_inband_index_range(bbc0, 1, :)

  vbc = view(bc0,1:2, 1:2)
  vbbc = view(bbc0,1:2, 1:2)
  inband_index_range(vbc, :, 1)
  upper_inband_index_range(vbc, :, 1)
  middle_inband_index_range(vbc, :, 1)
  lower_inband_index_range(vbc, :, 1)
  inband_index_range(vbc, 1, :)
  upper_inband_index_range(vbc, 1, :)
  middle_inband_index_range(vbc, 1, :)
  lower_inband_index_range(vbc, 1, :)

  inband_index_range(vbbc, :, 1)
  upper_inband_index_range(vbbc, :, 1)
  middle_inband_index_range(vbbc, :, 1)
  lower_inband_index_range(vbbc, :, 1)
  inband_index_range(vbbc, 1, :)
  upper_inband_index_range(vbbc, 1, :)
  middle_inband_index_range(vbbc, 1, :)
  lower_inband_index_range(vbbc, 1, :)

  sbc = bc0[1:2, 1:2]
  sbbc = bbc0[1:2, 1:2]
  inband_index_range(sbc, :, 1)
  upper_inband_index_range(sbc, :, 1)
  middle_inband_index_range(sbc, :, 1)
  lower_inband_index_range(sbc, :, 1)
  inband_index_range(sbc, 1, :)
  upper_inband_index_range(sbc, 1, :)
  middle_inband_index_range(sbc, 1, :)
  lower_inband_index_range(sbc, 1, :)

  inband_index_range(sbbc, :, 1)
  upper_inband_index_range(sbbc, :, 1)
  middle_inband_index_range(sbbc, :, 1)
  lower_inband_index_range(sbbc, :, 1)
  inband_index_range(sbbc, 1, :)
  upper_inband_index_range(sbbc, 1, :)
  middle_inband_index_range(sbbc, 1, :)
  lower_inband_index_range(sbbc, 1, :)
end

function run_bulge(::Type{E}) where {E}
  (bc0, bbc0) =
    BandStruct.standard_test_case(E, upper_rank_max = 3, lower_rank_max = 2)

  bc = copy(bc0)
  bulge!(bc, :, 1:2)
  bulge!(bc, 1:2, :)
  validate_rows_first_last(bc)
  bc = copy(bc0)
  bulge!(bc, 1, 6)

  bbc = copy(bbc0)
  bulge!(bbc, :, 1:2)
  bulge!(bbc, 1:2, :)
  validate_rows_first_last(bbc)
  bbc = copy(bbc0)
  bulge!(bbc, 1, 6)

  vbc = view(copy(bc0), 1:6, 1:6)
  bulge!(vbc, :, 1:2)
  bulge!(vbc, 1:2, :)
  bulge!(vbc, 1, 6)
  sbc = view(copy(bc0), 1:6, 1:6)
  bulge!(vbc, 1, 6)

  sbc = view(copy(bc0), 1:6, 1:6)
  bulge!(sbc, :, 1:2)
  bulge!(sbc, 1:2, :)
  bulge!(sbc, 1, 6)
  sbc = view(copy(bc0), 1:6, 1:6)
  bulge!(sbc, 1, 6)


end

function run_notch(::Type{E}) where {E}
  (bc0, bbc0) =
    BandStruct.standard_test_case(E, upper_rank_max = 3, lower_rank_max = 2)

  bc = copy(bc0)
  notch_upper!(bc, 1, 4)
  notch_lower!(bc, 6, 4)

  bbc = copy(bbc0)
  notch_upper!(bbc, 1, 4)
  notch_lower!(bbc, 6, 4)

  vbc = view(copy(bc0), 1:6, 1:6)
  notch_upper!(vbc, 1, 4)
  notch_lower!(vbc, 6, 4)

  sbc = copy(bc0)[1:6, 1:6]
  notch_upper!(sbc, 1, 4)
  notch_lower!(sbc, 6, 4)

end

function run_rotations(::Type{E}) where {E}
  bc0, bbc0 =
    BandStruct.standard_test_case(E)
  bc = copy(bc0)
  r = rgivens(bc0[2,5], bc0[2,6],5)
  bc ⊛ r
  bc ⊘ r
  bc = copy(bc0)
  r = lgivens(bc[6,4], bc[7,4], 6)
  r ⊘ bc
  r ⊛ bc

  bbc = copy(bbc0)
  r = rgivens(bbc0[2,5], bbc0[2,6],5)
  bbc ⊛ r
  bbc ⊘ r
  bbc = copy(bbc0)
  r = lgivens(bbc[6,4], bbc[7,4], 6)
  r ⊘ bbc
  r ⊛ bbc

  bc = view(copy(bc0), 1:6,1:6)
  r = rgivens(bc0[2,5], bc0[2,6],5)
  bc ⊛ r
  bc ⊘ r
  bc = copy(bc0)
  r = lgivens(bc[6,4], bc[7,4], 6)
  r ⊘ bc
  r ⊛ bc

  bc = copy(bc0)[1:6,1:6]
  r = rgivens(bc0[2,5], bc0[2,6],5)
  bc ⊛ r
  bc ⊘ r
  bc = copy(bc0)
  r = lgivens(bc[6,4], bc[7,4], 6)
  r ⊘ bc
  r ⊛ bc
end

function run_householder(::Type{E}) where {E}
  (bc0, bbc0) =
    BandStruct.standard_test_case(E, upper_rank_max = 2, lower_rank_max = 2)
  work = zeros(E, maximum(size(bc0)))
  v = zeros(E, 3)
  bc = copy(bc0)
  h = householder(bc, 2:4, 2, 1, 1, v, work)
  h ⊘ bc
  h ⊛ bc
  bc = copy(bc0)
  h = householder(bc, 4, 5:7, 3, 4, v, work)
  bc ⊛ h
  bc ⊘ h

  bbc = copy(bbc0)
  h = householder(bbc, 2:4, 2, 1, 1, v, work)
  h ⊘ bbc
  h ⊛ bbc
  bbc = copy(bbc0)
  h = householder(bbc, 4, 5:7, 3, 4, v, work)
  bbc ⊛ h
  bbc ⊘ h

  bc = view(copy(bc0), 1:6, 1:6)
  h = householder(bc, 2:4, 2, 1, 1, v, work)
  h ⊘ bc
  h ⊛ bc
  bc = copy(bc0)
  h = householder(bc, 4, 5:7, 3, 4, v, work)
  bc ⊛ h
  bc ⊘ h

  bc = copy(bc0)[1:6, 1:6]
  h = householder(bc, 2:4, 2, 1, 1, v, work)
  h ⊘ bc
  h ⊛ bc
  bc = copy(bc0)
  h = householder(bc, 4, 5:7, 3, 4, v, work)
  bc ⊛ h
  bc ⊘ h
end

function run_WY(::Type{E}) where {E}
  tol = 2e-15
  (bc, bbc) =
    BandStruct.standard_test_case(E, upper_rank_max = 2, lower_rank_max = 2)
  for bc0 ∈ [ bc, bbc ]

    (m, n) = size(bc0)
    mx_bc0 = Matrix(bc0)

    max_num_hs=2
    lw = maximum(size(bc0))
    work = zeros(E, lw)

    wyl = WYTrans(
      E,
      max_num_WY = 3,
      max_WY_size = m,
      work_size = n * (max_num_hs + m),
      max_num_hs = max_num_hs,
    )
    # set WY transformation 2 with offset 2 and size 3.
    selectWY!(wyl, 2)
    resetWYBlock!(wyl, block = 2, offset = 2, sizeWY = 3)

    # Do a QR factorization of bc[3:5,2:3]

    bcwy_35_23 = copy(bc0)
    bcwy_35_23h = copy(bc0)
    mx_bcwy_35_23 = Matrix(bcwy_35_23)

    bulge_lower!(bcwy_35_23, 5 , 2)
    h_35_2 = householder(
      bcwy_35_23,
      3:5,
      2,
      lw,
    )
    h_35_2 ⊘ bcwy_35_23h
    wyl ⊛ h_35_2
    wyl ⊘ bcwy_35_23

    h_45_3 = householder(
      bcwy_35_23,
      4:5,
      3,
      lw,
    )

    h_45_3 ⊘ bcwy_35_23h
    wyl ⊛ h_45_3

    bcwy_35_23w = copy(bc0)
    wyl ⊘ bcwy_35_23w

    # Right multiplication: Do an LQ factorization of bc[2:3,5:7]

    bcwy_23_57 = copy(bc0)
    bcwy_23_57[2,7]=one(E)
    bcwy_23_57[3,7]=2*one(E)

    bcwy_23_57h = copy(bcwy_23_57)
    bcwy_23_57w = copy(bcwy_23_57)
    mx_bcwy_23_57 = Matrix(bcwy_23_57)

    wyr = WYTrans(
      E,
      max_num_WY = 3,
      max_WY_size = m,
      work_size = n * (max_num_hs + m),
      max_num_hs = max_num_hs,
    )

    # set WY transformation 2 with offset 2 and size 3.
    selectWY!(wyr, 2)
    resetWYBlock!(wyr, block = 2, offset = 4, sizeWY = 3)

    h_2_57 = householder(
      bcwy_23_57,
      2,
      5:7,
      lw,
    )

    bcwy_23_57h ⊛ h_2_57
    wyr ⊛ h_2_57
    bcwy_23_57 ⊛ wyr
    
    h_3_67 = householder(
      bcwy_23_57,
      3,
      6:7,
      lw,
    )

    bcwy_23_57h ⊛ h_3_67
    wyr ⊛ h_3_67
    bcwy_23_57w ⊛ wyr
  end
end

function run_cases()

  run_first_last_init()
  run_wilkinson(Float64)
  run_wilkinson(Complex{Float64})
  run_index(Float64)
  run_index(Complex{Float64})
  run_submatrix(Float64)
  run_submatrix(Complex{Float64})
  run_range(Float64)
  run_range(Complex{Float64})
  run_bulge(Float64)
  run_bulge(Complex{Float64})
  run_notch(Float64)
  run_notch(Complex{Float64})
  run_rotations(Float64)
  run_rotations(Complex{Float64})
  run_householder(Float64)
  run_householder(Complex{Float64})
  run_WY(Float64)
  run_WY(Complex{Float64})
  nothing
end

run_cases()
