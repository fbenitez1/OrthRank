module GivensWeightQR

using Random,
  OrthWeight, BandStruct, Householder, InPlace, Rotations, LinearAlgebra

export random_blocks_generator,
  get_ranks,
  preparative_phase!,
  residual_phase!,
  swap,
  apply_rotations_forward!,
  apply_rotations_backward!,
  create_Q!,
  create_R,
  solve!,
  QRGivensWeight,
  QGivensWeight

struct QGivensWeight{M<:AbstractMatrix}
  Qb::M
  Qt::M
end

struct QRGivensWeight{S<:QGivensWeight,R<:AbstractGivensWeight} <:
       Factorization{R}
  Q::S
  R::R
end

function Base.:\(A::GivensWeight, b::AbstractVecOrMat)
  m, n = size(A.b)
  x = zeros(Float64, n)
  F = qr(A)
  for i = 1:min(m, n)
    F.R.b[i, i] == 0 ? throw(SingularException(i)) : nothing
  end
  ldiv!(x, F, b)
  return x
end

function LinearAlgebra.qr(gw::GivensWeight)
  preparative_phase!(gw)
  triang_rot = residual_phase!(gw)
  Qb = QGivensWeight(gw.lowerRots, triang_rot)
  return QRGivensWeight(Qb, gw)
end

function LinearAlgebra.ldiv!(
  x::AbstractVector,
  A::QRGivensWeight,
  b::AbstractMatrix,
)
  solve!(x, A, b)
end

function LinearAlgebra.ldiv!(
  x::AbstractVector,
  A::QRGivensWeight,
  b::AbstractVector,
)
  solve!(x, A, b)
end

function LinearAlgebra.ldiv!(A::QRGivensWeight, b::AbstractVector)
  solve!(A, b)
end

function LinearAlgebra.ldiv!(A::QRGivensWeight, b::AbstractMatrix)
  solve!(A, b)
end

function solve!(x::Vector{Float64}, A::QRGivensWeight, b::AbstractMatrix)
  m, n = size(A.R.b)
  s = min(m, n)
  x_copy = zeros(Float64, n, 1)

  #Apply lower block rotations to b
  for j in Iterators.reverse(axes(A.Q.Qb, 2))
    for i in axes(A.Q.Qb, 1)
      if A.Q.Qb[i, j].inds != 0
        apply_inv!(A.Q.Qb[i, j], b)
      end
    end
  end

  #Apply triangularizing rotations to b
  for j in axes(A.Q.Qt, 2)
    for i in axes(A.Q.Qt, 1)
      if A.Q.Qt[i, j].inds != 0
        apply_inv!(A.Q.Qt[i, j], b)
      end
    end
  end

  #Backward substitution
  x[s, 1] = b[s, 1] / A.R.b[s, s]
  x_copy[s, 1] = x[s, 1]
  ub_g_ind = length(A.R.b.upper_blocks)
  ub_mb = A.R.b.upper_blocks[ub_g_ind].mb
  for i = (s - 1):-1:1
    while i == ub_mb
      ub_rot_num = A.R.b.upper_blocks[ub_g_ind].num_rots
      for r = 1:ub_rot_num
        rot = A.R.upperRots[r, ub_g_ind]
        apply_inv!(rot, x_copy)
      end
      ub_g_ind -= 1
      if ub_g_ind == 0
        ub_mb = 0
      else
        ub_mb = A.R.b.upper_blocks[ub_g_ind].mb
      end
    end
    last_el_in_row_ind = last_inband_index(A.R.b, i, :)
    while A.R.b[i, last_el_in_row_ind] == 0.0
      last_el_in_row_ind -= 1
    end
    x[i, 1] =
      A.R.b[i, i] \ (
        b[i, 1] - sum(
          A.R.b[i, (i + 1):last_el_in_row_ind] .*
          x_copy[(i + 1):last_el_in_row_ind, 1],
        )
      )
    x_copy[i, 1] = x[i, 1]
  end

  return x
end

function solve!(x::Vector{Float64}, A::QRGivensWeight, b::AbstractVector)
  m, n = size(A.R.b)
  s = min(m, n)
  x_copy = zeros(Float64, n)

  #Apply lower block rotations to b
  for j in Iterators.reverse(axes(A.Q.Qb, 2))
    for i in axes(A.Q.Qb, 1)
      if A.Q.Qb[i, j].inds != 0
        apply_inv!(A.Q.Qb[i, j], b)
      end
    end
  end

  #Apply triangularizing rotations to b
  for j in axes(A.Q.Qt, 2)
    for i in axes(A.Q.Qt, 1)
      if A.Q.Qt[i, j].inds != 0
        apply_inv!(A.Q.Qt[i, j], b)
      end
    end
  end

  #Backward substitution
  x[s] = b[s] / A.R.b[s, s]
  x_copy[s] = x[s]
  ub_g_ind = length(A.R.b.upper_blocks)
  ub_mb = A.R.b.upper_blocks[ub_g_ind].mb
  for i = (s - 1):-1:1
    while i == ub_mb
      ub_rot_num = A.R.b.upper_blocks[ub_g_ind].num_rots
      for r = 1:ub_rot_num
        rot = A.R.upperRots[r, ub_g_ind]
        apply_inv!(rot, x_copy)
      end
      ub_g_ind -= 1
      if ub_g_ind == 0
        ub_mb = 0
      else
        ub_mb = A.R.b.upper_blocks[ub_g_ind].mb
      end
    end
    last_el_in_row_ind = last_inband_index(A.R.b, i, :)
    while A.R.b[i, last_el_in_row_ind] == 0.0
      last_el_in_row_ind -= 1
    end
    x[i] =
      A.R.b[i, i] \ (
        b[i] - sum(
          A.R.b[i, (i + 1):last_el_in_row_ind] .*
          x_copy[(i + 1):last_el_in_row_ind],
        )
      )
    x_copy[i] = x[i]
  end

  return x
end

function solve!(A::QRGivensWeight, b::AbstractMatrix)
  m, n = size(A.R.b)
  m != n ? throw(DimensionMismatch("A must be a square matrix")) : nothing
  m != size(b, 1) ?
  throw(DimensionMismatch("inputs should have the same number of rows")) :
  nothing
  b_copy = zeros(Float64, n, 1)

  #Apply lower block rotations to b
  for j in Iterators.reverse(axes(A.Q.Qb, 2))
    for i in axes(A.Q.Qb, 1)
      if A.Q.Qb[i, j].inds != 0
        apply_inv!(A.Q.Qb[i, j], b)
      end
    end
  end

  #Apply triangularizing rotations to b
  for j in axes(A.Q.Qt, 2)
    for i in axes(A.Q.Qt, 1)
      if A.Q.Qt[i, j].inds != 0
        apply_inv!(A.Q.Qt[i, j], b)
      end
    end
  end

  #Backward substitution
  b[m, 1] = b[m, 1] / A.R.b[m, m]
  b_copy[m, 1] = b[m, 1]
  ub_g_ind = length(A.R.b.upper_blocks)
  ub_mb = A.R.b.upper_blocks[ub_g_ind].mb
  for i = (m - 1):-1:1
    while i == ub_mb
      ub_rot_num = A.R.b.upper_blocks[ub_g_ind].num_rots
      for r = 1:ub_rot_num
        rot = A.R.upperRots[r, ub_g_ind]
        apply_inv!(rot, b_copy)
      end
      ub_g_ind -= 1
      if ub_g_ind == 0
        ub_mb = 0
      else
        ub_mb = A.R.b.upper_blocks[ub_g_ind].mb
      end
    end
    last_el_in_row_ind = last_inband_index(A.R.b, i, :)
    while A.R.b[i, last_el_in_row_ind] == 0.0
      last_el_in_row_ind -= 1
    end
    b[i, 1] =
      A.R.b[i, i] \ (
        b[i, 1] - sum(
          A.R.b[i, (i + 1):last_el_in_row_ind] .*
          b_copy[(i + 1):last_el_in_row_ind, 1],
        )
      )
    b_copy[i, 1] = b[i, 1]
  end

  return b
end

function solve!(A::QRGivensWeight, b::AbstractVector)
  m, n = size(A.R.b)
  m != n ? throw(DimensionMismatch("A must be a square matrix")) : nothing
  m != size(b, 1) ?
  throw(DimensionMismatch("inputs should have the same number of rows")) :
  nothing
  b_copy = zeros(Float64, n)
  #Apply lower block rotations to b
  for j in Iterators.reverse(axes(A.Q.Qb, 2))
    for i in axes(A.Q.Qb, 1)
      if A.Q.Qb[i, j].inds != 0
        apply_inv!(A.Q.Qb[i, j], b)
      end
    end
  end
  #Apply triangularizing rotations to b
  for j in axes(A.Q.Qt, 2)
    for i in axes(A.Q.Qt, 1)
      if A.Q.Qt[i, j].inds != 0
        apply_inv!(A.Q.Qt[i, j], b)
      end
    end
  end
  #Backward substitution
  b[m] = b[m] / A.R.b[m, m]
  b_copy[m] = b[m]
  ub_g_ind = length(A.R.b.upper_blocks)
  ub_mb = A.R.b.upper_blocks[ub_g_ind].mb
  for i = (m - 1):-1:1
    while i == ub_mb
      ub_rot_num = A.R.b.upper_blocks[ub_g_ind].num_rots
      for r = 1:ub_rot_num
        rot = A.R.upperRots[r, ub_g_ind]
        apply_inv!(rot, b_copy)
      end
      ub_g_ind -= 1
      if ub_g_ind == 0
        ub_mb = 0
      else
        ub_mb = A.R.b.upper_blocks[ub_g_ind].mb
      end
    end
    last_el_in_row_ind = last_inband_index(A.R.b, i, :)
    while A.R.b[i, last_el_in_row_ind] == 0.0
      last_el_in_row_ind -= 1
    end
    b[i] =
      A.R.b[i, i] \ (
        b[i] - sum(
          A.R.b[i, (i + 1):last_el_in_row_ind] .*
          b_copy[(i + 1):last_el_in_row_ind],
        )
      )
    b_copy[i] = b[i]
  end

  return b
end

macro swap(x, y)
  quote
    local tmp = $(esc(x))
    $(esc(x)) = $(esc(y))
    $(esc(y)) = tmp
  end
end

function random_blocks_generator(m::Int64, num_bl::Int64)
  num_bl <= m ? nothing :
  throw("Number of blocks $num_bl exceed the matrix row size $m.")
  num_bl > 0 ? nothing :
  throw("Number of blocks $num_bl cannot be zero or negative.")

  upper_rows = zeros(Int64, 1, num_bl)
  upper_cols = zeros(Int64, 1, num_bl)
  lower_rows = zeros(Int64, 1, num_bl)
  lower_cols = zeros(Int64, 1, num_bl)
  delta_m = m ÷ num_bl

  for i = 1:num_bl
    upper_rows[1, i] = rand(((i - 1) * delta_m + 1):(i * delta_m))
    upper_cols[1, i] = rand(((i - 1) * delta_m + 1):(i * delta_m))
    lower_rows[1, i] = rand(((i - 1) * delta_m + 1):(i * delta_m))
    lower_cols[1, i] = rand(((i - 1) * delta_m + 1):(i * delta_m))
    upper_rows[1, i] <= lower_rows[1, i] ||
      lower_cols[1, i] <= upper_cols[1, i] ?
    nothing : upper_rows[1, i] = lower_rows[1, i] #No block overlap
    #No block containing diagonal (i, i) element
    upper_rows[1, i] <= upper_cols[1, i] ? nothing :
    @swap(upper_rows[1, i], upper_cols[1, i])
    #No block containing diagonal (i, i) element
    lower_cols[1, i] <= lower_rows[1, i] ? nothing :
    @swap(lower_cols[1, i], lower_rows[1, i])
  end

  upper_block = givens_block_sizes([
    upper_rows
    upper_cols
  ])
  lower_block = givens_block_sizes([
    lower_rows
    lower_cols
  ])

  return upper_block, lower_block
end

function random_blocks_generator(m::Int64, n::Int64, num_bl::Int64)
  num_bl <= m ? nothing :
  throw("Number of blocks $num_bl exceed the matrix row size $m.")
  num_bl <= n ? nothing :
  throw("Number of blocks $num_bl exceed the matrix column size $n.")
  num_bl > 0 ? nothing :
  throw("Number of blocks $num_bl cannot be zero or negative.")

  s = min(m, n)
  upper_rows = zeros(Int64, 1, num_bl)
  upper_cols = zeros(Int64, 1, num_bl)
  lower_rows = zeros(Int64, 1, num_bl)
  lower_cols = zeros(Int64, 1, num_bl)
  delta = s ÷ num_bl

  for i = 1:num_bl
    upper_rows[1, i] = rand(((i - 1) * delta + 1):(i * delta))
    upper_cols[1, i] = rand(((i - 1) * delta + 1):(i * delta))
    lower_rows[1, i] = rand(((i - 1) * delta + 1):(i * delta))
    lower_cols[1, i] = rand(((i - 1) * delta + 1):(i * delta))
    upper_rows[1, i] <= lower_rows[1, i] ||
      lower_cols[1, i] <= upper_cols[1, i] ?
    nothing : upper_rows[1, i] = lower_rows[1, i] #No block overlap
    #No block containing diagonal (i, i) element
    upper_rows[1, i] <= upper_cols[1, i] ? nothing :
    @swap(upper_rows[1, i], upper_cols[1, i])
    #No block containing diagonal (i, i) element
    lower_cols[1, i] <= lower_rows[1, i] ? nothing :
    @swap(lower_cols[1, i], lower_rows[1, i])
  end

  upper_block = givens_block_sizes([
    upper_rows
    upper_cols
  ])
  lower_block = givens_block_sizes([
    lower_rows
    lower_cols
  ])

  return upper_block, lower_block
end

function get_ranks(gw::GivensWeight)
  num_lower_blocks = length(gw.b.lower_blocks)
  num_upper_blocks = length(gw.b.upper_blocks)
  lower_ranks = zeros(Int64, num_lower_blocks)
  upper_ranks = zeros(Int64, num_upper_blocks)

  for i = 1:num_lower_blocks
    lower_ranks[i] = gw.b.lower_blocks.data[i].block_rank
  end

  for i = 1:num_upper_blocks
    upper_ranks[i] = gw.b.upper_blocks.data[i].block_rank
  end

  return lower_ranks, upper_ranks
end

function preparative_phase!(gw::GivensWeight)
  m, n = size(gw.b)
  ub_g_ind = length(gw.b.upper_blocks)

  while gw.b.upper_blocks[ub_g_ind].mb == m
    ub_g_ind -= 1
  end
  ub_mb = gw.b.upper_blocks[ub_g_ind].mb

  for lb_ind in filter_compressed(Iterators.Reverse(gw.b.lower_blocks))

    lb_g_ind = gw.b.lower_blocks[lb_ind].givens_index
    lb_rot_num = gw.b.lower_blocks[lb_g_ind].num_rots

    for lb_rot_ind = 1:lb_rot_num
      lb_rot = gw.lowerRots[lb_rot_ind, lb_g_ind]
      lb_rank = gw.b.lower_blocks[lb_g_ind].block_rank

      while lb_rot.inds <= ub_mb < lb_rot.inds + lb_rank
        ub_rot_num = gw.b.upper_blocks[ub_g_ind].num_rots
        ub_active_rows = (ub_mb + 1):(lb_rot.inds + lb_rank)

        for ub_rot_num = 1:ub_rot_num
          ub_rot = gw.upperRots[ub_rot_num, ub_g_ind]
          ub_active_cols = 1:(ub_rot.inds + 1)
          ub_vb = view(gw.b, ub_active_rows, ub_active_cols)
          apply!(ub_vb, ub_rot)
        end

        #Reduce fill-up on upper block because of extension.
        for ub_fill_up_row in ub_active_rows
          start_column_ind = last_inband_index(gw.b, ub_fill_up_row - 1, :)
          finish_column_ind = last_inband_index(gw.b, ub_fill_up_row, :)
          while gw.b[ub_fill_up_row - 1, start_column_ind] == 0.0
            start_column_ind -= 1
          end
          while gw.b[ub_fill_up_row, finish_column_ind] == 0.0
            finish_column_ind -= 1
          end

          fill_up_vb = view(gw.b, ub_active_rows, 1:finish_column_ind)
          for j = (finish_column_ind - 1):-1:(start_column_ind + 1)
            surv = j
            kill = j + 1
            fill_up_rot = rgivens(gw.b[ub_fill_up_row, surv:kill]..., surv)
            apply!(fill_up_vb, fill_up_rot)
            # Avoid NaN problem. Another solution: Fix notch_upper to notch
            # below diagonal.
            gw.b[ub_fill_up_row, kill] = 0.0
            gw.b.upper_blocks[ub_g_ind].num_rots += 1
            gw.upperRots[gw.b.upper_blocks[ub_g_ind].num_rots, ub_g_ind] = fill_up_rot
          end
        end

        gw.b.upper_blocks[ub_g_ind].mb = lb_rot.inds + lb_rank
        ub_new_last_col_ind = last_inband_index(gw.b, lb_rot.inds + lb_rank, :)

        while gw.b[lb_rot.inds + lb_rank, ub_new_last_col_ind] == 0.0
          ub_new_last_col_ind -= 1
        end

        gw.b.upper_blocks[ub_g_ind].block_rank =
          ub_new_last_col_ind - gw.b.upper_blocks[ub_g_ind].nb #New rank
        ub_g_ind -= 1
        ub_g_ind == 0 ? ub_mb = 0 : ub_mb = gw.b.upper_blocks[ub_g_ind].mb
      end

      #Apply lb_rot in the compl. of lb_block
      _, lb_cols = lower_block_ranges(gw.b, lb_g_ind)
      if lb_cols[end] != n
        last_col_el = last_inband_index(gw.b, lb_rot.inds + 1, :)
        while gw.b[lb_rot.inds + 1, last_col_el] == 0
          last_col_el -= 1
        end
        lb_active_cols = (lb_cols[end] + 1):last_col_el
        lb_active_rows = 1:(lb_rot.inds + 1)
        lb_vb = view(gw.b, lb_active_rows, lb_active_cols)
        apply_inv!(lb_rot, lb_vb)
      end
    end
  end
  return nothing
end

function rotations_needed(
    ::Type{E},
    B::AbstractBandColumn,
    n::Int64
    ) where {R<:Real, E <: Union{R, Complex{R}}}
    m = 0
    for d_index in 1:n
        last_el_index = last_inband_index(B, :, d_index)
        amount  = last_el_index - d_index
        amount > m ? m = amount : nothing
    end
    matrix_of_rotations = Matrix{Rot{R, E, Int}}(undef, m, n)
    matrix_of_rotations .= Rot(zero(R), zero(E), 0)
    return matrix_of_rotations
end

function residual_phase!(gw::GivensWeight)
  m, n = size(gw.b)
  diag_size = min(m, n)
  triang_rot = rotations_needed(Float64, gw.b, diag_size)

  for d_ind = 1:diag_size
    last_el_und_diag = last_inband_index(gw.b, :, d_ind)
    num_el_und_diag = last_el_und_diag - d_ind
    last_el_und_diag == m ? virtual_last_el_und_diag = m + 1 :
    virtual_last_el_und_diag = last_el_und_diag
    # Regression action
    for ub_ind in filter_compressed(gw.b.upper_blocks)
      ub_g_ind = gw.b.upper_blocks[ub_ind].givens_index
      ub_mb = gw.b.upper_blocks[ub_g_ind].mb
      if d_ind <= ub_mb < virtual_last_el_und_diag
        ub_vb = view(gw.b, d_ind:ub_mb, :)
        for ub_rot_num = (gw.b.upper_blocks[ub_g_ind].num_rots):-1:1
          ub_rot = gw.upperRots[ub_rot_num, ub_g_ind]
          apply_inv!(ub_vb, ub_rot)
        end
        gw.b.upper_blocks[ub_ind].mb = d_ind - 1
      end
    end

    #Create zeros under diagonal
    for k = num_el_und_diag:-1:1
      surv = d_ind + k - 1
      kill = d_ind + k
      rot = lgivens(gw.b[surv:kill, d_ind]..., surv)
      last_el_in_row = last_inband_index(gw.b, d_ind + num_el_und_diag, :)
      vb = view(gw.b, 1:last_el_und_diag, d_ind:last_el_in_row)
      apply_inv!(rot, vb)
      triang_rot[1 + num_el_und_diag - k, d_ind] = rot
      #abs(gw.b[kill, d_ind])<1e-15 ? gw.b[kill, d_ind]=0.0 : nothing
    end
  end
  return triang_rot
end

function apply_rotations_forward!(A::Matrix, rot::Matrix)
  for j in axes(rot, 2)
    for i in Iterators.reverse(axes(rot, 1))
      if rot[i, j].inds != 0
        apply!(rot[i, j], A)
      end
    end
  end
  return nothing
end

function apply_rotations_backward!(A::Matrix, rot::Matrix)
  for j in Iterators.reverse(axes(rot, 2))
    for i in Iterators.reverse(axes(rot, 1))
      if rot[i, j].inds != 0
        apply!(rot[i, j], A)
      end
    end
  end
  return nothing
end

function create_Q!(A::Matrix, F::QRGivensWeight)
  apply_rotations_backward!(A, F.Q.Qt)
  apply_rotations_forward!(A, F.Q.Qb)
  return nothing
end


function create_R(F::QRGivensWeight)
  F.R.lowerRots .= Rot(one(Float64), zero(Float64), 1)
  A = Matrix(F.R)
  return A
end

end #module
