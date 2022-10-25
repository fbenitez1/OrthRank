Base.@propagate_inbounds function InPlace.apply!(
  ::Type{RightProduct},
  ::Type{GeneralMatrix{E}},
  A::AbstractArray{E,2},
  (wy, k)::Tuple{WYTrans{E}, Int};
  offset = 0,
) where {E<:Number}

  num_WY = wy.num_WY[]
  @boundscheck k ∈ 1:num_WY || throw_WYBlockNotAvailable(k, num_WY)
  @inbounds begin
    inds = 1:wy.sizes[k]
    total_offset = wy.offsets[k] + offset
    (ma, na) = size(A)
    num_hs = wy.num_hs[k]
  end

  @boundscheck begin

    inds .+ total_offset ⊆ 1:na ||
      throw_ColumnRange_DimensionMismatch(ma, na, inds .+ total_offset)

    length(wy.work) >= ma * num_hs ||
      throw_WorkSizeError(ma, na, ma * num_hs, length(wy.work))

  end

  @inbounds begin
    @views begin
      work = reshape(wy.work[1:(ma * num_hs)], ma, num_hs)
      W = wy.W[inds, 1:num_hs, k]
      Y = wy.Y[inds, 1:num_hs, k]
      A0 = A[:, inds .+ total_offset]
    end
    oneE = one(E)
    num_hs > 0 && length(inds) > 0 && begin
      matmul!(work, A0, W)
      matmul!(A0, work, Y', -oneE, oneE)
    end
  end
  nothing
end

Base.@propagate_inbounds function InPlace.apply!(
  ::Type{RightProduct},
  ::Type{GeneralMatrix{E}},
  A::AbstractArray{E,2},
  wy::WYTrans{E};
  offset = 0,
) where {E<:Number}
  InPlace.apply!(
    RightProduct,
    GeneralMatrix{E},
    A,
    (wy, wy.active_WY[]),
    offset = offset,
  )
end

Base.@propagate_inbounds function InPlace.apply_inv!(
  ::Type{RightProduct},
  ::Type{GeneralMatrix{E}},
  A::AbstractArray{E,2},
  (wy, k)::Tuple{WYTrans{E}, Int};
  offset = 0,
) where {E<:Number}

  num_WY = wy.num_WY[]
  @boundscheck k ∈ 1:num_WY || throw_WYBlockNotAvailable(k, num_WY)
  @inbounds begin
    num_hs = wy.num_hs[k]
    total_offset = wy.offsets[k] + offset
    inds = 1:wy.sizes[k]
    (ma, na) = size(A)
  end

  @boundscheck begin

    inds .+ total_offset ⊆ 1:na ||
      throw_ColumnRange_DimensionMismatch(ma, na, inds .+ total_offset)

    length(wy.work) >= ma * num_hs ||
      throw_WorkSizeError(ma, na, ma * num_hs, length(wy.work))

  end

  @inbounds begin
    @views begin
      work = reshape(wy.work[1:(ma * num_hs)], ma, num_hs)
      W = wy.W[inds, 1:num_hs, k]
      Y = wy.Y[inds, 1:num_hs, k]
      A0 = A[:, inds .+ total_offset]
    end
    oneE = one(E)
    num_hs > 0 && length(inds) > 0 && begin
      matmul!(work, A0, Y)
      matmul!(A0, work, W', -oneE, oneE)
    end
  end
  nothing
end

Base.@propagate_inbounds function InPlace.apply_inv!(
  ::Type{RightProduct},
  ::Type{GeneralMatrix{E}},
  A::AbstractArray{E,2},
  wy::WYTrans{E};
  offset = 0,
) where {E<:Number}
  apply_inv!(
    RightProduct,
    GeneralMatrix{E},
    A,
    (wy, wy.active_WY[]),
    offset = offset,
  )
end

Base.@propagate_inbounds function InPlace.apply!(
  ::Type{LeftProduct},
  ::Type{GeneralMatrix{E}},
  (wy, k)::Tuple{WYTrans{E}, Int},
  A::AbstractArray{E,2};
  offset = 0,
) where {E<:Number}

  num_WY = wy.num_WY[]
  @boundscheck k ∈ 1:num_WY || throw_WYBlockNotAvailable(k, num_WY)
  @inbounds begin
    num_hs = wy.num_hs[k]
    total_offset = wy.offsets[k] + offset
    inds = 1:wy.sizes[k]
    (ma, na) = size(A)
  end

  @boundscheck begin

    inds .+ total_offset ⊆ 1:ma ||
      throw_RowRange_DimensionMismatch(ma, na, inds .+ total_offset)

    length(wy.work) >= na * num_hs ||
      throw_WorkSizeError(ma, na, na * num_hs, length(wy.work))

  end

  @inbounds begin
    @views begin
      work = reshape(wy.work[1:(na * num_hs)], num_hs, na)
      W = wy.W[inds, 1:num_hs, k]
      Y = wy.Y[inds, 1:num_hs, k]
      A0 = A[inds .+ total_offset, :]
    end
    oneE = one(E)
    num_hs > 0 && length(inds) > 0 && begin
      matmul!(work, Y', A0)
      matmul!(A0, W, work, -oneE, oneE)
    end
  end
  nothing
end

Base.@propagate_inbounds function InPlace.apply!(
  ::Type{LeftProduct},
  ::Type{GeneralMatrix{E}},
  wy::WYTrans{E},
  A::AbstractArray{E,2};
  offset = 0,
) where {E<:Number}
  InPlace.apply!(
    LeftProduct,
    GeneralMatrix{E},
    (wy, wy.active_WY[]),
    A,
    offset = offset,
  )
end

Base.@propagate_inbounds function InPlace.apply_inv!(
  ::Type{LeftProduct},
  ::Type{GeneralMatrix{E}},
  (wy, k)::Tuple{WYTrans{E}, Int},
  A::AbstractArray{E,2};
  offset = 0,
) where {E<:Number}

  num_WY = wy.num_WY[]
  @boundscheck k ∈ 1:num_WY || throw_WYBlockNotAvailable(k, num_WY)

  @inbounds begin
    num_hs = wy.num_hs[k]
    total_offset = wy.offsets[k] + offset
    inds = 1:wy.sizes[k]
    (ma, na) = size(A)
  end
  @boundscheck begin
    inds .+ total_offset ⊆ 1:ma ||
      throw_RowRange_DimensionMismatch(ma, na, inds .+ total_offset)

    length(wy.work) >= na * num_hs ||
      throw_WorkSizeError(ma, na, na * num_hs, length(wy.work))

  end

  @inbounds begin
    @views begin
      work = reshape(wy.work[1:(na * num_hs)], num_hs, na)
      W = wy.W[inds, 1:num_hs, k]
      Y = wy.Y[inds, 1:num_hs, k]
      A0 = A[inds .+ total_offset, :]
    end
    oneE = one(E)
    num_hs > 0 && length(inds) > 0 && begin
      matmul!(work, W', A0)
      matmul!(A0, Y, work, -oneE, oneE)
    end
  end
  nothing
end

Base.@propagate_inbounds function InPlace.apply_inv!(
  ::Type{LeftProduct},
  ::Type{GeneralMatrix{E}},
  wy::WYTrans{E},
  A::AbstractArray{E,2};
  offset = 0,
) where {E<:Number}
  InPlace.apply_inv!(
    LeftProduct,
    GeneralMatrix{E},
    (wy, wy.active_WY[]),
    A,
    offset = offset,
  )
end
