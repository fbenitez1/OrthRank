# WY times WY

# Apply transform l2 of wy2 to transform l1 of wy1.
Base.@propagate_inbounds function InPlace.apply!(
  ::Type{RightProduct},
  ::Type{WYTrans{E}},
  wy1::WYTrans{E},
  wy2::WYTrans{E};
  offset = 0,
) where {E<:Number}
  InPlace.apply!(
    RightProduct,
    WYTrans{E},
    (wy1, wy1.active_WY[]),
    (wy2, wy2.active_WY[]),
    offset = offset,
  )
end

# Apply transform l2 of wy2 to transform l1 of wy1.
Base.@propagate_inbounds function InPlace.apply!(
  ::Type{RightProduct},
  ::Type{WYTrans{E}},
  (wy1, l1)::Tuple{WYTrans{E}, Int},
  (wy2, l2)::Tuple{WYTrans{E}, Int};
  offset = 0
) where {E<:Number}

  # Get number of available blocks.
  wy1_num_WY = wy1.num_WY[]
  wy2_num_WY = wy2.num_WY[]
  
  # Numer of Householders in each WY.
  wy1_num_hs = wy1.num_hs[l1]
  wy2_num_hs = wy2.num_hs[l2]

  @boundscheck l1 ∈ 1:wy1_num_WY || throw_WYBlockNotAvailable(l1, wy1_num_WY)
  @boundscheck l2 ∈ 1:wy2_num_WY || throw_WYBlockNotAvailable(l2, wy2_num_WY)

  wy1_offset = wy1.offsets[l1]
  wy2_offset = wy2.offsets[l2]
  total_wy2_offset = wy2_offset + offset

  wy1_size=wy1.sizes[l1]
  wy2_size=wy2.sizes[l2]
  
  # Active indices for each transformation.
  wy1_inds = (wy1_offset + 1):(wy1_offset + wy1_size)
  wy2_inds = (total_wy2_offset + 1):(total_wy2_offset + wy2_size)


  @boundscheck begin
    wy2_inds ⊆ wy1_inds || throw(WYIndexSubsetError)
    wy1_num_hs + wy2_num_hs <= wy1.max_num_hs || throw_WYMaxHouseholderError(l1)
  end

  @inbounds @views begin
    # Do a shift and unshift of wy2's offset to apply to wy1.
    InPlace.apply_inv!(
      LeftProduct,
      GeneralMatrix{E},
      (wy2, l2),
      wy1.Y[1:wy1_size, 1:wy1_num_hs, l1],
      offset = offset - wy1_offset,
    )

    col_inds2 = (wy1_num_hs + 1):(wy1_num_hs + wy2_num_hs)
    wy1.num_hs[l1] = wy1_num_hs + wy2_num_hs
    wy1.W[1:wy1_size, col_inds2, l1] .= zero(E)
    wy1.Y[1:wy1_size, col_inds2, l1] .= zero(E)
    # Row indices of wy2 within wy1, without the wy1_offset.
    inds_wy12 = wy2_inds .- wy1_offset
    Δj = first(inds_wy12) - 1
    Δk = first(col_inds2) - 1
    for k∈1:wy2_num_hs, j∈1:wy2_size
      wy1.W[j+Δj, k+Δk, l1] = wy2.W[j, k, l2]
      wy1.Y[j+Δj, k+Δk, l1] = wy2.Y[j, k, l2]
    end
    # JET prefers the explicit loop.
    # wy1.W[inds_wy12, col_inds2, l1] .= wy2.W[1:wy2_size, 1:wy2_num_hs, l2]
    # wy1.Y[inds_wy12, col_inds2, l1] .= wy2.Y[1:wy2_size, 1:wy2_num_hs, l2]
  end
  nothing
end

# Apply transform l2 of wy2 to transform l1 of wy1.
Base.@propagate_inbounds function InPlace.apply_inv!(
  ::Type{RightProduct},
  ::Type{WYTrans{E}},
  wy1::WYTrans{E},
  wy2::WYTrans{E};
  offset = 0,
) where {E<:Number}
  InPlace.apply_inv!(
    RightProduct,
    WYTrans{E},
    (wy1, wy1.active_WY[]),
    (wy2, wy2.active_WY[]),
    offset = offset,
  )
end

# Apply transform l2 of wy2 to transform l1 of wy1.
Base.@propagate_inbounds function InPlace.apply_inv!(
  ::Type{RightProduct},
  ::Type{WYTrans{E}},
  (wy1, l1)::Tuple{WYTrans{E}, Int},
  (wy2, l2)::Tuple{WYTrans{E}, Int};
  offset = 0
) where {E<:Number}

  # Get number of available blocks.
  wy1_num_WY = wy1.num_WY[]
  wy2_num_WY = wy2.num_WY[]
  
  # Numer of Householders in each WY.
  wy1_num_hs = wy1.num_hs[l1]
  wy2_num_hs = wy2.num_hs[l2]

  @boundscheck l1 ∈ 1:wy1_num_WY || throw_WYBlockNotAvailable(l1, wy1_num_WY)
  @boundscheck l2 ∈ 1:wy2_num_WY || throw_WYBlockNotAvailable(l2, wy2_num_WY)

  wy1_offset = wy1.offsets[l1]
  wy2_offset = wy2.offsets[l2]
  total_wy2_offset = wy2_offset + offset

  wy1_size=wy1.sizes[l1]
  wy2_size=wy2.sizes[l2]
  
  # Active indices for each transformation.
  wy1_inds = (wy1_offset + 1):(wy1_offset + wy1_size)
  wy2_inds = (total_wy2_offset + 1):(total_wy2_offset + wy2_size)


  @boundscheck begin
    wy2_inds ⊆ wy1_inds || throw(WYIndexSubsetError)
    wy1_num_hs + wy2_num_hs <= wy1.max_num_hs || throw_WYMaxHouseholderError(l1)
  end

  @inbounds @views begin
    # Do a shift and unshift of wy2's offset to apply to wy1.
    InPlace.apply!(
      LeftProduct,
      GeneralMatrix{E},
      (wy2, l2),
      wy1.Y[1:wy1_size, 1:wy1_num_hs, l1],
      offset = offset - wy1_offset,
    )

    col_inds2 = (wy1_num_hs + 1):(wy1_num_hs + wy2_num_hs)
    wy1.num_hs[l1] = wy1_num_hs + wy2_num_hs
    wy1.W[1:wy1_size, col_inds2, l1] .= zero(E)
    wy1.Y[1:wy1_size, col_inds2, l1] .= zero(E)
    # Row indices of wy2 within wy1, without the wy1_offset.
    inds_wy12 = wy2_inds .- wy1_offset
    Δj = first(inds_wy12) - 1
    Δk = first(col_inds2) - 1
    for k∈1:wy2_num_hs, j∈1:wy2_size
      wy1.W[j+Δj, k+Δk, l1] = wy2.Y[j, k, l2]
      wy1.Y[j+Δj, k+Δk, l1] = wy2.W[j, k, l2]
    end
    # JET prefers the explicit loop.
    # wy1.W[inds_wy12, col_inds2, l1] .= wy2.Y[1:wy2_size, 1:wy2_num_hs, l2]
    # wy1.Y[inds_wy12, col_inds2, l1] .= wy2.W[1:wy2_size, 1:wy2_num_hs, l2]
  end
  nothing
end

# Apply transform l1 of wy1 to transform l2 of wy2.
Base.@propagate_inbounds function InPlace.apply!(
  ::Type{LeftProduct},
  ::Type{WYTrans{E}},
  wy1::WYTrans{E},
  wy2::WYTrans{E};
  offset = 0,
) where {E<:Number}
  InPlace.apply!(
    LeftProduct,
    WYTrans{E},
    (wy1, wy1.active_WY[]),
    (wy2, wy2.active_WY[]),
    offset = offset,
  )
end

Base.@propagate_inbounds function InPlace.apply!(
  ::Type{LeftProduct},
  ::Type{WYTrans{E}},
  (wy1, l1)::Tuple{WYTrans{E}, Int},
  (wy2, l2)::Tuple{WYTrans{E}, Int};
  offset = 0
) where {E<:Number}

  # Get number of available blocks.
  wy1_num_WY = wy1.num_WY[]
  wy2_num_WY = wy2.num_WY[]
  
  # Numer of Householders in each WY.
  wy1_num_hs = wy1.num_hs[l1]
  wy2_num_hs = wy2.num_hs[l2]

  @boundscheck l1 ∈ 1:wy1_num_WY || throw_WYBlockNotAvailable(l1, wy1_num_WY)
  @boundscheck l2 ∈ 1:wy2_num_WY || throw_WYBlockNotAvailable(l2, wy2_num_WY)

  wy1_offset = wy1.offsets[l1]
  wy2_offset = wy2.offsets[l2]
  total_wy1_offset = wy1_offset + offset

  wy1_size=wy1.sizes[l1]
  wy2_size=wy2.sizes[l2]
  
  # Active indices for each transformation.
  wy1_inds = (total_wy1_offset + 1):(total_wy1_offset + wy1_size)
  wy2_inds = (wy2_offset + 1):(wy2_offset + wy2_size)

  @boundscheck begin
    wy1_inds ⊆ wy2_inds || throw(WYIndexSubsetError)
    wy1_num_hs + wy2_num_hs <= wy2.max_num_hs || throw_WYMaxHouseholderError(l2)
  end

  @inbounds @views begin
    # Do a shift and unshift of wy1's offset to apply to wy2.
    InPlace.apply!(
      LeftProduct,
      GeneralMatrix{E},
      (wy1, l1),
      wy2.W[1:wy2_size, 1:wy2_num_hs, l2],
      offset = offset - wy2_offset,
    )
    # column indices to sort W1 and Y1 in W2 and Y2
    col_inds1 = (wy2_num_hs + 1):(wy2_num_hs + wy1_num_hs)
    wy2.num_hs[l2] = wy2_num_hs + wy1_num_hs
    wy2.W[1:wy2_size, col_inds1, l2] .= zero(E)
    wy2.Y[1:wy2_size, col_inds1, l2] .= zero(E)
    # Row indices of wy1 within wy2, without the wy2_offset.
    inds_wy21 = wy1_inds .- wy2_offset
    Δj = first(inds_wy21) - 1
    Δk = first(col_inds1) - 1
    for k∈1:wy1_num_hs, j∈1:wy1_size
      wy2.W[j+Δj, k+Δk, l2] = wy1.W[j, k, l1]
      wy2.Y[j+Δj, k+Δk, l2] = wy1.Y[j, k, l1]
    end
    # wy2.W[inds_wy21, col_inds1, l2] .= wy1.W[1:wy1_size, 1:wy1_num_hs, l1]
    # wy2.Y[inds_wy21, col_inds1, l2] .= wy1.Y[1:wy1_size, 1:wy1_num_hs, l1]
  end
  nothing
end

# Apply transform l1 of wy1 to transform l2 of wy2.
Base.@propagate_inbounds function InPlace.apply_inv!(
  ::Type{LeftProduct},
  ::Type{WYTrans{E}},
  wy1::WYTrans{E},
  wy2::WYTrans{E};
  offset = 0,
) where {E<:Number}
  InPlace.apply_inv!(
    LeftProduct,
    WYTrans{E},
    (wy1, wy1.active_WY[]),
    (wy2, wy2.active_WY[]),
    offset = offset,
  )
end

Base.@propagate_inbounds function InPlace.apply_inv!(
  ::Type{LeftProduct},
  ::Type{WYTrans{E}},
  (wy1, l1)::Tuple{WYTrans{E}, Int},
  (wy2, l2)::Tuple{WYTrans{E}, Int};
  offset = 0
) where {E<:Number}

  # Get number of available blocks.
  wy1_num_WY = wy1.num_WY[]
  wy2_num_WY = wy2.num_WY[]
  
  # Numer of Householders in each WY.
  wy1_num_hs = wy1.num_hs[l1]
  wy2_num_hs = wy2.num_hs[l2]

  @boundscheck l1 ∈ 1:wy1_num_WY || throw_WYBlockNotAvailable(l1, wy1_num_WY)
  @boundscheck l2 ∈ 1:wy2_num_WY || throw_WYBlockNotAvailable(l2, wy2_num_WY)

  wy1_offset = wy1.offsets[l1]
  wy2_offset = wy2.offsets[l2]
  total_wy1_offset = wy1_offset + offset

  wy1_size=wy1.sizes[l1]
  wy2_size=wy2.sizes[l2]
  
  # Active indices for each transformation.
  wy1_inds = (total_wy1_offset + 1):(total_wy1_offset + wy1_size)
  wy2_inds = (wy2_offset + 1):(wy2_offset + wy2_size)


  @boundscheck begin
    wy1_inds ⊆ wy2_inds || throw(WYIndexSubsetError)
    wy1_num_hs + wy2_num_hs <= wy2.max_num_hs || throw_WYMaxHouseholderError(l2)
  end

  @inbounds @views begin
    # Do a shift and unshift of wy1's offset to apply to wy2.
    InPlace.apply_inv!(
      LeftProduct,
      GeneralMatrix{E},
      (wy1, l1),
      wy2.W[1:wy2_size, 1:wy2_num_hs, l2],
      offset = offset - wy2_offset,
    )
    # column indices to sort W1 and Y1 in W2 and Y2
    col_inds1 = (wy2_num_hs + 1):(wy2_num_hs + wy1_num_hs)
    wy2.num_hs[l2] = wy2_num_hs + wy1_num_hs
    wy2.W[1:wy2_size, col_inds1, l2] .= zero(E)
    wy2.Y[1:wy2_size, col_inds1, l2] .= zero(E)
    # Row indices of wy1 within wy2, without the wy2_offset.
    inds_wy21 = wy1_inds .- wy2_offset
    Δj = first(inds_wy21) - 1
    Δk = first(col_inds1) - 1
    for k∈1:wy1_num_hs
      @simd for j∈1:wy1_size
        wy2.W[j+Δj, k+Δk, l2] = wy1.Y[j, k, l1]
        wy2.Y[j+Δj, k+Δk, l2] = wy1.W[j, k, l1]
      end
    end
    # wy2.W[inds_wy21, col_inds1, l2] .= wy1.Y[1:wy1_size, 1:wy1_num_hs, l1]
    # wy2.Y[inds_wy21, col_inds1, l2] .= wy1.W[1:wy1_size, 1:wy1_num_hs, l1]
  end
  nothing
end
