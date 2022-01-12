struct WYIndexSubsetError <: Exception end

struct WYMaxHouseholderError <: Exception
  message::String
end

throw_WYMaxHouseholderError(block) =
  throw(WYMaxHouseholderError(@sprintf(
    "Too many Householders for block %d.",
    block
  )))

@inline function InPlace.apply_right!(
  wy::WYTrans{E},
  k::Int,
  h::HouseholderTrans{E};
  offset = 0
) where {E<:Number}
  
  num_WY = wy.num_WY[]
  active_WY = wy.active_WY[]

  @boundscheck k ∈ 1:wy.num_WY[] || throw_WYBlockNotAvailable(k, wy.num_WY[])
  
  @inbounds begin
    num_hs = wy.num_hs[k]
    wy_offset=wy.offsets[k]
    total_h_offset = h.offs + offset
    indsh = (total_h_offset + 1):(total_h_offset + h.size)
    indswy = 1:wy.sizes[k]
    indshwy = (total_h_offset - wy_offset + 1):(total_h_offset - wy_offset + h.size)
  end

  @boundscheck begin
    indsh ⊆ (indswy .+ wy_offset) || throw(WYIndexSubsetError)
    num_hs < wy.max_num_hs || throw_WYMaxHouseholderError(k)
  end

  @inbounds begin
    @views begin
      work = reshape(wy.work[1:num_hs],num_hs,1)

      W0 = wy.W[indswy, 1:num_hs, k]
      W1 = wy.W[indswy, (num_hs + 1):(num_hs + 1), k]
      Y0 = wy.Y[indswy, 1:num_hs, k]
      Y1 = wy.Y[indswy, (num_hs + 1):(num_hs + 1), k]
    end
    W1[:,:] .= zero(E)
    Y1[:,:] .= zero(E)
    W1[indshwy,:] .= h.β .* h.v
    Y1[indshwy,:] .= h.v
    num_hs > 0 && length(indswy) > 0 && begin
      matmul!(work, Y0', W1)
      matmul!(W1, W0, work, -one(E), one(E))
    end
    wy.num_hs[k] += 1
  end
  nothing
end

Base.@propagate_inbounds function InPlace.apply_right!(
  wyk::SelectWY{<:WYTrans{E}},
  h::HouseholderTrans{E};
  offset = 0,
) where {E<:Number}

  k = wyk.block
  wy = wyk.wy
  InPlace.apply_right!(wy, k, h, offset = offset)
end

Base.@propagate_inbounds function InPlace.apply_right!(
  wy::WYTrans{E},
  h::HouseholderTrans{E};
  offset = 0
) where {E<:Number}

  InPlace.apply_right!(wy, wy.active_WY[], h, offset = offset)
end

@inline function InPlace.apply_right_inv!(
  wy::WYTrans{E},
  k::Int,
  h::HouseholderTrans{E};
  offset = 0
) where {E<:Number}

  num_WY = wy.num_WY[]
  @boundscheck k ∈ 1:num_WY || throw_WYBlockNotAvailable(k, num_WY)

  @inbounds begin
    num_hs = wy.num_hs[k]
    wy_offset=wy.offsets[k]
    total_h_offset = h.offs + offset

    indsh = (total_h_offset + 1):(total_h_offset + h.size)
    indswy = 1:wy.sizes[k]
    indshwy = (total_h_offset - wy_offset + 1):(total_h_offset - wy_offset + h.size)
  end

  @boundscheck begin
    indsh ⊆ (indswy .+ wy_offset) || throw(WYIndexSubsetError)
    num_hs < wy.max_num_hs || throw_WYMaxHouseholderError(k)
  end

  @inbounds begin
    @views begin
      work = reshape(wy.work[1:num_hs],num_hs,1)

      W0 = wy.W[indswy, 1:num_hs, k]
      W1 = wy.W[indswy, (num_hs + 1):(num_hs + 1), k]
      Y0 = wy.Y[indswy, 1:num_hs, k]
      Y1 = wy.Y[indswy, (num_hs + 1):(num_hs + 1), k]
    end

    W1[:,:] .= zero(E)
    Y1[:,:] .= zero(E)
    W1[indshwy,:] .= conj(h.β) .* h.v
    Y1[indshwy,:] = h.v
    num_hs > 0 && length(indswy) > 0 && begin
      matmul!(work, Y0', W1)
      matmul!(W1, W0, work, -one(E), one(E))
    end

    wy.num_hs[k] += 1
  end
  nothing
end

Base.@propagate_inbounds function InPlace.apply_right_inv!(
  wyk::SelectWY{<:WYTrans{E}},
  h::HouseholderTrans{E};
  offset = 0,
) where {E<:Number}

  k = wyk.block
  wy = wyk.wy
  InPlace.apply_right_inv!(wy, k, h, offset = offset)
end

Base.@propagate_inbounds function InPlace.apply_right_inv!(
  wy::WYTrans{E},
  h::HouseholderTrans{E};
  offset = 0,
) where {E<:Number}

  InPlace.apply_right_inv!(wy, wy.active_WY[], h, offset = offset)
end

@inline function InPlace.apply_left!(
  h::HouseholderTrans{E},
  wy::WYTrans{E},
  k::Int;
  offset = 0
) where {E<:Number}

  num_WY = wy.num_WY[]
  @boundscheck k ∈ 1:num_WY || throw_WYBlockNotAvailable(k, num_WY)

  @inbounds begin
    num_hs = wy.num_hs[k]
    wy_offset = wy.offsets[k]
    total_h_offset = h.offs + offset
    v=reshape(h.v,length(h.v),1)

    indsh = (total_h_offset + 1):(total_h_offset + h.size)
    indswy = 1:wy.sizes[k]
    indshwy = (total_h_offset - wy_offset + 1):(total_h_offset - wy_offset + h.size)
  end

  @boundscheck begin
    indsh ⊆ (indswy .+ wy_offset) || throw(WYIndexSubsetError)
    num_hs < wy.max_num_hs || throw_WYMaxHouseholderError(k)
  end

  @inbounds begin

    @views begin
      work = reshape(wy.work[1:num_hs],num_hs,1)

      W0 = wy.W[indswy, 1:num_hs, k]
      W1 = wy.W[indswy, num_hs + 1:num_hs+1, k]
      Y0 = wy.Y[indswy, 1:num_hs, k]
      Y1 = wy.Y[indswy, num_hs + 1:num_hs+1, k]
    end
    W1[:,:] .= zero(E)
    Y1[:,:] .= zero(E)
    W1[indshwy,:] = v
    Y1[indshwy,:] .= conj(h.β) .* v
    
    num_hs > 0 && length(indswy) > 0 && begin
      matmul!(work, W0', Y1)
      matmul!(Y1, Y0, work, -one(E), one(E))
    end

    wy.num_hs[k] += 1
  end
  nothing
end

Base.@propagate_inbounds function InPlace.apply_left!(
  h::HouseholderTrans{E},
  wyk::SelectWY{<:WYTrans{E}};
  offset = 0
) where {E<:Number}

  k=wyk.block
  wy=wyk.wy

  InPlace.apply_left!(h, wy, k, offset = offset)
end

Base.@propagate_inbounds function InPlace.apply_left!(
  h::HouseholderTrans{E},
  wy::WYTrans{E};
  offset = 0
) where {E<:Number}

  InPlace.apply_left!(h, wy, wy.active_WY[], offset = offset)
end

@inline function InPlace.apply_left_inv!(
  h::HouseholderTrans{E},
  wy::WYTrans{E},
  k::Int;
  offset = 0
) where {E<:Number}

  num_WY = wy.num_WY[]
  @boundscheck k ∈ 1:num_WY || throw_WYBlockNotAvailable(k, num_WY)

  @inbounds begin
    num_hs = wy.num_hs[k]
    wy_offset = wy.offsets[k]
    total_h_offset = h.offs + offset
    v=reshape(h.v,length(h.v),1)

    indsh = (total_h_offset + 1):(total_h_offset + h.size)
    indswy = 1:wy.sizes[k]
    indshwy = (total_h_offset - wy_offset + 1):(total_h_offset - wy_offset + h.size)
  end

  @boundscheck begin
    indsh ⊆ (indswy .+ wy_offset) || throw(WYIndexSubsetError)
    num_hs < wy.max_num_hs || throw_WYMaxHouseholderError(k)
  end

  @inbounds begin
    @views begin
      work = reshape(wy.work[1:num_hs],num_hs,1)

      W0 = wy.W[indswy, 1:num_hs, k]
      W1 = wy.W[indswy, num_hs + 1:num_hs+1, k]
      Y0 = wy.Y[indswy, 1:num_hs, k]
      Y1 = wy.Y[indswy, num_hs + 1:num_hs+1, k]
    end
    W1[:,:] .= zero(E)
    Y1[:,:] .= zero(E)
    W1[indshwy,:] = v
    Y1[indshwy,:] .= h.β .* v

    num_hs > 0 && length(indswy) > 0 && begin
      matmul!(work, W0', Y1)
      matmul!(Y1, Y0, work, -one(E), one(E))
    end

    wy.num_hs[k] += 1
  end
  nothing

end

Base.@propagate_inbounds function InPlace.apply_left_inv!(
  h::HouseholderTrans{E},
  wyk::SelectWY{<:WYTrans{E}};
  offset = 0
) where {E<:Number}

  k=wyk.block
  wy=wyk.wy
  InPlace.apply_left_inv!(h, wy, k, offset = offset)
end

Base.@propagate_inbounds function InPlace.apply_left_inv!(
  h::HouseholderTrans{E},
  wy::WYTrans{E};
  offset = 0
) where {E<:Number}

  InPlace.apply_left_inv!(h, wy, wy.active_WY[], offset = offset)
end
