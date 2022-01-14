"""
    SweepForward{T}

A struct for a forward product of WY transformations.  If
`wy=SweepForward.wy` and `n=wy.num_WY[]`, then this is conceptually
equivalent to the product

    wy₁ * wy₂ * ⋯ * wyₙ.
"""
struct SweepForward{T}
  wy::T
end

"""
    SweepBackward{T}

A struct for a forward product of WY transformations.  If
`wy=SweepForward.wy` and `n=wy.num_WY[]`, then this is conceptually
equivalent to the product

    wyₙ * wyₙ₋₁ * ⋯ * wy₁.
"""
struct SweepBackward{T}
  wy::T
end

InPlace.product_side(::Type{<:SweepForward}, _) = InPlace.LeftProduct()
InPlace.product_side(::Type{<:SweepBackward}, _) = InPlace.LeftProduct()
InPlace.product_side(_, ::Type{<:SweepForward}) = InPlace.RightProduct()
InPlace.product_side(_, ::Type{<:SweepBackward}) = InPlace.RightProduct()

# InPlace.structure_type(::Type{S}) where E = WYTrans{E}
# InPlace.structure_type(::Type{Tuple{W, Int}}) where {E, W <: WYTrans{E}} = WYTrans{E}


@inline function InPlace.apply_right!(
  ::Type{GeneralMatrix{E}},
  A::AbstractArray{E,2},
  sfwy::SweepForward{<:WYTrans{E}};
  offset = 0
) where {E<:Number}

  wy = sfwy.wy
  n = wy.num_WY[]
  for k ∈ 1:n
    InPlace.apply_right!(A, (wy, k), offset = offset)
  end
end

@inline function InPlace.apply_left!(
  ::Type{GeneralMatrix{E}},
  sfwy::SweepForward{<:WYTrans{E}},
  A::AbstractArray{E,2};
  offset = 0
) where {E<:Number}
  wy = sfwy.wy
  n = wy.num_WY[]
  for k ∈ n:-1:1
    InPlace.apply_left!((wy, k), A, offset = offset)
  end
end

@inline function InPlace.apply_right_inv!(
  ::Type{GeneralMatrix{E}},
  A::AbstractArray{E,2},
  sfwy::SweepForward{<:WYTrans{E}};
  offset = 0
) where {E<:Number}

  wy = sfwy.wy
  n = wy.num_WY[]
  for k ∈ n:-1:1
    InPlace.apply_right_inv!(A, (wy, k), offset = offset)
  end
end

@inline function InPlace.apply_left_inv!(
  ::Type{GeneralMatrix{E}},
  sfwy::SweepForward{<:WYTrans{E}},
  A::AbstractArray{E,2};
  offset = 0
) where {E<:Number}

  wy = sfwy.wy
  n = wy.num_WY[]
  for k ∈ 1:n
    InPlace.apply_left_inv!((wy, k), A, offset = offset)
  end
end

@inline function InPlace.apply_right!(
  ::Type{GeneralMatrix{E}},
  A::AbstractArray{E,2},
  sfwy::SweepBackward{<:WYTrans{E}};
  offset = 0
) where {E<:Number}

  wy = sfwy.wy
  n = wy.num_WY[]
  for k ∈ n:-1:1
    InPlace.apply_right!(A, (wy, k), offset = offset)
  end
end

@inline function InPlace.apply_left!(
  ::Type{GeneralMatrix{E}},
  sfwy::SweepBackward{<:WYTrans{E}},
  A::AbstractArray{E,2};
  offset = 0
) where {E<:Number}

  wy = sfwy.wy
  n = wy.num_WY[]
  for k ∈ 1:n
    InPlace.apply_left!((wy, k), A, offset = offset)
  end
end

@inline function InPlace.apply_right_inv!(
  ::Type{GeneralMatrix{E}},
  A::AbstractArray{E,2},
  sfwy::SweepBackward{<:WYTrans{E}};
  offset = 0
) where {E<:Number}

  wy = sfwy.wy
  n = wy.num_WY[]
  for k ∈ 1:n
    InPlace.apply_right_inv!(A, (wy, k), offset = offset)
  end
end

@inline function InPlace.apply_left_inv!(
  ::Type{GeneralMatrix{E}},
  sfwy::SweepBackward{<:WYTrans{E}},
  A::AbstractArray{E,2};
  offset = 0
) where {E<:Number}

  wy = sfwy.wy
  n = wy.num_WY[]
  for k ∈ n:-1:1
    InPlace.apply_left_inv!((wy, k), A, offset = offset)
  end
end
