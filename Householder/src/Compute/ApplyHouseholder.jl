macro real_tturbo(t, ex)
  return esc(quote
               if $t <: Real
                 @tturbo $ex
               else
                 @inbounds @fastmath $ex
               end
             end)
end

@inline function InPlace.apply_left!(
  ::Type{GeneralMatrix{E}},
  h::HouseholderTrans{E},
  A::AbstractArray{E,2};
  offset = 0
) where {E<:Number}
  m = h.size
  (ma, na) = size(A)
  offs = h.offs + offset
  @boundscheck begin
    if !((offs + 1):(offs + m) ⊆ 1:ma)
      throw(DimensionMismatch(@sprintf(
        "In h ⊛ A, A is of dimension %d×%d and h acts on rows %d:%d",
        ma,
        na,
        offs + 1,
        offs + m
      )))
    end
    lw = length(h.work)
    (lw >= na) || throw(DimensionMismatch(@sprintf(
      """
        In h ⊛ A, h has work array of length %d. A is %d×%d and requires
        a work array of length %d (the number of columns of A).
        """,
      lw,
      ma,
      na,
      na
    )))
  end
  v = h.v
  β = h.β
  work = h.work
  if na > 0 && m > 0
    @real_tturbo E for k ∈ 1:na
      x = zero(E)
      for j ∈ 1:m
        x += conj(v[j]) * A[offs+j,k]
      end
      work[k] = x
    end
    @real_tturbo E for k ∈ 1:na
      x = work[k]
      for j ∈ 1:m
        A[offs + j, k] -= β * v[j] * x
      end
    end
  end
  nothing
end

@inline function InPlace.apply_left_inv!(
  ::Type{GeneralMatrix{E}},
  h :: HouseholderTrans{E},
  A::AbstractArray{E,2};
  offset = 0
) where {E<:Number}
  m = h.size
  (ma,na) = size(A)
  offs = h.offs + offset
  @boundscheck begin
    if !((offs + 1):(offs + m) ⊆ 1:ma)
      throw(DimensionMismatch(@sprintf(
        "In h ⊘ A, A is of dimension %d×%d and h acts on rows %d:%d",
        ma,
        na,
        offs + 1,
        offs + m
      )))
    end
    lw = length(h.work)
    (lw >= na) || throw(DimensionMismatch(@sprintf(
      """
          In h ⊛ A, h has work array of length %d. A is %d×%d and requires
          a work array of length %d (the number of columns of A).
          """,
      lw,
      ma,
      na,
      na
    )))
  end
  v = reshape(h.v, m, 1)
  α = conj(h.β)
  work=h.work
  if na > 0 && m > 0
    @real_tturbo E for k ∈ 1:na
      x = zero(E)
      for j ∈ 1:m
        x += conj(v[j]) * A[offs+j,k]
      end
      work[k] = x
    end
    @real_tturbo E for k ∈ 1:na
      x = work[k]
      for j ∈ 1:m
        A[offs+j,k] -= α * v[j] * x
      end
    end
  end
  nothing
end


@inline function InPlace.apply_right!(
  ::Type{GeneralMatrix{E}},
  A::AbstractArray{E,2},
  h::HouseholderTrans{E};
  offset = 0
) where {E<:Number}
  m = h.size
  v = reshape(h.v, m, 1)
  offs = h.offs + offset
  work = h.work
  (ma,na) = size(A)
  @boundscheck begin
    if !((offs + 1):(offs + m) ⊆ 1:na)
      throw(DimensionMismatch(@sprintf(
        "In A ⊛ h, A is of dimension %d×%d and h acts on columns %d:%d",
        ma,
        na,
        offs + 1,
        offs + m
      )))
    end
    lw = length(h.work)
    (lw >= ma) || throw(DimensionMismatch(@sprintf(
      """
      In A ⊛ h, h has work array of length %d. A is %d×%d and requires
      a work array of length %d (the number of rows of A).
      """,
      lw,
      ma,
      na,
      ma
    )))
  end

  work[1:ma] .= zero(E)
  β = h.β
  if ma > 0 && m > 0
    @real_tturbo E for k ∈ 1:m
      x = v[k]
      for j ∈ 1:ma
        work[j] += A[j,k+offs] * x
      end
    end
    @real_tturbo E for k ∈ 1:m
      x=conj(v[k])
      for j ∈ 1:ma
        A[j,k+offs] -= β * work[j] * x
      end
    end
  end
  nothing
end

@inline function InPlace.apply_right_inv!(
  ::Type{GeneralMatrix{E}},
  A::AbstractArray{E,2},
  h::HouseholderTrans{E};
  offset = 0
) where {E<:Number}
  m = h.size
  v = reshape(h.v, m, 1)
  offs = h.offs + offset
  work = h.work
  (ma, na) = size(A)
  @boundscheck begin
    if !((offs + 1):(offs + m) ⊆ 1:na)
      throw(DimensionMismatch(@sprintf(
        "In A ⊘ h, A is of dimension %d×%d and h acts on columns %d:%d",
        ma,
        na,
        offs + 1,
        offs + m
      )))
    end
    lw=length(h.work)
    (lw >= ma) || throw(DimensionMismatch(@sprintf(
      """
      In A ⊘ h, h has work array of length %d. A is %d×%d and requires
      a work array of length %d (the number of rows of A).
      """,
      lw,
      ma,
      na,
      ma
    )))
  end
  work[1:ma] .= zero(E)
  β̃ = conj(h.β)
  if ma > 0 && m > 0
    @real_tturbo E for k ∈ 1:m
      x = v[k]
      for j ∈ 1:ma
        work[j] += A[j,k+offs] * x
      end
    end
    @real_tturbo E for k ∈ 1:m
      x = conj(v[k])
      for j ∈ 1:ma
        A[j,k+offs] -= β̃ * work[j] * x
      end
    end
  end  
  nothing
end

# Adjoint operations.

@inline function InPlace.apply_left!(
  ::Type{GeneralMatrix{E}},
  h::HouseholderTrans{E},
  Aᴴ::Adjoint{E,<:AbstractArray{E,2}};
  offset = 0
) where {E<:Number}
  m = h.size
  na = size(Aᴴ,2)
  v = reshape(h.v, m, 1)
  offs = h.offs + offset
  work = h.work
  (ma, na) = size(Aᴴ)
  @boundscheck begin
    if !((offs + 1):(offs + m) ⊆ 1:ma)
      throw(DimensionMismatch(@sprintf(
        "In h ⊛ A, A is of dimension %d×%d and h acts on rows %d:%d",
        ma,
        na,
        offs + 1,
        offs + m
      )))
    end
    lw = length(h.work)
    (lw >= na) || throw(DimensionMismatch(@sprintf(
      """
      In h ⊛ A, h has work array of length %d. A is %d×%d and requires
      a work array of length %d (the number of columns of A).
      """,
      lw,
      ma,
      na,
      na
    )))
  end
  work[1:na] .= zero(E)
  β = h.β
  if na > 0 && m > 0
    @real_tturbo E for j ∈ 1:m
      x = conj(v[j])
      for k ∈ 1:na
        work[k] += Aᴴ[j + offs, k] * x
      end
    end
    @real_tturbo E for j ∈ 1:m
      x = v[j]
      for k ∈ 1:na
        Aᴴ[j + offs, k] -= β * work[k] * x
      end
    end
  end
  nothing
end

@inline function InPlace.apply_left_inv!(
  ::Type{GeneralMatrix{E}},
  h::HouseholderTrans{E},
  Aᴴ::Adjoint{E,<:AbstractArray{E,2}};
  offset = 0
) where {E<:Number}
  m = h.size
  na = size(Aᴴ,2)
  v = reshape(h.v, m, 1)
  offs = h.offs + offset
  work = h.work
  (ma,na) = size(Aᴴ)
  @boundscheck begin
    if !((offs + 1):(offs + m) ⊆ 1:ma)
      throw(DimensionMismatch(@sprintf(
        "In h ⊘ A, A is of dimension %d×%d and h acts on rows %d:%d",
        ma,
        na,
        offs + 1,
        offs + m
      )))
    end
    lw = length(h.work)
    (lw >= na) || throw(DimensionMismatch(@sprintf(
      """
      In h ⊘ A, h has work array of length %d. A is %d×%d and requires
      a work array of length %d (the number of columns of A).
      """,
      lw,
      ma,
      na,
      na
    )))
  end
  work[1:na] .= zero(E)
  β̃ = conj(h.β)
  if na > 0 && m > 0
    @real_tturbo E for j ∈ 1:m
      x = conj(v[j])
      for k ∈ 1:na
        work[k] += Aᴴ[j + offs, k] * x
      end
    end
    @real_tturbo E for j ∈ 1:m
      x = v[j]
      for k ∈ 1:na
        Aᴴ[j + offs, k] -= β̃ * work[k] * x
      end
    end
  end
  nothing
end

@inline function InPlace.apply_right!(
  ::Type{GeneralMatrix{E}},
  Aᴴ::Adjoint{E,<:AbstractArray{E,2}},
  h::HouseholderTrans{E};
  offset = 0
) where {E<:Number}
  m = h.size
  v = reshape(h.v, m, 1)
  offs = h.offs + offset
  (ma, na) = size(Aᴴ)
  @boundscheck begin
    if !((offs + 1):(offs + m) ⊆ 1:na)
      throw(DimensionMismatch(@sprintf(
        "In A ⊛ h, A is of dimension %d×%d and h acts on columns %d:%d",
        ma,
        na,
        offs + 1,
        offs + m
      )))
    end
    lw = length(h.work)
    (lw >= ma) || throw(DimensionMismatch(@sprintf(
      """In A⊛h, h has work array of length %d. A is %d×%d and requires
        a work array of length %d (the number of rows of A).
        """,
      lw,
      ma,
      na,
      ma
    )))
  end
  β = h.β
  work=h.work
  if ma > 0 && m > 0
    @real_tturbo E for j ∈ 1:ma
      x = zero(E)
      for k ∈ 1:m
        x = x + Aᴴ[j,k+offs] * v[k]
      end
      work[j] = x
    end
    @real_tturbo E for j ∈ 1:ma
      x = work[j]
      for k ∈ 1:m
        Aᴴ[j, k + offs] -= β * conj(v[k]) * x
      end
    end
  end
  nothing
end

@inline function InPlace.apply_right_inv!(
  ::Type{GeneralMatrix{E}},
  Aᴴ::Adjoint{E,<:AbstractArray{E,2}},
  h::HouseholderTrans{E};
  offset = 0
) where {E<:Number}
  m = h.size
  v = reshape(h.v, m, 1)
  offs = h.offs + offset
  work = h.work
  (ma,na) = size(Aᴴ)
  @boundscheck begin
    if !((offs + 1):(offs + m) ⊆ 1:na)
      throw(DimensionMismatch(@sprintf(
        "In A ⊘ h, A is of dimension %d×%d and h acts on columns %d:%d",
        ma,
        na,
        offs + 1,
        offs + m
      )))
    end
    lw = length(work)
    (lw >= ma) || throw(DimensionMismatch(@sprintf(
      """
        In A ⊘ h, h has work array of length %d. A is %d×%d and requires
        a work array of length %d (the number of columns of A).
        """,
      lw,
      ma,
      na,
      ma
    )))
  end
  β̃ = conj(h.β)
  if ma > 0 && m > 0
    @real_tturbo E for j ∈ 1:ma
      x = zero(E)
      for k ∈ 1:m
        x = x + Aᴴ[j, k + offs] * v[k]
      end
      work[j] = x
    end
    @real_tturbo E for j ∈ 1:ma
      x = work[j]
      for k ∈ 1:m
        Aᴴ[j, k + offs] -= β̃ * conj(v[k]) * x
      end
    end
  end
  nothing
end
