module Compute

using Printf

using LinearAlgebra
import InPlace

export HouseholderTrans,
  update_norm,
  lhouseholder,
  rhouseholder,
  householder,
  column_nonzero!,
  row_nonzero!

"""

# HouseholderTrans

    HouseholderTrans{E,AEV<:AbstractArray{E,1},AEW<:AbstractArray{E,1}}

A Householder data structure.

## `h ⊛ A` is equivalent to

    v = reshape(h.v, h.size, 1)
    A₁ = A[(h.offs + 1) : (h.offs+h.size), :]
    A₁ = A₁ - h.β * v[1:h.size,1] * ( v[1:h.size,1]' * A₁ )

## `h ⊘ A` is equivalent to

    v = reshape(h.v, h.size, 1)
    A₁ = A[(h.offs + 1) : (h.offs+h.size), :]
    A₁ = A₁ - conj(h.β) * v[1:h.size,1] * ( v[1:h.size,1]' * A₁ )

## `A ⊛ h` is equivalent to

    v = reshape(h.v, h.size, 1)
    A₁ = A[:, (h.offs + 1) : (h.offs+h.size)]
    A₁ = A₁ - h.β * ( A₁ * v[1:h.size,1] ) * v[1:h.size,1]'

## `A ⊘ h` is equivalent to

    v = reshape(h.v, h.size, 1)
    A₁ = A[:, (h.offs + 1) : (h.offs+h.size)]
    A₁ = A₁ - conj(h.β) * ( A₁ * v[1:h.size,1] ) * v[1:h.size,1]'

"""
struct HouseholderTrans{E,AEV<:AbstractArray{E,1},AEW<:AbstractArray{E,1}}
  β::E
  # Householder vector.
  v::AEV
  # element to leave nonzero.
  l::Int64
  # size of transformation.
  size::Int64
  # offset for applying to a matrix.
  offs::Int64
  # Size = opposite side size of A.  For m × n A:
  # h ⊛ A requires work space of size n.
  # A ⊛ h requires work space of size m.
  work::AEW
end

@inline function update_norm(a::R, b::E) where {R<:Real,E<:Union{R,Complex{R}}}
  a_abs = abs(a)
  b_abs = abs(b)
  z = max(a_abs, b_abs)
  iszero(z) ? z : z * sqrt((a_abs / z)^2 + (b_abs / z)^2)
end


@inline function maybe_complex(::Type{E}, a::E, ::E) where {E<:Real}
  a
end

@inline function maybe_complex(
  ::Type{E},
  a::R,
  b::R,
) where {R<:Real,E<:Complex{R}}
  complex(a, b)
end

"""

  Compute a Householder such that for h = I - β v vᴴ,
  hᴴ * a = ||a|| eₗ.

"""
function lhouseholder(
  a::AbstractArray{E,1},
  l::Int64,
  offs::Int64,
  work::AbstractArray{E,1}
) where {E<:Number}
  m = length(a)
  a1 = a[l]
  if m == 1
    a[1] = 1
    sign_a1 = iszero(a1) ? one(a1) : sign(a1)
    β = (sign_a1 - one(a1)) / sign_a1
    HouseholderTrans(conj(β), a, l, m, offs, work)
  else
    norm_a2 = @views update_norm(norm(a[1:(l - 1)]), norm(a[(l + 1):m]))
    norm_a = update_norm(norm_a2, a1)
    if iszero(norm_a)
      HouseholderTrans(zero(E), a, l, m, offs, work)
    else
      alpha = if real(a1) <= 0
        a1 - norm_a
      else
        a1i = imag(a1)
        a1r = real(a1)
        x = update_norm(norm_a2, a1i)
        y = a1r + norm_a
        z = y / x
        maybe_complex(E, -x, a1i * z) / z
      end
      β = -conj(alpha) / norm_a
      a[l] = one(a1)
      rdiv!(view(a,1:(l-1)), alpha)
      rdiv!(view(a,(l+1):m), alpha)
      HouseholderTrans(conj(β), a, l, m, offs, work)
    end
  end
end

function lhouseholder(
  a::AbstractArray{E,1},
  l::Int64,
  offs::Int64,
  work_size::Int64,
) where {E<:Number}
  work = zeros(E, work_size)
  lhouseholder(a,l,offs,work)
end

"""

  Compute a Householder such that for h = I - β v vᴴ,
  a * h = ||a|| eₗᵀ.

"""
function rhouseholder(
  a::AbstractArray{E,1},
  l::Int64,
  offs::Int64,
  work::AbstractArray{E,1},
) where {E<:Number}
  m = length(a)
  a1 = a[l]
  if m == 1
    a[1] = 1
    sign_a1 = iszero(a1) ? one(a1) : sign(a1)
    β = (sign_a1 - one(a1)) / sign_a1
    conj!(a)
    HouseholderTrans(β, a, l, m, offs, work)
  else
    norm_a2 = @views update_norm(norm(a[1:(l - 1)]), norm(a[(l + 1):m]))
    norm_a = update_norm(norm_a2, a1)
    if iszero(norm_a)
      HouseholderTrans(zero(E), a, l, m, offs, work)
    else
      alpha = if real(a1) <= 0
        a1 - norm_a
      else
        a1i = imag(a1)
        a1r = real(a1)
        x = update_norm(norm_a2, a1i)
        y = a1r + norm_a
        z = y / x
        maybe_complex(E, -x, a1i * z) / z
      end

      β = -conj(alpha) / norm_a
      a[l] = one(a1)
      rdiv!(view(a,1:(l-1)), alpha)
      rdiv!(view(a,(l+1):m), alpha)
      conj!(a)
      HouseholderTrans(β, a, l, m, offs, work)
    end
  end
end

function rhouseholder(
  a::AbstractArray{E,1},
  l::Int64,
  offs::Int64,
  work_size::Int64,
) where {E<:Number}
  work = zeros(E, work_size)
  rhouseholder(a,l,offs,work)
end

function householder(
  a::AbstractArray{E,2},
  js::UnitRange{Int},
  k::Int,
  l::Int64,
  offs::Int64,
  work_size::Int64,
) where {E<:Number}
  work = zeros(E, work_size)
  @views lhouseholder(a[js,k],l,offs,work)
end

function householder(
  a::AbstractArray{E,2},
  j::Int,
  ks::UnitRange{Int},
  l::Int64,
  offs::Int64,
  work_size::Int64,
) where {E<:Number}
  work = zeros(E, work_size)
  @views rhouseholder(a[j,ks],l,offs,work)
end

function householder(
  a::AbstractArray{E,2},
  js::UnitRange{Int},
  k::Int,
  l::Int64,
  offs::Int64,
  v::AbstractArray{E,1},
  work_size::Int64,
) where {E<:Number}
  work = zeros(E, work_size)
  @views v[1:length(js)] = a[js,k]
  lhouseholder(v,l,offs,work)
end

function householder(
  a::AbstractArray{E,2},
  j::Int,
  ks::UnitRange{Int},
  l::Int64,
  offs::Int64,
  v::AbstractArray{E,1},
  work_size::Int64,
) where {E<:Number}
  work = zeros(E, work_size)
  @views v[1:length(ks)] = a[j,ks]
  rhouseholder(v,l,offs,work)
end

function householder(
  a::AbstractArray{E,2},
  js::UnitRange{Int},
  k::Int,
  l::Int64,
  offs::Int64,
  v::AbstractArray{E,1},
  work::AbstractArray{E,1},
) where {E<:Number}
  @views v[1:length(js)] = a[js,k]
  lhouseholder(v,l,offs,work)
end

function householder(
  a::AbstractArray{E,2},
  j::Int,
  ks::UnitRange{Int},
  l::Int64,
  offs::Int64,
  v::AbstractArray{E,1},
  work::AbstractArray{E,1},
) where {E<:Number}
  @views v[1:length(ks)] = a[j,ks]
  rhouseholder(v,l,offs,work)
end

@inline function column_nonzero!(
  A::AbstractArray{E,2},
  l::Int,
  k::Int,
) where {E<:Number}
  A[1:(l - 1), k] .= zero(E)
  A[(l + 1):end, k] .= zero(E)
end

@inline function row_nonzero!(
  A::AbstractArray{E,2},
  j::Int,
  l::Int,
) where {E<:Number}
  A[j, 1:(l - 1)] .= zero(E)
  A[j, (l + 1):end] .= zero(E)
end

@inline function InPlace.:⊛(
  h::HouseholderTrans{E},
  A::AbstractArray{E,2},
) where {E<:Number}
  m = h.size
  (ma, na) = size(A)
  offs = h.offs
  @boundscheck if !((offs + 1):(offs + m) ⊆ 1:ma)
    throw(DimensionMismatch(@sprintf(
      "In h ⊛ A, A is of dimension %d×%d and h acts on rows %d:%d",
      ma,
      na,
      offs + 1,
      offs + m
    )))
  end
  v = reshape(h.v, m, 1)
  @inbounds for k ∈ 1:na
    x = zero(E)
    @simd for j ∈ 1:m
      x = x + conj(v[j]) * A[offs+j,k]
    end
    @simd for j ∈ 1:m
      A[offs+j,k] -= h.β * v[j] * x
    end
  end
  nothing
end

@inline function InPlace.:⊘(
  h :: HouseholderTrans{E},
  A::AbstractArray{E,2},
) where {E<:Number}
  m = h.size
  (ma,na) = size(A)
  offs = h.offs
  @boundscheck if !((offs + 1):(offs + m) ⊆ 1:ma)
    throw(DimensionMismatch(@sprintf(
      "In h ⊘ A, A is of dimension %d×%d and h acts on rows %d:%d",
      ma,
      na,
      offs + 1,
      offs + m
    )))
  end
  v = reshape(h.v, m, 1)
  α = conj(h.β)
  @inbounds for k ∈ 1:na
    x = zero(E)
    @simd for j ∈ 1:m
      x = x + conj(v[j]) * A[offs+j,k]
    end
    @simd for j ∈ 1:m
      A[offs+j,k] -= α * v[j] * x
    end
  end
  nothing
end


@inline function InPlace.:⊛(
  A::AbstractArray{E,2},
  h::HouseholderTrans{E},
) where {E<:Number}
  m = h.size
  v = reshape(h.v, m, 1)
  offs = h.offs
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
  @inbounds begin
    for k ∈ 1:m
      @simd for j ∈ 1:ma
        work[j] += A[j,k+offs] * v[k]
      end
    end
    for k ∈ 1:m
      @simd for j ∈ 1:ma
        A[j,k+offs] -= β * work[j] * conj(v[k])
      end
    end
  end
  nothing
end

@inline function InPlace.:⊘(
  A::AbstractArray{E,2},
  h::HouseholderTrans{E},
) where {E<:Number}
  m = h.size
  v = reshape(h.v, m, 1)
  offs = h.offs
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
  @inbounds begin
    for k ∈ 1:m
      @simd for j ∈ 1:ma
        work[j] += A[j,k+offs] * v[k]
      end
    end
    for k ∈ 1:m
      @simd for j ∈ 1:ma
        A[j,k+offs] -= conj(h.β) * work[j] * conj(v[k])
      end
    end
  end
  nothing
end

# Adjoint operations.

@inline function InPlace.:⊛(
  h::HouseholderTrans{E},
  Aᴴ::Adjoint{E,<:AbstractArray{E,2}},
) where {E<:Number}
  m = h.size
  na = size(Aᴴ,2)
  v = reshape(h.v, m, 1)
  offs = h.offs
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
  @inbounds begin
    for j ∈ 1:m
      @simd for k ∈ 1:na
        work[k] += Aᴴ[j + offs, k] * conj(v[j])
      end
    end
    for j ∈ 1:m
      @simd for k ∈ 1:na
        Aᴴ[j + offs, k] -= h.β * work[k] * v[j]
      end
    end
  end
  nothing
end

@inline function InPlace.:⊘(
  h::HouseholderTrans{E},
  Aᴴ::Adjoint{E,<:AbstractArray{E,2}},
) where {E<:Number}
  m = h.size
  na = size(Aᴴ,2)
  v = reshape(h.v, m, 1)
  offs = h.offs
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
  @inbounds begin
    for j ∈ 1:m
      @simd for k ∈ 1:na
        work[k] += Aᴴ[j + offs, k] * conj(v[j])
      end
    end
    for j ∈ 1:m
      @simd for k ∈ 1:na
        Aᴴ[j + offs, k] -= conj(h.β) * work[k] * v[j]
      end
    end
  end
  nothing
end

@inline function InPlace.:⊛(
  Aᴴ::Adjoint{E,<:AbstractArray{E,2}},
  h::HouseholderTrans{E},
) where {E<:Number}
  m = h.size
  v = reshape(h.v, m, 1)
  offs = h.offs
  (ma, na) = size(Aᴴ)
  @boundscheck if !((offs + 1):(offs + m) ⊆ 1:na)
    throw(DimensionMismatch(@sprintf(
      "In A ⊛ h, A is of dimension %d×%d and h acts on columns %d:%d",
      ma,
      na,
      offs + 1,
      offs + m
    )))
  end
  @inbounds for j ∈ 1:ma
    x = zero(E)
    @simd for k ∈ 1:m
      x = x + Aᴴ[j,k+offs] * v[k]
    end
    @simd for k ∈ 1:m
      Aᴴ[j,k+offs] -= h.β * conj(v[k]) * x
    end
  end
  nothing
end

@inline function InPlace.:⊘(
  Aᴴ::Adjoint{E,<:AbstractArray{E,2}},
  h::HouseholderTrans{E},
) where {E<:Number}
  m = h.size
  v = reshape(h.v, m, 1)
  offs = h.offs
  work = h.work
  (ma,na) = size(Aᴴ)
  @boundscheck if !((offs + 1):(offs + m) ⊆ 1:na)
    throw(DimensionMismatch(@sprintf(
      "In A ⊘ h, A is of dimension %d×%d and h acts on columns %d:%d",
      ma,
      na,
      offs + 1,
      offs + m
    )))
  end
  @inbounds for j ∈ 1:ma
    x = zero(E)
    @simd for k ∈ 1:m
      x = x + Aᴴ[j, k + offs] * v[k]
    end
    @simd for k ∈ 1:m
      Aᴴ[j, k + offs] -= conj(h.β) * conj(v[k]) * x
    end
  end
  nothing
end


end # module
