# Operators for in place multiplication by a linear transformation and
# its inverse.
module InPlace

export ⊛, ⊘, Linear

struct Linear{A}
  a :: A
end

@inline function ⊛(
  t::Linear{AE},
  b::AE
) where {E<:Number, AE <: AbstractArray{E,2}}
  b[1:end,1:end] = t.a * b
  nothing
end

@inline function ⊛(
  b::AE,
  t::Linear{AE}
) where {E<:Number, AE <: AbstractArray{E,2}}
  b[1:end,1:end] = b * t.a 
  nothing
end

@inline function ⊘(
  t::Linear{AE},
  b::AE
) where {E<:Number, AE <: AbstractArray{E,2}}
  b[1:end,1:end] = t.a \ b
  nothing
end

@inline function ⊘(
  b::AE,
  t::Linear{AE}
) where {E<:Number, AE <: AbstractArray{E,2}}
  b[1:end,1:end] = b / t.a 
  nothing
end

end # module
