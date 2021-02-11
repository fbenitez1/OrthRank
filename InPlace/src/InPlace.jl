# Operators for in place multiplication by a linear transformation and
# its inverse.
module InPlace

export ⊛, ⊘, Linear, apply!, apply_inv!

struct Linear{A}
  a :: A
end

@inline function apply!(
  t::Linear{<:AbstractArray{E,2}},
  b::AbstractArray{E,2}
) where {E<:Number}
  b[1:end,1:end] = t.a * b
  nothing
end

@inline function apply!(
  b::AbstractArray{E,2},
  t::Linear{<:AbstractArray{E,2}}
) where {E<:Number}
  b[1:end,1:end] = b * t.a 
  nothing
end

@inline function ⊛(a, b)
  apply!(a, b)
end

@inline function apply_inv!(
  t::Linear{<:AbstractArray{E,2}},
  b::AbstractArray{E,2}
) where {E<:Number}
  b[1:end,1:end] = t.a \ b
  nothing
end

@inline function apply_inv!(
  b::AbstractArray{E,2},
  t::Linear{<:AbstractArray{E,2}}
) where {E<:Number}
  b[1:end,1:end] = b / t.a 
  nothing
end


@inline function ⊘(a, b)
  apply_inv!(a,b)
end

end # module
