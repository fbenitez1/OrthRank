# C ← α A B + β C 
function LinearAlgebra.mul!(
  C::AbstractMatrix,
  A::GivensWeight,
  B::AbstractMatrix,
  α,
  β,
  #  tmp = similar(C)
) 

  Base.require_one_based_indexing(C, B, tmp)
  
  ma, na = size(A.b)
  mb, nb = size(B)
  mc, nc = size(C)

  na != mb &&
    throw(DimensionMismatch(
      lazy"A has dimensions ($ma,$na) but B has dimensions ($mb,$nb)"))

  mc != ma || nc != nb &&
    throw(DimensionMismatch(lazy"C has dimensions $(size(C)), should have ($ma,$nb)"))

  @. C = β * C
  for l in 1:nc
    for k in 1:na
      αbkl = α*B[k,l]
      for j in middle_inband_index_range(A.b, :, k)
        C[j, l] += A.b[j, k] * αbkl
      end
    end
  end

  if A.lower_decomp[] isa LeadingDecomp
    for l in 1:nc
      for lb_ind ∈ gw.b.lower_blocks
        
      end
    end
  end

  if A.lower_decomp[] isa TrailingDecomp
  end

  if A.upper_decomp[] isa LeadingDecomp
  end

  if A.upper_decomp[] isa TrailingDecomp
  end

end
