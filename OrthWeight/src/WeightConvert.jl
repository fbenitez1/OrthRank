module WeightConvert

using BandStruct
using Householder # : WYTrans, householder, selectWY!, resetWYBlock!, SweepForward
using InPlace: apply!, apply_inv!
import InPlace
using LinearAlgebra

using ..WYWeightMatrices

# Move into Householder
get_wy_size(wy::WYTrans) = (wy.max_WY_size, wy.max_num_hs, wy.max_num_WY)
get_wy_size(wy::WYTrans, l::Int) = (wy.sizes[l], wy.num_hs[l])

set_wy_size(wy::WYTrans, l::Int, s::Int, n::Int) =
  wy.sizes[l], wy.num_hs[l] = s, n

# Insert wyw2 into wyw1.
function insertWY!(wy1::WYTrans{E}, l1, wy2::WYTrans{E}, l2) where {E<:Number}
  size2, num_hs2 = get_wy_size(wy2, l2)
  wy1.W[1:size2, 1:num_hs2, l1] = wy2.W[1:size2, 1:num_hs2, l2]
  wy1.Y[1:size2, 1:num_hs2, l1] = wy2.Y[1:size2, 1:num_hs2, l2]
  set_wy_size(wy1, l1, size2, num_hs2)
  wy1.offsets[l1] = wy2.offsets[l2]
end


# given a p×r nonzero block in the upper right of lower block l, do a
# SpanStep row compression (i.e. just a QR factorization) and store
# the transformations in a WYTrans.
function row_compress_lower_block(
  B::AbstractBandColumn,
  l::Int, # block number to compress
  p::Int, # Number of current nonzero rows.
  r::Int, # rank, current nonzero columns.
  wy::WYTrans{E},
  v::Vector{E},
  work::Vector{E};
  column_block_size::Int = 16
) where {E<:Number}

  m, _ = size(B)
  Bl = view_lower_block(B, l)
  ml, nl = size(Bl)
  p > ml || r > nl && error("Bad block size given to row_compress_lower_block.")
  
  # No compression needed.
  if p <= r
    wy.num_WY[] = 0
    return
  end

  column_blocks, rem = divrem(r, column_block_size)
  # include the last incomplete block.
  column_blocks = rem > 0 ? column_blocks + 1 : column_blocks

  workh = @views work[1:p]

  # offsets into the block to be compressed.
  roffs = m - ml
  coffs = nl - r

  @views for cb ∈ 1:column_blocks
    selectWY!(wy, cb)
    # Note this offset is for just the lower block, not the entire
    # weight matrix.
    offs = roffs + (cb - 1) * column_block_size
    resetWYBlock!(
      wy,
      block = cb,
      offset = offs,
      sizeWY = p - (cb - 1) * column_block_size
    )
    block_end = min(cb * column_block_size, r)
    for k ∈ ((cb-1)*column_block_size+1):block_end
      h = householder(B, roffs .+ (k:p), coffs + k, v, workh)
      h ⊘ B[:, coffs .+ (k:block_end)]
      k < p && notch_lower!(B, roffs + k + 1, coffs+k)
      wy ⊛ h
    end
    wy ⊘ B[:, coffs .+ (block_end+1:r)]
  end
  (SweepForward(wy), B)
end

function weight_convert!(wyw::WYWeight, tmp_lwy::WYTrans, tmp_rwy::WYTrans)
  weight_convert!(typeof(wyw.decomp[]), wyw, tmp_lwy, tmp_rwy)
end

# Leading to trailing
function weight_convert!(
  ::Type{LeadingDecomp},
  wyw::WYWeight,
  tmp_lwy::WYTrans,
  tmp_rwy::WYTrans,
)
  bbc = wyw.b
  (m, n) = size(bbc)
  lwy = wyw.leftWY
  rwy = wyw.rightWY
  num_blocks = bbc.num_blocks
  # Need to do an initial compression
  @views for l ∈ num_blocks:-1:1
    rows_lb, cols_lb = lower_block_ranges(bbc, l)
    r_lb = bbc.lower_ranks[l]
    # Undo the lower leading transformation.
    apply_inv!(bbc[lrows, :], rwy, l)
    compressed_rows_lb = first(rows_lb, r)
    compressed_cols_lb = last(cols_lb, r)
    last_compressed_row_lb =
      isempty(compressed_rows_lb) ? m : last(compressed_rows_lb)
    # Reset the old transformation.
    # restWYBlock(rwy, block=l, offset=, sizeWY=)
    # restWYBlock(rwy, block=l, offset=, sizeWY=)

    # # Upper
    # rows_ub, cols_ub = upper_block_ranges(bbc, l)
    # ru = bbc.upper_ranks[l]
    # apply!(lwy, l, bbc[:, ucols])

    # Now recompress.


  end
  a

end


end
