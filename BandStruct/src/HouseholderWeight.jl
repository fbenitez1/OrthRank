module HouseholderWeight
using Printf
using Random

import Base: size, getindex, setindex!, @propagate_inbounds, show, print, copy

import BandStruct.BandColumnMatrices:
  get_m_els,
  get_m,
  get_n,
  get_roffset,
  get_coffset,
  get_rbws,
  get_cbws,
  get_upper_bw_max,
  get_middle_bw_max,
  get_lower_bw_max,
  upper_bw,
  middle_bw,
  lower_bw,
  first_super,
  first_sub,
  get_band_element,
  set_band_element!,
  viewbc,
  wilk
using BandStruct.BandColumnMatrices


export HouseholderWeightMatrix
  # size,
  # getindex,
  # setindex!,
  # leading_lower_ranks_to_cbws!,
  # leading_upper_ranks_to_cbws!,
  # leading_constrain_lower_ranks,
  # leading_constrain_upper_ranks

"""

A rank structured matrix stored in Householder Weight format,
including a banded matrix with Leading Band Structure defined by
leading blocks and stored in a compressed column-wise format.  The
weight matrix parameters are the same as for a LeadingBandColumn.

The Householder vectors are stored in arrays with indices as follows:

  left_num_trans[leading_block_number]

  left_householder_vectors[vector_element, transform_number,
                           leading_block_number]

  left_betas[transform_number,leading_block_number]

  left_js[transform_number, leading_block_number]

with the right transformations defined similarly.

"""
struct HouseholderWeightMatrix{
  E<:Number,
  AE1<:AbstractArray{E,1},
  AE2<:AbstractArray{E,2},
  AE3<:AbstractArray{E,3},
  AI1<:AbstractArray{Int,1},
  AI2<:AbstractArray{Int,2},
} <: AbstractBandColumn{E,AE2,AI2}
  m::Int              # Matrix number of rows.
  n::Int              # Matrix number of columns.
  m_els::Int          # number of elements rows.
  num_blocks::Int     # Number of leading blocks.
  upper_bw_max::Int   # maximum upper bandwidth.
  middle_bw_max::Int  # maximum middle bandwidth.
  lower_bw_max::Int   # maximum lower bandwidth.
  rbws::AI2           # mx4 matrix: row-wise lower, middle, and upper bw +
                      # first subdiagonal postion in A.
  cbws::AI2           # 4xn matrix: column-wise upper, middle, and lower bw +
                      # first superdiagonal postion in A.
  leading_blocks::AI2 # 2xn matrix, leading block row and column counts.
  band_elements::AE2

  left_num_trans::AI1
  left_householder_vectors::AE3
  left_betas::AE2
  left_js::AI2

  right_num_trans::AI1
  right_householder_vectors::AE3
  right_betas::AE2
  right_ks::AI2
end

end
