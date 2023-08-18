module Precompile

using LinearAlgebra
using InPlace
using Random
using OrthWeight
using BandStruct

function make_givens_weight(
  ::Type{E};
  l_or_t::Union{LeadingDecomp, TrailingDecomp},
  m::Int,
  n::Int,
  lower_blocks::Vector{<:AbstractBlockData},
  upper_blocks::Vector{<:AbstractBlockData},
  lower_ranks::Vector{Int},
  upper_ranks::Vector{Int},
) where {E}

  gw = GivensWeight(
    E,
    l_or_t,
    MersenneTwister(),
    m,
    n;
    upper_rank_max = maximum(upper_ranks),
    lower_rank_max = maximum(lower_ranks),
    upper_ranks = upper_ranks,
    lower_ranks = lower_ranks,
    upper_blocks = upper_blocks,
    max_num_upper_blocks = 2 * length(upper_blocks),
    lower_blocks = lower_blocks,
    max_num_lower_blocks = 2 * length(lower_blocks),
  )
  
  upper_ranks =
    constrain_upper_ranks(m, n, blocks = upper_blocks, ranks = upper_ranks)

  lower_ranks =
    constrain_lower_ranks(m, n, blocks = lower_blocks, ranks = lower_ranks)

  Matrix(gw)
  
end

function run_all()

  m = 60
  n = 50

  lower_blocks = givens_block_sizes([
    10 20 30 40
    10 20 25 37
  ])

  upper_blocks = givens_block_sizes([
    10 20 30 40
    10 20 25 37
  ])
  upper_ranks = [10, 5, 7, 8]
  lower_ranks = [10, 5, 7, 8]

  make_givens_weight(
    Float64,
    l_or_t = TrailingDecomp(),
    m = m,
    n = n,
    upper_blocks = upper_blocks,
    lower_blocks = lower_blocks,
    upper_ranks = upper_ranks,
    lower_ranks = lower_ranks,
  )

  make_givens_weight(
    Complex{Float64},
    l_or_t = TrailingDecomp(),
    m = m,
    n = n,
    upper_blocks = upper_blocks,
    lower_blocks = lower_blocks,
    upper_ranks = upper_ranks,
    lower_ranks = lower_ranks,
  )

  make_givens_weight(
    Float64,
    l_or_t = LeadingDecomp(),
    m = m,
    n = n,
    upper_blocks = upper_blocks,
    lower_blocks = lower_blocks,
    upper_ranks = upper_ranks,
    lower_ranks = lower_ranks,
  )

  make_givens_weight(
    Complex{Float64},
    l_or_t = LeadingDecomp(),
    m = m,
    n = n,
    upper_blocks = upper_blocks,
    lower_blocks = lower_blocks,
    upper_ranks = upper_ranks,
    lower_ranks = lower_ranks,
  )

  return nothing
end

end
