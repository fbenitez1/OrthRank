module WYWeightMatrices

export WYWeight,
  BigStep,
  SmallStep,
  Left,
  Right,
  get_transform_params!,
  get_transform_params,
  get_max_WY_sizes

using BandStruct
using Householder: WYTrans

using Random

"""
      struct BigStep end

  A type marking a WYWeight with steps that are large relative to
  the off-diagonal ranks.
  """
struct BigStep end

"""
        struct SmallStep end

  A type marking a WYWeight with steps that are small relative to
  the off-diagonal ranks.
  """
struct SmallStep end

struct Left end
struct Right end

struct WYWeight{Step<:Union{BigStep,SmallStep},LWY,B,RWY}
  decomp::Base.RefValue{Union{Nothing,LeadingDecomp,TrailingDecomp}}
  step::Step
  leftWY::LWY
  b::B
  rightWY::RWY
  upper_ranks::Array{Int,1}
  lower_ranks::Array{Int,1}
end

# Generic BigStep WYWeight with zero ranks but room for either type of
# decomposition with ranks up to upper_rank_max and Lower_rank_max.
function WYWeight(
  ::Type{E},
  ::Type{BigStep},
  m::Int,
  n::Int;
  upper_rank_max::Int,
  lower_rank_max::Int,
  upper_blocks::Array{Int,2},
  lower_blocks::Array{Int,2},
) where {E<:Number}
  
  lbc = BlockedBandColumn(
    E,
    m,
    n,
    upper_rank_max = upper_rank_max,
    lower_rank_max = lower_rank_max,
    upper_blocks = upper_blocks,
    lower_blocks = lower_blocks,
  )
  
  num_blocks = size(upper_blocks,2)
  size(lower_blocks, 2) == num_blocks ||
    error("""In a WYWeight, the number of upper blocks should equal the number
             of lower blocks""")

  rank_max = max(upper_rank_max, lower_rank_max)

  left_sizes = get_max_WY_sizes(
    Left,
    m,
    n,
    upper_blocks = upper_blocks,
    lower_blocks = lower_blocks,
    upper_rank_max = upper_rank_max,
    lower_rank_max = lower_rank_max,
  )

  left_WY_size = maximum(left_sizes)
  # n * (max_WY_size + max_num_hs) where max_num_hs is potentially as
  # large as max_WY_size.
  left_work_size = n * (2*left_WY_size)

  leftWY = WYTrans(
    E,
    max_num_WY = num_blocks,
    max_WY_size = left_WY_size,
    max_num_hs = rank_max,
    work_size = left_work_size
  )

  right_sizes = get_max_WY_sizes(
    Right,
    m,
    n,
    upper_blocks=upper_blocks,
    lower_blocks=lower_blocks,
    upper_rank_max=upper_rank_max,
    lower_rank_max=lower_rank_max,
  )

  right_WY_size = maximum(right_sizes)
  right_work_size = m * (2*right_WY_size)

  rightWY = WYTrans(
    E,
    max_num_WY = num_blocks,
    max_WY_size = right_WY_size,
    max_num_hs = rank_max,
    work_size = right_work_size
  )
  upper_ranks = zeros(Int, num_blocks)
  lower_ranks = zeros(Int, num_blocks)
  ref = Base.RefValue{Union{Nothing, LeadingDecomp, TrailingDecomp}}(nothing)
  WYWeight(ref, BigStep(), leftWY, lbc, rightWY, upper_ranks, lower_ranks)

end

# Generic BigStep WYWeight with zero ranks but room for either type of
# decomposition with ranks up to upper_rank_max and Lower_rank_max.
function WYWeight(
  ::Type{E},
  ::Type{BigStep},
  ::Type{LeadingDecomp},
  rng::AbstractRNG,
  m::Int,
  n::Int;
  upper_ranks::Array{Int,1},
  lower_ranks::Array{Int,1},
  upper_rank_max::Int = maximum(upper_ranks),
  lower_rank_max::Int = maximum(lower_ranks),
  upper_blocks::Array{Int,2},
  lower_blocks::Array{Int,2},
) where {E<:Number}

  lbc = BlockedBandColumn(
    E,
    LeadingDecomp,
    rng,
    m,
    n,
    upper_rank_max = upper_rank_max,
    lower_rank_max = lower_rank_max,
    upper_blocks = upper_blocks,
    lower_blocks = lower_blocks,
    upper_ranks = upper_ranks,
    lower_ranks = lower_ranks,
  )
  
  num_blocks = size(upper_blocks,2)
  size(lower_blocks, 2) == num_blocks ||
    error("""In a WYWeight, the number of upper blocks should equal the number
             of lower blocks""")

  rank_max = max(upper_rank_max, lower_rank_max)

  left_sizes = get_max_WY_sizes(
    Left,
    m,
    n,
    upper_blocks = upper_blocks,
    lower_blocks = lower_blocks,
    upper_rank_max = upper_rank_max,
    lower_rank_max = lower_rank_max,
  )

  left_WY_size = maximum(left_sizes)
  # n * (max_WY_size + max_num_hs) where max_num_hs is potentially as
  # large as max_WY_size.
  left_work_size = n * (2*left_WY_size)

  leftWY = WYTrans(
    E,
    max_num_WY = num_blocks,
    max_WY_size = left_WY_size,
    max_num_hs = rank_max,
    work_size = left_work_size
  )
  
  get_transform_params!(
    LeadingDecomp,
    Left,
    BigStep,
    m,
    n,
    upper_blocks = upper_blocks,
    upper_ranks = upper_ranks,
    sizes = leftWY.sizes,
    num_hs = leftWY.num_hs,
    offsets = leftWY.offsets,
  )
  rand!(rng, leftWY)

  right_sizes = get_max_WY_sizes(
    Right,
    m,
    n,
    upper_blocks=upper_blocks,
    lower_blocks=lower_blocks,
    upper_rank_max=upper_rank_max,
    lower_rank_max=lower_rank_max,
  )

  right_WY_size = maximum(right_sizes)
  right_work_size = m * (2*right_WY_size)

  rightWY = WYTrans(
    E,
    max_num_WY = num_blocks,
    max_WY_size = right_WY_size,
    max_num_hs = rank_max,
    work_size = right_work_size
  )

  get_transform_params!(
    LeadingDecomp,
    Right,
    BigStep,
    m,
    n,
    lower_blocks = lower_blocks,
    lower_ranks = lower_ranks,
    sizes = leftWY.sizes,
    num_hs = leftWY.num_hs,
    offsets = leftWY.offsets,
  )
  rand!(rng, rightWY)

  ref = Base.RefValue{Union{Nothing, LeadingDecomp, TrailingDecomp}}(nothing)
  WYWeight(ref, BigStep(), leftWY, lbc, rightWY, upper_ranks, lower_ranks)
end

function get_transform_params(
  decomp::Union{Type{LeadingDecomp}, Type{TrailingDecomp}},
  side::Union{Type{Right}, Type{Left}},
  step::Union{Type{BigStep},Type{SmallStep}},
  m::Int,
  n::Int;
  lower_blocks::Union{AbstractArray{Int,2}, Nothing} = nothing,
  lower_ranks::Union{AbstractArray{Int,1}, Nothing} = nothing,
  upper_blocks::Union{AbstractArray{Int,2}, Nothing} = nothing,
  upper_ranks::Union{AbstractArray{Int,1}, Nothing} = nothing,
  find_max::Bool=false
)
  if isa(lower_ranks, AbstractArray{Int})
    num_blocks = size(lower_ranks)
  elseif isa(upper_ranks, AbstractArray{Int})
    num_blocks = size(upper_ranks)
  else
    error("get_transform_params requires either lower or upper ranks and blocks.")
  end
  num_hs = zeros(Int, num_blocks)
  sizes = zeros(Int, num_blocks)
  offsets = zeros(Int, num_blocks)
  get_transform_params!(
    decomp,
    side,
    step,
    m,
    n,
    lower_blocks = lower_blocks,
    lower_ranks = lower_ranks,
    upper_blocks = upper_blocks,
    upper_ranks = upper_ranks,
    num_hs = num_hs,
    sizes = sizes,
    offsets = offsets,
    find_max,
  )
  
end

function get_transform_params!(
  ::Type{LeadingDecomp},
  ::Type{Right},
  step::Union{Type{BigStep},Type{SmallStep}},
  m::Int,
  n::Int;
  lower_blocks::Union{AbstractArray{Int,2}, Nothing} = nothing,
  lower_ranks::Union{AbstractArray{Int,1}, Nothing} = nothing,
  upper_blocks::Union{AbstractArray{Int,2}, Nothing} = nothing,
  upper_ranks::Union{AbstractArray{Int,1}, Nothing} = nothing,
  sizes::Union{AbstractArray{Int,1}, Nothing}=nothing,
  num_hs::Union{AbstractArray{Int,1}, Nothing}=nothing,
  offsets::Union{AbstractArray{Int,1}, Nothing}=nothing,
  find_max::Bool=false
)
  num_blocks = size(lower_blocks, 2)

  # leading lower
  old_cols_lb = 1:0
  old_rank = 0
  for lb ∈ 1:num_blocks
    _, cols_lb = lower_block_ranges(lower_blocks, m, n, lb)
    dᵣ = setdiffᵣ(cols_lb, old_cols_lb)
    trange = dᵣ ∪ᵣ last(old_cols_lb, old_rank)
    tsize = length(trange)
    if isa(sizes, AbstractArray{Int})
      if find_max
        sizes[lb] = max(sizes[lb], tsize)
      else
        sizes[lb] = tsize
      end
    end
    if isa(num_hs, AbstractArray{Int})
      if step == BigStep
        num_hs[lb] = lower_ranks[lb]
      else
        num_hs[lb] = tsize - lower_ranks[lb]
      end
    end
    if isa(offsets, AbstractArray{Int})
      offsets[lb] =
        if !isempty(trange)
          first(trange) - 1        
        elseif !isempty(cols_lb)
          last(cols_lb)
        else
          0
        end
    end
    old_cols_lb = cols_lb
    old_rank = lower_ranks[lb]
  end
  nothing
end

function get_transform_params!(
  ::Type{LeadingDecomp},
  ::Type{Left},
  step::Union{Type{BigStep},Type{SmallStep}},
  m::Int,
  n::Int;
  lower_blocks::Union{AbstractArray{Int,2}, Nothing} = nothing,
  lower_ranks::Union{AbstractArray{Int,1}, Nothing} = nothing,
  upper_blocks::Union{AbstractArray{Int,2}, Nothing} = nothing,
  upper_ranks::Union{AbstractArray{Int,1}, Nothing} = nothing,
  sizes::Union{AbstractArray{Int,1}, Nothing}=nothing,
  num_hs::Union{AbstractArray{Int,1}, Nothing}=nothing,
  offsets::Union{AbstractArray{Int,1}, Nothing}=nothing,
  find_max::Bool=false
)

  num_blocks = size(upper_blocks, 2)

  # leading upper
  old_rows_ub = 1:0
  old_rank = 0
  for ub ∈ 1:num_blocks
    rows_ub, _ = upper_block_ranges(upper_blocks, m, n, ub)
    dᵣ = setdiffᵣ(rows_ub, old_rows_ub)
    trange = dᵣ ∪ᵣ last(old_rows_ub, old_rank)
    tsize = length(trange)

    if isa(sizes, AbstractArray{Int})
      if find_max
        sizes[ub] = max(tsize, sizes[ub])
      else
        sizes[ub] = tsize
      end
    end
    if isa(num_hs, AbstractArray{Int})
      if step == BigStep
        num_hs[ub] = upper_ranks[ub]
      else
        num_hs[ub] = tsize - upper_ranks[ub]
      end
    end
    if isa(offsets, AbstractArray{Int})
      offsets[ub] =
        if !isempty(trange)
          first(trange) - 1        
        elseif !isempty(rows_ub)
          last(rows_ub)
        else
          0
        end
    end
    old_rows_ub = rows_ub
    old_rank = upper_ranks[ub]
  end
  nothing
end

function get_transform_params!(
  ::Type{TrailingDecomp},
  ::Type{Right},
  step::Union{Type{BigStep},Type{SmallStep}},
  m::Int,
  n::Int;
  lower_blocks::Union{AbstractArray{Int,2}, Nothing} = nothing,
  lower_ranks::Union{AbstractArray{Int,1}, Nothing} = nothing,
  upper_blocks::Union{AbstractArray{Int,2}, Nothing} = nothing,
  upper_ranks::Union{AbstractArray{Int,1}, Nothing} = nothing,
  sizes::Union{AbstractArray{Int,1}, Nothing}=nothing,
  num_hs::Union{AbstractArray{Int,1}, Nothing}=nothing,
  offsets::Union{AbstractArray{Int,1}, Nothing}=nothing,
  find_max::Bool=false
)
  num_blocks = size(upper_blocks, 2)

  # trailing upper
  old_cols_ub = 1:0
  old_rank = 0
  for ub ∈ num_blocks:-1:1
    (_, cols_ub) = upper_block_ranges(upper_blocks, m, n, ub)
    dᵣ = setdiffᵣ(cols_ub, old_cols_ub)
    trange = dᵣ ∪ᵣ first(old_cols_ub, old_rank)
    tsize = length(trange)
                         
    if isa(sizes, AbstractArray{Int})
      if find_max
        sizes[ub] = max(sizes[ub], tsize)
      else
        sizes[ub] = tsize
      end
    end
    if isa(num_hs, AbstractArray{Int})
      if step == BigStep
        num_hs[ub] = upper_ranks[ub]
      else
        num_hs[ub] = tsize - upper_ranks[ub]
      end
    end
    if isa(offsets, AbstractArray{Int})
      offsets[ub] =
        if !isempty(trange)
          first(trange) - 1        
        elseif !isempty(cols_ub)
          last(cols_ub)
        else
          0
        end
    end
    old_cols_ub = cols_ub
    old_rank = upper_ranks[ub]
  end
  nothing
end

function get_transform_params!(
  ::Type{TrailingDecomp},
  ::Type{Left},
  step::Union{Type{BigStep},Type{SmallStep}},
  m::Int,
  n::Int;
  lower_blocks::Union{AbstractArray{Int,2}, Nothing} = nothing,
  lower_ranks::Union{AbstractArray{Int,1}, Nothing} = nothing,
  upper_blocks::Union{AbstractArray{Int,2}, Nothing} = nothing,
  upper_ranks::Union{AbstractArray{Int,1}, Nothing} = nothing,
  sizes::Union{AbstractArray{Int,1}, Nothing}=nothing,
  num_hs::Union{AbstractArray{Int,1}, Nothing}=nothing,
  offsets::Union{AbstractArray{Int,1}, Nothing}=nothing,
  find_max::Bool=false
)
  num_blocks = size(lower_blocks, 2)

  # trailing lower
  old_rows_lb = 1:0
  old_rank = 0
  for lb ∈ num_blocks:-1:1
    rows_lb, _ = lower_block_ranges(lower_blocks, m, n, lb)
    dᵣ = setdiffᵣ(rows_lb, old_rows_lb)
    trange = dᵣ ∪ᵣ first(old_rows_lb, old_rank)
    tsize = length(trange)
    if isa(sizes, AbstractArray{Int})
      if find_max
        sizes[lb] = max(sizes[lb], tsize)
      else
        sizes[lb] = tsize
      end
    end
    if isa(num_hs, AbstractArray{Int})
      if step == BigStep
        num_hs[lb] = lower_ranks[lb]
      else
        num_hs[lb] = tsize - lower_ranks[lb]
      end
    end
    if isa(offsets, AbstractArray{Int})
      offsets[lb] =
        if !isempty(trange)
          first(trange) - 1        
        elseif !isempty(rows_lb)
          last(rows_lb)
        else
          0
        end
    end
    old_rows_lb = rows_lb
    old_rank = lower_ranks[lb]
  end
  nothing
end

function get_max_WY_sizes(
  ::Type{Left},
  m::Int,
  n::Int;
  upper_blocks::AbstractArray{Int,2},
  lower_blocks::AbstractArray{Int,2},
  upper_rank_max::Int,
  lower_rank_max::Int,
)
  num_blocks = size(lower_blocks, 2)
  sizes = zeros(Int, num_blocks)
  upper_ranks = fill(upper_rank_max, num_blocks)
  lower_ranks = fill(lower_rank_max, num_blocks)

  get_transform_params!(
    LeadingDecomp,
    Left,
    BigStep,
    m,
    n,
    upper_blocks = upper_blocks,
    upper_ranks = upper_ranks,
    sizes = sizes,
    find_max=true
  )

  get_transform_params!(
    TrailingDecomp,
    Left,
    BigStep,
    m,
    n,
    lower_blocks = lower_blocks,
    lower_ranks = lower_ranks,
    sizes = sizes,
    find_max=true
  )

  sizes

end

function get_max_WY_sizes(
  ::Type{Right},
  m::Int,
  n::Int;
  upper_blocks::AbstractArray{Int,2},
  lower_blocks::AbstractArray{Int,2},
  upper_rank_max::Int,
  lower_rank_max::Int,
)
  num_blocks = size(lower_blocks, 2)
  sizes = zeros(Int, num_blocks)
  upper_ranks = fill(upper_rank_max, num_blocks)
  lower_ranks = fill(lower_rank_max, num_blocks)

  get_transform_params!(
    LeadingDecomp,
    Right,
    BigStep,
    m,
    n,
    lower_blocks = lower_blocks,
    lower_ranks = lower_ranks,
    sizes = sizes,
    find_max=true
  )

  get_transform_params!(
    TrailingDecomp,
    Right,
    BigStep,
    m,
    n,
    upper_blocks = upper_blocks,
    upper_ranks = upper_ranks,
    sizes = sizes,
    find_max=true
  )

  sizes

end


end
