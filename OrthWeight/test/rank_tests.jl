tol_r = 1e-14

lower_blocks_r = [
  10 20 30 40
  10 20 25 37
]

upper_blocks_r = [
  10 20 30 40
  10 20 25 37
]

r = 2

wyw_r = WYWeight(
  Float64,
  SpanStep,
  LeadingDecomp,
  Random.default_rng(),
  60,
  50,
  upper_rank_max = r,
  lower_rank_max = r,
  upper_blocks = upper_blocks_r,
  lower_blocks = lower_blocks_r,
)

A = Matrix(wyw_r)

let uflag::Union{Bool,Int} = false
  for l ∈ 1:size(upper_blocks_r, 2)
    ru = rank(Matrix(view(wyw_r.b, upper_block_ranges(wyw_r.b, l)...)), tol_r)
    ru == r || (uflag = l; break)
  end
  show_equality_result("Upper blocks rank test", uflag, false)
  typeof(uflag) == Int &&
    (println("     Incorrect rank in upper block: ", uflag); println())
  nothing
end

let lflag::Union{Bool,Int} = false
  for l ∈ 1:size(lower_blocks_r, 2)
    rl = rank(Matrix(view(wyw_r.b, upper_block_ranges(wyw_r.b, l)...)), tol_r)
    rl == r || (lflag = l; break)
  end
  show_equality_result("Lower blocks rank test", lflag, false)
  typeof(lflag) == Int &&
    (println("     Incorrect rank in lower block: ", lflag); println())
  nothing
end

