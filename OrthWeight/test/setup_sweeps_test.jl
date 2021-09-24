# Test that the sweeps in constructing a random WY-weight
# decomposition really are sweeps of orthogonal WY transformations.

tol = 1e-14

lower_blocks_sweep = [
  2 4 5 7
  2 3 4 6
]

upper_blocks_sweep = [
  1 3 4 6
  3 4 6 7
]

wyw_sweep = WYWeight(
  Float64,
  SpanStep,
  LeadingDecomp,
  Random.default_rng(),
  8,
  7,
  upper_ranks = [2 for j∈1:size(upper_blocks_sweep,2)],
  lower_ranks = [2 for j∈1:size(lower_blocks_sweep,2)],
  upper_blocks = upper_blocks_sweep,
  lower_blocks = lower_blocks_sweep,
)

sweep = SweepForward(wyw_sweep.leftWY)
A = Matrix{Float64}(I, 8, 8)
sweep ⊛ A
sweep ⊘ A
show_error_result(
  "Real Left Sweep Error",
  norm(Matrix{Float64}(I, 8, 8) - A, Inf),
  tol,
)

sweep = SweepForward(wyw_sweep.rightWY)
A = Matrix{Float64}(I, 7, 7)
A ⊛ sweep
A ⊘ sweep
show_error_result(
  "Real Right Sweep Error",
  norm(Matrix{Float64}(I, 7, 7) - A, Inf),
  tol,
)

# wyw_sweep = WYWeight(
#   Complex{Float64},
#   SpanStep,
#   LeadingDecomp,
#   Random.default_rng(),
#   8,
#   7,
#   upper_ranks = [2 for j∈1:size(upper_blocks_sweep,2)],
#   lower_ranks = [2 for j∈1:size(lower_blocks_sweep,2)],
#   upper_blocks = upper_blocks_sweep,
#   lower_blocks = lower_blocks_sweep,
# )

# sweep = SweepForward(wyw_sweep.leftWY)
# A = Matrix{Complex{Float64}}(I, 8, 8)
# sweep ⊛ A
# sweep ⊘ A
# show_error_result(
#   "Complex Left Sweep Error",
#   norm(Matrix{Complex{Float64}}(I, 8, 8) - A, Inf),
#   tol,
# )

# sweep = SweepForward(wyw_sweep.rightWY)
# A = Matrix{Complex{Float64}}(I, 7, 7)
# A ⊛ sweep
# A ⊘ sweep
# show_error_result(
#   "Complex Right Sweep Error",
#   norm(Matrix{Complex{Float64}}(I, 7, 7) - A, Inf),
#   tol,
# )
