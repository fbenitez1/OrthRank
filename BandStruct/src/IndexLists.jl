module IndexLists

using ErrorTypes

export IndexList,
  ListIndex,
  SortedError,
  FreeIndexError,
  is_first,
  is_last,
  is_free,
  is_sorted,
  Before,
  After,
  BeforeAfterError,
  next_list_index,
  prev_list_index,
  first_list_index,
  last_list_index,
  ListOutOfStorageError,
  unsafe_push_free!,
  unsafe_pop_free!,
  remove!,
  insert_after!,
  insert_before!,
  exchange_data!,
  exchange_indices!,
  validate,
  sort_indices!,
  print_list,
  collect_elements

"""
    mutable struct IndexList{E}
      sorted::Bool
      length::Int
      data::Vector{E}
      max_length::Int
      first_index::Int
      last_index::Int
      prev_next_indices::Matrix{Int}
      # maintain a stack with free indices in
      free_length::Int
      free_indices::Vector{Int}
      is_free::Vector{Bool}
    end

A preallocated doubly linked list.  If a value is stored in xs.data[i]
then the next value is in `xs.data[prev_next_indices[2,i]]` and the
previous value is in `xs.data[prev_next_indices[1,i]]`.  The vector
`free_indices` contains unused indices in positions
`1:xs.free_length`.  Functions should make sure `xs.free_length +
xs.length[] = xs.max_length` when list structure is fully restored.
The last element with index `i` should have
`xs.prev_next_indices[2,i]==0` and the first element with index `j`
should have `xs.prev_next_indices[1,j]==0`.  If the list is empty,
`xs.first_index` and `xs.last_index` should both be zero.

"""
mutable struct IndexList{E}
  sorted::Bool
  length::Int
  data::Vector{E}
  max_length::Int
  first_index::Int
  last_index::Int
  prev_next_indices::Matrix{Int}
  # maintain a stack with free indices in
  free_length::Int
  free_indices::Vector{Int}
  is_free::Vector{Bool}
end

function IndexList(
  v::Union{AbstractVector{E}, IndexList{E}};
  max_length::Int = length(v),
  copy = true,
) where {E}
  n = length(v)
  max_length = max(max_length, n)
  v1 = Vector{E}(undef, max_length)
  if v isa IndexList
    for (j, li) ∈ zip(1:n, v)
      v1[j] = copy ? deepcopy(v[li]) : v[li]
    end
  else
    for (j, x) ∈ zip(1:n, v)
      v1[j] = copy ? deepcopy(x) : x
    end
  end
  indices = zeros(Int, 2, max_length)
  indices[1, 2:n] = 1:(n - 1)
  indices[2, 1:(n - 1)] = 2:n
  free = zeros(Int, max_length)
  free_length = max_length - n
  free[1:free_length] = (n + free_length):-1:(n + 1)
  is_free = fill(true, max_length)
  is_free[1:n] .= false
  first_index = n == 0 ? 0 : 1
  last_index = n == 0 ? 0 : n
  IndexList(
    true,
    n,
    v1,
    max_length,
    first_index,
    last_index,
    indices,
    free_length,
    free,
    is_free,
  )
end

function IndexList(E::DataType; max_length::Int)
  v=E[]
  IndexList(v; max_length = max_length)
end

struct ListIndex
  index::Int
end

Base.size(xs::IndexList) = (xs.length,)
Base.length(xs::IndexList) = xs.length
Base.isempty(xs::IndexList) = xs.length == 0

is_free(xs::IndexList, i::ListIndex) = xs.is_free[i.index]

is_sorted(xs::IndexList) = xs.sorted

struct FreeIndexError end

function Base.getindex(xs::IndexList, i::ListIndex)
  xs.is_free[i.index] && throw(FreeIndexError)
  xs.data[i.index]
end

function Base.setindex!(xs::IndexList, x, i::ListIndex)
  xs.is_free[i.index] && throw(FreeIndexError)
  xs.data[i.index] = x
end

Base.view(xs::IndexList, i::ListIndex) = xs[i]

struct SortedError <: Exception
  arr::IndexList
  index::Int
end

# For sorted, a linear index is OK.
function Base.getindex(xs::IndexList, i::Int)
  n = xs.length
  (i >= 1 && i <= xs.length) || throw(BoundsError(xs,i))
  if xs.sorted
    xs.data[i]
  elseif i == 1
    xs.data[xs.first_index]
  elseif i == n
    xs.data[xs.last_index]
  else
    throw(SortedError(xs,i))
  end
end

function Base.lastindex(xs::IndexList)
  isempty(xs) && throw(ArgumentError("List must be nonempty."))
  ListIndex(xs.last_index)
end

function Base.firstindex(xs::IndexList)
  isempty(xs) && throw(ArgumentError("List must be nonempty."))
  ListIndex(xs.first_index)
end

is_first(xs::IndexList, xsi::ListIndex) = xsi.index == xs.first_index

is_last(xs::IndexList, xsi::ListIndex) = xsi.index == xs.last_index

Base.iterate(xs::IndexList) =
  isempty(xs) ? nothing : (ListIndex(xs.first_index), ListIndex(xs.first_index))

function Base.iterate(xs::IndexList, i::ListIndex)
  if xs.prev_next_indices[2, i.index] == 0
    return nothing
  else
    result = ListIndex(xs.prev_next_indices[2, i.index])
    # perhaps this index was freed during iteration...
    if is_free(xs, i)
      xs.prev_next_indices[1, i.index] = 0
      xs.prev_next_indices[2, i.index] = 0
    end
    return (result, result)
  end
end

Base.length(rxs::Iterators.Reverse{<:IndexList}) = rxs.itr.length

Base.isempty(rxs::Iterators.Reverse{<:IndexList}) = rxs.itr.length == 0

Base.iterate(rxs::Iterators.Reverse{<:IndexList}) = isempty(rxs) ? nothing :
  (ListIndex(rxs.itr.last_index), ListIndex(rxs.itr.last_index))

function Base.iterate(rxs::Iterators.Reverse{<:IndexList}, i::ListIndex)
  if rxs.itr.prev_next_indices[1, i.index] == 0
    return nothing
  else
    result = ListIndex(
      rxs.itr.prev_next_indices[1, i.index],
    )
    if is_free(rxs.itr, i)
      rxs.itr.prev_next_indices[1, i.index] = 0
      rxs.itr.prev_next_indices[2, i.index] = 0
    end
    return (result, result)
  end
end

collect_elements(xs) = [xs[j] for j ∈ xs]

struct After end
struct Before end

struct BeforeAfterError <: Exception
  err::Union{Before, After}
end

function next_list_index(
  xs::IndexList,
  i::ListIndex,
)::Result{ListIndex,BeforeAfterError}
  xs.is_free[i.index] && throw(FreeIndexError)
  isempty(xs) && throw(ArgumentError("List must be nonempty."))
  is_last(xs, i) && (return Err(BeforeAfterError(After())))
  return Ok(ListIndex(xs.prev_next_indices[2, i.index]))
end

function next_list_index(
  ::IndexList,
  ::After
)::Result{ListIndex,BeforeAfterError}
  return Err(BeforeAfterError(After()))
end

function next_list_index(
  ::IndexList,
  ::Before
)::Result{ListIndex,BeforeAfterError}
  return Err(BeforeAfterError(Before()))
end

function prev_list_index(
  xs::IndexList,
  i::ListIndex,
)::Result{ListIndex,BeforeAfterError}
  xs.is_free[i.index] && throw(FreeIndexError)
  isempty(xs) && throw(ArgumentError("List must be nonempty."))
  is_first(xs, i) && (return Err(BeforeAfterError(Before())))
  return Ok(ListIndex(xs.prev_next_indices[1, i.index]))
end

function prev_list_index(
  ::IndexList,
  ::After
)::Result{ListIndex,BeforeAfterError}
  return Err(BeforeAfterError(After()))
end

function prev_list_index(
  ::IndexList,
  ::Before
)::Result{ListIndex,BeforeAfterError}
  return Err(BeforeAfterError(Before()))
end

function last_list_index(xs::IndexList) 
  isempty(xs) && throw(ArgumentError("List must be nonempty."))
  ListIndex(xs.last_index)
end

function first_list_index(xs::IndexList) 
  isempty(xs) && throw(ArgumentError("List must be nonempty."))
  ListIndex(xs.first_index)
end

struct ListOutOfStorageError <: Exception
  list::IndexList
end

# unsafe because it adjusts the free_length but doesn't otherwise
# adjust the list in any way, including the length.  These should be
# used as primitives for other functions for manipulating the list.
function unsafe_pop_free!(xs::IndexList)
  xs.length == xs.max_length && throw(ListOutOfStorageError(xs))
  i = xs.free_indices[xs.max_length - xs.length]
  xs.free_indices[xs.max_length - xs.length] = 0
  xs.free_length = xs.free_length - 1
  xs.is_free[i] = false
  return i
end

function unsafe_push_free!(xs::IndexList, i::Int)
  xs.free_length = xs.free_length + 1
  xs.free_indices[xs.free_length] = i
  xs.is_free[i] = true
  return nothing
end

# Push and pop to the end don't make the list unsorted.
function Base.push!(xs::IndexList{E}, x::E) where E
  inew = unsafe_pop_free!(xs)
  xs.prev_next_indices[2,inew] = 0 # no successor.
  if isempty(xs)
    # adding to an empty list means inew has no predecessor and inew
    # is both first and last.
    xs.prev_next_indices[1,inew] = 0
    xs.first_index=inew
  else
    # the old last index has a successor now.
    ilast = xs.last_index
    xs.prev_next_indices[2,ilast] = inew
    xs.prev_next_indices[1,inew] = ilast
  end
  xs.data[inew] = x
  xs.last_index=inew
  xs.length = xs.length+1
  return nothing
end

function Base.pop!(xs::IndexList)
  isempty(xs) && throw(ArgumentError("List must be nonempty."))
  xs.length = xs.length-1
  ilast = xs.last_index
  prev = xs.prev_next_indices[1,ilast]
  xs.last_index = prev # sets to zero if list is now empty.
  if xs.length == 0
    xs.first_index = 0
  else
    # if there was a previous element, it now has no successor.
    xs.prev_next_indices[2, prev] = 0
  end
  unsafe_push_free!(xs, ilast)
  return xs.data[ilast]
end

# push and pop to the front do make the List unsorted.
function Base.pushfirst!(xs::IndexList{E}, x::E) where E
  inew = unsafe_pop_free!(xs)
  xs.prev_next_indices[1,inew] = 0 # no predecessor for inew.
  if isempty(xs)
    # inew has no successor with an empty list and inew is both first
    # and last.
    xs.prev_next_indices[2,inew] = 0
    xs.last_index=inew
  else
    # the old first index now has a predecessor.
    ifirst = xs.first_index
    xs.prev_next_indices[1,ifirst] = inew
    xs.prev_next_indices[2,inew] = ifirst
  end
  xs.data[inew] = x
  xs.sorted = false
  xs.first_index=inew
  xs.length = xs.length+1
  return nothing
end

function Base.popfirst!(xs::IndexList)
  isempty(xs) && throw(ArgumentError("List must be nonempty."))
  xs.length = xs.length-1
  ifirst = xs.first_index
  next = xs.prev_next_indices[2,ifirst]
  xs.first_index = next # sets to zero if list is now empty.
  if xs.length == 0
    xs.last_index = 0
  else
    # if there was a next element, it now has no predecessor.
    xs.prev_next_indices[1, next] = 0
  end
  xs.sorted = false
  unsafe_push_free!(xs, ifirst)
  return xs.data[ifirst]
end

function remove!(xs::IndexList, i::ListIndex)
  isempty(xs) && throw(ArgumentError("List must be nonempty."))
  is_free(xs, i) && throw(FreeIndexError)
  if is_first(xs,i)
    Base.popfirst!(xs)
  elseif is_last(xs,i)
    Base.pop!(xs)
  else
    prev = xs.prev_next_indices[1,i.index]
    next = xs.prev_next_indices[2,i.index]
    xs.prev_next_indices[2,prev] = next
    xs.prev_next_indices[1,next] = prev
    xs.length = xs.length - 1
    if xs.length == 0
      xs.first_index = 0
      xs.last_index = 0
    end
    unsafe_push_free!(xs, i.index)
    xs.is_free[i.index] = true
    xs.sorted = false
  end
  return nothing
end

function insert_after!(xs::IndexList{E}, i::ListIndex, x::E) where {E}
  isempty(xs) && throw(ArgumentError("List must be nonempty."))
  is_free(xs, i) && throw(FreeIndexError)
  if is_last(xs, i)
    push!(xs, x)
  else
    index_new = unsafe_pop_free!(xs)
    next = xs.prev_next_indices[2,i.index]
    xs.prev_next_indices[1,next] = index_new
    xs.prev_next_indices[2,i.index] = index_new
    xs.prev_next_indices[1,index_new] = i.index
    xs.prev_next_indices[2,index_new] = next
    xs.length = xs.length + 1
    xs.data[index_new] = x
    xs.is_free[index_new] = false
    xs.sorted = false
  end
end

function insert_before!(xs::IndexList{E}, i::ListIndex, x::E) where {E}
  isempty(xs) && throw(ArgumentError("List must be nonempty."))
  is_free(xs, i) && throw(FreeIndexError)
  if is_first(xs, i)
    pushfirst!(xs, x)
  else
    index_new = unsafe_pop_free!(xs)
    prev = xs.prev_next_indices[1,i.index]
    xs.prev_next_indices[2,prev] = index_new
    xs.prev_next_indices[1,i.index] = index_new
    xs.prev_next_indices[1,index_new] = prev
    xs.prev_next_indices[2,index_new] = i.index
    xs.length = xs.length + 1
    xs.data[index_new] = x
    xs.is_free[index_new] = false
    xs.sorted = false
  end
end

function exchange_data!(xs::IndexList, i::ListIndex, j::ListIndex)
  (is_free(xs, i) || is_free(xs, j)) && throw(FreeIndexError)
  i == j && (return nothing)
  tmp = xs.data[i.index]
  xs.data[i.index] = xs.data[j.index]
  xs.data[j.index] = tmp
  return nothing
end

function exchange_indices!(xs::IndexList, i::ListIndex, j::ListIndex)
  (is_free(xs, i) || is_free(xs, j)) && throw(FreeIndexError)
  i == j && (return nothing)

  if i.index == xs.first_index
    xs.first_index = j.index
  elseif j.index == xs.first_index
    xs.first_index = i.index
  end

  if i.index == xs.last_index
    xs.last_index = j.index
  elseif j.index == xs.last_index
    xs.last_index = i.index
  end

  i_prev = xs.prev_next_indices[1, i.index]
  i_next = xs.prev_next_indices[2, i.index]
  j_prev = xs.prev_next_indices[1, j.index]
  j_next = xs.prev_next_indices[2, j.index]

  i_prev != 0 && (xs.prev_next_indices[2, i_prev] = j.index)
  i_next != 0 && (xs.prev_next_indices[1, i_next] = j.index)

  j_prev != 0 && (xs.prev_next_indices[2, j_prev] = i.index)
  j_next != 0 && (xs.prev_next_indices[1, j_next] = i.index)
  if i_next == j.index
    xs.prev_next_indices[1, i.index] = j.index
    xs.prev_next_indices[2, i.index] = j_next
    xs.prev_next_indices[1, j.index] = i_prev
    xs.prev_next_indices[2, j.index] = i.index
  elseif j_next == i.index
    xs.prev_next_indices[1, j.index] = i.index
    xs.prev_next_indices[2, j.index] = i_next
    xs.prev_next_indices[1, i.index] = j_prev
    xs.prev_next_indices[2, i.index] = j.index
  else  
    xs.prev_next_indices[1, i.index] = j_prev
    xs.prev_next_indices[2, i.index] = j_next
    xs.prev_next_indices[1, j.index] = i_prev
    xs.prev_next_indices[2, j.index] = i_next
  end

  xs.sorted = false
  return nothing
end

function validate(xs::IndexList)
  if isempty(xs)
    (xs.first_index == 0 && xs.first_index == 0) ||
      error("First and last indices incorrect for an empty list.")
    xs.free_length == xs.max_length ||
      error("Free length incorrect for empty list.")
  else
    # Forward traversal.
    j = ListIndex(xs.first_index)
    n = 1
    xs.prev_next_indices[1, j.index] == 0 ||
      error("First element has a predecessor.")
    indices = [j]
    while xs.prev_next_indices[2, j.index] != 0
      xs.is_free[j.index] == false || error("Incorrectly marked as free.")
      jnext = unwrap(next_list_index(xs, j))
      jnext == ListIndex(xs.prev_next_indices[2, j.index]) ||
        error("next_list_index, bad successor.")
      j == ListIndex(xs.prev_next_indices[1, jnext.index]) ||
        error("next_list_index, bad predecessor.")
      j = jnext
      n = n + 1
      push!(indices, j)
    end
    n == xs.length || error("Bad length.")
    xs.free_length == xs.max_length - xs.length || error("Bad free length.")
    indices == collect(xs) || error("Bad collected indices.")

    # Reverse traveral
    j = ListIndex(xs.last_index)
    n = 1
    xs.prev_next_indices[2, j.index] == 0 ||
      error("Last element has a successor.")
    indices = [j]
    while xs.prev_next_indices[1, j.index] != 0
      xs.is_free[j.index] == false || error("Reverse incorrectly marked as free.")
      jprev = unwrap(prev_list_index(xs, j))
      jprev == ListIndex(xs.prev_next_indices[1, j.index]) ||
        error("prev_list_index, bad predecessor.")
      j == ListIndex(xs.prev_next_indices[2, jprev.index]) ||
        error("prev_list_index, bad successor.")
      j = jprev
      n = n + 1
      push!(indices, j)
    end
    n == xs.length || error("Reverse traversal bad length.")
    indices == collect(Iterators.Reverse(xs)) ||
      error("Reverse bad collected indices.")
    if xs.sorted == true
      collect(xs) == map(ListIndex, 1:(xs.length)) || error("Bad sorted indices.")
    end
  end
  num_free = 0
  for f ∈ xs.is_free
    f && (num_free += 1)
  end
  num_free == xs.free_length || error("Incorrect number of marked free indices.")
  return true
end

function sort_indices!(xs::IndexList)
  n = xs.length
  if n == 0
    xs.prev_next_indices .= 0
    xs.free_length = xs.max_length
    xs.free_indices[1:xs.max_length] = xs.max_length:-1:1
    xs.is_free .= 1
    xs.first_index = 0
    xs.last_index = 0
  else
    j = 1
    for i ∈ xs
      # A map from order to location.
      xs.prev_next_indices[1,j] = i.index
      j += 1
    end
    xs.prev_next_indices[2,:] .= 0
    for j ∈ 1:xs.length
      # A map from location to order.
      ind = xs.prev_next_indices[1,j]
      xs.prev_next_indices[2,ind] = j
    end
    for j ∈ 1:xs.length
      if xs.prev_next_indices[1,j] != j
        # In the wrong location..
        if is_free(xs, ListIndex(j))
          xs.data[j] = xs.data[xs.prev_next_indices[1,j]]
        else
          tmp = xs.data[j]
          xs.data[j] = xs.data[xs.prev_next_indices[1,j]]
          xs.data[xs.prev_next_indices[1,j]] = tmp
          # Now element originally at location j and order
          # xs.prev_next_indices[2,j] is at location
          # xs.prev_next_indices[1,j].  It should have
          # the same order as before:
          xs.prev_next_indices[2, xs.prev_next_indices[1, j]] =
            xs.prev_next_indices[2, j]
          xs.prev_next_indices[1, xs.prev_next_indices[2, j]] = 
            xs.prev_next_indices[1, j]
        end
      end
    end
  end
  xs.is_free[1:n] .= false
  xs.is_free[(n + 1):(xs.max_length)] .= true
  xs.free_indices[1:(xs.free_length)] = (n + xs.free_length):-1:(n + 1)
  xs.free_indices[(xs.free_length + 1):(xs.max_length)] .= 0
  xs.prev_next_indices .= 0
  xs.prev_next_indices[1, 2:n] = 1:(n - 1)
  xs.prev_next_indices[2, 1:(n - 1)] = 2:n
  xs.first_index = 1
  xs.last_index = n
  xs.sorted = true
end

function print_list(xs::IndexList)
  println("Length $(xs.length[]) list with max_length=$(xs.max_length):")
  println("sorted = $(xs.sorted[])")
  println("first_index = $(xs.first_index[])")
  println("last_index = $(xs.last_index[])")
  println("prev_next_indices=")
  display(xs.prev_next_indices)
  println("free_length = $(xs.free_length[])")
  println("free_indices=")
  display(xs.free_indices)
  println("is_free=")
  display(xs.is_free)
  println("data=")
  display(xs.data)
end

end
