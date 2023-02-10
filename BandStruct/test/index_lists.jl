@safetestset "IndexLists" begin
  using BandStruct.IndexLists

  xs = IndexList([1,2,3], max_length=7)
  @testset "Collection." begin
    @test validate(xs)
    @test collect_elements(xs) == [1, 2, 3]
    @test [xs[j] for j in collect(Iterators.Reverse(xs))] == [3, 2, 1]
  end

  @testset "Push!" begin
    push!(xs, 4)
    @test validate(xs)
    @test collect_elements(xs) == [1, 2, 3, 4]
  end

  @testset "Popped!" begin
    result = pop!(xs)
    @test validate(xs)
    @test result == 4
    @test collect_elements(xs) == [1, 2, 3]
  end

  @testset "Pushed first" begin
    pushfirst!(xs, 0)
    @test validate(xs)
    @test collect_elements(xs) == [0, 1, 2, 3]
  end

  @testset "Popped first" begin
    popfirst!(xs)
    @test validate(xs)
    @test collect_elements(xs) == [1, 2, 3]
  end

  # Start with empty list
  xs = IndexList(Int, max_length=7)

  @testset "Empty push!" begin
    push!(xs, 1)
    @test validate(xs)
    @test collect_elements(xs) == [1]
  end

  @testset "Pop to empty" begin
    pop!(xs)
    @test validate(xs)
    @test collect_elements(xs) == Int[]
  end

  @testset "Push 3 back back to empty" begin
    push!(xs, 1)
    @test validate(xs)
    @test collect_elements(xs) == [1]
    push!(xs, 2)
    @test collect_elements(xs) == [1, 2]
    @test validate(xs)
    push!(xs, 3)
    @test validate(xs)
    @test collect_elements(xs) == [1, 2, 3]
  end

  @testset "Pop 3 back back to empty" begin
    pop!(xs)
    @test validate(xs)
    @test collect_elements(xs) == [1, 2]
    pop!(xs)
    @test validate(xs)
    @test collect_elements(xs) == [1]
    pop!(xs)
    @test validate(xs)
    @test collect_elements(xs) == Int[]
  end

  @testset "Push first 3 to empty" begin
    pushfirst!(xs, 3)
    @test validate(xs)
    @test collect_elements(xs) == [3]
    pushfirst!(xs, 2)
    @test validate(xs)
    @test collect_elements(xs) == [2, 3]
    pushfirst!(xs, 1)
    @test validate(xs)
    @test collect_elements(xs) == [1, 2, 3]
  end

  @testset "Another push!" begin
    push!(xs, 4)
    validate(xs)
    @test collect_elements(xs) == [1, 2, 3, 4]
  end

  @testset "Another pushfirst!" begin
    pushfirst!(xs, 0)
    @test validate(xs)
    @test collect_elements(xs) == [0, 1, 2, 3, 4]
  end

  @testset "Pop 3 indices" begin
    pop!(xs)
    @test validate(xs)
    @test collect_elements(xs) == [0, 1, 2, 3]
    pop!(xs)
    @test validate(xs)
    @test collect_elements(xs) == [0, 1, 2]
    pop!(xs)
    @test validate(xs)
    @test collect_elements(xs) == [0, 1]
  end

  xs = IndexList([1, 2, 3, 4, 5], max_length = 10)

  @testset "Various removals" begin
    inds = collect(xs)
    remove!(xs, inds[2])
    @test validate(xs)
    @test collect_elements(xs) == [1, 3, 4, 5]
    remove!(xs, inds[4])
    @test validate(xs)
    @test collect_elements(xs) == [1, 3, 5]
    remove!(xs, inds[1])
    @test validate(xs)
    @test collect_elements(xs) == [3, 5]
    remove!(xs, inds[5])
    @test validate(xs)
    @test collect_elements(xs) == [3]
  end

  xs = IndexList([1, 5], max_length = 10)

  @testset "Various inserts" begin
    inds = collect(xs)
    insert_after!(xs, inds[1], 2)
    @test validate(xs)
    @test collect_elements(xs) == [1, 2, 5]
    inds = collect(xs)
    insert_before!(xs, inds[3], 4)
    @test validate(xs)
    @test collect_elements(xs) == [1, 2, 4, 5]
    inds = collect(xs)
    insert_before!(xs, inds[3], 3)
    @test validate(xs)
    @test collect_elements(xs) == [1, 2, 3, 4, 5]
    inds = collect(xs)
    insert_before!(xs, inds[1], 0)
    @test validate(xs)
    @test collect_elements(xs) == [0, 1, 2, 3, 4, 5]
    inds = collect(xs)
    insert_after!(xs, inds[6], 6)
    @test validate(xs)
    @test collect_elements(xs) == [0, 1, 2, 3, 4, 5, 6]
  end

  @testset "Exchange indices" begin
    inds = collect(xs)
    exchange_indices!(xs, inds[2], inds[4])
    @test validate(xs)
    @test collect_elements(xs) == [0, 3, 2, 1, 4, 5, 6]
    inds = collect(xs)
    exchange_indices!(xs, inds[1], inds[7])
    @test validate(xs)
    @test collect_elements(xs) == [6, 3, 2, 1, 4, 5, 0]
    inds = collect(xs)
    exchange_indices!(xs, inds[3], inds[4])
    @test validate(xs)
    @test collect_elements(xs) == [6, 3, 1, 2, 4, 5, 0]
    inds = collect(xs)
    exchange_indices!(xs, inds[1], inds[2])
    @test validate(xs)
    @test collect_elements(xs) == [3, 6, 1, 2, 4, 5, 0]
    inds = collect(xs)
    exchange_indices!(xs, inds[7], inds[6])
    @test validate(xs)
    @test collect_elements(xs) == [3, 6, 1, 2, 4, 0, 5]
  end

  @testset "Exchange data" begin
    inds = collect(xs)
    exchange_data!(xs, inds[3], inds[1])
    @test validate(xs)
    @test collect_elements(xs) == [1, 6, 3, 2, 4, 0, 5]
    exchange_data!(xs, inds[3], inds[4])
    @test validate(xs)
    @test collect_elements(xs) == [1, 6, 2, 3, 4, 0, 5]
    exchange_data!(xs, inds[3], inds[2])
    @test validate(xs)
    @test collect_elements(xs) == [1, 2, 6, 3, 4, 0, 5]
    exchange_data!(xs, inds[3], inds[5])
    @test validate(xs)
    @test collect_elements(xs) == [1, 2, 4, 3, 6, 0, 5]
    exchange_data!(xs, inds[3], inds[4])
    @test validate(xs)
    @test collect_elements(xs) == [1, 2, 3, 4, 6, 0, 5]
  end

  @testset "Sort indices" begin
    sort_indices!(xs)
    @test validate(xs)
    @test collect_elements(xs) == [1, 2, 3, 4, 6, 0, 5]
    @test xs.first_index == 1
    @test xs.last_index == 7
    @test xs.is_free == [0,0,0,0,0,0,0,1,1,1]
    @test xs.prev_next_indices ==
      [0 1 2 3 4 5 6 0 0 0;
       2 3 4 5 6 7 0 0 0 0]
    @test xs.sorted == true
    @test xs.length == 7
    @test xs.free_indices == [10, 9, 8, 0 , 0, 0, 0, 0, 0, 0]
    @test xs.free_length == 3
  end

end
