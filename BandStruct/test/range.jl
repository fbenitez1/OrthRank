@safetestset "Column/row Range Tests" begin
  using BandStruct
  using Random

  @testset "$E" for
    E ∈ [ Float64,
          Complex{Float64} ]

    (bc, bbc) =
      BandStruct.standard_test_case(E, upper_rank_max = 3, lower_rank_max = 2)

    @testset "$T" for
      (bc0, T) ∈ [ (bc, "Unblocked"),
                   (bbc, "Blocked") ]


      # bc0 =
      #                 1   2       3   4
      #       X   X   X | U | O   O | N |
      #                 +---+-----------+
      #       X   X   X   X | U   U | O |
      #     1 ------+       |       |   |
      #       O   L | X   X | U   U | O |
      #             |       +-------+---+
      #       O   L | X   X   X   X | U |
      #     2 ------+---+           +---+
      #       O   O | L | X   X   X   X |
      #     3 ------+---+---+           |
      #       O   O | O | L | X   X   X |
      #             |   |   |           +
      #       N   N | N | L | X   X   X 
      #     4 ------+---+---+-------+   
      #       N   N | N | N | O   L | X 

      @testset "Full BandColumn index ranges." begin
        @testset "Columns" begin
          column_inband_index_ranges_bc0 = [1:2, 1:4, 1:5, 1:7, 2:7, 2:8, 4:8]

          @test column_inband_index_ranges_bc0 ==
            [inband_index_range(bc0, :, k) for k = 1:bc0.n]
          @test column_inband_index_ranges_bc0 ==
            [inband_index_range(bc0, :, k) for k = 1:bc0.n]

          upper_column_inband_index_ranges_bc0 =
            [1:0, 1:0, 1:0, 1:1, 2:3, 2:3, 4:4]
          @test upper_column_inband_index_ranges_bc0 ==
            [upper_inband_index_range(bc0, :, k) for k = 1:bc0.n]
          @test upper_column_inband_index_ranges_bc0 ==
            [upper_inband_index_range(bc0, :, k) for k = 1:bc0.n]

          middle_column_inband_index_ranges_bc0 =
            [1:2, 1:2, 1:4, 2:5, 4:7, 4:7, 5:8]
          @test middle_column_inband_index_ranges_bc0 ==
            [middle_inband_index_range(bc0, :, k) for k = 1:bc0.n]
          @test middle_column_inband_index_ranges_bc0 ==
            [middle_inband_index_range(bc0, :, k) for k = 1:bc0.n]

          lower_column_inband_index_ranges_bc0 =
            [1:0, 3:4, 5:5, 6:7, 1:0, 8:8, 1:0]
          @test lower_column_inband_index_ranges_bc0 ==
            [lower_inband_index_range(bc0, :, k) for k = 1:bc0.n]
          @test lower_column_inband_index_ranges_bc0 ==
            [lower_inband_index_range(bc0, :, k) for k = 1:bc0.n]
        end

        @testset "Rows" begin
          row_inband_index_ranges_bc0 = [1:4, 1:6, 2:6, 2:7, 3:7, 4:7, 4:7, 6:7]
          @test row_inband_index_ranges_bc0 ==
            [inband_index_range(bc0, j, :) for j = 1:bc0.m]
          @test row_inband_index_ranges_bc0 ==
            [inband_index_range(bc0, j, :) for j = 1:bc0.m]

          upper_row_inband_index_ranges_bc0 =
            [4:4, 5:6, 5:6, 7:7, 8:7, 8:7, 8:7, 8:7]
          @test upper_row_inband_index_ranges_bc0 ==
            [upper_inband_index_range(bc0, j, :) for j = 1:bc0.m]
          @test upper_row_inband_index_ranges_bc0 ==
            [upper_inband_index_range(bc0, j, :) for j = 1:bc0.m]

          lower_row_inband_index_ranges_bc0 =
            [1:0, 1:0, 2:2, 2:2, 3:3, 4:4, 4:4, 6:6]
          @test lower_row_inband_index_ranges_bc0 ==
            [lower_inband_index_range(bc0, j, :) for j = 1:bc0.m]
          @test lower_row_inband_index_ranges_bc0 ==
            [lower_inband_index_range(bc0, j, :) for j = 1:bc0.m]
        end

      end
      
      # bc1 = 
      # X X U U O
      # X X U U O
      # X X X X U
      # L X X X X
      # O L X X X

      @testset "Submatrix/View BandColumn index ranges, $x." for
        (bc1, x) ∈ [ (bc0[2:6, 3:7], "Submatrix"),
                     (view(bc0,2:6, 3:7), "View") ]
        @testset "Columns" begin
          column_inband_index_ranges_bc1 = [1:4, 1:5, 1:5, 1:5, 3:5]
          @test column_inband_index_ranges_bc1 ==
            [inband_index_range(bc1, :, k) for k = 1:bc1.n]

          column_upper_inband_index_ranges_bc1 = [1:0, 1:0, 1:2, 1:2, 3:3]
          @test column_upper_inband_index_ranges_bc1 ==
            [upper_inband_index_range(bc1, :, k) for k = 1:bc1.n]

          column_middle_inband_index_ranges_bc1 = [1:3, 1:4, 3:5, 3:5, 4:5]
          @test column_middle_inband_index_ranges_bc1 ==
            [middle_inband_index_range(bc1, :, k) for k = 1:bc1.n]

          column_lower_inband_index_ranges_bc1 = [4:4, 5:5, 1:0, 1:0, 1:0]
          @test column_lower_inband_index_ranges_bc1 ==
            [lower_inband_index_range(bc1, :, k) for k = 1:bc1.n]
        end

        @testset "Rows" begin
          row_inband_index_ranges_bc1 = [1:4, 1:4, 1:5, 1:5, 2:5]
          @test row_inband_index_ranges_bc1 ==
            [inband_index_range(bc1, j, :) for j = 1:bc1.m]

          row_upper_inband_index_ranges_bc1 = [3:4, 3:4, 5:5, 1:0, 1:0]
          @test row_upper_inband_index_ranges_bc1 ==
            [upper_inband_index_range(bc1, j, :) for j = 1:bc1.m]

          row_middle_inband_index_ranges_bc1 = [1:2, 1:2, 1:4, 2:5, 3:5]
          @test row_middle_inband_index_ranges_bc1 ==
            [middle_inband_index_range(bc1, j, :) for j = 1:bc1.m]

          row_lower_inband_index_ranges_bc1 = [1:0, 1:0, 1:0, 1:1, 2:2]
          @test row_lower_inband_index_ranges_bc1 ==
            [lower_inband_index_range(bc1, j, :) for j = 1:bc1.m]
        end
      end

    end
  end
end
nothing
