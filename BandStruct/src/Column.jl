module Column
import Base.size, Base.getindex, Base.setindex!
export BandColumnLeading, print_wilk, size, get_wilk, getindex, setindex!,
    lower_ranks_to_bw, upper_ranks_to_bw

#=
A banded matrix with structure defined by leading blocks and
stored in a compressed column-wise format.

Given

A = X | U | O   O   O | O | O
    _ ⌋   |           |   |  
    L   X | O   O   O | O | O
          |           |   |  
    L   X | U   U   U | O | O
    _ _ _ ⌋           |   |  
    O   L   X   X   X | O | O
    _ _ _ _ _ _ _ _ _ |   |  
    O   O   O   O   L | O | O
                      |   |  
    O   O   O   O   L | U | U
    _ _ _ _ _ _ _ _ _ ⌋ _ |  
    O   O   O   O   O   L   X

    O   O   O   O   O   L   X

The matrix is stored as

B = 

Num rows|Elements
----------------------
ubwmax  |O O O O O O O
        |O O O O O O O
        |O O O O O O O
        |O U U U U U U
----------------------
mbwmax+ |X X X X X L X
lbwmax  |L X O O L L X
        |L L O O L O O
        |O O O O O O O

where
    first_super =        [ 0, 1, 3, 3, 3, 6, 6]
    upper_bws =          [ 1, 1, 1, 1, 1, 1, 1]
    middle_bws =         [ 1, 2, 1, 1, 1, 0, 2]
    lower_bws =          [ 1, 1, 0, 0, 2, 1, 0]
    leading_block_rows = [ 1, 3, 4, 6, 6, 8]
    leading_block_cols = [ 1, 2, 5, 5, 6, 7]
    upper_bw_max =       4
    middle_bw_max =      2
    lower_bw_max =       2
    m = 8
    n = 7
    l = 6
=#


struct BandColumnLeading{T <: Number}
    m :: Int # Matrix number of rows.
    n :: Int # Matrix number of columns.
    num_blocks :: Int # Number of leading blocks.
    upper_bw_max :: Int # maximum upper bandwidth.
    middle_bw_max :: Int # maximum middle bandwidth.
    lower_bw_max :: Int # maximum lower bandwidth.
    upper_bws :: Array{Int} # Upper BW in each column.
    middle_bws :: Array{Int} # Middle BW in each column.
    lower_bws :: Array{Int} # Lower BW in each column
    first_super :: Array{Int} # first superdiagonal row in A(:,k)
    leading_block_rows :: Array{Int} # leading block row counts.
    leading_block_cols :: Array{Int} # leading block column counts.
    band_elements :: Array{T,2}
end

# Construct an empty (all zero) structure from the matrix size, bounds
# on the upper and lower bandwidth, and blocksizes.
function BandColumnLeading(::Type{T}, m :: Int, n :: Int, upper_bw_max :: Int,
                           lower_bw_max :: Int, leading_block_rows ::
                           Array{Int}, leading_block_cols ::
                           Array{Int}) where {T}
    num_blocks = length(leading_block_rows)
    num_blocks == length(leading_block_cols) || 
        error("BandColumnLeading: length of leading_block_rows " *
              "and leading_block__cols should be equal")
    upper_bws = zeros(Int,n)
    middle_bws = zeros(Int,n)
    lower_bws = zeros(Int,n)
    first_super = zeros(Int,n)
    middle_bws[1] = leading_block_rows[1]
    first_super[1] = 0
    block=1
    for k=2:n
        cols_in_block = leading_block_cols[block]
        k <= cols_in_block ||
            while (leading_block_cols[block] == cols_in_block)
                block += 1
            end
        middle_bws[k] = leading_block_rows[block] - leading_block_rows[block-1]
        first_super[k] = leading_block_rows[block-1]
    end
    middle_bw_max = maximum(middle_bws)
    band_elements = zeros(T, upper_bw_max + middle_bw_max + lower_bw_max, n)
    BandColumnLeading(m, n, num_blocks, upper_bw_max, middle_bw_max,
                      lower_bw_max, upper_bws, middle_bws, lower_bws,
                      first_super, leading_block_rows,
                      leading_block_cols, band_elements)
 end

# Overload the size function and array index syntax.

@inline size(bcl :: BandColumnLeading{T}) where {T} = (bcl.m, bcl.n)

@inline function getindex( bcl :: BandColumnLeading{T}, j :: Int,
                           k :: Int) where {T <: Number}

    offset = bcl.first_super[k]-bcl.upper_bw_max
    bcl.band_elements[j - offset, k]
end

@inline function setindex!( bcl :: BandColumnLeading{T}, x :: T,
                            j :: Int, k :: Int) where {T <: Number}

    offset = bcl.first_super[k]-bcl.upper_bw_max
    bcl.band_elements[j - offset, k] = x
end

# Find bounds for lower left block l in A.
@inline function get_lower_block(bcl :: BandColumnLeading{T},
                                 l :: Integer) where {T <: Number}
    (m,_)=size(bcl)
    j1 = bcl.leading_block_rows[l]+1
    k2 = bcl.leading_block_cols[l]
    ((j1,m),(1,k2))
end 

# Find bounds for upper right block l in A.
@inline function get_upper_block(bcl :: BandColumnLeading{T},
                                 l :: Integer) where {T <: Number}
    (_,n)=size(bcl)
    j2 = bcl.leading_block_rows[l]
    k1 = bcl.leading_block_cols[l] + 1
    ((1,j2),(k1,n))
end 

# Set lower bandwidth appropriate for a given lower rank sequence.
function lower_ranks_to_bw(bcl :: BandColumnLeading{T},
                           rs :: Array{I}) where {T <: Number, I <: Integer} 

    (m,n) = size(bcl)
    bcl.lower_bws .= 0
    for l=1:bcl.num_blocks-1
        ((j0,_), (_,k0)) = get_lower_block(bcl,l)
        ((j1,_), (_,k1)) = get_lower_block(bcl,l+1)
        d = j1-j0
        r = min(rs[l], j1-j0, k0)
        bcl.lower_bws[k0-r+1:k0] .+= d
    end
end

# Set upper bandwidth appropriate for a given lower rank sequence.
function upper_ranks_to_bw(bcl :: BandColumnLeading{T},
                           rs :: Array{I}) where {T <: Number, I <: Integer} 

    (m,n) = size(bcl)
    bcl.upper_bws .= 0
    for l=1:bcl.num_blocks-1
        ((_,j0), (k0,_)) = get_upper_block(bcl,l)
        ((_,j1), (k1,_)) = get_upper_block(bcl,l+1)
        r = min(rs[l], j0, k1-k0)
        bcl.upper_bws[k0:k1-1] .= r
    end
end


# Functions for printing out the matrix structure associated with
# a BandColumnLeading struct.
function print_wilk(a :: Array{Char,2})
    (m,n) = size(a)
    for j=1:m
        println()
        for k=1:n
            print(a[j,k], " ")
        end
    end
end

function print_wilk(bcl :: BandColumnLeading{T}) where {T}
    print_wilk(get_wilk(bcl))
end

@views function get_wilk(bcl :: BandColumnLeading{T}) where {T}
    (m,n) = size(bcl)
    num_blocks = bcl.num_blocks - 1 # leave off the full matrix.
    a = fill('O', (2*m,2*n))
    # insert spaces
    fill!(a[2:2:2*m, :], ' ')
    fill!(a[:, 2:2:2*n], ' ')
    # insert boundaries for leading blocks.
    for j = 1:num_blocks
        row = bcl.leading_block_rows[j]
        col = bcl.leading_block_cols[j]
        fill!(a[1:2*row-1, 2*col], '|')
        a[2*row,2*col] = '⌋'
        fill!(a[2*row, 1:2*col-1], '_')
    end
    for k = 1:n
        j=bcl.first_super[k]+1
        fill!(a[2*(j-bcl.upper_bws[k])-1:2:2*(j-1)-1,2*k-1], 'U')
        fill!(a[2*(j+bcl.middle_bws[k])-1:2:2*
                (j+bcl.middle_bws[k]+bcl.lower_bws[k]-1)-1, 2*k-1], 'L')
        fill!(a[2*j-1:2:2*(j+bcl.middle_bws[k]-1)-1, 2*k-1], 'X')
    end
    a :: Array{Char,2}
end

end
