module BandColumnMatrices

using MLStyle
using ConstructionBase
using Printf

import Base: size, getindex, setindex!, showerror, show, print, @propagate_inbounds

export BandColumn, AbstractBandColumn, MatchError,
    first_storage_el, last_storage_el, NoStorageForIndex,
    get_offset, bc_index_stored, check_bc_storage_bounds,
    get_m_els, get_offset_all, get_upper_bw_max, get_upper_bw, get_middle_bw,
    get_lower_bw, get_first_super, get_band_element, set_band_element!,
    extend_lower_band, extend_upper_band


"MatchError: An exception for pattern match errors in MLStyle."
struct MatchError <: Exception end

"""

NoStorageForIndex: An exception to indicate that there is no
storage available to represent a particular element of a BandColumn.

"""
struct NoStorageForIndex <: Exception
    arr
    ix
end

showerror(io :: IO, e :: NoStorageForIndex) =
    print(io, "Attempt to access ", typeof(e.arr), " at index ", e.ix,
          " for which there is no storage.")

abstract type AbstractBandColumn{E,AE,AI} <: AbstractArray{E,2} end

"""

A simplified band structure that does not include leading blocks but
does include a uniform offset that can be changed to give different
submatrices.  This can be used to represent submatrices of leading and
trailing band structures.

The method show for this structure represents elements for which there
is no storage with `N`.  Elements that have available storage but are
not actually stored are represented by `O`.  These are elements that
are outside the current bandwidth but not outside the maximum
bandwidth.

"""
struct BandColumn{E <: Number,
                         AE <: AbstractArray{E,2},
                         AI <: AbstractArray{Int,2}} <:
                             AbstractBandColumn{E,AE,AI}
    m :: Int # Matrix number of rows.
    n :: Int # Matrix and elements number of columns.
    m_els :: Int # Elements number of rows.
    offset_all :: Int # uniform offset.
    upper_bw_max :: Int # maximum upper bandwidth.
    middle_bw_max :: Int # maximum middle bandwidth.
    lower_bw_max :: Int # maximum lower bandwidth.
    bws :: AI # Bandwidths and first superdiagonal in each column.
    band_elements :: AE
end

@as_record BandColumn
@as_record UnitRange

##
## Functions that should be implemented as part of the
## AbstractBandColumn interface.
##

@inline get_m_els(bc :: BandColumn) = bc.m_els
@inline get_offset_all(bc :: BandColumn) = bc.offset_all
@inline get_upper_bw_max(bc :: BandColumn) = bc.upper_bw_max
@inline get_upper_bw(bc :: BandColumn, k :: Int) = bc.bws[1,k]
@inline get_middle_bw(bc :: BandColumn, k :: Int) = bc.bws[2,k]
@inline get_lower_bw(bc :: BandColumn, k :: Int) = bc.bws[3,k]
@inline get_first_super(bc :: BandColumn, k :: Int) = bc.bws[4,k]

@inline function extend_upper_band(bc :: BandColumn, j :: Int, k :: Int)
    bc.bws[1,k]=max(bc.bws[1,k], bc.bws[4,k]-j+1)
end

@inline function extend_lower_band(bc :: BandColumn, j :: Int, k :: Int)
    bc.bws[3,k]=max(bc.bws[3,k], j - bc.bws[4,k] - bc.bws[2,k])
end

@propagate_inbounds @inline get_band_element(bc :: BandColumn,
                                             j :: Int, k :: Int) =
    bc.band_elements[j,k]

@propagate_inbounds @inline function set_band_element!(bc :: BandColumn{E,AE,AI}, x :: E,
                      j :: Int, k :: Int) where {E,AE,AI}
    bc.band_elements[j,k] = x
end

@inline size(bc :: BandColumn) = (bc.m, bc.n)

##
## Generic functions defined for AbstractBandColumn
##

@inline get_offset(bc :: AbstractBandColumn, k :: Int) =
    get_offset_all(bc) + get_first_super(bc,k) - get_upper_bw_max(bc)

@inline function check_bc_storage_bounds( ::Type{Bool},
                                           bc :: AbstractBandColumn,
                                           j :: Int,
                                           k :: Int )
    j1 = j - get_offset(bc, k)
    j1 >= 1 && j1 <= get_m_els(bc)
end

@inline check_bc_storage_bounds( bc :: AbstractBandColumn, j :: Int,
                                  k :: Int ) =
    check_bc_storage_bounds(Bool, bc, j, k) || throw(NoStorageForIndex(bc, (j,k)))

@inline first_storage_el(bc :: AbstractBandColumn, k) =
    get_upper_bw_max(bc) - get_upper_bw(bc,k) + 1

@inline last_storage_el(bc :: AbstractBandColumn, k) =
    get_upper_bw_max(bc) + get_middle_bw(bc,k) + get_lower_bw(bc,k)

@inline function bc_index_stored( bc :: AbstractBandColumn,
                                   j :: Int,
                                   k :: Int )
    j1 = j - get_offset(bc, k)
    j1 >= first_storage_el(bc,k) && j1 <= last_storage_el(bc,k)
end

##
## Print and show.
##

function show(io::IO,
              bc :: BandColumn)
    print(io, typeof(bc),"(", bc.m, ", ", bc.n, ", ", bc.m_els, ", ",
          bc.offset_all, ", ", bc.upper_bw_max, ", ", bc.middle_bw_max,
          ", ", bc.lower_bw_max, ", ", bc.bws, "): ")
    for j=1:bc.m
        println()
        for k=1:bc.n
            if check_bc_storage_bounds(Bool, bc, j, k)
                if bc_index_stored(bc,j,k)
                    @printf("%10.2e", bc[j,k])
                else
                    print("         O")
                end
            else
                print("         N")
            end
        end
    end
end

show(io :: IO, ::MIME"text/plain",
     bc :: BandColumn) = show(io, bc)

print(io::IO,
      bc :: BandColumn) =
          print(io, typeof(bc),"(", bc.m, ", ", bc.n, ", ", bc.m_els, ", ",
                bc.offset_all, ", ", bc.upper_bw_max, ", ", bc.middle_bw_max,
                ", ", bc.lower_bw_max, ", ", bc.bws, ", ", bc.band_elements,")")

print(io :: IO, ::MIME"text/plain",
      bc :: BandColumn) = print(io, bc)

##
## Index operations
##

# (BandColumn{Float64, Array{Float64,2}, Array{Int}}, Int, Int))
@inline function getindex( bc :: AbstractBandColumn{E,AE,AI}, j :: Int,
                           k :: Int) where {E,AE,AI}
    @boundscheck begin
        checkbounds(bc, j, k)
        check_bc_storage_bounds(bc, j, k)
        bc_index_stored(bc,j,k) || return zero(E)
    end
    j1 = j - get_offset(bc,k)
    @inbounds get_band_element(bc,j1, k)
end

@inline function setindex!( bc :: AbstractBandColumn{E,AE,AI}, x :: E,
                            j :: Int, k :: Int) where {E,AE,AI}
    @boundscheck begin
        checkbounds(bc, j, k)
        check_bc_storage_bounds(bc, j, k)
    end
    extend_upper_band(bc,j,k)
    extend_lower_band(bc,j,k)
    j1 = j - get_offset(bc,k)
    @inbounds set_band_element!(bc,x,j1, k)
end

@inline function view(bc :: BandColumn, rows :: UnitRange{Int},
                      cols :: UnitRange{Int})
    @when let BandColumn(;m, n, m_els, offset_all=oall, upper_bw_max=ubw_max,
                         middle_bw_max=mbw_max, lower_bw_max=lbw_max, 
                         bws, band_elements=els) = bc,
              UnitRange(j0,j1) = rows,
              UnitRange(k0,k1) = cols
        @boundscheck begin
            checkbounds(bc,j0,k0)
            checkbounds(bc,j1,k1)
        end
        BandColumn(max(j1-j0+1,0), max(k1-k0+1,0), m_els, oall-j0+1,
                          ubw_max, mbw_max, lbw_max, view(bws,:,cols),
                          view(els,:,cols))
        @otherwise
        throw(MatchError)
    end
end    
                        
@inline function getindex(bc :: BandColumn, rows :: UnitRange{Int},
                          cols :: UnitRange{Int})

    @when let BandColumn(;m, n, m_els, offset_all=oall, upper_bw_max=ubw_max,
                                middle_bw_max=mbw_max, lower_bw_max=lbw_max, 
                                bws, band_elements=els) = bc,
              UnitRange(j0,j1) = rows,
              UnitRange(k0,k1) = cols
        @boundscheck begin
            checkbounds(bc,j0,k0)
            checkbounds(bc,j1,k1)
        end
        BandColumn(max(j1-j0+1,0), max(k1-k0+1,0), m_els, oall-j0+1,
                          ubw_max, mbw_max, lbw_max, bws[:,cols], els[:,cols])
        @otherwise
        throw(MatchError)
    end
end

@inline getindex(bc :: AbstractBandColumn, ::Colon, cols :: UnitRange{Int}) =
    getindex(bc,1:bc.m, cols)

@inline getindex(bc :: AbstractBandColumn, rows :: UnitRange{Int}, ::Colon) =
    getindex(bc,rows,1:bc.n)

@inline getindex(bc :: AbstractBandColumn, ::Colon, ::Colon) = bc

@inline view(bc :: AbstractBandColumn, ::Colon, cols :: UnitRange{Int}) =
    view(bc,1:bc.m, cols)

@inline view(bc :: AbstractBandColumn, rows :: UnitRange{Int}, ::Colon) =
    view(bc,rows,1:bc.n)

@inline view(bc :: AbstractBandColumn, ::Colon, ::Colon) = bc

end # module
