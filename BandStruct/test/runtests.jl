using BandStruct.BandColumnMatrices
using BandStruct.LeadingBandColumnMatrices


leading_rows = [1,3,4,6,7,9,9,11,12]
leading_cols = [1,2,5,5,6,7,8,9,10]

lbc = LeadingBandColumn(Float64, 12,10,2,2,leading_rows, leading_cols)
lower_ranks_to_bw(lbc, [1,1,1,1,1,1,1,1,1])
upper_ranks_to_bw(lbc, [1,1,1,1,1,1,1,1,1])

print(lbc.bws)

print_wilk(lbc)

a=randn(3,5)

bc=BandColumn(lbc.m, lbc.n, lbc.m_els, 0, lbc.upper_bw_max,
              lbc.middle_bw_max, lbc.lower_bw_max,
              lbc.bws, lbc.band_elements)
