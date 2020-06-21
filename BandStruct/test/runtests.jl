using BandStruct.Column

leading_rows = [1,3,4,6,7,9,9,11,12]
leading_cols = [1,2,5,5,6,7,8,9,10]

bcl = BandColumnLeading(Float64, 12,10,2,2,leading_rows, leading_cols)
lower_ranks_to_bw(bcl, [1,1,1,1,1,1,1,1,1])
upper_ranks_to_bw(bcl, [1,1,1,1,1,1,1,1,1])

print(bcl.middle_bws)
print(bcl.first_super)

print_wilk(bcl)
