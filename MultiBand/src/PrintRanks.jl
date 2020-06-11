module PrintRanks

using LinearAlgebra
using Printf

export printRanks, printSV

function printSV(A,l,k)
    n = size(A,1);
    lA = convert(Int64, round(log2(n)-1))
    if l > lA || l < 1 
        error("Invalid level in printSV")
    end
    b = 2^l
    b2 = b ÷ 2
    if k > n ÷ b - 1 || k < 1
        error("Invalid region in printSV")
    end
    j1 = 2^l+1 + (k - 1)*b
    k1 = 1 + (k - 1)*b
    B=A[ j1:j1+b2-1, k1:k1+b2-1 ]
    s = svdvals(B)
    print(s[1:min(length(s),15)], "\n")
end
    

function printRanks(A,tol)
    n = size(A,1);
    l = convert(Int64, round(log2(n)-1))
    maxrank = 0
    for j = 2:l
        @printf("Level: %d\n\n", j);
        b = 2^j;
        b2  =  b ÷ 2
        j1 = 2^j+1;
        k1 = 1;
        for k = 1:(n ÷ b-1)
            r1 = rank(A[ j1:j1+b2-1, k1:k1+b2-1 ],rtol = tol);
            r2 = rank(A[ j1+b2:j1+b-1, k1:k1+b2-1 ],rtol = tol);
            r3 = rank(A[ j1+b2:j1+b-1, k1+b2:k1+b-1 ], rtol = tol);
            @printf("k: %d, ranks: %d, %d, %d\n", k, r1, r2, r3);
            j1 = j1+b;
            k1 = k1+b;
        end
    end
end

end
