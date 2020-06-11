module MakeMultiBand
using LinearAlgebra

export makeMultiBand, applyLevelTrans

makeMultiBand(n, blsize) = makeMultiBand(n,blsize, Float64)

function makeMultiBand(n :: Integer,
                       blsize :: Integer,
                       ::Type{T}) where {X <: AbstractFloat,
                                         T <: Union{X, Complex{X}}}
    begin
        A = diagm(1 => randn(T,n-1), -1 => randn(T,n-1))
        l = convert(Int64, round(log2(n)-1))
        for j = 1 : l
            b = 2^j
            j1 = 2^j+1
            k1 = 1
            b2 = b ÷ 2
            if (blsize <= b2)
                for k = 1 : (n ÷ b - 1)
                    A[j1 : j1+blsize-1, k1+b2-blsize : k1+b2-1] = 
                        randn(T,(blsize, blsize));
                    A[j1+b2 : j1+b2+blsize-1, k1+b-blsize : k1+b-1] =
                        randn(T,(blsize,blsize))
                    A[j1+b2 : j1+b2+blsize-1, k1+b2-blsize : k1+b2-1] =
                        randn(T,(blsize,blsize));
                    j1 = j1+b;
                    k1 = k1+b;
                end
            end
        end
        A = A+A';
        A = A/norm(A,Inf)
        A = A + 0.06*I
    end :: AbstractArray{T, 2}
end

function applyLevelTrans(A :: AbstractArray{T,2},
                         bs :: Integer) where {X <: AbstractFloat,
                                               T <: Union{X, Complex{X}}}
    begin
        n = size(A,1);
        l = convert(Int64, round(log2(n)-1))
        for j = l : -1 : 2
            b = 2^j;
            b2 = b ÷ 2;
            b4 = b ÷ 4;
            j1 =2^j+1;
            k1 = 1;
            if (bs <= b4)
                for k = 1 : (n ÷ b - 1)
                    q = Matrix(qr(randn(2*bs,2*bs)).Q)
                    ks = [ Vector(k1+b4-bs : k1+b4-1);
                           Vector(k1+b2-bs : k1+b2-1) ]
                    A[ j1 : n, ks ] = A[ j1 : n, ks ] * q

                    q = Matrix(qr(randn(2*bs,2*bs)).Q)
                    js = [ Vector(j1+b2 : j1+b2+bs-1);
                           Vector(j1+b2+b4 : j1+b2+b4+bs-1) ]
                    A[ js, 1 : k1+b-1 ] = q*A[ js, 1 : k1+b-1 ]

                    q = Matrix(qr(randn(2*bs,2*bs)).Q)
                    js = [ Vector(j1 : j1+bs-1);
                           Vector(j1+b4 : j1+b4+bs-1) ]
                    A[ js, 1 : k1+b2-1 ] = q * A[ js, 1 : k1+b2-1 ]

                    q = Matrix(qr(randn(2*bs,2*bs)).Q)
                    ks = [ Vector(k1+b2+b4-bs : k1+b2+b4-1);
                           Vector(k1+b-bs : k1+b-1) ]
                    A[ j1+b2 : n, ks ] = A[ j1+b2 : n,ks ]*q

                    j1 = j1+b
                    k1 = k1+b
                end
            end
        end
        A = A';
        for j = l : -1 : 2
            b = 2^j;
            b2 = b ÷ 2;
            b4 = b ÷ 4;
            j1 = 2^j+1;
            k1 = 1;
            if (bs <= b4)
                for k = 1 : (n/b - 1)
                    q = Matrix(qr(randn(2*bs,2*bs)).Q)
                    ks = [ Vector(k1+b4-bs : k1+b4-1);
                           Vector(k1+b2-bs : k1+b2-1) ]
                    A[ j1 : n, ks ] = A[ j1 : n, ks ] * q

                    q = Matrix(qr(randn(2*bs,2*bs)).Q)
                    js = [ Vector(j1+b2 : j1+b2+bs-1);
                           Vector(j1+b2+b4 : j1+b2+b4+bs-1) ]
                    A[ js, 1 : k1+b-1] = q*A[ js, 1 : k1+b-1 ]

                    q = Matrix(qr(randn(2*bs,2*bs)).Q)
                    js = [ Vector(j1 : j1+bs-1);
                           Vector(j1+b4 : j1+b4+bs-1) ]
                    A[ js, 1 : k1+b2-1 ] = q * A[ js, 1 : k1+b2-1 ]

                    q = Matrix(qr(randn(2*bs,2*bs)).Q)
                    ks = [ Vector(k1+b2+b4-bs : k1+b2+b4-1);
                           Vector(k1+b-bs : k1+b-1) ]
                    A[ j1+b2 : n,ks ] = A[ j1+b2 : n,ks ]*q

                    j1 = j1+b
                    k1 = k1+b
                end
            end
        end
        A=A';
    end :: AbstractArray{T,2}
end

end
