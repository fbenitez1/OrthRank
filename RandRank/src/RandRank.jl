module RandRank
using LinearAlgebra

export rand_svd, rand_rank, rand_QB

# Here for debugging; print all of a matrix.
showall(x) = show(IOContext(stdout, :limit=>false), MIME"text/plain"(), x)

# Compute a fast randomized SVD.
function rand_svd( A :: AbstractArray{T,2},
                   rmax :: Integer;
                   tol=1e-15,
                   rel=false ) where { R <: AbstractFloat,
                                       T <: Union{R,Complex{R}} }
    begin
        tol=convert(R,tol)
        (m,n)=size(A)
        Y=randn(T,n,2*rmax)
        Y=A*Y
        Q_Y = Matrix(qr(Y).Q)
        (U,S,V)=svd(Q_Y'*A)
        U_B = Q_Y * U
        r = rel ? rank(Diagonal(S),tol/S[1]) : rank(Diagonal(S),tol)
        if r > rmax
            error("r > rmax in rand_svd.")
        end
        (U_B[:,1:r],S[1:r], V[:,1:r])
    end :: Tuple{AbstractArray{T,2}, AbstractArray{T,1}, AbstractArray{T,2}}
end

# Invert a permutation.
function inv_perm(p :: AbstractArray{I,1}) where {I <: Integer}
    n=length(p)
    i = zeros(I,n)
    for j in 1:n
        i[p[j]] = j
    end
    i
end

# A randomized low rank factorization.
function rand_QB( A :: AbstractArray{T,2},
                  rmax :: Integer;
                  tol=1e-15,
                  rel=false ) where { R <: AbstractFloat,
                                      T <: Union{R, Complex{R}} }
    tol=convert(R,tol)
    (m,n)=size(A)
    Y=randn(T,n,2*rmax)
    Y=A*Y
    Q_Y = Matrix(qr(Y).Q)
    B=Q_Y'*A
    LQ_B = lq!(B)
    L_B = LQ_B.L
    QR_L_B = qr(L_B, Val(true))
    D = Diagonal(diag(QR_L_B.R))
    r = rel ? rank(D,tol/abs(D[1,1])) : rank(D,tol)
    if r > rmax
        error("r > rmax in rand_QB")
    end
    (_,l)=size(QR_L_B.R)
    i = inv_perm(QR_L_B.p)
    # Q :: Array{T,2} = Q_Y*(Matrix(QR_L_B.Q)[:,1:r])
    # R :: Array{T,2} = QR_L_B.R[1:r,i] * LQ_B.Q
    Qout = Q_Y*(Matrix(QR_L_B.Q)[:,1:r])
    Rout = QR_L_B.R[1:r,i] * LQ_B.Q
    (Qout,Rout)
end

function rand_rank(A :: AbstractArray{T,2}, rmax :: Integer;
                   tol=1e-15, rel=true) where { R <: AbstractFloat,
                                                T <: Union{R, Complex{R}} }
    tol=convert(R,tol)
    (m,n)=size(A)
    Y=randn(T,n,2*rmax)
    Y=A*Y
    Q_Y = Matrix(qr(Y).Q)
    s=svdvals(Q_Y'*A)
    rel ? rank(Diagonal(s),tol/S[1]) : rank(Diagonal(s), tol)
end

end # module
