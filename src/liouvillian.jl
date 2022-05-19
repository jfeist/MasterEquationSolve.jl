using SparseArrays
using LinearAlgebra
using CUDA, CUDA.CUSPARSE, CUDA.CUBLAS
CUDA.allowscalar(false)

# vec for CuArrays returns a ReshapedArray in some versions, which do not work for mul! with CUDA.CUSPARSE matrices
# so define our own version
myvec(A::CuArray) = unsafe_wrap(CuArray,pointer(A),length(A))
myvec(A) = vec(A)

# L = -i (Heff ρ - ρ Heff^†) + sum_i J_i ρ J_i^†
# L = -i Heff ρ + i ρ Heff^† + sum_i J_i ρ J_i^†
# L = (-i Heff ρ) + (-i Heff ρ^†)^† + sum_i J_i ρ J_i^†
function L!(Lρ,ρ,ps,t)
    miHeff, J, tempρ, tempρ2, hermρ = ps
    LHeff!(Lρ,ρ,miHeff,tempρ,hermρ)
    Lrefill!(Lρ,ρ,J,tempρ,tempρ2)
    Lρ
end

# assume that ρ is Hermitian, so (-i Heff ρ^†)^† = (-i Heff ρ)^†
# miHeff = -i Heff
function LHeff!(Lρ,ρ,miHeff,tempρ,hermρ::Val{true})
    mul!(tempρ,miHeff,ρ)
    adjoint!(Lρ,tempρ)
    Lρ .+= tempρ
end
# Take into account that ρ might not be Hermitian
# (because of numerical error or because we propagate a non-physical ρ)
function LHeff!(Lρ,ρ,miHeff,tempρ,hermρ::Val{false})
    mul!(Lρ,miHeff,ρ)
    mulacc!(Lρ,ρ,miHeff',tempρ)
end

function Lrefill!(Lρ,ρ,J::AbstractVector,tempρ,tempρ2)
    for j in J
        mul!(tempρ,j,ρ)
        mulacc!(Lρ,tempρ,j',tempρ2)
    end
end
# if J is a matrix, we assume it is a superoperator
function Lrefill!(Lρ,ρ,J::AbstractMatrix,temp,temp2)
    mul!(myvec(Lρ),J,myvec(ρ),1,1)
end

# calculate C += A*B
function mulacc!(C,A,B,tmp)
    mul!(C,A,B,1,1)
end
# if B is CuSparseMatrixCSR or its adjoint, calculate tmp = B'*A', and then C += tmp'
function mulacc!(C,A,B::U,tmp) where U <: Union{CuSparseMatrixCSR{T},Adjoint{T,<:CuSparseMatrixCSR{T}}} where T
    mul!(tmp,B',A')
    CUBLAS.geam!('N','C',1,C,1,tmp,C)
end
