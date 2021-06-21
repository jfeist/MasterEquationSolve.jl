using LinearAlgebra
using SparseArrays
using Printf
using CUDA, CUDA.CUSPARSE, CUDA.CUBLAS
CUDA.allowscalar(false)

using DifferentialEquations

# mul! for CUDA.CUSPARSE arrays does not work with the ReshapedArray returned by vec, define our own
cuvec(A::CuArray) = unsafe_wrap(CuArray,pointer(A),length(A))
cuvec(A) = vec(A)

"calculate -i Heff = -i * (H - i/2*∑_i c_i^† c_i)"
get_miHeff(H,J) = -1im*(H - 0.5im*sum(j'*j for j in J))

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
    mul!(cuvec(Lρ),J,cuvec(ρ),1,1)
end

# calculate C += A*B
function mulacc!(C,A,B,tmp)
    mul!(C,A,B,1,1)
end
# if B is CuSparseMatrixCSR or its adjoint, calculate tmp = B'*A', and then C += tmp'
function mulacc!(C,A,B::U,tmp) where U <: Union{CuSparseMatrixCSR{T},Adjoint{T,CuSparseMatrixCSR{T}}} where T
    mul!(tmp,B',A')
    CUBLAS.geam!('N','C',1,C,1,tmp,C)
end

mutable struct qutip_saver{T}
    saver::T
    outdir::String
    istep::Int
    qutip_saver(saver,outdir) = new{typeof(saver)}(saver,outdir,0)
end

function (f::qutip_saver)(u,t,integrator)
    data = f.saver(u,t,integrator)
    outfi = @sprintf("%s/%08d",f.outdir,f.istep+=1)
    qt.qsave((t,data),outfi)
    data
end

function get_savingcallback(saveat, saveddata=:probabilities, write_qutip=true, outdir="")
    if saveddata == :probabilities
        savefunc = (u,t,integrator) -> collect(real(diag(u)))
        sv = SavedValues(Float64,Vector{Float64})
    elseif saveddata == :full
        savefunc = (u,t,integrator) -> u
        sv = SavedValues(Float64,Matrix{ComplexF64})
    else
        raise(ArgumentError("No function implemented for saveddata: $saveddata."))
    end

    if write_qutip
        mkdir(outdir)
        saver = qutip_saver(savefunc,outdir)
    else
        saver = savefunc
    end

    SavingCallback(saver, sv, saveat=saveat)
end


function setup(H,J,ρ0,ts; backend, assume_herm=false)
    miHeff = get_miHeff(H,J)
    Lrefill = sum(kron.(conj.(J),J))
    if backend == :CUDA
        miHeffc = CuSparseMatrixCSR(miHeff)
        Lrefillc = CuSparseMatrixCSR(Lrefill)
        ρc = CuArray(ρ0)
        tmpρc = copy(ρc)
        tmpρc2 = copy(ρc)
        ps = (miHeffc,Lrefillc,tmpρc,tmpρc2,Val(assume_herm))
        return ρc, ts, ps
    elseif backend == :CPU
        tmpρ = copy(ρ0)
        ps = (miHeff,Lrefill,tmpρ,tmpρ,Val(assume_herm))
        return ρ0, ts, ps
    else
        raise(ArgumentError("Unknown backend $backend."))
    end
end

function mesolve_qutip(infile,outdir;backend=:CUDA,saveddata=:probabilities)
    input = load_input_from_qutip(infile)
    (ρ0,ts,ps) = setup(input...; backend=backend)
    tspan = (ts[1],ts[end])
    cb = get_savingcallback(ts, saveddata, true, outdir)
    sv = cb.affect!.saved_values

    println("writing output for propagation defined in '$infile' to directory '$outdir/'")

    prob = ODEProblem(L!,ρ0,tspan,ps)
    sol = solve(prob,Tsit5(),save_start=false,save_end=false,save_everystep=false,
                callback=cb,reltol=1e-9,abstol=1e-9,maxiters=1e18)
    sol, sv
end
