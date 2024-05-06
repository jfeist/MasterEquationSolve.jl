using Printf
# use only the DifferentialEquations sublibraries we actually need
using OrdinaryDiffEq, DiffEqCallbacks

export mesolve, mesolve_qutip, sesolve, sesolve_qutip

"calculate -i Heff = -i * (H - i/2*∑_i c_i^† c_i)"
get_miHeff(H,J) = begin
    Heff = H
    if !isempty(J)
        Heff -= 0.5im*sum(j'*j for j in J)
    end
    -1im*Heff
end
get_miHeff(H,J,J_offdiag::Nothing) = get_miHeff(H,J)
get_miHeff(H,J,J_offdiag) = begin
    (γ, Fs) = J_offdiag
    # 0.5 is -i * -i/2
    get_miHeff(H,J) - 0.5*sum(γ[i,j] * Fs[j]' * Fs[i] for i in eachindex(Fs), j in eachindex(Fs))
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

get_saver(savefunc,::Nothing) = savefunc
get_saver(savefunc,outdir) = (outdir != "." && mkdir(outdir); qutip_saver(savefunc,outdir))

function get_savingcallback(saveat, saveddata=:probabilities, outdir=nothing)
    if saveddata == :probabilities
        savefunc = (u,t,integrator) -> collect(real(diag(u)))
        sv = SavedValues(Float64,Vector{Float64})
    elseif saveddata == :full
        savefunc = (u,t,integrator) -> collect(u) # collect to get a copy
        sv = SavedValues(Float64,Matrix{ComplexF64})
    elseif saveddata isa AbstractVector # e_ops
        savefunc = (u,t,integrator) -> ComplexF64[tr(e*u) for e in saveddata]
        sv = SavedValues(Float64,Vector{ComplexF64})
    else
        throw(ArgumentError("No function implemented for saveddata: $saveddata."))
    end

    saver = get_saver(savefunc,outdir)
    SavingCallback(saver, sv, saveat=saveat)
end

function get_savingcallback_se(saveat, saveddata=:probabilities, outdir=nothing)
    if saveddata == :full
        savefunc = (u,t,integrator) -> collect(u) # collect to get a copy
        sv = SavedValues(Float64,Vector{ComplexF64})
    elseif saveddata isa AbstractVector # e_ops
        savefunc = (u,t,integrator) -> ComplexF64[dot(u,e*u) for e in saveddata]
        sv = SavedValues(Float64,Vector{ComplexF64})
    else
        throw(ArgumentError("No function implemented for saveddata: $saveddata."))
    end

    saver = get_saver(savefunc,outdir)
    SavingCallback(saver, sv, saveat=saveat)
end

function setup(H,J,ρ0,ts,J_offdiag,saveddata; backend, assume_herm=false)
    miHeff = get_miHeff(H,J,J_offdiag)
    Lrefill = isempty(J) ? kron(0*miHeff, 0*miHeff) : sum(kron.(conj.(J),J))
    if J_offdiag !== nothing
        (γ, Fs) = J_offdiag
        Lrefill += sum(kron(γ[i,j] * conj(Fs[j]), Fs[i]) for i in eachindex(Fs), j in eachindex(Fs))
    end
    if backend == :CUDA
        miHeffc = CuSparseMatrixCSR(miHeff)
        Lrefillc = CuSparseMatrixCSR(Lrefill)
        ρc = CuArray(ρ0)
        tmpρc = copy(ρc)
        tmpρc2 = copy(ρc)
        ps = (miHeffc,Lrefillc,tmpρc,tmpρc2,Val(assume_herm))
        if saveddata isa AbstractVector
            saveddata = CuSparseMatrixCSR.(saveddata)
        end
        return ρc, ts, ps, saveddata
    elseif backend == :CPU
        tmpρ = copy(ρ0)
        ps = (miHeff,Lrefill,tmpρ,tmpρ,Val(assume_herm))
        return ρ0, ts, ps, saveddata
    else
        throw(ArgumentError("Unknown backend $backend."))
    end
end

function mesolve(H, J, ρ0, ts, J_offdiag=nothing; backend=:CUDA, saveddata=:probabilities, outdir=nothing, reltol=1e-10, abstol=1e-12)
    (ρ0,ts,ps,saveddata) = setup(H,J,ρ0,ts,J_offdiag,saveddata; backend=backend)
    tspan = (ts[1],ts[end])
    cb = get_savingcallback(ts, saveddata, outdir)
    sv = cb.affect!.saved_values

    prob = ODEProblem(L!,ρ0,tspan,ps)
    sol = solve(prob,Tsit5(),save_start=false,save_end=false,save_everystep=false,
                callback=cb,reltol=reltol,abstol=abstol,maxiters=1e18)
    sol, sv
end

function mesolve_qutip(infile, outdir; backend=:CUDA, saveddata=nothing)
    input, saveddata_qu, reltol, abstol = load_input_from_qutip(infile)
    saveddata = saveddata === nothing ? saveddata_qu : saveddata
    println("writing output for propagation defined in '$infile' to directory '$outdir/'")
    # use that keyword argument names are inferred, i.e., f(; a) = f(; a=a)
    mesolve(input...; backend, saveddata, outdir, reltol, abstol)
end

miH!(miHψ,ψ,miH,t) = mul!(miHψ,miH,ψ)

function setup_se(H,ψ0,ts,saveddata; backend)
    miH = -1im*H
    if backend == :CUDA
        miHc = CuSparseMatrixCSR(miH)
        ψc = CuArray(ψ0)
        ps = miHc
        if saveddata isa AbstractVector
            saveddata = CuSparseMatrixCSR.(saveddata)
        end
        return ψc, ts, ps, saveddata
    elseif backend == :CPU
        ps = miH
        return ψ0, ts, ps, saveddata
    else
        throw(ArgumentError("Unknown backend $backend."))
    end
end

function sesolve(H,ψ0,ts;backend=:CUDA,saveddata=:full,outdir=nothing)
    (ψ0,ts,ps,saveddata) = setup_se(H,ψ0,ts,saveddata; backend=backend)
    tspan = (ts[1],ts[end])
    cb = get_savingcallback_se(ts, saveddata, outdir)
    sv = cb.affect!.saved_values

    prob = ODEProblem(miH!,ψ0,tspan,ps)
    sol = solve(prob,Tsit5(),save_start=false,save_end=false,save_everystep=false,
                callback=cb,reltol=1e-10,abstol=1e-12,maxiters=1e18)
    sol, sv
end

function sesolve_qutip(infile,outdir;backend=:CUDA,saveddata=nothing)
    input, saveddata_qu = load_se_input_from_qutip(infile)
    saveddata = saveddata === nothing ? saveddata_qu : saveddata
    println("writing output for propagation defined in '$infile' to directory '$outdir/'")
    sesolve(input...;backend=backend,saveddata=saveddata,outdir=outdir)
end
