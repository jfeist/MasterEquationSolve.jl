using MasterEquationSolve
using MasterEquationSolve: qt, qobj_to_jl, mesolve, mesolve_qutip, myqload
using Printf
using Test
import CUDA

# make the destroy operators in a tensor product space of dimensions dims
function destroy_ops(dims)
    ops = qt.qeye.(dims)
    return ntuple(length(dims)) do i
        ops[i] = qt.destroy(dims[i])
        A = qt.tensor(ops)
        ops[i] = qt.qeye(dims[i])
        A
    end
end

@testset "MasterEquationSolve.jl" begin
    ωe = 1.5
    ωc = 1.45
    g = 0.5
    γc = 0.4
    γdephasing = 0.5
    ts = LinRange(0,20,400)

    Nmol = 4
    Nc = Nmol+1

    (acav, σs...) = destroy_ops([Nc;fill(2,Nmol)])

    H0 = ωc * acav.dag() * acav + sum(ωe*σ.dag()*σ + g*(σ.dag()*acav + σ*acav.dag()) for σ in σs)

    # This is the dephasing operator (2 σp σm - 1)
    makedeph(σ) = sqrt(γdephasing)*(2*σ.dag()*σ-1)
    c_ops = [sqrt(γc)*acav, makedeph.(σs)...];
    e_ops = [σs[1].dag()*σs[1], acav.dag()*acav]
    ψ0 = qt.tensor([qt.basis(Nc); qt.basis.(fill(2,Nmol),1)])

    infile = "mesolve_runtests_in"
    outdir = "mesolve_runtests_out"
    qt.qsave(Dict("H"=>H0, "c_ops"=>c_ops, "ρ0"=>ψ0, "ts"=>ts, "e_ops"=>e_ops, "abstol"=>1e-12, "reltol"=>1e-10),infile)

    sol_qt = qt.mesolve(H0,ψ0,ts,c_ops,options=qt.Options(atol=1e-12,rtol=1e-10))
    sol_qt_e = qt.mesolve(H0,ψ0,ts,c_ops,e_ops=e_ops,options=qt.Options(atol=1e-12,rtol=1e-10))

    H = qobj_to_jl(H0)
    J = qobj_to_jl.(c_ops)
    ρ0 = qobj_to_jl(ψ0,true);
    e_ops_j = qobj_to_jl.(e_ops)

    backends = CUDA.functional() ? (:CPU, :CUDA) : (:CPU,)
    @testset "backend $backend" for backend in backends
        sol, sv = mesolve(H,J,ρ0,ts;backend=backend,saveddata=:full)
        @test all([s ≈ q.full() for (s,q) in zip(sv.saveval,sol_qt.states)])

        sol, sv = mesolve(H,J,ρ0,ts;backend=backend,saveddata=e_ops_j)
        @test reduce(hcat,sol_qt_e.expect) ≈ transpose(reduce(hcat,sv.saveval))

        rm(outdir,force=true,recursive=true)
        sol, sv = mesolve_qutip(infile,outdir; backend=backend,saveddata=:full)
        @test all([s ≈ q.full() for (s,q) in zip(sv.saveval,sol_qt.states)])
        sv2 = [myqload(@sprintf("%s/%08d",outdir,i)) for i in 1:length(ts)]
        @test ts == [s[1] for s in sv2]
        @test all([s[2] ≈ q.full() for (s,q) in zip(sv2,sol_qt.states)])

        rm(outdir,force=true,recursive=true)
        # not passing saveddata corresponds to taking it from the saved qutip file
        sol, sv = mesolve_qutip(infile,outdir; backend=backend)
        @test reduce(hcat,sol_qt_e.expect) ≈ transpose(reduce(hcat,sv.saveval))
        sv2 = [myqload(@sprintf("%s/%08d",outdir,i)) for i in 1:length(ts)]
        @test ts == [s[1] for s in sv2]
        @test reduce(hcat,sol_qt_e.expect) ≈ transpose(reduce(hcat,[s[2] for s in sv2]))
    end

    rm(outdir,force=true,recursive=true)
    rm(infile*".qu",force=true)
end
