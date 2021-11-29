function expectation_value_normalized(psi,A,temp)
    # Calculate normalized expectation value <psi|A|psi> / <psi|psi>
    mul!(temp,A,psi)
    return dot(psi,temp)/sum(abs2,psi)
end

function wf_collapse!(psi,A,temp)
    # Calculate |psi_c> = A|psi> / sqrt(<psi|A† A|psi>)
    mul!(temp,A,psi)
    psi .= (temp ./ norm(temp))
    return psi
end

apply_H!(psires,psi,H,t,β,α=-1im) = mul!(psires,H,psi,α,β)
apply_H!(psires,psi,H_f::Tuple,t,β) = apply_H!(psires,psi,H_f[1],t,β,-1im*H_f[2](t))

# prefactor is (-i * -i/2) = -1/2, with -i from (-i Heff) and -i/2 from Heff = H - i/2 c'c
apply_cdc!(psires,psi,cdc,t,α=-0.5) = mul!(psires,cdc,psi,α,1.0)
apply_cdc!(psires,psi,cdc_f::Tuple,t) = apply_cdc!(psires,psi,cdc_f[1],t,-0.5*abs2(cdc_f[2](t)))

function minus_im_Heff_mcsolve!(psires,psi,ps,t)
    Hs, cdc_ops = ps

    # Hamiltonian part
    if isempty(Hs)
        psires .= 0
    else
        apply_H!(psires,psi,Hs[1],t,0)
        apply_H!.((psires,),(psi,),Hs[2:end],t,1)
    end

    # collapse operator part
    apply_cdc!.((psires,),(psi,),cdc_ops,t)

    return psires
end

collapse_prob(psi,c,t,temp,α=1) = (mul!(temp, c, psi, α, 0); sum(abs2,temp))
collapse_prob(psi,c::Tuple,t,temp) = collapse_prob(psi, c[1], t, temp, c[2](t))

get_c(c) = c
get_c(c::Tuple) = c[1]

to_cpu(c::CuSparseMatrixCSR) = SparseArrays.SparseMatrixCSC(c.dims...,collect.((c.rowPtr,c.colVal,c.nzVal))...) |> transpose

get_cdc_op(c) = c'c
get_cdc_op(c::CuSparseMatrixCSR) = c |> to_cpu |> get_cdc_op |> CuSparseMatrixCSR
get_cdc_op(c::Tuple{T,U}) where {T,U} = (get_cdc_op(c[1]), c[2])

get_Heff(Hs_ti,cs_ti) = sum(Hs_ti) - 0.5im * sum(cs_ti)
get_Heff(Hs_ti::NTuple{N,CuSparseMatrixCSR},cs_ti::NTuple{M,CuSparseMatrixCSR}) where {N,M} = get_Heff(to_cpu.(Hs_ti),to_cpu.(cs_ti)) |> CuSparseMatrixCSR

istuple(x) = x isa Tuple
function group_ops(Hs,cdc_ops)
    # collect all time-independent operators into a single one:
    # Heff = sum_n H_{ti,n} - 0.5i sum_m c_{ti,m}^† c_{ti,m}
    Heff = get_Heff(filter(!istuple,Hs),filter(!istuple,cdc_ops))
    Hs_new = (Heff, filter(istuple,Hs)...)
    return Hs_new, filter(istuple,cdc_ops)
end

function mcsolve(Hs, c_ops, psi0, ts, e_ops; ntrajs=1, verbose=false, alg=Tsit5(), kwargs...)
    # MCWF solver for a single psi0. If you want to propagate an initial state that is
    # rho0 = \sum_i p_i psi_i * psi_i ^dag you just need to run this wavefunction and sum all results * p_i
    # H0 is the time independent Hamiltonian
    # H0_td is a list of a operators such as 
    # H(t) = H0 + sum_i H0_td[i] fs_H0_td[i](t)
    # c_ops_no_td is a list of collapse operators
    # c_ops_td is a list of collapse operators is a list of collapse operators with time-dependency
    # psi0 is the initial wavefunction
    # ts is an array of times for the propagation
    # e_ops is a list of operators to perform expectation values

    results_all = zeros(ComplexF64,length(e_ops),length(ts))
    results = copy(results_all)
    temp = copy(psi0)
    cdc_ops = get_cdc_op.(c_ops)
    ps = group_ops(Hs, cdc_ops)

    deltap = rand()

    function condition(u,t,integrator)
        #println(norm(u)^2, " ", deltap, " ",t)
        sum(abs2,u) - deltap
    end

    function affect!(integrator)
        probabilities = Vector{Float64}(undef,length(c_ops))
        cumprob = 0.0
        for i=1:length(c_ops)
            cumprob += collapse_prob(integrator.u,c_ops[i],integrator.t,temp)
            probabilities[i] = cumprob
        end
        rand_number = rand() * probabilities[end]

        i_op = searchsortedfirst(probabilities, rand_number)
        c = get_c(c_ops[i_op])
        wf_collapse!(integrator.u, c, temp)

        deltap = rand()
    end
    
    function get_callback(ts,results,e_ops,temp)
        it = 0
        savefunc(u,t,integrator) = (it+=1; results[:,it] .= expectation_value_normalized.((u,),e_ops,(temp,)))
        FunctionCallingCallback(savefunc; funcat=ts)
    end

    cb1 = ContinuousCallback(condition,affect!,save_positions=(false,false))
    prob = ODEProblem(minus_im_Heff_mcsolve!,psi0,(ts[1],ts[end]),ps)
    for itraj in 1:ntrajs
        t1 = time()
        deltap = rand()
        cb2 = get_callback(ts,results,e_ops,temp)
        cb = CallbackSet(cb1,cb2);
        t2 = time()
        sol = solve(prob, alg; save_start=false, save_end=false, save_everystep=false, callback=cb, maxiters=1e18, kwargs...)
        results_all .+= results
        if verbose
            println("Performing : ", itraj, " , out of ntrajs: ", ntrajs , "  Time for traj  : ", time()-t1, " time for setup : ",t2-t1)
        end
    end
    return results_all .* inv(ntrajs)
end
