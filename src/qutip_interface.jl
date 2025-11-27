function pysparse_to_julia(A)
    A = sp.csc_matrix(A)
    m, n = A.shape
    colPtr = Int[i+1 for i in PyArray(A."indptr")]
    rowVal = Int[i+1 for i in PyArray(A."indices")]
    nzVal = A.data
    SparseMatrixCSC{eltype(nzVal),Int}(m, n, colPtr, rowVal, nzVal)
end

function qobj_to_jl(A,getρ=false)
    if A."type" == "oper"
        return getρ ? A.full() : pysparse_to_julia(A.data.as_scipy())
    elseif A."type" == "ket"
        ψ = reshape(A.full(),:)
        return getρ ? ψ * ψ' : ψ
    else
        error("Unkown type $(A."type") of Qobj!")
    end
end

function myqload(file,silent=true)
    # redirect python stdout to avoid "Loaded tuple object." message from qload
    if silent
        tmp = nothing # because of scoping rules, tmp has to be defined before assignment
        @pywith pyimport("contextlib").redirect_stdout(nothing) begin
            tmp = qt.qload(file)
        end
        tmp
    else
        qt.qload(file)
    end
end

function load_input_from_qutip(filename)
    data = myqload(filename)
    H_q = pop!(data,"H")
    c_ops_q = pop!(data,"c_ops")
    ρ0_q = pop!(data,"ρ0")
    ts = pop!(data,"ts")
    e_ops_q = pop!(data,"e_ops")
    reltol = pop!(data,"reltol",1e-10)
    abstol = pop!(data,"abstol",1e-12)
    c_ops_offdiag_q = pop!(data,"c_ops_offdiag",nothing)
    @assert isempty(data)

    saveddata = e_ops_q isa String ? Symbol(e_ops_q) : qobj_to_jl.(e_ops_q)
    H = qobj_to_jl(H_q)
    J = qobj_to_jl.(c_ops_q)
    ρ0 = qobj_to_jl(ρ0_q,true)
    J_offdiag = if c_ops_offdiag_q === nothing
        nothing
    else
        length(c_ops_offdiag_q)==2 || error("length of c_ops_offdiag must be 2, got $(length(c_ops_offdiag_q))")
        γ = collect(c_ops_offdiag_q[1])
        @show γ
        Fs = qobj_to_jl.(c_ops_offdiag_q[2])
        @show Fs
        @show size(γ), size(Fs)
        @assert size(γ) == (size(Fs)..., size(Fs)...)
        (γ, Fs)
    end
    (H,J,ρ0,ts,J_offdiag), saveddata, reltol, abstol
end

function load_se_input_from_qutip(filename)
    data = myqload(filename)
    if length(data)==4
        H_q, ψ0_q, ts, e_ops_q = data
        saveddata = e_ops_q isa String ? Symbol(e_ops_q) : qobj_to_jl.(e_ops_q)
    else
        throw(ValueError("Expected 4 objects in qutip input file $filename, got $(length(data))."))
    end
    H = qobj_to_jl(H_q)
    ψ0 = qobj_to_jl(ψ0_q)
    (H,ψ0,ts), saveddata
end
