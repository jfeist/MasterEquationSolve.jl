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
        return getρ ? A.full() : pysparse_to_julia(A.data)
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
    if length(data) == 4
        H_q, c_ops_q, ρ0_q, ts = data
        saveddata = :probabilities
    elseif length(data) == 5
        H_q, c_ops_q, ρ0_q, ts, e_ops_q = data
        saveddata = e_ops_q isa String ? Symbol(e_ops_q) : qobj_to_jl.(e_ops_q)
    else
        throw(ValueError("length(data) must be 4 or 5, got length(data) = $(length(data))."))
    end
    H = qobj_to_jl(H_q)
    J = qobj_to_jl.(c_ops_q)
    ρ0 = qobj_to_jl(ρ0_q,true)
    (H,J,ρ0,ts), saveddata
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
