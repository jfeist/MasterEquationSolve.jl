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

function load_input_from_qutip(filename)
    H_q, c_ops_q, ρ0_q, ts = qt.qload(filename)
    H = qobj_to_jl(H_q)
    J = qobj_to_jl.(c_ops_q)
    ρ0 = qobj_to_jl(ρ0_q,true)
    (H,J,ρ0,ts)
end
