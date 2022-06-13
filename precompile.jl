using MasterEquationSolve
using CUDA

function dorun(func,suff)
    rm("precompile_outdir",force=true,recursive=true)
    CUDA.@time func("precompile_input$(suff)","precompile_outdir")
end

dorun(mesolve_qutip,"")
dorun(mesolve_qutip,"_eops")
dorun(sesolve_qutip,"_se")
dorun(sesolve_qutip,"_se_eops")
