using MasterEquationSolve
using CUDA

rm("precompile_outdir",force=true,recursive=true)
CUDA.@time MasterEquationSolve.mesolve_qutip("precompile_input","precompile_outdir")

rm("precompile_outdir",force=true,recursive=true)
CUDA.@time MasterEquationSolve.mesolve_qutip("precompile_input_eops","precompile_outdir")
