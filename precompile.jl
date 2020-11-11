using MasterEquationSolve

rm("precompile_outdir",force=true,recursive=true)

MasterEquationSolve.mesolve_qutip("precompile_input","precompile_outdir")
