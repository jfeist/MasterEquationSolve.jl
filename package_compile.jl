using PackageCompiler
using Pkg

Pkg.activate(".")
create_sysimage(:MasterEquationSolve,sysimage_path="mesolve_sysimage.so",precompile_execution_file="precompile.jl")
