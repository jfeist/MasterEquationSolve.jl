module MasterEquationSolve

using PyCall
function __init__()
    global qt = pyimport("qutip")
    global sp = pyimport("scipy.sparse")
end

include("qutip_interface.jl")
include("liouvillian.jl")
include("mesolve.jl")
include("main.jl")

end

# this has to be outside the module so __init__() has been called already
if abspath(PROGRAM_FILE) == @__FILE__
    MasterEquationSolve.julia_main()
end
