function julia_main()::Cint
    try
        real_main()
    catch
        Base.invokelatest(Base.display_error, Base.catch_stack())
        return 1
    end
    return 0
end

function real_main()
    # TODO: "real" argument handling
    infile = ARGS[1]
    outdir = ARGS[2]
    mesolve_qutip(infile, outdir; backend=:CUDA)
end
