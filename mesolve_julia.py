import os, contextlib, subprocess
from tempfile import mkdtemp
from glob import glob
from qutip import Qobj, qsave, qload
import numpy as np

MESOLVE_DIR = os.path.dirname(os.path.abspath(__file__))
JOBFILE = "mesolve_julia.job"
JOBIDFILE = "jobid.dat"
JOBFILE_STR = f"""\
#!/usr/bin/bash -l
#SBATCH -J julia_mesolve
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --time=0
#SBATCH -n 3
#SBATCH -N 1

module load julia/1.7

echo "$(date '+%Y-%m-%d %H:%M:%S'): running mesolve_julia"

julia --project="{MESOLVE_DIR}" -J "{MESOLVE_DIR}/mesolve_sysimage_julia17.so" -e "using MasterEquationSolve: julia_main; julia_main()" "input" "."

echo "$(date '+%Y-%m-%d %H:%M:%S'): finished"
"""

class mesolve_julia:
    """mesolve_julia is an object that allows solution of master
    equation using the https://github.com/jfeist/MasterEquationSolve.jl julia
    library, which in particular makes use of the GPU to run significantly
    faster than on CPUs. There are two ways to create a mesolve_julia object:

        1. sol = mesolve_julia(H, ρ0, ts, c_ops, e_ops="probabilities") to start
           a new simulation. The input parameters are the same as for mesolve
           from qutip, with the only difference that e_ops can be either a list
           of operators (for which expectation values are then calculated) or a
           string, either "probabilities" or "full". For e_ops="probabilities"
           (the default), only the diagonal elements of the density matrix are
           saved, for "full" it is the full density matrix. So you need to use
           e_ops="full" to get the default behavior of qutip.

        2. sol = mesolve_julia(directory), where directory is a string, to load
           an existing simulation that has run (or is running) in directory.

    When submitting a new simulation, a random name is generated and saved in
    the property sol.dir. The directory names are of the form
    "mesolve_julia_XXXXXXXX", where XXXXXXXX are random characters. In order to
    later access the simulation again, it can be worthwhile to create some
    "database" (manually or automatically) mapping the relevant information
    (input parameters etc) to the random strings. Note that
    mesolve_julia(directory) can be called with
    directory="mesolve_julia_XXXXXXXX" or just directory="XXXXXXXX", in which
    case "mesolve_dir_" is added automatically.

    Note that in order to access the simulation results, you have to call
    sol.load_results(). They are then available in sol.times and sol.Ps. Note
    that the saved results are always available through sol.Ps regardless of the
    value of e_ops (i.e., the name does not change even if full density matrices
    or expectation values are saved).
    """
    def __init__(self, H_or_dir, ρ0=None, ts=None, c_ops=None, e_ops="probabilities"):
        if isinstance(H_or_dir,str):
            dir = H_or_dir
            self.dir = dir if dir.startswith("mesolve_julia_") or dir.startswith("./mesolve_julia_") else "./mesolve_julia_"+dir
            # this "with" statement suppresses unnecessary "loaded xy" statements from qload
            with open(os.devnull, 'w') as devnull:
                with contextlib.redirect_stdout(devnull):
                    self.data = qload(os.path.join(self.dir,"input"))
            # [] to take the only element this should have - raises error if more than one file found
            with open(os.path.join(self.dir,JOBIDFILE),"r") as f:
                self.jobid = f.read()
        elif isinstance(H_or_dir,Qobj):
            if ρ0 is None or ts is None or c_ops is None:
                raise ValueError(f"H_or_dir is a Qobj, but one of ρ0, ts, c_ops is None. They must have a value. Got ρ0 = {ρ0}, ts = {ts}, c_ops = {c_ops}.")
            self.data = (H_or_dir,c_ops,ρ0,ts,e_ops)
            self.dir = mkdtemp(prefix="mesolve_julia_",dir=".")
            qsave(self.data,os.path.join(self.dir,"input"))
            with open(os.path.join(self.dir,JOBFILE),"w") as f:
                f.write(JOBFILE_STR)
            out = subprocess.run(["sbatch",JOBFILE], cwd=self.dir, check=True, capture_output=True, text=True)
            self.jobid = out.stdout.strip().replace("Submitted batch job ","")
            with open(os.path.join(self.dir,JOBIDFILE),"w") as f:
                f.write(self.jobid)
            print(out.stdout.strip(),"with data in",self.dir)
        else:
            raise ValueError(f"H_or_dir must be either a string or a Qobj, got: {H_or_dir}")
            
    def load_results(self):
        resfis = sorted(glob(os.path.join(self.dir,"[0-9]*.qu")))
        # these "with" statements suppress unnecessary "loaded xy" statements from qload
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull):
                tPs = [qload(f.replace(".qu","")) for f in resfis]
        self.times = np.array([t for (t,P) in tPs])
        self.Ps = np.array([P for (t,P) in tPs])
