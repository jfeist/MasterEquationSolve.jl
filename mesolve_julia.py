import os, contextlib, subprocess
from pathlib import Path
from tempfile import mkdtemp
from qutip import Qobj, qsave, qload
import numpy as np

MESOLVE_DIR = Path(__file__).resolve().parent
JOBFILE = "mesolve_julia.job"
JOBIDFILE = "jobid.dat"
JOBFILE_STR = f"""\
#!/usr/bin/bash -l
#SBATCH -J {{solvefun}}_julia
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --time=0
#SBATCH -n 3
#SBATCH -N 1

module load julia/1.8

echo "$(date '+%Y-%m-%d %H:%M:%S'): running {{solvefun}}_julia"

julia --project="{MESOLVE_DIR}" -J "{MESOLVE_DIR}/mesolve_sysimage_julia18.so" -e 'using MasterEquationSolve; {{solvefun}}_qutip("input",".")'

echo "$(date '+%Y-%m-%d %H:%M:%S'): finished"
"""

class mesolve_julia:
    """mesolve_julia is an object that allows solution of master
    equations using the https://github.com/jfeist/MasterEquationSolve.jl julia
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
    def __init__(self, H_or_dir, ρ0=None, ts=None, c_ops=None, e_ops="probabilities", **kwargs):
        if isinstance(H_or_dir,str) or isinstance(H_or_dir,Path):
            dir = Path(H_or_dir)
            self.dir = dir if dir.name.startswith("mesolve_julia_") else Path("mesolve_julia_"+str(dir))
            # these "with" statements suppress unnecessary "loaded xy" statements from qload
            with open(os.devnull, 'w') as devnull:
                with contextlib.redirect_stdout(devnull):
                    self.data = qload(str(self.dir/"input"))
            self.jobid = (self.dir / JOBIDFILE).read_text()
        elif isinstance(H_or_dir,Qobj):
            if ρ0 is None or ts is None or c_ops is None:
                raise ValueError(f"H_or_dir is a Qobj, but one of ρ0, ts, c_ops is None. They must have a value. Got ρ0 = {ρ0}, ts = {ts}, c_ops = {c_ops}.")
            self.data = dict(H=H_or_dir, c_ops=c_ops, ρ0=ρ0, ts=ts, e_ops=e_ops, **kwargs)
            self.dir = Path(mkdtemp(prefix="mesolve_julia_",dir="."))
            qsave(self.data, str(self.dir/"input"))
            (self.dir/JOBFILE).write_text(JOBFILE_STR.format(solvefun="mesolve"))
            out = subprocess.run(["sbatch",JOBFILE], cwd=self.dir, check=True, capture_output=True, text=True)
            self.jobid = out.stdout.strip().replace("Submitted batch job ","")
            (self.dir/JOBIDFILE).write_text(self.jobid)
            print(out.stdout.strip(),"with data in",self.dir)
        else:
            raise ValueError(f"H_or_dir must be either a string or a Qobj, got: {H_or_dir}")

    def load_results(self):
        resfis = sorted(self.dir.glob("[0-9]*.qu"))
        # these "with" statements suppress unnecessary "loaded xy" statements from qload
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull):
                tPs = [qload(str(f).replace(".qu","")) for f in resfis]
        self.times = np.array([t for (t,P) in tPs])
        self.Ps = np.array([P for (t,P) in tPs])

class sesolve_julia:
    """sesolve_julia is an object that allows solution of Schrödinger
    equations using the https://github.com/jfeist/MasterEquationSolve.jl julia
    library, which in particular makes use of the GPU to run significantly
    faster than on CPUs. There are two ways to create a sesolve_julia object:

        1. sol = sesolve_julia(H, ψ0, ts, e_ops="full") to start a new
           simulation. The input parameters are the same as for sesolve from
           qutip, with the only difference that e_ops can be either a list of
           operators (for which expectation values are then calculated) or the
           string "full". For e_ops="full" (the default), the full wave
           functions are saved.

        2. sol = sesolve_julia(directory), where directory is a string, to load
           an existing simulation that has run (or is running) in directory.

    When submitting a new simulation, a random directory is generated and saved
    in the property sol.dir. The directory names are of the form
    "sesolve_julia_XXXXXXXX", where XXXXXXXX are random characters. In order to
    later access the simulation again, it can be worthwhile to create some
    "database" (manually or automatically) mapping the relevant information
    (input parameters etc) to the random strings. Note that
    sesolve_julia(directory) can be called with
    directory="sesolve_julia_XXXXXXXX" or just directory="XXXXXXXX", in which
    case "sesolve_dir_" is added automatically.

    Note that in order to access the simulation results, you have to call
    sol.load_results(). They are then available in sol.times and sol.values.
    Note that the saved results are always available through sol.values
    regardless of the value of e_ops (i.e., the name does not change no matter
    if wave functions or expectation values are saved).
    """
    def __init__(self, H_or_dir, ψ0=None, ts=None, e_ops="full"):
        if isinstance(H_or_dir,str) or isinstance(H_or_dir,Path):
            dir = Path(H_or_dir)
            self.dir = dir if dir.name.startswith("sesolve_julia_") else Path("sesolve_julia_"+str(dir))
            # these "with" statements suppress unnecessary "loaded xy" statements from qload
            with open(os.devnull, 'w') as devnull:
                with contextlib.redirect_stdout(devnull):
                    self.data = qload(str(self.dir/"input"))
            self.jobid = (self.dir / JOBIDFILE).read_text()
        elif isinstance(H_or_dir,Qobj):
            if ψ0 is None or ts is None:
                raise ValueError(f"H_or_dir is a Qobj, but one of ψ0 or ts is None. They must have a value. Got ψ0 = {ψ0}, ts = {ts}.")
            self.data = (H_or_dir,ψ0,ts,e_ops)
            self.dir = Path(mkdtemp(prefix="sesolve_julia_",dir="."))
            qsave(self.data, str(self.dir/"input"))
            (self.dir/JOBFILE).write_text(JOBFILE_STR.format(solvefun="sesolve"))
            out = subprocess.run(["sbatch",JOBFILE], cwd=self.dir, check=True, capture_output=True, text=True)
            self.jobid = out.stdout.strip().replace("Submitted batch job ","")
            (self.dir/JOBIDFILE).write_text(self.jobid)
            print(out.stdout.strip(),"with data in",self.dir)
        else:
            raise ValueError(f"H_or_dir must be either a string or a Qobj, got: {H_or_dir}")

    def load_results(self):
        resfis = sorted(self.dir.glob("[0-9]*.qu"))
        # these "with" statements suppress unnecessary "loaded xy" statements from qload
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull):
                tvs = [qload(str(f).replace(".qu","")) for f in resfis]
        self.times  = np.array([t for (t,v) in tvs])
        self.values = np.array([v for (t,v) in tvs])
