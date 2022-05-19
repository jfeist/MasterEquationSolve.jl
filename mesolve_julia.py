import os, contextlib, subprocess
from tempfile import mkdtemp
from glob import glob
from qutip import qsave, qload
import numpy as np

MESOLVE_DIR = os.path.dirname(os.path.abspath(__file__))
JOBFILE = "mesolve_julia.job"
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
    def __init__(self, H, ρ0, ts, c_ops, e_ops="probabilities", dir=None):
        self.data = (H,c_ops,ρ0,ts,e_ops)
        if dir is None:
            self.dir = mkdtemp(prefix="mesolve_julia_",dir=".")
            qsave(self.data,os.path.join(self.dir,"input"))
            with open(os.path.join(self.dir,JOBFILE),"w") as f:
                f.write(JOBFILE_STR)
            out = subprocess.run(["sbatch",JOBFILE], cwd=self.dir, check=True, capture_output=True, text=True)
            self.jobid = out.stdout.strip().replace("Submitted batch job ","")
            print(out.stdout.strip(),"with data in",self.dir)
        else:
            self.dir = dir if dir.startswith("mesolve_julia_") else "mesolve_julia_"+dir
            # [] to take the only element this should have - raises error if more than one file found
            [outfile] = glob(os.path.join(self.dir,"slurm-*.out"))
            name, ext = os.path.splitext(os.path.basename(outfile))
            self.jobid = name.replace('slurm-','')
    def load_results(self):
        resfis = sorted(glob(os.path.join(self.dir,"[0-9]*.qu")))
        # these "with" statements suppress unnecessary "loaded xy" statements from qload
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull):
                tPs = [qload(f.replace(".qu","")) for f in resfis]
        self.times = np.array([t for (t,P) in tPs])
        self.Ps = np.array([P for (t,P) in tPs])
