import os
from typing import Tuple, Optional, Dict, Any
from pathlib import Path
import shutil

import pandas as pd
from iMiner.cmd import run_command, find_executable, ExecutableNotFoundError, set_directory
from iMiner.md.common import parse_index_file, mk_index_file
from iMiner.md.gbsa.parameters import generate_input_file, DEFAULT_PARAMS
from iMiner.log import init_logger
from iMiner.utils import file_abspath

LOGGER = init_logger("iMiner.log")

try:
    MPIRUN_EXEC = find_executable("mpirun")
except ExecutableNotFoundError as e:
    MPIRUN_EXEC = None


def preprocess_index_file(
    f_ndx_ori: os.PathLike,
    f_ndx_new: os.PathLike,
    overwrite: bool = True,
    ligand_name: str = "MOL"
) -> Tuple[int, int]:
    """
    Pre-process index file
    
    Combine protein with ions except [Na+] and [Cl-] and get group id of 
    receptor and ligand
    
    Parameters
    ----------
    f_ndx_ori: os.PathLike
        Path to original index file
    f_ndx_new: os.PathLike
        Path to pre-processed index file
    overwrite: bool
        Whether to overwrite index file, default True
    ligand_name: str
        Group name of ligand, default "MOL"
    
    Return
    ------
    (r_grp_idx, l_grp_idx): Tuple[int, int]
        Group index of receptor and ligand
    """
    receptor_name = 'Receptor_GBSA'
    group_dict = parse_index_file(f_ndx_ori)
    group_dict[receptor_name] = group_dict['Protein'].copy()
    for ndx in group_dict.get('Ion', []):
        if ndx in group_dict.get('NA', []):
            continue
        if ndx in group_dict.get('CL', []):
            continue
        group_dict[receptor_name].append(ndx)
    
    mk_index_file(group_dict, f_ndx_new, overwrite)
    keys = list(group_dict.keys())
    r_grp_idx = keys.index(receptor_name)
    l_grp_idx = keys.index(ligand_name)
    return r_grp_idx, l_grp_idx

    
class GBSA:
    def __init__(self, workdir: os.PathLike, use_mpi: bool = True, num_threads: int = 1):
        """
        Initialize a GBSA calculation workflow

        workdir: os.PathLike
            Working directory
        use_mpi: bool
            Whether to use MPI, default True
        num_threads: int
            Number of cores used in mpirun, default 1
        """
        workdir = Path(workdir).resolve()
        workdir.mkdir(exist_ok=True)
        self.workdir = workdir
        self.use_mpi = use_mpi
        self.mpirun_exec = None

        self.params = {
            "gmx_mmpbsa_exec": find_executable("gmx_MMPBSA"),
            "o_file": self.workdir / "FINAL_RESULTS_MMPBSA.dat",
            "do_file": self.workdir / "FINAL_DECOMP_MMPBSA.dat",
            "eo_file": self.workdir / "FINAL_EO_MMPBSA.csv",
            "deo_file": self.workdir / "FINAL_DEL_MMPBSA.csv",
            "num_thread": num_threads
        }
        
        if self.use_mpi:
            try:
                self.params['mpi_exec'] = find_executable("mpirun")
            except ExecutableNotFoundError as e:
                self.use_mpi = False
                LOGGER.warning("mpirun is not found. MMPB/GBSA calculation will not run with MPI.")
                
    def set_params(
        self,
        complex_file: os.PathLike, 
        traj_file: os.PathLike, 
        topol_file: os.PathLike, 
        index_file: os.PathLike, 
        mmpbsa_params: Dict[str, Any] = DEFAULT_PARAMS, 
        mmpbsa_in_file: Optional[os.PathLike] = None
    ):
        """
        Set GBSA Parameters
        
        Parameters
        ----------
        complex_file: os.PathLike
            Path to complex strcture file, argument '-cs' in gmx_MMPBSA  
        traj_file: os.PathLike
            Path to MD trajectory file, argument '-ct' in gmx_MMPBSA
        topol_file: os.PathLike
            Path to complex topology, argument '-cp' in gmx_MMPBSA
        index_file: os.PathLike
            Path to index file, argument '-ci' in gmx_MMPBSA
        mmpbsa_params: Dict[str, Any]
            Parameters to control gmx_MMPBSA calculation. Default is `iMiner.md.gbsa.parameters.DEFAULT_PARAMS`
            See more, please refer to https://valdes-tresanco-ms.github.io/gmx_MMPBSA/v1.5.7/input_file/  
        mmpbsa_in_file: os.PathLike or None
            Path to gmx_MMPBSA input file, default None
            If None, iMiner will use default parameters and parameters specified in `mmpbsa_params`
        """
        ori_index_file = file_abspath(index_file)
        new_index_file = self.workdir / "index_gbsa.ndx"
        r_grp_idx, l_grp_idx = preprocess_index_file(ori_index_file, new_index_file)
        in_file = self.workdir / "mmpbsa.in"
        if mmpbsa_in_file:
            shutil.copyfile(
                file_abspath(mmpbsa_in_file),
                in_file
            )
        else:
            generate_input_file(mmpbsa_params, in_file)

        self.params.update({
            "mmpbsa_in_file": in_file,
            "complex_file": file_abspath(complex_file),
            "traj_file": file_abspath(traj_file),
            "topol_file": file_abspath(topol_file),
            "index_file": new_index_file,
            "r_grp_idx": r_grp_idx,
            "l_grp_idx": l_grp_idx,
        })
        
    def run(self, clean: bool = True):
        """
        Run gmx_MMPBSA program

        Parameters
        ----------
        clean: bool
            Whether to clean the working directory
        """
        LOGGER.log(f"Working directory set to: {self.workdir}")
        
        # Set up commands
        cmd = (
            "{gmx_mmpbsa_exec} MPI "
            "-i {mmpbsa_in_file} -cs {complex_file} -ci {index_file} -ct {traj_file} -cp {topol_file} -cg {r_grp_idx} {l_grp_idx} "
            "-o {o_file} -do {do_file} -nogui" 
        )
        # if decomp:
        #     cmd += " -eo {eo_file} -deo {deo_file}"
        if self.use_mpi:
            cmd = "{mpi_exec} --use-hwthread-cpus --allow-run-as-root -np {num_thread} " + cmd
        
        # start
        cmd = cmd.format(**self.params)
        LOGGER.log(f"Start MMPB/GBSA calculation with command: {cmd}")
        with set_directory(self.workdir):
            code, out, err = run_command(cmd)
        with open(self.workdir / 'mmpbsa.log', 'w') as f:
            f.write(out)
        LOGGER.log("MMPB/GBSA calculation finished.")
        
        # analyze result
        LOGGER.log(f"Parsing results to {self.workdir / 'Energy.csv'}")
        self.analyze_results()
        LOGGER.log(f"The average binding affinity is {self.delta_G:.4f} kcal/mol. ({self.result_df.shape[0]} frames evaulated)")
        
        # clean
        if clean:
            run_command(f"{self.params['gmx_mmpbsa_exec']} --clean")

    def analyze_results(self) -> float:
        """
        Analyze MMPB/GBSA calculation results

        Return
        ------
        delta_G: float
            Average binding free energy, in kcal/mol
        """
        with set_directory(self.workdir):
            run_command([
                find_executable("mmxsaparse"),
                "-i"
                "COMPACT_MMXSA_RESULTS.mmxsa",
                "-o",
                "."
            ])
        self.result_df = pd.read_csv(str(self.workdir / "Energy.csv"))
        self.delta_G = float(self.result_df['TOTAL'].mean())
        return self.delta_G