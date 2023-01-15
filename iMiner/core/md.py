"""
Author: Eric Wang
Data Created: 10/24/2022

This package contains class for running molecular dynamics project
"""
import os
import shutil
from pathlib import Path
from typing import Optional, Dict, Any
import json

from iMiner.cmd import run_command, set_directory, find_executable, CommandExecuteError
from iMiner.core.project import BaseProject
from iMiner.md.prep.ligand import run_acpype
from iMiner.md.prep.protein import run_tleap
from iMiner.md.prep.complex import make_complex
from iMiner.md.runner.gromacs import run_preprocess_workflow, run_md_workflow
from iMiner.log import init_logger

LOGGER = init_logger("iMiner.log")

def log_step(n: int, msg: str):
    LOGGER.info(f"===== Step {n}: {msg.capitalize()} =====")


class MDProject(BaseProject):
    def __init__(self, project_name:str, project_path: Optional[os.PathLike] = None, engine: str = "gromacs") -> None:
        '''
        Initialize a MD project with a project name and a project path

        When project_path is None, the project will be initialized in the current working directory,
        using the project_name as the project folder name

        :param engine: molecular dynamics engine, only support "gromacs" currently
        :type engine: str
        '''
        super().__init__(project_name, project_path)
        self.md_path = Path(self.project_path) / "md"
        self.md_path.mkdir(exist_ok=True, parents=True)
        assert engine in ['gromacs'], f"Not supported MD engine: {engine}"
        self.params = {
            "md": {
                "enforce_gpu": True,
                "em": {},
                "nvt": {},
                "npt": {},
                "prod": {},
                "interaction_analysis": {
                    "dt": 200, # ps
                    "use_mpi": True,
                    "chuncksize": 1,
                    "gen_short_dt": 1000
                }
            }
        }
    
    @property
    def md_params(self) -> Dict[str, Any]:
        return self.params['md']
    
    def read_params_json(self, fname: os.PathLike):
        with open(fname, 'r') as f:
            self.params.update(json.load(f))

    def parametrize_ligand(self, name: str, wdir: Path, **kwargs):
        """
        Parametrize ligand
        """
        prep_path = wdir.resolve() / "ligand"
        # delete all existing acpype files, otherwise acpype reuses old files
        if prep_path.exists() and prep_path.is_dir():
            shutil.rmtree(prep_path) 
        prep_path.mkdir(parents=True)
        shutil.copyfile(self.get_ligand_with_name(name), prep_path / "ligand.sdf")
        with set_directory(prep_path):
            obabel = find_executable(['obabel'])
            try:
                run_acpype("ligand.sdf", **kwargs)
            except CommandExecuteError:
                LOGGER.error(f"Error in preparing ligand {self.get_ligand_with_name(name)}. See details in acpype log file.")
                return False
            try:
                run_command([obabel, 'ligand.sdf', '-O', 'MOL.gro'])
            except CommandExecuteError:
                LOGGER.error(f"Error in converting ligand {self.get_ligand_with_name(name)} with obabel.")
                return False
        return True
    
    def parametrize_protein(self, name: str, wdir: Path):
        """
        Parametrize protein
        """
        prep_path = wdir.resolve() / "protein"
        prep_path.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.get_protein_with_name(name), prep_path / "protein.pdb")
        with set_directory(prep_path):
            try:
                run_tleap("protein.pdb")
            except CommandExecuteError:
                LOGGER.error(f"Error in preparing protein {self.get_protein_with_name(name)}. See details in tleap log file.")
                return False
            try:
                run_acpype(args=["-p", "protein.prmtop", "-x", "protein.inpcrd"])
            except CommandExecuteError:
                LOGGER.error(f"Error in preparing protein {self.get_protein_with_name(name)}. See details in acpype log file.")
                return False
        return True
    
    def make_complex(self, wdir: Path):
        """
        Make protein-ligand complex
        TODO: error handlings
        """
        prep_path = wdir.resolve() / "complex"
        prep_path.mkdir(parents=True, exist_ok=True)
        make_complex(
            wdir.resolve() / "protein" / "protein.amb2gmx" / "protein_GMX.top",
            wdir.resolve() / "ligand" / "MOL.acpype" / "MOL_GMX.itp",
            wdir.resolve() / "protein" / "protein.amb2gmx" / "protein_GMX.gro",
            wdir.resolve() / "ligand" / "MOL.gro",
            prep_path
        )
    
    def analyze_md_traj(self, wdir: Path, lig_name: str):
        """
        Analyze MD trajectories: calculating RMSD and do interaction analysis
        """
        from iMiner.md.common import preprocess_index_file
        from iMiner.md.analysis import (
            analyze_multiple_frames,
            plot_interact,
            plot_rmsd,
            read_xvg,
            gmx_rms,
            gmx_extract_and_align_traj,
            xtc_to_pdb,
            gmx_genidx
        )
            
        prod_dir = wdir.resolve() / "prod"
        traj_file = prod_dir / "prod.xtc"
        ref_tpr = prod_dir / "prod.tpr"
        ref_gro = prod_dir / "prod_align.gro"
        index_file = prod_dir / "index.ndx"
        traj_nopbc_file = prod_dir / "prod_align.xtc"
        with set_directory(prod_dir):
            # generate index file
            gmx_genidx("prod.gro", "index.ndx")
            r_grp_idx, l_grp_idx = preprocess_index_file(index_file, index_file)
            LOGGER.info(f"Index file generated: {index_file}")
            LOGGER.info(f"Receptor group index: {r_grp_idx}")
            LOGGER.info(f"Ligand group index: {l_grp_idx}")

            # post-process traj file
            LOGGER.info("Post-process MD trajectory")
            gmx_extract_and_align_traj(
                ref_tpr,
                traj_file,
                index_file,
                traj_nopbc_file,
                center_grp = l_grp_idx,
                align_grp = r_grp_idx,
                output_grp = r_grp_idx,
                gen_short_dt = self.md_params['interaction_analysis']['gen_short_dt']
            )
            LOGGER.info(f"Dry MD traj with pbc fixed: {traj_nopbc_file}")
        
            # rmsd
            f_xvg = prod_dir / "prod_rmsd.xvg"
            gmx_rms(
                prod_dir / "prod_align.gro",
                traj_nopbc_file,
                f_xvg,
                index_file,
            )
            tlist, rmslist = read_xvg(f_xvg, tuni='ns', dunit='A')
            fig, ax = plot_rmsd(tlist, rmslist, name=lig_name)
            fig.savefig("prod_rmsd.png", dpi=300)
            LOGGER.info(f"RMSD Calculated: {prod_dir / 'prod_rmsd.png'}")

            # analyze interaction
            LOGGER.info("Analyze interaction...")
            trajdir = Path(traj_file).parent / "traj"
            f_csv = "interaction.csv"
            if not trajdir.is_dir():
                trajdir.mkdir(exist_ok=True)
                pdbs = xtc_to_pdb(
                    ref_gro, traj_nopbc_file, trajdir, 
                    self.md_params['interaction_analysis']['dt']
                )
                LOGGER.info(f"Convert trajectory to seperate pdb files: {trajdir}")
            else:
                pdbs = list(trajdir.glob("*.pdb"))
            df = analyze_multiple_frames(
                pdbs,
                f_csv,
                add_hydrogen=False,
                resnr_renum=None,
                use_mpi=self.md_params['interaction_analysis']['use_mpi'], 
                chunksize=self.md_params['interaction_analysis']['chunksize']
            )
            fig, ax = plot_interact(f_csv, title=lig_name)
            fig.savefig("interaction.png")
            LOGGER.info(f"Intearction analysis result: {prod_dir / 'interaction.png'}")
        
        return

    def prep_md(self, wdir: Path):
        """
        Run MD preparation workflow
        """
        complex_dir = wdir.resolve() / "complex"
        try:
            run_preprocess_workflow("topol.top", "complex.gro", complex_dir, verbose=True)
        except CommandExecuteError:
            LOGGER.info("Error in gromacs prep steps. See complex folder.")
            return False
        shutil.copyfile(complex_dir / "ions.gro", wdir / "ions.gro")
        shutil.copyfile(complex_dir / "processed.top", wdir / 'processed.top')
        return True
    
    def run_md(self, wdir: Path):
        """
        Run molecular dynamics workflow
        """
        try:
            run_md_workflow(
            "processed.top", 
            "ions.gro", 
            wdir, 
            restart=True, 
            params=self.md_params, 
            enforce_gpu=self.md_params['enforce_gpu'],
            verbose=True
            )
        except CommandExecuteError:
            LOGGER.error("Error in running gromacs. See complex folder.")
            return False
        return True
    
    def run(self, lig_name: str, prot_name: str, task_name: Optional[str] = None, 
            lig_charge = "guess"):
        """
        Run iMiner Molecular Dynamics Workflow
        
        Returns pre-md topology and md production run trajectory, to be used as trajectory
        rmsd calculation inputs
        """
        task_name = f"{lig_name}_{prot_name}" if task_name is None else task_name
        LOGGER.info(f"running md for {lig_name}_{prot_name}")
        wdir = self.md_path / task_name
        
        log_step(1, "Parametrize Ligand")
        succ = self.parametrize_ligand(lig_name, wdir, net_charge=lig_charge)
        if not succ:
            return
        log_step(2, "Parametrize Protein")
        succ = self.parametrize_protein(prot_name, wdir)
        if not succ:
            return
        log_step(3, "Make Complex")
        self.make_complex(wdir)
        log_step(4, "MD Preparation")
        succ = self.prep_md(wdir)
        if not succ:
            return
        log_step(5, "Run MD")
        succ = self.run_md(wdir)
        if not succ:
            return
        log_step(6, "Post MD Analysis")
        self.analyze_md_traj(wdir, lig_name)
        complex_dir = wdir.resolve() / "complex"
        prod_dir = wdir.resolve() / "prod"
        return complex_dir / "ions.tpr", prod_dir/ "prod.xtc" 
